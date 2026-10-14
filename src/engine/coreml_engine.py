import inspect
import os
import torch
import torch.nn as nn
import logging
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.models.controlnet import ControlNetOutput
from python_coreml_stable_diffusion.coreml_model import CoreMLModel
from PIL import Image

CONTROLNET_ID = "lllyasviel/sd-controlnet-scribble"
CONTROLNET_BACKENDS = ("coreml", "pytorch")
# One unit for every Core ML model: mixing units across the UNet and ControlNet made them
# contend for the ANE/GPU (54-148 s renders). CPU_AND_NE was fastest and ~1.1 GB lighter
# than ALL on an 8 GB M2 (see CHG-20261014-01, E15).
COMPUTE_UNITS = ("CPU_AND_NE", "ALL", "CPU_AND_GPU", "CPU_ONLY")
NUM_DOWN_RESIDUALS = 12  # additional_residual_0..11 = down blocks, additional_residual_12 = mid block


# --- SHARED CORE ML I/O HELPERS (used by the UNet and ControlNet wrappers) ---
def safe_predict(coreml_wrapper, inputs):
    if hasattr(coreml_wrapper, "predict"):
        return coreml_wrapper.predict(inputs)
    if hasattr(coreml_wrapper, "model") and hasattr(coreml_wrapper.model, "predict"):
        return coreml_wrapper.model.predict(inputs)
    raise AttributeError(f"Could not find .predict() on {type(coreml_wrapper)}")


def sanitize_input(val, target_meta, do_pad):
    target_shape = target_meta["shape"]
    target_dtype_str = target_meta["dtype"]
    target_dtype = np.float16 if "float16" in str(target_dtype_str) else np.float32

    # 1. Scalar Expansion
    if torch.is_tensor(val) and val.ndim == 0: val = val.unsqueeze(0)
    if isinstance(val, np.ndarray) and val.ndim == 0: val = np.expand_dims(val, 0)

    # 2. Batch Padding
    if do_pad and target_shape[0] == 2:
        if torch.is_tensor(val) and val.shape[0] == 1: val = torch.cat([val, val])
        elif isinstance(val, np.ndarray) and val.shape[0] == 1: val = np.concatenate([val, val])

    # 3. Torch -> Numpy
    if torch.is_tensor(val): val = val.detach().cpu().numpy()

    # 4. Rank Expansion & Transpose
    current_rank = len(val.shape)
    target_rank = len(target_shape)

    if current_rank < target_rank:
        diff = target_rank - current_rank
        for _ in range(diff):
            val = np.expand_dims(val, axis=-1)

    if len(val.shape) == 4 and val.shape[1] == 77 and val.shape[2] == 768:
        if target_shape[1] == 768 and target_shape[3] == 77:
             val = np.transpose(val, (0, 2, 3, 1))

    # 5. Dtype Casting
    if val.dtype != target_dtype: val = val.astype(target_dtype)

    return val


def prepare_inputs(model_meta, inputs_pool, do_pad):
    prepared = {}
    for k in model_meta:
        if k in inputs_pool:
            val = inputs_pool[k]
            prepared[k] = sanitize_input(val, model_meta[k], do_pad)
        else:
            meta = model_meta[k]
            prepared[k] = np.zeros(meta["shape"], dtype=np.float32)
    return prepared


def needs_batch_pad(coreml_model, sample):
    """The models are compiled for batch 2 (CFG); at guidance <= 1 the pipeline sends batch 1."""
    meta = coreml_model.expected_inputs.get("sample")
    return bool(meta) and meta["shape"][0] == 2 and sample.shape[0] == 1


class CoreMLControlNetWrapper(ControlNetModel):
    """
    Runs the compiled ControlNet.mlpackage on the ANE behind the ControlNetModel interface.

    StableDiffusionControlNetPipeline type-checks `isinstance(controlnet, ControlNetModel)`
    and reads `config.global_pool_conditions` / `dtype`, so this subclasses ControlNetModel
    but skips its __init__: only nn.Module state and the real config are set up, no weights.
    The Core ML graph returns unscaled residuals, so `conditioning_scale` is applied here
    exactly as ControlNetModel.forward does.
    """

    def __init__(self, coreml_model, config):
        nn.Module.__init__(self)
        # ControlNetModel's @register_to_config also records __init__ defaults for keys an
        # older config.json lacks (e.g. global_pool_conditions); mirror that here.
        full_config = {name: p.default for name, p in inspect.signature(ControlNetModel.__init__).parameters.items()
                       if name != "self" and p.default is not inspect.Parameter.empty}
        full_config.update({k: v for k, v in config.items() if not k.startswith("_")})
        self.register_to_config(**full_config)
        self.coreml_model = coreml_model

    @property
    def device(self): return torch.device("cpu")

    @property
    def dtype(self): return torch.float32

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond,
                conditioning_scale=1.0, guess_mode=False, return_dict=True, **kwargs):
        if guess_mode:
            raise NotImplementedError("guess_mode is not supported by the Core ML ControlNet")
        inputs_pool = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
        }
        do_pad = needs_batch_pad(self.coreml_model, sample)
        out = safe_predict(self.coreml_model,
                           prepare_inputs(self.coreml_model.expected_inputs, inputs_pool, do_pad))

        batch = sample.shape[0]
        residuals = [torch.from_numpy(np.asarray(out[f"additional_residual_{i}"][:batch], dtype=np.float32))
                     * conditioning_scale for i in range(NUM_DOWN_RESIDUALS + 1)]
        down_block_res_samples, mid_block_res_sample = residuals[:NUM_DOWN_RESIDUALS], residuals[-1]

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        return ControlNetOutput(down_block_res_samples=down_block_res_samples,
                                mid_block_res_sample=mid_block_res_sample)


class CoreMLEngine:
    def __init__(self, hf_repo_id="tekraj/sketch-render-coreml", model_dir="models/coreml",
                 controlnet_backend="coreml", compute_unit="CPU_AND_NE"):
        print(f" ** Initializing CoreML Engine (Frankenstein Mode)...")
        if controlnet_backend not in CONTROLNET_BACKENDS:
            raise ValueError(f"controlnet_backend {controlnet_backend!r} not in {CONTROLNET_BACKENDS}")
        if compute_unit not in COMPUTE_UNITS:
            raise ValueError(f"compute_unit {compute_unit!r} not in {COMPUTE_UNITS}")
        self.controlnet_backend = controlnet_backend
        self.compute_unit = compute_unit

        self._restore_apple_filenames(model_dir)

        prefix = "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_"
        text_enc_path = os.path.join(model_dir, f"{prefix}text_encoder.mlpackage")
        vae_dec_path = os.path.join(model_dir, f"{prefix}vae_decoder.mlpackage")
        unet_c1_path = os.path.join(model_dir, f"{prefix}unet_chunk1.mlpackage")
        unet_c2_path = os.path.join(model_dir, f"{prefix}unet_chunk2.mlpackage")
        control_path = os.path.join(model_dir, "ControlNet.mlpackage")

        # The Core ML ControlNet is built first and handed to the pipeline, so the
        # PyTorch fp32 ControlNet weights are never loaded on that path.
        if controlnet_backend == "coreml":
            if not os.path.exists(control_path):
                raise FileNotFoundError(
                    f"{control_path} not found. Compile it with `python -m src.engine.coreml_compiler`, "
                    f"or use controlnet_backend='pytorch'.")
            print(" ** Loading Core ML ControlNet (ANE)...")
            controlnet = CoreMLControlNetWrapper(CoreMLModel(control_path, compute_unit),
                                                 ControlNetModel.load_config(CONTROLNET_ID))
        else:
            print(" ** Loading PyTorch ControlNet (fp32, CPU)...")
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID)

        print(" ** Loading PyTorch config skeleton...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True
        ).to("cpu")

        # --- COMPONENT 1: TEXT ENCODER ---
        class CoreMLTextEncoderWrapper(nn.Module):
            def __init__(self, coreml_model):
                super().__init__()
                self.model = coreml_model
                self.config = type('Config', (), {'use_attention_mask': True})()
                self.register_buffer("dummy", torch.empty(0))
            @property
            def device(self): return torch.device("cpu")
            @property
            def dtype(self): return torch.float32
            def forward(self, input_ids, **kwargs):
                inputs = {"input_ids": input_ids.int().numpy()}
                res = safe_predict(self.model, inputs)
                last_hidden_state = None
                for k, v in res.items():
                    if len(v.shape) == 3:
                        last_hidden_state = torch.from_numpy(v)
                        break
                if last_hidden_state is None:
                    last_hidden_state = torch.from_numpy(list(res.values())[0])
                B, _, Dim = last_hidden_state.shape
                pooled_output = torch.from_numpy(res["pooled_output"]) if "pooled_output" in res else torch.zeros((B, Dim))
                return [last_hidden_state, pooled_output]

        if os.path.exists(text_enc_path):
            print(" ** Swapping Text Encoder...")
            self.pipe.text_encoder = CoreMLTextEncoderWrapper(CoreMLModel(text_enc_path, compute_unit))

        # --- COMPONENT 2: VAE DECODER ---
        class CoreMLVAEDecoderWrapper(nn.Module):
            def __init__(self, coreml_model):
                super().__init__()
                self.model = coreml_model
                self.config = type('Config', (), {'scaling_factor': 0.18215})() 
                self.register_buffer("dummy", torch.empty(0))
            @property
            def device(self): return torch.device("cpu")
            @property
            def dtype(self): return torch.float32
            
            # --- FIX: Added **kwargs to accept 'generator' argument ---
            def decode(self, z, return_dict=True, **kwargs):
                inputs = {"z": z.numpy()}
                res = safe_predict(self.model, inputs)
                
                # VAE Output Handling
                # Often returns 'image' or 'sample'. We grab the first key.
                first_val = list(res.values())[0]
                img = torch.from_numpy(first_val)
                
                from diffusers.models.autoencoders.vae import DecoderOutput
                return DecoderOutput(sample=img)

        if os.path.exists(vae_dec_path):
            print(" ** Swapping VAE Decoder...")
            self.pipe.vae.decode = CoreMLVAEDecoderWrapper(CoreMLModel(vae_dec_path, compute_unit)).decode

        # --- COMPONENT 3: UNET (CHUNKED) ---
        class CoreMLChunkedUNetWrapper(nn.Module):
            def __init__(self, chunk1, chunk2):
                super().__init__()
                self.chunk1 = chunk1
                self.chunk2 = chunk2
                self.register_buffer("dummy", torch.empty(0))
                self.config = type('Config', (), {
                    'sample_size': 64, 'in_channels': 4, 'time_cond_proj_dim': None,
                    'addition_embed_type': None, 'class_embed_type': None, 'encoder_hid_dim': None,
                    'encoder_hid_dim_type': None, 'cross_attention_dim': 768
                })()
            
            @property
            def device(self): return torch.device("cpu")
            @property
            def dtype(self): return torch.float32

            def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
                inputs_pool = {
                    "sample": sample,
                    "timestep": timestep,
                    "encoder_hidden_states": encoder_hidden_states
                }

                if "down_block_additional_residuals" in kwargs:
                    for i, res in enumerate(kwargs["down_block_additional_residuals"]):
                        inputs_pool[f"additional_residual_{i}"] = res
                if "mid_block_additional_residual" in kwargs:
                    inputs_pool["additional_residual_12"] = kwargs["mid_block_additional_residual"]

                do_pad = needs_batch_pad(self.chunk1, sample)

                chunk1_inputs = prepare_inputs(self.chunk1.expected_inputs, inputs_pool, do_pad)
                out1 = safe_predict(self.chunk1, chunk1_inputs)

                inputs_pool.update(out1)
                chunk2_inputs = prepare_inputs(self.chunk2.expected_inputs, inputs_pool, do_pad)
                res = safe_predict(self.chunk2, chunk2_inputs)
                
                final_out = list(res.values())[0]
                if do_pad and final_out.shape[0] == 2: final_out = final_out[:1]
                
                return torch.from_numpy(final_out)

        if os.path.exists(unet_c1_path) and os.path.exists(unet_c2_path):
            print("   ↳ Swapping UNet (Chunked)...")
            c1 = CoreMLModel(unet_c1_path, compute_unit)
            c2 = CoreMLModel(unet_c2_path, compute_unit)
            self.pipe.unet = CoreMLChunkedUNetWrapper(c1, c2)
        else:
             raise FileNotFoundError("UNet chunks missing!")

        print("  ✅ Pipeline Assembled Successfully.")
        print("  -> Switching to LCM Scheduler...")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config, beta_schedule="scaled_linear")

    def _restore_apple_filenames(self, model_dir):
        prefix = "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5_"
        mapping = {
            "TextEncoder.mlpackage": f"{prefix}text_encoder.mlpackage",
            "VAEDecoder.mlpackage": f"{prefix}vae_decoder.mlpackage",
            "VAEEncoder.mlpackage": f"{prefix}vae_encoder.mlpackage",
            "UnetChunk1.mlpackage": f"{prefix}unet_chunk1.mlpackage",
            "UnetChunk2.mlpackage": f"{prefix}unet_chunk2.mlpackage",
        }
        for clean, ugly in mapping.items():
            clean_path = os.path.join(model_dir, clean)
            ugly_path = os.path.join(model_dir, ugly)
            if os.path.exists(clean_path) and not os.path.exists(ugly_path):
                os.rename(clean_path, ugly_path)

    def generate(
        self, 
        prompt, 
        negative_prompt, 
        control_image, 
        steps=4, 
        guidance=1.0, 
        control_scale=0.8):
        if control_image.size != (512, 512): 
            control_image = control_image.resize((512, 512), Image.LANCZOS)
        return self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            image=control_image, 
            num_inference_steps=int(steps), 
            guidance_scale=float(guidance), 
            controlnet_conditioning_scale=float(control_scale)
        ).images[0]
