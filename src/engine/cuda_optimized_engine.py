import torch
import os
import time
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# --- ZeroGPU Wrapper ---
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

# --- GLOBAL HOOK FUNCTION (to be pickleable) ---
def vae_cast_hook(module, inputs):
    """
    Forces all inputs to the VAE to be Float32.
    This prevents the VAE from crashing when receiving Float16 latents.
    """
    if isinstance(inputs, tuple):
        return tuple(t.to(dtype=torch.float32) if isinstance(t, torch.Tensor) else t for t in inputs)
    return inputs

class CUDAOptimizedEngine:
    def __init__(self):
        print(" ** Initializing CUDA Optimized Engine (PyTorch 2.0)...")
        
        # 1. Force CUDA
        if not torch.cuda.is_available():
            raise RuntimeError(" !! CUDA not found! This engine requires an NVIDIA GPU.")
            
        self.device = "cuda"
        self.dtype = torch.float16 # Half-precision for UNet (Speed)
        
        print(f" ** Active GPU: {torch.cuda.get_device_name(0)}")

        # 2. Load Models
        print(" ** Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=self.dtype
        )
        
        print(" ** Loading Stable Diffusion...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        # ---------------------------------------------------------
        # 3. CRITICAL FIX: VAE MUST STAY ON CUDA WITH FLOAT32
        # ---------------------------------------------------------
        print(" ** Casting VAE to Float32 on CUDA...")
        # CRITICAL: Explicitly keep on CUDA device to prevent CPU migration
        self.pipe.vae = self.pipe.vae.to(device=self.device, dtype=torch.float32)

        print(" ** Registering VAE Type-Safety Hook...")
        # We use the global function 'vae_cast_hook' here
        self.pipe.vae.post_quant_conv.register_forward_pre_hook(vae_cast_hook)
        
        # Ensure ControlNet is also on CUDA
        self.pipe.controlnet = self.pipe.controlnet.to(device=self.device, dtype=self.dtype)

        # 4. Inject LCM (Speed)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        # NOTE: torch.compile is DISABLED for ZeroGPU compatibility
        # ZeroGPU has its own optimization and torch.compile causes issues
        print(" ✅ Engine Ready (ZeroGPU optimized)")
        
    @spaces.GPU(duration=60) 
    def generate(self, prompt, negative_prompt, control_image, steps=4, guidance=1.0, control_scale=1.0):
        # 1. Resize & Format
        if control_image.size != (512, 512):
            control_image = control_image.resize((512, 512), Image.LANCZOS)
        
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        # 2. Inference with PIL image (let pipeline handle preprocessing)
        # CRITICAL: Pass PIL image directly, pipeline's VaeImageProcessor handles conversion
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale)
        ).images[0]
        
        return result