import torch
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    LCMScheduler,
    AutoencoderTiny,
    AutoencoderKL
)
from PIL import Image
import yaml
import gc

class SketchToRenderEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        print("Initializing engine...")
        self.dtype = torch.float32 # Force Float32 for Mac stability
        
        # 1. Load ControlNet (Scribble)
        print("Loading ControlNet (Scribble)...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.config['pipeline']['controlnet_model'], 
            torch_dtype=self.dtype
        )
        
        # 2. Load TinyVAE (TAESD)
        try:
            vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype)
            print("** TinyVAE Loaded.")
        except:
            print("!! TinyVAE Failed. Using heavy VAE.")
            base_model_id = self.config['pipeline']['base_model']
            vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=self.dtype)

        # 3. Load Main Pipeline
        print("Loading Stable Diffusion...")
        base_model_id = self.config['pipeline']['base_model']
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            safety_checker=None
        )

        # 4. HYBRID MEMORY STRATEGY
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("** Device: MPS. Applying Hybrid Memory Strategy for CPU Offloading...")
            
            # Move heavy compute models to GPU
            self.pipe.unet.to("mps")
            self.pipe.controlnet.to("mps")
            self.pipe.vae.to("mps")
            
            # Keep Text Encoder on CPU (Saves VRAM)
            self.pipe.text_encoder.to("cpu")
            
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_tiling()
            
        else:
            self.device = "cpu"
            self.pipe.to("cpu")
        
        # 5. LCM Distillation
        if self.config['pipeline'].get('distillation_type') == "lcm":
            self.pipe.load_lora_weights(self.config['pipeline']['lcm_lora_id'])
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

    def generate(
        self, 
        prompt: str, 
        negative_prompt: str, 
        control_image: Image.Image,
        num_inference_steps: int = 4, 
        guidance_scale: float = 1.0,
        controlnet_conditioning_scale: float = 1.0
    ) -> Image.Image:
        
        # 1. Clean Memory
        if self.device == "mps":
            torch.mps.empty_cache()
            gc.collect()

        generator = torch.Generator(device="cpu").manual_seed(42)
        
        # 2. Determine Classifier Free Guidance
        # If guidance_scale > 1.0, we NEED negative embeddings.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Manual Encoding (Hybrid Mode - CPU ONLY)
        with torch.no_grad():
            prompt_embeds, negative_embeds = self.pipe.encode_prompt(
                prompt, 
                "cpu", # Force CPU encoding
                1, 
                do_classifier_free_guidance, # Dynamic boolean (Fixes the logic error)
                negative_prompt
            )

        # 4. Move Embeddings to MPS (If they exist)
        if self.device == "mps":
            prompt_embeds = prompt_embeds.to("mps")
            if negative_embeds is not None:
                negative_embeds = negative_embeds.to("mps") # Fixes the NoneType crash

        # 5. Inference
        with torch.inference_mode():
            output = self.pipe(
                prompt_embeds=prompt_embeds, 
                negative_prompt_embeds=negative_embeds,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=float(guidance_scale),
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                generator=generator
            )
            
        return output.images[0]