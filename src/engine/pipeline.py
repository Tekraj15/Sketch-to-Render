import torch
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    LCMScheduler,
    AutoencoderKL
)
from PIL import Image
import yaml

class SketchToRenderEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        print("Initializing engine...")
        
        # 1. Force Float32 for Stability on macOS < 14.0
        # As Float16 is unstable on my macOS < 14.0 OS version without NaNs.
        self.dtype = torch.float32
        
        # 2. Load ControlNet
        print("Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.config['pipeline']['controlnet_model'], 
            torch_dtype=self.dtype
        )
        
        # 3. Load Main Pipeline
        print("Loading Stable Diffusion...")
        base_model_id = self.config['pipeline']['base_model']
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )

        # 4. MEMORY OPTIMIZATION STRATEGY (Fix for slower memory swapping)
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Mac Silicon Detected. Applying Smart Offloading...")
            
            # self.pipe.to(self.device)
            self.pipe.enable_model_cpu_offload() # It drastically reduces peak VRAM usage
            
            # Enable Attention Slicing (Saves VRAM, slightly slower but safer)
            self.pipe.enable_attention_slicing()
            
            # VAE Tiling (Critical for Float32 VAEs to prevent OOM)
            self.pipe.enable_vae_tiling()
            
        else:
            self.device = "cpu"
            self.pipe.to("cpu")
        
        # 5. LCM Distillation
        if self.config['pipeline'].get('distillation_type') == "lcm":
            print("⚡ Loading LCM-LoRA...")
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
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # Standardize inputs
        guidance_scale = float(guidance_scale)
        controlnet_conditioning_scale = float(controlnet_conditioning_scale)

        # Inference
        # Relying on the pipeline's internal handling now that it's  pure Float32
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            )
            
        return output.images[0]