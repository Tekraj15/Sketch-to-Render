import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# --- ZEROGPU SETUP ---
try:
    import spaces
    print("   ✅ ZeroGPU Library Active")
except ImportError:
    # Local fallback
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

class PyTorchEngine:
    def __init__(self):
        print("** Initializing Pro Engine (ZeroGPU Ready)...")
        
        # 1. Determine Precision
        # We use Float16 for speed on GPU, but we must be careful with the VAE.
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # 2. Load Models
        print("   ** Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=self.dtype
        )
        
        print("   ** Loading Stable Diffusion...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None, # Disable safety checker to prevent false positives
            torch_dtype=self.dtype
        )
        
        # 3. CRITICAL FIX: FORCE VAE TO FLOAT32
        # SD v1.5 VAE cracks in Float16, producing black/grey images.
        # Forcing the VAE (Decoder) to keep full precision.
        print("   !! Forcing VAE to Float32 to prevent black images...")
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
        
        # 4. Inject LCM
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        print("   ✅ Engine Loaded.")
        
    # 5. ZeroGPU Execution
    @spaces.GPU(duration=60) 
    def generate(self, prompt, negative_prompt, control_image, steps=4, guidance=1.0, control_scale=1.0):
        # Move pipeline to GPU (The decorator ensures we have one)
        if torch.cuda.is_available():
            self.pipe.to("cuda")
            # Ensure VAE stays in Float32 even after moving to CUDA
            self.pipe.vae.to(dtype=torch.float32)
            
        # Preprocess
        if control_image.size != (512, 512):
            control_image = control_image.resize((512, 512), Image.LANCZOS)
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        print("   <> Generating...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale)
        ).images[0]
        
        return result