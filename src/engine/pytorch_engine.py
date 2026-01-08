import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# --- ZeroGPU Compatibility ---
# Hugging Face ZeroGPU uses the 'spaces' library.
# If running locally, this import fails, so we create a dummy wrapper.
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

class PyTorchEngine:
    def __init__(self):
        print("🚀 Initializing Universal PyTorch Engine...")
        
        # 1. Hardware Detection
        # Priority: CUDA (Cloud GPU) > MPS (Mac GPU) > CPU (Fallback)
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            
        print(f"   ↳ Active Device: {self.device.upper()} | Precision: {self.dtype}")

        # 2. Load Models (Automatic Download from HF Cache)
        print("   ↳ Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=self.dtype
        )
        
        print("   ↳ Loading Stable Diffusion...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=self.dtype
        ).to(self.device)
        
        # 3. Apply LCM (Latent Consistency Model) for Speed
        print("   ↳ Injecting LCM Scheduler...")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        print("   ✅ PyTorch Engine Ready.")
        
    # Decorate with @spaces.GPU for free GPU on Hugging Face
    @spaces.GPU(duration=60) 
    def generate(self, prompt, negative_prompt, control_image, steps=4, guidance=1.0, control_scale=1.0):
        # Resize inputs to 512x512
        if control_image.size != (512, 512):
            control_image = control_image.resize((512, 512), Image.LANCZOS)
            
        # Ensure RGB
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        # Run Inference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale)
        ).images[0]
        
        return result