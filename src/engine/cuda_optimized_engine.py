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

class CUDAOptimizedEngine:
    def __init__(self):
        print(" ** initializing CUDA Optimized Engine (PyTorch 2.0)...")
        
        # 1. Force CUDA (This engine is ONLY for Nvidia GPUs)
        if not torch.cuda.is_available():
            raise RuntimeError(" !! CUDA not found! This engine requires an NVIDIA GPU.")
            
        self.device = "cuda"
        self.dtype = torch.float16 # Half-precision is mandatory for compilation speed
        
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
        
        # 3. Inject LCM (Speed)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        # 4. CRITICAL: The "TensorRT-Lite" Optimization
        # This compiles the UNet into a fused CUDA Graph.
        # mode="reduce-overhead" is optimal for real-time inference.
        print(" ** Compiling UNet with torch.compile (This takes ~1 min on first run)...")
        start_compile = time.time()
        
        # We wrap the UNet with the compiler
        self.pipe.unet = torch.compile(
            self.pipe.unet, 
            mode="reduce-overhead", 
            fullgraph=True
        )
        
        # Trigger a dummy run to force the compilation NOW (instead of waiting for first user)
        # This "warms up" the engine.
        print(" ** Warming up compilation engine...")
        try:
            dummy_img = Image.new("RGB", (512, 512), (0,0,0))
            self.pipe(
                prompt="warmup", 
                image=dummy_img, 
                num_inference_steps=1
            )
        except Exception as e:
            print(f"   ⚠️ Warmup warning: {e}")
            
        print(f"   ✅ Engine Compiled & Ready ({time.time() - start_compile:.1f}s)")
        
    @spaces.GPU(duration=60) 
    def generate(self, prompt, negative_prompt, control_image, steps=4, guidance=1.0, control_scale=1.0):
        if control_image.size != (512, 512):
            control_image = control_image.resize((512, 512), Image.LANCZOS)
        
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        # No need for manual offloading; the compiled graph handles memory efficiently on GPU
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(control_scale)
        ).images[0]
        
        return result