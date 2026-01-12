import torch
import os
import time
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image

# --- ZeroGPU Wrapper ---
try:
    import spaces
    ZEROGPU_AVAILABLE = True
    print(" ** ZeroGPU detected - using lazy GPU loading")
except ImportError:
    ZEROGPU_AVAILABLE = False
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

# --- GLOBAL HOOK FUNCTION ---
def vae_cast_hook(module, inputs):
    """
    Forces all inputs to the VAE to be Float32.
    Prevents the VAE from crashing when receiving Float16 latents.
    """
    if isinstance(inputs, tuple):
        return tuple(t.to(dtype=torch.float32) if isinstance(t, torch.Tensor) else t for t in inputs)
    return inputs

class CUDAOptimizedEngine:
    def __init__(self):
        print(" ** Initializing CUDA Optimized Engine...")
        print(f" ** ZeroGPU Mode: {ZEROGPU_AVAILABLE}")
        
        # NOTE: In ZeroGPU, CUDA is NOT available at init time!
        # We defer all GPU operations to the generate() function.
        
        self.device = "cuda"
        self.dtype = torch.float16
        
        # Check if CUDA is available (for local testing)
        # On ZeroGPU this will be False at init, which is expected
        if torch.cuda.is_available():
            print(f" ** CUDA Available at init: {torch.cuda.get_device_name(0)}")
        else:
            print(" ** CUDA not available at init (expected for ZeroGPU)")

        # 2. Load Models on CPU (required for ZeroGPU)
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
        )
        
        # 3. Configure scheduler and load LoRA (can be done on CPU)
        print(" ** Configuring LCM scheduler...")
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        # 4. Register VAE hook
        self.pipe.vae.post_quant_conv.register_forward_pre_hook(vae_cast_hook)
        
        # Track GPU state
        self._gpu_ready = False
        
        print(" ✅ Engine initialized (awaiting GPU allocation)")
    
    def _move_to_gpu(self):
        """Move pipeline to GPU. Must be called inside @spaces.GPU function."""
        if self._gpu_ready:
            # Verify still on GPU
            try:
                unet_device = next(self.pipe.unet.parameters()).device
                if unet_device.type == 'cuda':
                    return  # Already on GPU
                print(f" ** Models moved to {unet_device}, re-moving to CUDA...")
            except:
                pass
        
        print(" ** Moving pipeline to CUDA...")
        
        # Move entire pipeline to GPU
        self.pipe = self.pipe.to(self.device)
        
        # Cast VAE to float32 for numerical stability
        self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
        
        self._gpu_ready = True
        
        # Print device info
        print(" ** Device placement after GPU move:")
        print(f"    UNet:        {next(self.pipe.unet.parameters()).device}")
        print(f"    VAE:         {next(self.pipe.vae.parameters()).device}")
        print(f"    ControlNet:  {next(self.pipe.controlnet.parameters()).device}")
        print(f"    TextEncoder: {next(self.pipe.text_encoder.parameters()).device}")
        
    @spaces.GPU(duration=120)
    def generate(self, prompt, negative_prompt, control_image, steps=6, guidance=1.5, control_scale=0.8):
        print("\n" + "="*60)
        print(" ** GENERATE CALLED")
        print("="*60)
        
        # CRITICAL: Move to GPU inside @spaces.GPU decorated function
        self._move_to_gpu()
        
        # Analyze control image
        print(f"\n ** Control Image: size={control_image.size}, mode={control_image.mode}")
        img_array = np.array(control_image)
        print(f"    Pixels: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.1f}")
        
        # Resize if needed
        if control_image.size != (512, 512):
            control_image = control_image.resize((512, 512), Image.LANCZOS)
        
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")
        
        print(f" ** Prompt: {prompt[:80]}...")
        print(f" ** Steps: {steps}, Guidance: {guidance}, ControlScale: {control_scale}")
        
        # Final device verification
        unet_device = next(self.pipe.unet.parameters()).device
        vae_device = next(self.pipe.vae.parameters()).device
        print(f" ** Final check: UNet={unet_device}, VAE={vae_device}")
        
        # Run inference
        print(" ** Running inference...")
        start = time.time()
        
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                controlnet_conditioning_scale=float(control_scale)
            )
            
            output_image = result.images[0]
            output_array = np.array(output_image)
            
            print(f" ** Inference done in {time.time() - start:.2f}s")
            print(f" ** Output: min={output_array.min()}, max={output_array.max()}, std={output_array.std():.1f}")
            
            if output_array.std() < 10:
                print(" ❌ WARNING: Output appears to be blank/gray!")
            else:
                print(" ✅ Output looks valid!")
            
            return output_image
            
        except Exception as e:
            print(f" ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise