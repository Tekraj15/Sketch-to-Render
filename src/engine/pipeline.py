import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from PIL import Image
from typing import Optional, List
import yaml
import os

class SketchToRenderEngine:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['pipeline'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
            
        print(f"Initializing engine on device: {self.device}")
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            self.config['pipeline']['controlnet_model'], 
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )
        
        # Load Pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config['pipeline']['base_model'],
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # Load LCM LoRA for fast inference
        self.pipe.load_lora_weights(self.config['pipeline']['lcm_lora_id'])
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Enable Attention Slicing for Mac/Low VRAM
        if self.device == "mps" or self.device == "cuda":
            self.pipe.enable_attention_slicing()

    def generate(
        self, 
        prompt: str, 
        negative_prompt: str, 
        control_image: Image.Image,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        controlnet_conditioning_scale: float = 1.0
    ) -> Image.Image:
        """
        Performs inference to transform sketch into render.
        """
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images[0]
        
        return output

    def load_custom_lora(self, lora_path: str, weight_name: Optional[str] = None):
        """
        Loads a custom design LoRA (e.g., Porsche design).
        """
        if os.path.exists(lora_path):
            self.pipe.load_lora_weights(lora_path, weight_name=weight_name)
            print(f"Loaded custom LoRA from {lora_path}")
        else:
            print(f"LoRA path {lora_path} does not exist.")
