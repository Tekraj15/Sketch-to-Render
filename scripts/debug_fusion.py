import torch
from diffusers import StableDiffusionPipeline
import os
import shutil

def test_fusion_and_load():
    base_model_id = "runwayml/stable-diffusion-v1-5"
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    temp_model_path = os.path.abspath("debug_temp_model")
    
    print(f"1. Loading Base Model: {base_model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32
    )
    
    print(f"2. Loading LoRA: {lcm_lora_id}")
    pipe.load_lora_weights(lcm_lora_id)
    print("3. Fusing LoRA...")
    pipe.fuse_lora(lora_scale=1.0)
    
    # DEBUG: Check if UNet is a PeftModel
    print(f"DEBUG: UNet type: {type(pipe.unet)}")
    
    # Try to unload the LoRA weights mechanism (assuming fusion is permanent)
    print("3.5 Unloading LoRA weights (cleanup)...")
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        print(f"Warning: unload_lora_weights failed: {e}")
    
    
    print(f"4. Saving Fused Model to {temp_model_path} (safe_serialization=False)...")
    if os.path.exists(temp_model_path):
        shutil.rmtree(temp_model_path)
    pipe.save_pretrained(temp_model_path, safe_serialization=False)
    
    print("5. Attempting to Reload Model...")
    try:
        loaded_pipe = StableDiffusionPipeline.from_pretrained(temp_model_path)
        print("SUCCESS: Model reloaded successfully!")
    except Exception as e:
        print(f"FAILURE: Could not reload model. Error:\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fusion_and_load()
