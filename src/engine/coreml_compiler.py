import torch
import os
import shutil
import sys
import subprocess
import glob
from diffusers import StableDiffusionPipeline

def rename_models(output_dir):
    """
    Cleans up the messy auto-generated filenames from Apple's script.
    """
    print("\n[3/3] Renaming models to standard format...")
    replacements = {
        "vae_decoder": "VAEDecoder.mlpackage",
        "vae_encoder": "VAEEncoder.mlpackage",
        "text_encoder": "TextEncoder.mlpackage",
        "controlnet": "ControlNet.mlpackage",
    }
    
    # Handle UNet chunks specifically
    for file_path in glob.glob(os.path.join(output_dir, "*.mlpackage")):
        filename = os.path.basename(file_path)
        
        # Rename standard components
        for key, new_name in replacements.items():
            if key in filename and "unet" not in filename: # Avoid confusion
                new_path = os.path.join(output_dir, new_name)
                if os.path.exists(new_path): shutil.rmtree(new_path)
                os.rename(file_path, new_path)
                print(f"   ↳ Renamed {filename} -> {new_name}")

        # Rename UNet Chunks
        if "unet_chunk1" in filename:
            new_path = os.path.join(output_dir, "UnetChunk1.mlpackage")
            if os.path.exists(new_path): shutil.rmtree(new_path)
            os.rename(file_path, new_path)
            print(f"   ↳ Renamed {filename} -> UnetChunk1.mlpackage")
            
        elif "unet_chunk2" in filename:
            new_path = os.path.join(output_dir, "UnetChunk2.mlpackage")
            if os.path.exists(new_path): shutil.rmtree(new_path)
            os.rename(file_path, new_path)
            print(f"   ↳ Renamed {filename} -> UnetChunk2.mlpackage")

def fuse_and_compile(base_model_id, controlnet_id, lcm_lora_id, output_dir="models/coreml"):
    print(f"** Starting CoreML Compilation Pipeline...")
    
    # --- STEP 1: LOAD & FUSE ---
    print("\n[1/3] Loading and Fusing Weights...")
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
    print("   ↳ Merging LCM-LoRA into UNet...")
    pipe.load_lora_weights(lcm_lora_id)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    
    temp_model_path = os.path.abspath("temp_fused_model")
    if os.path.exists(temp_model_path): shutil.rmtree(temp_model_path)
    pipe.save_pretrained(temp_model_path, safe_serialization=True)
    print(f"   ↳ Fused model saved to {temp_model_path}")

    # --- STEP 2: CONVERT VIA CLI (TWO PASSES) ---
    print("\n[2/3] Running Apple CoreML Converter (Multi-Pass)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # PASS A: Convert Fused UNet + VAEs + Text Encoder
    # skipping ControlNet here to avoid the version mismatch error
    print("   ↳ Pass A: Converting Fused UNet, VAE, Text Encoder...")
    cmd_base = [
        sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
        "-o", output_dir,
        "--latent-h", "64", "--latent-w", "64",
        "--attention-implementation", "SPLIT_EINSUM",
        "--compute-unit", "ALL",
        "--quantize-nbits", "6",
        "--chunk-unet",
        "--unet-support-controlnet", # imp: prepares UNet input ports
        "--convert-unet", 
        "--convert-text-encoder", 
        "--convert-vae-decoder", 
        "--convert-vae-encoder",
        "--model-version", temp_model_path # Use local fused model
    ]
    subprocess.check_call(cmd_base)

    # PASS B: Convert ControlNet Separately
    # Using the standard repo ID here to satisfy the sanity check
    print("\n   ↳ Pass B: Converting ControlNet...")
    cmd_control = [
        sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
        "-o", output_dir,
        "--compute-unit", "ALL",
        "--quantize-nbits", "6",
        "--convert-controlnet", controlnet_id,
        "--model-version", base_model_id # Use Standard ID (e.g. runwayml/...)
    ]
    subprocess.check_call(cmd_control)

    # --- STEP 3: RENAME ---
    rename_models(output_dir)
    
    # Cleanup
    if os.path.exists(temp_model_path): shutil.rmtree(temp_model_path)
    print(f"\n** SUCCESS!! All models compiled and renamed in {output_dir}")

if __name__ == "__main__":
    BASE = "runwayml/stable-diffusion-v1-5"
    CONTROL = "lllyasviel/sd-controlnet-scribble"
    LCM = "latent-consistency/lcm-lora-sdv1-5"
    fuse_and_compile(BASE, CONTROL, LCM)