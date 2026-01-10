import os
import subprocess

def run_command(command):
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()

def setup_models():
    print("Setting up models for Sketch-to-Render...")
    
    # 1. Disable Xet transfers which are flaky on slow connections
    os.environ["HF_HUB_DISABLE_XET_TRANSFERS"] = "1"
    
    # 2. Define models to download
    models = [
        "runwayml/stable-diffusion-v1-5",
        "lllyasviel/sd-controlnet-canny",
        "latent-consistency/lcm-lora-sdv1-5"
    ]
    
    # 3. Use huggingface-cli to download models robustly
    # We prioritize safetensors and fp16 to save space and time
    for model in models:
        print(f"\n--- Downloading {model} ---")
        # Standard download command
        cmd = f"huggingface-cli download {model} --exclude \"*.bin\" \"*.pth\" \"*.msgpack\" --include \"*.safetensors\" \"*.json\" \"*.txt\""
        run_command(cmd)

    print("\n** All models pre-cached successfully!")

if __name__ == "__main__":
    setup_models()
