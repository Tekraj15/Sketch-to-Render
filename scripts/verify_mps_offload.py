import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from src.engine.pipeline import SketchToRenderEngine
from PIL import Image
import numpy as np
import os

def test_mps_offload():
    print("Testing MPS Offload Fix...")
    config_path = "configs/inference_lcm.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    try:
        engine = SketchToRenderEngine(config_path)
        print("Engine initialized successfully.")
        print(f"Engine device: {engine.device}")
        print(f"Pipeline device: {engine.pipe.device}")
        if hasattr(engine.pipe, "_execution_device"):
            print(f"Execution device: {engine.pipe._execution_device}")
        
        # Create a dummy white image for testing
        dummy_image = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 255)
        
        print("Running generation...")
        output = engine.generate(
            prompt="A futuristic car",
            negative_prompt="low quality",
            control_image=dummy_image,
            num_inference_steps=1
        )
        print("Generation successful!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mps_offload()
