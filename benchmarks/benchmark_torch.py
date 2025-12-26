# verify model loading and inference performance on Mac (MPS)
import torch
import time
import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import numpy as np
from src.engine.pipeline import SketchToRenderEngine
from src.processing.preprocessor import Preprocessor

def run_benchmark():
    print("Starting Benchmark...")
    
    # Initialize components
    config_path = "configs/inference_lcm.yaml"
    try:
        engine = SketchToRenderEngine(config_path)
        preprocessor = Preprocessor()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # Create a dummy sketch (a circle)
    dummy_sketch = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2_img = np.zeros((512, 512), dtype=np.uint8)
    import cv2
    cv2.circle(cv2_img, (256, 256), 100, (255), 2)
    dummy_sketch[:,:,0] = cv2_img
    dummy_sketch[:,:,1] = cv2_img
    dummy_sketch[:,:,2] = cv2_img
    sketch_pil = Image.fromarray(dummy_sketch)
    
    # Preprocess
    start_time = time.time()
    control_image = preprocessor.get_canny(sketch_pil)
    pre_time = (time.time() - start_time) * 1000
    print(f"Preprocessing time: {pre_time:.2f}ms")
    
    # Warmup
    print("Performing warmup run...")
    prompt = "A sleek silver Porsche 911 in a studio setting, front view"
    _ = engine.generate(prompt, "low quality", control_image, num_inference_steps=2)
    
    # Benchmark
    print("Performing benchmark run (4 steps)...")
    start_time = time.time()
    output = engine.generate(prompt, "low quality", control_image, num_inference_steps=4)
    inf_time = (time.time() - start_time) * 1000
    
    print(f"Inference time (4 steps): {inf_time:.2f}ms")
    
    # Save output
    os.makedirs("assets", exist_ok=True)
    output.save("assets/benchmark_output.png")
    print("Output saved to assets/benchmark_output.png")

if __name__ == "__main__":
    run_benchmark()
