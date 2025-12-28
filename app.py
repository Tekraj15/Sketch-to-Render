import gradio as gr
from src.engine.pipeline import SketchToRenderEngine
from src.processing.preprocessor import Preprocessor
from PIL import Image
import numpy as np
import os

# Initialize components
CONFIG_PATH = "configs/inference_lcm.yaml"
engine = SketchToRenderEngine(CONFIG_PATH)
preprocessor = Preprocessor()

def process_sketch(sketch_data, prompt, neg_prompt, steps, guidance, control_scale):
    """
    Callback for UI interaction.
    """
    if sketch_data is None:
        return None, "0ms"
    
    if isinstance(sketch_data, dict):
        sketch_image = sketch_data.get("composite")
    else:
        sketch_image = sketch_data
        
    if sketch_image is None:
        return None, "0ms"
        
    import time
    start_time = time.time()
    
    # Resize and preprocess
    sketch_image = Image.fromarray(np.uint8(sketch_image)).convert("RGB")
    control_image = preprocessor.get_canny(sketch_image)
    
    # Generate
    output = engine.generate(
        prompt=prompt,
        negative_prompt=neg_prompt,
        control_image=control_image,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        controlnet_conditioning_scale=control_scale
    )
    
    latency = f"{(time.time() - start_time) * 1000:.0f}ms"
    
    return output, latency

# UI Layout
with gr.Blocks(title="Sketch-to-Render: Automotive Design Studio Using Stable Diffusion") as demo:
    gr.Markdown("# 🏎️ Sketch-to-Render: Real-Time Automotive Design Studio")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(
                label="Sketch your car", 
                type="numpy",
                layers=False,
                canvas_size=(512, 512)
            )
            
            with gr.Row():
                prompt = gr.Textbox(
                    label="Design Prompt", 
                    value=engine.config['ui']['default_prompt'],
                    scale=4
                )
                latency_meter = gr.Label(label="Latency", value="0ms", scale=1)
            
            neg_prompt = gr.Textbox(
                label="Negative Prompt", 
                value=engine.config['ui']['negative_prompt']
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                steps = gr.Slider(1, 20, value=4, step=1, label="Inference Steps")
                guidance = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Guidance Scale")
                control_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="ControlNet Strength")
            
            run_btn = gr.Button("Manual Render", variant="primary")
            
        with gr.Column():
            output_render = gr.Image(label="AI Render", interactive=False)
            
    # Event Handlers
    sketchpad.change(
        fn=process_sketch, 
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale], 
        outputs=[output_render, latency_meter],
        show_progress="hidden"
    )
    
    run_btn.click(
        fn=process_sketch,
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale],
        outputs=[output_render, latency_meter]
    )

if __name__ == "__main__":
    demo.launch(share=False)
