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
    sketch_data is a dict with 'background' and 'layers' or a list of images.
    In modern Gradio Sketchpad, it might be different. 
    We'll assume it returns the composite image.
    """
    if sketch_data is None:
        return None
    
    # Extract image from Gradio data
    # Gradio 5.x Sketchpad returns a dict with "composite"
    if isinstance(sketch_data, dict):
        sketch_image = sketch_data.get("composite")
    else:
        sketch_image = sketch_data
        
    if sketch_image is None:
        return None
        
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
    
    return output

# UI Layout
with gr.Blocks(title="Sketch-to-Render: Automotive Design Studio Using Stable Diffusion") as demo:
    gr.Markdown("# 🏎️ Sketch-to-Render: Real-Time Automotive Design Studio")
    gr.Markdown("Draw your automotive concept on the left and see it rendered in real-time on the right.")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(
                label="Sketch your car", 
                type="numpy",
                layers=False,
                canvas_size=(512, 512)
            )
            
            prompt = gr.Textbox(
                label="Design Prompt", 
                value=engine.config['ui']['default_prompt']
            )
            
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
    # 'change' on sketchpad triggers real-time. 
    # Note: might need 'step' or 'release' if it's too frequent.
    sketchpad.change(
        fn=process_sketch, 
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale], 
        outputs=output_render,
        show_progress="hidden" # For smoother real-time feel
    )
    
    run_btn.click(
        fn=process_sketch,
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale],
        outputs=output_render
    )

if __name__ == "__main__":
    demo.launch(share=False)
