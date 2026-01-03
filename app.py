import gradio as gr
from src.engine.pipeline import SketchToRenderEngine
from src.processing.preprocessor import Preprocessor
from PIL import Image
import numpy as np
import time

# Initialize components
CONFIG_PATH = "configs/inference_lcm.yaml"
engine = SketchToRenderEngine(CONFIG_PATH)
preprocessor = Preprocessor()

def process_sketch(sketch_data, prompt, neg_prompt, steps, guidance, control_scale, style_choice):
    if sketch_data is None: return None, "0ms"
    
    # Handle Gradio Dict input
    sketch_raw = sketch_data.get("composite") if isinstance(sketch_data, dict) else sketch_data
    if sketch_raw is None: return None, "0ms"
        
    start_time = time.time()
    
    # Preprocess
    sketch_image = preprocessor.process_sketch_input(sketch_raw)
    control_image = preprocessor.get_canny(sketch_image)
    
    # Style Prompt Logic
    styles = {
        "Minimalist": "modern minimalist design, clean lines, white studio background, 8k, architectural digest style",
        "Cyberpunk": "cyberpunk automotive, neon lights, night city background, rain reflections, futuristic, glowing",
        "Vintage": "vintage classic car, 1960s style, film grain, warm Kodak portra colors, retro poster aesthetic",
        "Sketch": "highly detailed pencil sketch, technical drawing, blueprint style, white lines on blue paper"
    }
    final_prompt = f"{prompt}, {styles.get(style_choice, '')}"
    
    # Generate
    output = engine.generate(
        prompt=final_prompt,
        negative_prompt=neg_prompt,
        control_image=control_image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        controlnet_conditioning_scale=float(control_scale)
    )
    
    latency = f"{(time.time() - start_time):.2f}s"
    return output, latency

# --- CSS STYLING ---
custom_css = """
#render_btn {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    color: white;
    border: none;
    font-weight: bold;
}
.gradio-container {
    font-family: 'Helvetica Neue', sans-serif;
}
"""

with gr.Blocks(title="AutoDesign AI", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🏎️ **AutoDesign AI:** Real-Time Sketch Studio")
    
    with gr.Row():
        # LEFT COLUMN: Controls
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Sketch")
                sketchpad = gr.Sketchpad(
                    label="Draw Here", 
                    type="numpy", 
                    image_mode="RGBA",
                    canvas_size=(512, 512),
                    brush=gr.Brush(colors=["#000000"], default_size=4)
                )

            with gr.Group():
                gr.Markdown("### 2. Style DNA")
                # Stylish Radio Buttons instead of Dropdown
                style_dna = gr.Radio(
                    choices=["Minimalist", "Cyberpunk", "Vintage", "Sketch"],
                    value="Minimalist",
                    label="Select Aesthetic",
                    info="Applies specific LoRA and prompt weights"
                )
                
                prompt = gr.Textbox(
                    label="Additional Details", 
                    placeholder="e.g., Red spoiler, open roof...", 
                    lines=1
                )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    steps = gr.Slider(1, 10, value=4, step=1, label="Steps (Speed)")
                    control_scale = gr.Slider(0.1, 2.0, value=0.8, label="ControlNet Strength")
                guidance = gr.Slider(0.1, 10.0, value=1.0, label="Guidance Scale")
                neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad anatomy, blurry, messy")

            # Big Render Button
            run_btn = gr.Button("✨ Render Concept", elem_id="render_btn", size="lg")
            latency_meter = gr.Label(label="Latency", value="0.0s")

        # RIGHT COLUMN: Output
        with gr.Column(scale=6):
            gr.Markdown("### 3. Visualization")
            output_render = gr.Image(
                label="Final Render", 
                type="pil", 
                interactive=False,
                show_download_button=True,
                elem_id="output_image"
            )

    # Event wiring
    run_btn.click(
        fn=process_sketch,
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale, style_dna],
        outputs=[output_render, latency_meter]
    )

if __name__ == "__main__":
    demo.launch(share=False)