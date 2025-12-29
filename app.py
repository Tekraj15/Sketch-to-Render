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
with gr.Blocks(title="Sketch-to-Render: Automotive Design Studio Using Stable Diffusion", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏎️ Sketch-to-Render: Automotive Design Studio
    *Elevating rough automotive sketches to high-fidelity concepts in real-time.*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 1. Concept Input")
                sketchpad = gr.Sketchpad(
                    label="Sketch Area", 
                    type="numpy",
                    layers=False,
                    canvas_size=(512, 512),
                    brush=gr.Brush(colors=["#000000"]),
                    interactive=True
                )
                
            with gr.Group():
                gr.Markdown("### 2. Design DNA")
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Style Prompt", 
                        placeholder="e.g., A minimalist Porsche exterior...",
                        value=engine.config['ui']['default_prompt'],
                        scale=4
                    )
                    latency_meter = gr.Label(label="Latency", value="0ms", scale=1)
                
                style_dna = gr.Dropdown(
                    label="Design Style", 
                    choices=["Modern Minimalist", "Cyberpunk / Futuristic", "Vintage Classic", "Aerodynamic Race"],
                    value="Modern Minimalist"
                )
                
            with gr.Group():
                gr.Markdown("### 3. Performance Settings")
                perf_mode = gr.Radio(
                    label="Inference Profile", 
                    choices=["Production (50 steps)", "Creative (4 steps)", "Instant (1 step)"],
                    value="Creative (4 steps)"
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    steps = gr.Slider(1, 50, value=4, step=1, label="Denoising Steps")
                    guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="Prompt Strength")
                    control_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Sketch Constraint")
            
            run_btn = gr.Button("🚀 Render Concept", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 4. Output Render")
            output_render = gr.Image(label="AI Rendered Concept", interactive=False, type="pil")
            gr.Markdown("""
            ---
            **Pro Tip**: Use the "Creative" mode for live sketching. Switch to "Production" for final high-fidelity results.
            """)
            
    # Helper to append style to prompt
    def get_styled_prompt(p, style):
        styles = {
            "Modern Minimalist": ", clean lines, minimalist design, high-end studio lighting, 8k",
            "Cyberpunk / Futuristic": ", neon accents, glowing panels, cyberpunk aesthetic, rainy night, cinematic",
            "Vintage Classic": ", retro automotive design, nostalgic tones, classic car show, film grain",
            "Aerodynamic Race": ", aggressive body kit, spoiler, racing decals, high-speed motion blur, track background"
        }
        return p + styles.get(style, "")

    # Event Handlers
    def on_change(sketch, p, neg, s, g, c, style):
        styled_p = get_styled_prompt(p, style)
        return process_sketch(sketch, styled_p, neg, s, g, c)

    sketchpad.change(
        fn=on_change, 
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale, style_dna], 
        outputs=[output_render, latency_meter],
        show_progress="hidden"
    )
    
    # Update steps slider automatically based on mode
    def update_steps(mode):
        if mode == "Production (50 steps)": return 50
        if mode == "Creative (4 steps)": return 4
        if mode == "Instant (1 step)": return 1
        return 4

    perf_mode.change(fn=update_steps, inputs=perf_mode, outputs=steps)
    
    run_btn.click(
        fn=process_sketch,
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale],
        outputs=[output_render, latency_meter]
    )

if __name__ == "__main__":
    demo.launch(share=False)
