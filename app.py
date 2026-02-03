import gradio as gr
# IMPORT THE NEW ENGINE
from src.engine.coreml_engine import CoreMLEngine 
from src.processing.preprocessor import Preprocessor
from PIL import Image
import numpy as np
import time
from PIL import Image, ImageOps

# --- INITIALIZE COREML ENGINE ---
# YAML config not required because CoreML models are self-contained.
engine = CoreMLEngine(model_dir="models/coreml")


# --- ENGINE SELECTION for Hugging Face Space. Commented out here as this script will serve Apple hardware optimization(ANE) ---
# try:
#     from src.engine.cuda_optimized_engine import CUDAOptimizedEngine
#     # This will fail locally on your Mac (No CUDA), catching the error automatically
#     engine = CUDAOptimizedEngine()
# except (ImportError, RuntimeError) as e:
#     print(f" !! Could not load CUDA Engine ({e}). Falling back to Universal Selector...")
#     from src.engine.engine_selector import get_engine
#     engine = get_engine()


preprocessor = Preprocessor()

def process_sketch(sketch_data, prompt, neg_prompt, steps, guidance, control_scale, style_choice):
    # --- 1. Safety Checks ---
    if sketch_data is None: 
        return None, "0ms"
    
    # Handle Gradio Dict input safely
    sketch_raw = sketch_data.get("composite") if isinstance(sketch_data, dict) else sketch_data
    if sketch_raw is None: 
        return None, "0ms"
        
    start_time = time.time()
    
    # --- 2. Preprocessing (Standardize Size/Mode) ---
    # This handles resizing to 512x512 and basic RGBA conversion
    control_image = preprocessor.process_sketch_input(sketch_raw)
    
    # --- 3. CRITICAL ACCURACY FIX: Invert Colors ---
    # Sketch is Black-on-White. ControlNet needs White-on-Black.
    # We ensure it's RGB first (removing alpha if preprocessor didn't)
    if control_image.mode == 'RGBA':
        background = Image.new("RGB", control_image.size, (255, 255, 255))
        background.paste(control_image, mask=control_image.split()[3])
        control_image = background
    elif control_image.mode != 'RGB':
        control_image = control_image.convert("RGB")
        
    # Apply Inversion
    control_image = ImageOps.invert(control_image)
    
    # --- 4. Style Prompt Logic ---
    styles = {
        "Minimalist": "modern minimalist design, clean lines, white studio background, 8k, architectural digest style",
        "Cyberpunk": "cyberpunk automotive, neon lights, night city background, rain reflections, futuristic, glowing",
        "Vintage": "vintage classic car, 1960s style, film grain, warm Kodak portra colors, retro poster aesthetic",
        "Sketch": "highly detailed pencil sketch, technical drawing, blueprint style, white lines on blue paper",
        "Hyper-Gloss": "professional automotive studio photography, 3-point lighting, dramatic shadows, 8k resolution, metallic paint, highly reflective surface, unreal engine 5 render"
    }
    # Add style to prompt if not 'None' or empty
    # --- 4. Style Prompt Logic ---
    styles = {
        "Minimalist (Arch)": "modern minimalist design, clean lines, white studio background, 8k, architectural digest style",
        "Cyberpunk (Auto)": "cyberpunk automotive, neon lights, night city background, rain reflections, futuristic, glowing",
        "Vintage (Auto)": "vintage classic car, 1960s style, film grain, warm Kodak portra colors, retro poster aesthetic",
        "Sketch": "highly detailed pencil sketch, technical drawing, blueprint style, white lines on blue paper",
        "Hyper-Gloss": "professional studio photography, 3-point lighting, dramatic shadows, 8k resolution, metallic paint, highly reflective surface, unreal engine 5 render",
        "Fashion": "high fashion runway photography, elegant clothing design, silk and leather textures, spotlight, vanity fair style",
        "Urban / City": "bustling metropolis, skyscrapers, wet street level view, cinematic lighting, gotham city vibe",
        "Nature / Landscape": "national geographic nature photography, majestic mountains, lush forest, golden hour sunlight, dramatic clouds",
        "Ocean / Beach": "tropical paradise, crystal clear turquoise water, white sand, sunset reflection, cinematic seascape",
        "Floral": "macro photography of flowers, morning dew, bokeh background, vibrant botanical garden, soft natural light",
        "Wildlife": "wildlife photography telephoto lens, detailed fur texture, national geographic style, natural habitat"
    }
    # Add style to prompt if not 'None' or empty
    style_suffix = styles.get(style_choice, "")
    final_prompt = f"{prompt}, {style_suffix}".strip(", ")
    
    # --- 5. Generate using CoreML ---
    output = engine.generate(
        prompt=final_prompt,
        negative_prompt=neg_prompt,
        control_image=control_image,  # Pass the INVERTED image
        steps=steps,
        guidance=guidance,
        control_scale=1.0  # Force strict adherence to the control image
    )
    
    latency = f"{(time.time() - start_time):.2f}s"
    return output, latency

# --- CSS STYLING ---
# 1. Button and containers
custom_css = """
#render_btn {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    color: white;
    border: none;
    font-weight: bold;
    font-size: 1.1em;
}
#render_btn:hover {
    background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.gradio-container { font-family: 'Helvetica Neue', sans-serif; }
.canvas-container { border: 2px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
"""


# 2. New Fixes (Cursor Alignment & Canvas Containment)
cursor_fixes = """
.gradio-container {
    max_width: 1200px !important;
}
/* Force the canvas to match the cursor location */
canvas {
    cursor: crosshair !important;
}
/* Fix offset by resetting margins on the image container */
.image-container {
    margin: 0 !important;
    padding: 0 !important;
}
"""

# 3. Combine them
combined_css = custom_css + cursor_fixes

# --- UI LAYOUT ---
with gr.Blocks(title="Sketch-to-Render AI", css=combined_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ## **⛩️ Sketch-to-Render AI**
    ### ***Fast Inference Design Studio powered by Stable Diffusion***
    """)
    
    with gr.Row():
        # LEFT COLUMN: Controls
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Sketch")
                sketchpad = gr.Sketchpad(
                    label="Draw Here", 
                    type="numpy", 
                    image_mode="RGBA",
                    # COREML REQUIREMENT: Must be 512x512 to match compiled model
                    canvas_size=(512, 512), 
                    brush=gr.Brush(colors=["#000000"], default_size=4),
                    elem_classes=["canvas-container"]
                )

            with gr.Group():
                gr.Markdown("### 2. Style DNA")
                style_dna = gr.Radio(
                    choices=[
                        "Minimalist (Arch)", 
                        "Cyberpunk (Auto)", 
                        "Vintage (Auto)", 
                        "Sketch", 
                        "Hyper-Gloss",
                        "Fashion", 
                        "Urban / City", 
                        "Nature / Landscape", 
                        "Ocean / Beach", 
                        "Floral", 
                        "Wildlife"
                    ],
                    value="Minimalist (Arch)",
                    label="Select Aesthetic"
                )
                
                prompt = gr.Textbox(label="Additional Details", placeholder="e.g., Red spoiler...", lines=1)

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    # LCM allows low steps (4 is usually optimal)
                    steps = gr.Slider(1, 10, value=4, step=1, label="Steps")
                    control_scale = gr.Slider(0.1, 2.0, value=0.8, label="Control Strength")
                guidance = gr.Slider(0.1, 10.0, value=1.0, label="Guidance Scale")
                neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad anatomy, blurry")

            run_btn = gr.Button(" Render Concept", elem_id="render_btn", size="lg")
            latency_meter = gr.Label(label="Latency", value="0.0s")

        # RIGHT COLUMN: Output
        with gr.Column(scale=6):
            gr.Markdown("### 3. Visualization")
            output_render = gr.Image(label="Final Render", type="pil", interactive=False)

    # Event Wiring
    run_btn.click(
        fn=process_sketch,
        inputs=[sketchpad, prompt, neg_prompt, steps, guidance, control_scale, style_dna],
        outputs=[output_render, latency_meter]
    )

if __name__ == "__main__":
    demo.queue()
    # Force IPv4 localhost to prevent MacOS IPv6 confusion
    demo.launch(server_name="127.0.0.1", share=False, show_error=True)