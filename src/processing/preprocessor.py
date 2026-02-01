import numpy as np
from PIL import Image

class Preprocessor:
    def process_sketch_input(self, sketch_input) -> Image.Image:
        """
        Handles Gradio input and returns a PIL Image with
        BLACK lines on a WHITE background.
        """
        if sketch_input is None: return None
        
        # Gradio Dict handling
        if isinstance(sketch_input, dict):
            sketch_input = sketch_input.get("composite", None)
        if sketch_input is None: return None
            
        # Convert to PIL
        if isinstance(sketch_input, np.ndarray):
            image = Image.fromarray(sketch_input.astype('uint8'))
        else:
            image = sketch_input

        # Handle Transparency -> White Background
        # This ensures we have a solid white sheet with black lines
        if image.mode != 'RGB':
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image)
            image = background
            
        return image

    def get_canny(self, image: Image.Image) -> Image.Image:
        """
        PREPARES IMAGE FOR CONTROLNET SCRIBBLE
        
        CRITICAL CHANGE:
        We do NOT invert colors anymore.
        ControlNet Scribble expects: BLACK lines on WHITE background.
        """
        # We simply return the cleaned "Black-on-White" image.
        # No inversion (255 - img) needed.
        return image