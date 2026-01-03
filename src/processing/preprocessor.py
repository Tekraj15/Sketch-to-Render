import cv2
import numpy as np
from PIL import Image, ImageOps

class Preprocessor:
    def process_sketch_input(self, sketch_input) -> Image.Image:
        """
        Handles Gradio input and converts to a clean RGB image.
        """
        if sketch_input is None:
            return None

        # Handle Dict input from new Gradio versions
        if isinstance(sketch_input, dict):
            sketch_input = sketch_input.get("composite", None)
        
        if sketch_input is None:
            return None
            
        # Convert to PIL
        if isinstance(sketch_input, np.ndarray):
            image = Image.fromarray(sketch_input.astype('uint8'))
        else:
            image = sketch_input

        # Handle Alpha Channel (Transparency) -> White Background
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
        Processes the sketch for ControlNet.
        """
        img_array = np.array(image)
        
        # 1. Invert the image
        img_array = 255 - img_array
        # 2. Edge Detection (on the inverted image)
        canny = cv2.Canny(img_array, 100, 200)
        
        # 3. Format for ControlNet (H, W, 3)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        
        return Image.fromarray(canny)