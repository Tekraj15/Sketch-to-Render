import cv2
import numpy as np
from PIL import Image

class Preprocessor:
    def __init__(self):
        pass

    def get_canny(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """
        Generates a Canny edge detection map from an input image.
        """
        # Convert PIL to CV2
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Canny
        canny = cv2.Canny(img, low_threshold, high_threshold)
        
        # Canny is 1 channel, need to replicate to 3 or keep as 1 depending on ControlNet
        # Most ControlNets expect 3 channels
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        
        return Image.fromarray(canny)

    def get_scribble(self, image: Image.Image) -> Image.Image:
        """
        Generates a scribble map. For user-drawn sketches, this is often 
        just a normalization/thresholding step if the input is already a sketch.
        """
        # For simplicity, if it's already a black and white sketch, we ensure it's high contrast
        img = np.array(image.convert("L"))
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Re-invert if necessary, scribbles usually want black lines on white or vice versa?
        # Standard ControlNet Scribble usually expects white lines on black background.
        
        scribble = binary[:, :, None]
        scribble = np.concatenate([scribble, scribble, scribble], axis=2)
        
        return Image.fromarray(scribble)

    def prepare_sketch(self, image: Image.Image, size: tuple = (512, 512)) -> Image.Image:
        """
        Resizes and prepares the user input sketch for the pipeline.
        """
        return image.resize(size, Image.Resampling.LANCZOS)
