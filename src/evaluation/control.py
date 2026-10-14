from typing import Optional, Tuple

from PIL import Image, ImageOps

from src.processing.preprocessor import Preprocessor

CONTROL_MODES = ("app_draw", "app_import", "auto")


def load_sketch(path: str) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")


def letterbox(image: Image.Image, fill: Tuple[int, int, int]) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def app_draw_control(image: Image.Image, canvas: int = 512) -> Image.Image:
    """
    Emulates the Draw-tab path of app.py (process_sketch_input -> RGB -> ImageOps.invert).
    A non-square sketch is centred on a white square first, as if placed on the
    square canvas, then resized to the canvas size the engines expect.
    """
    rgb = Preprocessor().process_sketch_input(image)
    rgb = letterbox(rgb, (255, 255, 255))
    if rgb.size != (canvas, canvas):
        rgb = rgb.resize((canvas, canvas), Image.LANCZOS)
    return ImageOps.invert(rgb)


def has_import_path() -> bool:
    return hasattr(Preprocessor, "process_uploaded_sketch")


def make_control_map(image: Image.Image, mode: str, input_mode: Optional[str],
                     canvas: int = 512) -> Tuple[Image.Image, str]:
    """Returns (white-on-black control map, effective mode used)."""
    if mode not in CONTROL_MODES:
        raise ValueError(f"control mode {mode!r} not in {CONTROL_MODES}")
    use_import = mode == "app_import" or (mode == "auto" and has_import_path() and input_mode)
    if use_import:
        if not has_import_path():
            raise RuntimeError("control_mode=app_import but Preprocessor.process_uploaded_sketch "
                               "is not available on this branch")
        control = Preprocessor().process_uploaded_sketch(image.convert("RGB"),
                                                         input_mode=input_mode or "Digital Sketch")
        if control.size != (canvas, canvas):
            control = control.resize((canvas, canvas), Image.NEAREST)
        return control, "app_import"
    return app_draw_control(image, canvas), "app_draw"
