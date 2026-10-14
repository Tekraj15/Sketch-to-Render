import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image

ENGINE_TYPES = ("coreml", "cuda", "pytorch", "mps_legacy")


def seed_everything(seed: int) -> None:
    # The engines take no seed/generator, so diffusers samples from the global
    # torch RNG; seeding it here makes renders reproducible without engine edits.
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


@dataclass
class EngineAdapter:
    """Uniform call surface over the engines' differing generate() signatures."""
    type: str
    engine: object
    device: str
    seed_respected: bool
    _generate: Callable[..., Image.Image]

    @property
    def pipe(self):
        return getattr(self.engine, "pipe", None)

    def render(self, prompt: str, negative_prompt: str, control_image: Image.Image,
               steps: int, guidance: float, control_scale: float, seed: int) -> Image.Image:
        seed_everything(seed)
        return self._generate(prompt, negative_prompt, control_image, steps, guidance, control_scale)


def build_engine(engine_type: str, options: Optional[Dict] = None) -> EngineAdapter:
    options = options or {}
    if engine_type == "coreml":
        from src.engine.coreml_engine import CoreMLEngine
        engine = CoreMLEngine(model_dir=options.get("model_dir", "models/coreml"))
        # Core ML predictions are synchronous; the PyTorch parts of this engine run on CPU.
        return EngineAdapter("coreml", engine, "cpu", True, _steps_guidance_call(engine))

    if engine_type == "cuda":
        from src.engine.cuda_optimized_engine import CUDAOptimizedEngine
        engine = CUDAOptimizedEngine()
        return EngineAdapter("cuda", engine, "cuda", True, _steps_guidance_call(engine))

    if engine_type == "pytorch":
        from src.engine.pytorch_engine import PyTorchEngine
        engine = PyTorchEngine()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return EngineAdapter("pytorch", engine, device, True, _steps_guidance_call(engine))

    if engine_type == "mps_legacy":
        from src.engine.pipeline import SketchToRenderEngine
        engine = SketchToRenderEngine(options.get("config_path", "configs/inference_lcm.yaml"))

        def call(prompt, negative_prompt, control_image, steps, guidance, control_scale):
            return engine.generate(prompt, negative_prompt, control_image,
                                   num_inference_steps=int(steps), guidance_scale=float(guidance),
                                   controlnet_conditioning_scale=float(control_scale))

        # pipeline.py hardcodes torch.Generator().manual_seed(42), so per-seed variation is lost.
        return EngineAdapter("mps_legacy", engine, engine.device, False, call)

    raise ValueError(f"engine type {engine_type!r} not in {ENGINE_TYPES}")


def _steps_guidance_call(engine):
    def call(prompt, negative_prompt, control_image, steps, guidance, control_scale):
        return engine.generate(prompt=prompt, negative_prompt=negative_prompt,
                               control_image=control_image, steps=int(steps),
                               guidance=float(guidance), control_scale=float(control_scale))
    return call
