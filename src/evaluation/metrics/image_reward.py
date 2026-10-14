import sys
import types
from typing import Dict, List

from src.evaluation.metrics.base import Scorer, open_image


def import_image_reward():
    """
    ImageReward's package __init__ imports its ReFL *training* module, which imports
    `datasets`. Installing `datasets` would bump huggingface_hub / fsspec past the
    app's pins, and scoring never touches that code path, so a placeholder module is
    registered only for the duration of this import. transformers must be imported
    first: its availability probe (find_spec) rejects a module without __spec__.
    """
    import transformers  # noqa: F401  (see docstring)
    if "ImageReward" in sys.modules:
        return sys.modules["ImageReward"]
    placeholder_added = "datasets" not in sys.modules
    if placeholder_added:
        stub = types.ModuleType("datasets")
        stub.load_dataset = None
        sys.modules["datasets"] = stub
    try:
        import ImageReward
    finally:
        if placeholder_added:
            sys.modules.pop("datasets", None)
    return ImageReward


class ImageRewardScorer(Scorer):
    """ImageReward-v1.0 (Xu et al. 2023) against the full prompt."""
    name = "imagereward"
    requires = ("ImageReward", "fairscale", "clip")

    def load(self, device: str) -> None:
        # BLIP inside ImageReward has MPS op gaps; CPU is fast enough for benchmark sizes.
        super().load("cuda" if device == "cuda" else "cpu")
        self.model = import_image_reward().load("ImageReward-v1.0", device=self.device)

    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        return [{"imagereward": float(self.model.score(r["prompt_full"],
                                                        open_image(run_dir, r["image"])))}
                for r in records]
