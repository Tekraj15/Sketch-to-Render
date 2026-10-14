import gc
import importlib.util
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


class Scorer:
    """
    One metric. `per_image` scorers return one dict per record; `set` scorers
    return a single dict for the whole run. Scorers are loaded one at a time
    by the score phase and unloaded before the next (8 GB machines).
    """
    name = "base"
    kind = "per_image"
    requires: Tuple[str, ...] = ()
    needs_reference = False

    def available(self) -> Tuple[bool, str]:
        missing = [m for m in self.requires if importlib.util.find_spec(m) is None]
        return (False, f"missing packages: {', '.join(missing)}") if missing else (True, "")

    def load(self, device: str) -> None:
        self.device = device

    def score(self, records: List[Dict], run_dir: str):
        raise NotImplementedError

    def unload(self) -> None:
        for attr in list(vars(self)):
            if attr != "device":
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def open_image(run_dir: str, rel_path: str) -> Image.Image:
    return Image.open(os.path.join(run_dir, rel_path)).convert("RGB")


def open_reference(record: Dict, repo_root: str = ".") -> Optional[Image.Image]:
    ref = record.get("reference")
    return Image.open(os.path.join(repo_root, ref)).convert("RGB") if ref else None


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
