from typing import Dict, List

import numpy as np
import torch

from src.evaluation.metrics.base import Scorer, open_image, open_reference

MIN_SAMPLES = 20


class KIDScorer(Scorer):
    """
    Set-level Kernel Inception Distance between renders and the distinct reference
    photos (paired items only). KID is unbiased at small n, unlike FID, but still
    needs a minimum sample count; below MIN_SAMPLES it is reported as skipped.
    """
    name = "kid"
    kind = "set"
    requires = ("torchmetrics", "torch_fidelity")
    needs_reference = True

    def score(self, records: List[Dict], run_dir: str) -> Dict:
        paired = [r for r in records if r.get("reference")]
        refs = sorted({r["reference"] for r in paired})
        n = min(len(paired), len(refs))
        if n < MIN_SAMPLES:
            return {"kid_mean": None, "kid_std": None,
                    "kid_note": f"skipped: {len(refs)} distinct references (< {MIN_SAMPLES})"}
        from torchmetrics.image.kid import KernelInceptionDistance
        kid = KernelInceptionDistance(subset_size=min(n, 50), normalize=True)

        def to_tensor(images):
            arr = np.stack([np.asarray(im.resize((299, 299))) for im in images]) / 255.0
            return torch.from_numpy(arr).permute(0, 3, 1, 2).float()

        kid.update(to_tensor([open_reference({"reference": p}) for p in refs]), real=True)
        kid.update(to_tensor([open_image(run_dir, r["image"]) for r in paired]), real=False)
        mean, std = kid.compute()
        return {"kid_mean": float(mean), "kid_std": float(std),
                "kid_note": f"{len(paired)} renders vs {len(refs)} references"}
