"""
Structure fidelity: does the render follow the sketch?

Input side  = the control map the engine actually received (white lines on black),
              binarized and thinned to 1-px skeletons.
Output side = HED soft edges of the render -- a different extractor family from the
              app's threshold/invert preprocessing, so the metric is not circular --
              thresholded and thinned.
Scores      = precision / recall / F1 of skeleton pixels matched within a tolerance
              band, plus symmetric Chamfer distance. Distances are reported in px at
              512 scale so runs at other resolutions stay comparable.
"""
import math
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin

from src.evaluation.metrics.base import Scorer, open_image

REFERENCE_SIDE = 512
DEFAULT_TOL_PX = 3.0  # at 512 px


def skeleton(mask: np.ndarray) -> np.ndarray:
    return thin(mask.astype(bool)) if mask.any() else mask.astype(bool)


def control_to_mask(control: Image.Image, size: int) -> np.ndarray:
    """White-on-black control map -> thinned boolean line mask at `size`."""
    gray = control.convert("L")
    if gray.size != (size, size):
        gray = gray.resize((size, size), Image.BOX)  # area resampling keeps thin strokes
    return skeleton(np.asarray(gray) > 64)


def edge_scores(pred: np.ndarray, target: np.ndarray, tol_px: float) -> Dict[str, float]:
    """pred/target: boolean skeleton masks of equal shape."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() or not target.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "chamfer_px": math.nan,
                "empty": True}
    dist_to_target = distance_transform_edt(~target)
    dist_to_pred = distance_transform_edt(~pred)
    d_pred = dist_to_target[pred]      # each predicted edge pixel -> nearest sketch line
    d_target = dist_to_pred[target]    # each sketch line pixel -> nearest predicted edge
    precision = float(np.mean(d_pred <= tol_px))
    recall = float(np.mean(d_target <= tol_px))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    chamfer = 0.5 * (float(np.mean(d_pred)) + float(np.mean(d_target)))
    return {"precision": precision, "recall": recall, "f1": f1, "chamfer_px": chamfer,
            "empty": False}


def structure_scores(pred: np.ndarray, target: np.ndarray, tol_px_at_512: float = DEFAULT_TOL_PX
                     ) -> Dict[str, float]:
    scale = pred.shape[0] / REFERENCE_SIDE
    out = edge_scores(pred, target, tol_px_at_512 * scale)
    if not math.isnan(out["chamfer_px"]):
        out["chamfer_px"] /= scale
    return out


class HEDEdges:
    def __init__(self, threshold: int = 96):
        from controlnet_aux import HEDdetector
        self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        self.threshold = threshold

    def to(self, device):
        self.detector.to(device)
        return self

    def __call__(self, image: Image.Image) -> np.ndarray:
        side = image.size[0]
        soft = self.detector(image, detect_resolution=side, image_resolution=side)
        soft = np.asarray(soft.convert("L").resize(image.size, Image.BILINEAR))
        return skeleton(soft > self.threshold)


class StructureScorer(Scorer):
    name = "structure"
    requires = ("controlnet_aux", "skimage", "scipy")

    def __init__(self, tol_px_at_512: float = DEFAULT_TOL_PX):
        self.tol = tol_px_at_512

    def load(self, device: str) -> None:
        super().load(device)
        # HED is a small CNN; MPS works but CPU avoids MPS op gaps in controlnet_aux.
        self.edges = HEDEdges().to("cpu")

    def score(self, records: List[Dict], run_dir: str) -> List[Dict]:
        out = []
        for rec in records:
            render = open_image(run_dir, rec["image"])
            control = open_image(run_dir, rec["control"])
            side = render.size[0]
            pred = self.edges(render)
            target = control_to_mask(control, side)
            s = structure_scores(pred, target, self.tol)
            out.append({"struct_f1": s["f1"], "struct_precision": s["precision"],
                        "struct_recall": s["recall"], "struct_chamfer_px": s["chamfer_px"]})
        return out
