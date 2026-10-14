import math

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.metrics.structure import control_to_mask, edge_scores, structure_scores


def _line_mask(size=512, x=256, width=1):
    m = np.zeros((size, size), bool)
    m[50:450, x:x + width] = True
    return m


def test_identical_is_perfect():
    m = _line_mask()
    s = edge_scores(m, m, tol_px=3)
    assert s["f1"] == 1.0 and s["chamfer_px"] == 0.0 and not s["empty"]


def test_shift_within_tolerance_matches():
    s = edge_scores(_line_mask(x=258), _line_mask(x=256), tol_px=3)
    assert s["f1"] == 1.0 and math.isclose(s["chamfer_px"], 2.0)


def test_shift_beyond_tolerance_fails():
    s = edge_scores(_line_mask(x=270), _line_mask(x=256), tol_px=3)
    assert s["f1"] == 0.0 and math.isclose(s["chamfer_px"], 14.0)


def test_precision_vs_recall():
    target = _line_mask(x=100)
    pred = _line_mask(x=100) | _line_mask(x=400)  # correct line + a hallucinated one
    s = edge_scores(pred, target, tol_px=3)
    assert s["recall"] == 1.0 and math.isclose(s["precision"], 0.5)


def test_empty_prediction_is_flagged():
    s = edge_scores(np.zeros((64, 64), bool), _line_mask(64, 10), tol_px=3)
    assert s["f1"] == 0.0 and s["empty"] and math.isnan(s["chamfer_px"])


def test_tolerance_and_chamfer_scale_with_resolution():
    # 4 px shift at 1024 == 2 px at 512: inside the 3 px@512 band, chamfer reported in 512-px units.
    s = structure_scores(_line_mask(1024, x=516), _line_mask(1024, x=512))
    assert s["f1"] == 1.0 and math.isclose(s["chamfer_px"], 2.0)


def test_control_to_mask_thins_thick_strokes():
    img = Image.new("RGB", (512, 512), "black")
    ImageDraw.Draw(img).line([(100, 256), (400, 256)], fill="white", width=9)
    mask = control_to_mask(img, 512)
    assert mask.any()
    assert mask[:, 250].sum() <= 2   # one skeleton line crosses each column
