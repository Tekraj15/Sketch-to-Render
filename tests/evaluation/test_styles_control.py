import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.evaluation import control
from src.evaluation.styles import compose_prompt, load_style_table


def test_live_style_table_from_app():
    table = load_style_table("app.py")
    # app.py assigns `styles` twice; the second (live) table has the domain-tagged names.
    assert "Hyper-Gloss" in table and "Fashion" in table and "Vintage (Auto)" in table


def test_last_assignment_wins(tmp_path):
    app = tmp_path / "app.py"
    app.write_text("def process_sketch():\n    styles = {'A': 'old'}\n    styles = {'A': 'new'}\n")
    assert load_style_table(str(app)) == {"A": "new"}


def test_missing_table_raises(tmp_path):
    app = tmp_path / "app.py"
    app.write_text("def process_sketch():\n    pass\n")
    with pytest.raises(ValueError):
        load_style_table(str(app))


def test_compose_prompt_matches_app():
    assert compose_prompt("a car", "studio") == "a car, studio"
    assert compose_prompt("", "studio") == "studio"
    assert compose_prompt("a car", "") == "a car"


def _sketch(w, h):
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).line([(5, h // 2), (w - 5, h // 2)], fill="black", width=4)
    return img


def test_draw_path_inverts_and_squares():
    out = control.app_draw_control(_sketch(300, 150), canvas=512)
    arr = np.asarray(out.convert("L"))
    assert out.size == (512, 512)
    assert arr[0, 0] == 0                 # white paper (incl. letterbox) -> black background
    assert arr[256, 256] > 200            # black stroke -> white line


def test_draw_path_handles_transparent_canvas():
    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).line([(10, 256), (500, 256)], fill=(0, 0, 0, 255), width=4)
    arr = np.asarray(control.app_draw_control(rgba).convert("L"))
    assert arr[10, 10] == 0 and arr[256, 256] > 200


def test_auto_falls_back_to_draw_without_import_path(monkeypatch):
    monkeypatch.setattr(control, "has_import_path", lambda: False)
    _, mode = control.make_control_map(_sketch(64, 64), "auto", "Paper Sketch")
    assert mode == "app_draw"
    with pytest.raises(RuntimeError):
        control.make_control_map(_sketch(64, 64), "app_import", "Paper Sketch")
    with pytest.raises(ValueError):
        control.make_control_map(_sketch(64, 64), "bogus", None)
