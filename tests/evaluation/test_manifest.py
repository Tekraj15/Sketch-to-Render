import json

import pytest
from PIL import Image

from src.evaluation.manifest import (BenchItem, ManifestError, append_item, load_manifest,
                                     select_items)


def _item(tmp_path, **kw):
    sketch = tmp_path / "s.png"
    if not sketch.exists():
        Image.new("RGB", (8, 8), "white").save(sketch)
    base = dict(id="a", stratum="scribble", sketch="s.png", prompt="a car", style_dna="Sketch",
                domain="auto", owner="user")
    base.update(kw)
    return BenchItem(**base)


def _write(path, items):
    path.write_text("\n".join(json.dumps(vars(i)) for i in items) + "\n")


def test_roundtrip_and_comments(tmp_path):
    m = tmp_path / "m.jsonl"
    _write(m, [_item(tmp_path)])
    m.write_text("# comment\n\n" + m.read_text())
    items = load_manifest(str(m), root=str(tmp_path))
    assert [i.id for i in items] == ["a"]


@pytest.mark.parametrize("kw, msg", [
    ({"stratum": "doodle"}, "stratum"),
    ({"owner": "someone"}, "owner"),
    ({"input_mode": "Fax"}, "input_mode"),
    ({"prompt": "  "}, "empty prompt"),
    ({"sketch": "missing.png"}, "not found"),
    ({"id": "has space"}, "invalid id"),
])
def test_validation_errors(tmp_path, kw, msg):
    m = tmp_path / "m.jsonl"
    _write(m, [_item(tmp_path, **kw)])
    with pytest.raises(ManifestError, match=msg):
        load_manifest(str(m), root=str(tmp_path))


def test_duplicate_ids_rejected(tmp_path):
    m = tmp_path / "m.jsonl"
    _write(m, [_item(tmp_path), _item(tmp_path)])
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(str(m), root=str(tmp_path))


def test_unknown_field_rejected(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps({**vars(_item(tmp_path)), "extra": 1}) + "\n")
    with pytest.raises(ManifestError):
        load_manifest(str(m), root=str(tmp_path))


def test_append_item(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = tmp_path / "m.jsonl"
    append_item(str(m), _item(tmp_path, id="one"))
    append_item(str(m), _item(tmp_path, id="two"))
    assert [i.id for i in load_manifest(str(m))] == ["one", "two"]
    with pytest.raises(ManifestError, match="duplicate"):
        append_item(str(m), _item(tmp_path, id="one"))


def test_select_items(tmp_path):
    items = [_item(tmp_path, id="x", stratum="scribble"), _item(tmp_path, id="y", stratum="real_photo")]
    assert [i.id for i in select_items(items, strata=["real_photo"])] == ["y"]
    assert [i.id for i in select_items(items, limit=1)] == ["x"]
    with pytest.raises(ManifestError, match="unknown"):
        select_items(items, ids=["nope"])


def test_repo_manifest_schema_is_valid():
    # File existence is machine-specific (third-party images are local-only), so schema only.
    items = load_manifest("eval/sketchbench/manifest.jsonl", check_files=False)
    assert len(items) >= 10
    assert all(i.owner == "user" for i in items if i.sketch.startswith("eval/"))
