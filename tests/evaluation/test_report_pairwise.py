import json
import math
import os

import pytest
from PIL import Image

from src.evaluation import pairwise, report
from src.evaluation.metrics import REGISTRY, get_scorer


# ---------- report ----------
def test_bootstrap_ci_basic():
    mean, lo, hi = report.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and lo < 2.5 < hi
    assert report.bootstrap_ci([5.0]) == (5.0, None, None)
    assert math.isnan(report.bootstrap_ci([None, float("nan")])[0])


def test_seeds_averaged_per_item_first():
    rows = [{"item_id": "a", "seed": 0, "m": 1.0}, {"item_id": "a", "seed": 1, "m": 3.0},
            {"item_id": "b", "seed": 0, "m": 10.0}]
    assert report.per_item_means(rows, "m") == {"a": 2.0, "b": 10.0}
    assert report.summarize(rows, "m")["mean"] == 6.0   # not (1+3+10)/3


def test_merge_metric_rows_incremental():
    a = [{"item_id": "x", "seed": 0, "stratum": "s", "clip": 1.0}]
    b = [{"item_id": "x", "seed": 0, "stratum": "s", "dino": 0.5}]
    assert report.merge_metric_rows(a, b) == [{"item_id": "x", "seed": 0, "stratum": "s",
                                               "clip": 1.0, "dino": 0.5}]


def _record(item, seed, total, unet):
    return {"item_id": item, "seed": seed, "stratum": "scribble", "warm": True,
            "control": f"controls/{item}.png", "image": f"images/{item}__s{seed}.png",
            "reference": None, "control_mode": "app_draw",
            "timings": {"total_s": total, "components": {"unet": {"seconds": unet, "calls": 4}}},
            "memory": {"peak_rss_mb": 100.0}}


def test_latency_summary_other_bucket():
    lat = report.latency_summary([_record("a", 0, 2.0, 1.5), _record("a", 1, 4.0, 3.0)])
    assert lat["total"]["p50"] == 3.0
    assert lat["components"]["unet"]["calls_per_image"] == 4
    assert lat["components"]["other (scheduler/pre/post)"]["mean"] == pytest.approx(0.75)


def test_build_report_end_to_end(tmp_path):
    run = tmp_path
    os.makedirs(run / "images"); os.makedirs(run / "controls")
    records = []
    for item in ("a", "b"):
        Image.new("RGB", (64, 64), "black").save(run / "controls" / f"{item}.png")
        for seed in (0, 1):
            Image.new("RGB", (64, 64), "red").save(run / "images" / f"{item}__s{seed}.png")
            records.append(_record(item, seed, 2.0 + seed, 1.0))
    (run / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    meta = {"run_name": "t", "config": {"engine": {"type": "coreml"}, "seeds": [0, 1],
            "params": {"steps": 4, "guidance": 1.0, "control_scale": 1.0}},
            "git": {"sha": "abc", "dirty": False, "branch": "b"}, "hardware": "hw",
            "started_at": "now", "seed_respected": True}
    (run / "run.json").write_text(json.dumps(meta))
    report.write_metrics_csv(str(run / "metrics.csv"),
                             [{"item_id": r["item_id"], "seed": r["seed"], "stratum": "scribble",
                               "struct_f1": 0.5} for r in records])
    out = report.build_report(str(run))
    text = open(out).read()
    assert "Structure F1" in text and "Latency" in text
    assert os.path.exists(run / "grid.png")


# ---------- pairwise ----------
def _rating(item, choice, left="A", right="B", q="quality"):
    return {"item_id": item, "seed": "0", "question": q, "left_run": left, "right_run": right,
            "choice": choice}


def test_win_rate_with_sides_and_ties():
    ratings = [_rating("1", "left"), _rating("2", "right", left="B", right="A"),
               _rating("3", "tie")]
    w = pairwise.win_rate(ratings, "A")
    assert w["win_rate"] == pytest.approx(2.5 / 3) and w["n_items"] == 3
    assert pairwise.win_rate(ratings, "B")["win_rate"] == pytest.approx(0.5 / 3)
    assert pairwise.win_rate(ratings, "A", question="structure")["win_rate"] is None


def test_elo_and_bradley_terry_rank_the_winner():
    matches = [("A", "B", 1.0)] * 3 + [("A", "B", 0.0)]
    ratings = pairwise.elo(matches)
    assert ratings["A"] > ratings["B"]
    p = pairwise.bradley_terry(matches)
    assert p["A"] / (p["A"] + p["B"]) == pytest.approx(0.75, abs=1e-3)


# ---------- metrics registry ----------
def test_registry_classes_import_without_heavy_deps():
    for name in REGISTRY:
        scorer = get_scorer(name)
        assert scorer.name == name


def test_vqascore_is_reported_unavailable():
    ok, reason = get_scorer("vqascore").available()
    assert not ok and "16 GB" in reason


def test_unknown_metric():
    with pytest.raises(KeyError):
        get_scorer("fid")
