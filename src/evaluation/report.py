import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

KEY_FIELDS = ("item_id", "seed", "stratum")
# metric -> (label, higher_is_better)
METRIC_INFO = {
    "struct_f1": ("Structure F1 @3px (HED)", True),
    "struct_precision": ("Structure precision", True),
    "struct_recall": ("Structure recall", True),
    "struct_chamfer_px": ("Structure Chamfer (px @512)", False),
    "clip_prompt": ("CLIP-L/14 x100 vs user prompt", True),
    "clip_full": ("CLIP-L/14 x100 vs full prompt", True),
    "dino_ref_sim": ("DINOv2 sim to reference", True),
    "pickscore": ("PickScore", True),
    "hpsv2": ("HPSv2.1", True),
    "imagereward": ("ImageReward", True),
}


# ---------- io ----------
def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_metrics_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seed"] = int(row["seed"])
        for k, v in list(row.items()):
            if k not in KEY_FIELDS:
                row[k] = _to_float(v)
    return rows


def write_metrics_csv(path: str, rows: List[Dict]) -> None:
    cols = list(KEY_FIELDS) + sorted({k for r in rows for k in r} - set(KEY_FIELDS))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def merge_metric_rows(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge per-record metric dicts keyed by (item_id, seed) so metrics can be scored incrementally."""
    by_key = {(r["item_id"], r["seed"]): dict(r) for r in existing}
    for r in new:
        by_key.setdefault((r["item_id"], r["seed"]), {}).update(r)
    return [by_key[k] for k in sorted(by_key)]


def _to_float(v) -> Optional[float]:
    if v in ("", None):
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except ValueError:
        return None


# ---------- statistics ----------
def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, Optional[float], Optional[float]]:
    vals = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if len(vals) == 0:
        return math.nan, None, None
    if len(vals) == 1:
        return float(vals[0]), None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def per_item_means(rows: List[Dict], metric: str) -> Dict[str, float]:
    """Seeds are averaged within an item first; items are the unit of the CI."""
    acc = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            acc[r["item_id"]].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def summarize(rows: List[Dict], metric: str) -> Dict:
    items = per_item_means(rows, metric)
    mean, lo, hi = bootstrap_ci(items.values())
    return {"mean": mean, "lo": lo, "hi": hi, "n_items": len(items)}


def percentile(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def latency_summary(records: List[Dict]) -> Dict:
    warm = [r for r in records if r.get("warm")]
    totals = [r["timings"]["total_s"] for r in warm]
    comps = defaultdict(list)
    calls = defaultdict(list)
    for r in warm:
        for name, s in r["timings"].get("components", {}).items():
            comps[name].append(s["seconds"])
            calls[name].append(s["calls"])
        known = sum(s["seconds"] for s in r["timings"].get("components", {}).values())
        comps["other (scheduler/pre/post)"].append(max(r["timings"]["total_s"] - known, 0.0))
    return {
        "n_warm": len(warm),
        "total": {"p50": percentile(totals, 50), "p95": percentile(totals, 95),
                  "mean": float(np.mean(totals)) if totals else None},
        "components": {k: {"p50": percentile(v, 50), "mean": float(np.mean(v)),
                           "calls_per_image": float(np.mean(calls[k])) if calls.get(k) else None}
                       for k, v in comps.items()},
        "peak_rss_mb": max((r["memory"]["peak_rss_mb"] for r in warm), default=None),
    }


# ---------- rendering ----------
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(s: Dict, nd=3) -> str:
    if s["lo"] is None:
        return f"{_fmt(s['mean'], nd)}"
    return f"{_fmt(s['mean'], nd)} [{_fmt(s['lo'], nd)}, {_fmt(s['hi'], nd)}]"


def build_grid(run_dir: str, records: List[Dict], thumb: int = 192, max_seeds: int = 4) -> str:
    by_item = defaultdict(list)
    for r in records:
        if r.get("warm"):
            by_item[r["item_id"]].append(r)
    cols = 2 + max_seeds
    label_h = 18
    grid = Image.new("RGB", (cols * thumb, len(by_item) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    for row, (item_id, recs) in enumerate(sorted(by_item.items())):
        recs = sorted(recs, key=lambda r: r["seed"])[:max_seeds]
        y = row * (thumb + label_h)
        draw.text((4, y + 2), f"{item_id}  [{recs[0]['stratum']}]", fill="black")
        tiles = [os.path.join(run_dir, recs[0]["control"])] + \
                [os.path.join(run_dir, r["image"]) for r in recs]
        ref = recs[0].get("reference")
        for col, path in enumerate(tiles):
            grid.paste(_thumb(path, thumb), (col * thumb, y + label_h))
        if ref and os.path.exists(ref):
            grid.paste(_thumb(ref, thumb), ((cols - 1) * thumb, y + label_h))
    out = os.path.join(run_dir, "grid.png")
    grid.save(out)
    return out


def _thumb(path: str, side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    tile = Image.new("RGB", (side, side), (235, 235, 235))
    tile.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    return tile


def build_report(run_dir: str) -> str:
    meta = json.load(open(os.path.join(run_dir, "run.json")))
    records = read_jsonl(os.path.join(run_dir, "records.jsonl"))
    rows = read_metrics_csv(os.path.join(run_dir, "metrics.csv"))
    set_path = os.path.join(run_dir, "set_metrics.jsonl")
    set_metrics = read_jsonl(set_path)[-1] if os.path.exists(set_path) else {}
    skipped_path = os.path.join(run_dir, "skipped_metrics.jsonl")
    skipped = read_jsonl(skipped_path) if os.path.exists(skipped_path) else []

    lat = latency_summary(records)
    cfg = meta["config"]
    lines = [f"# Eval report: `{meta['run_name']}`", "",
             "| | |", "|---|---|",
             f"| Engine | `{cfg['engine']['type']}` |",
             f"| Params | steps={cfg['params']['steps']}, guidance={cfg['params']['guidance']}, "
             f"control_scale={cfg['params']['control_scale']} |",
             f"| Control mode | `{cfg.get('control_mode', 'auto')}` (effective: "
             f"{', '.join(sorted({r['control_mode'] for r in records}))}) |",
             f"| Seeds respected | {meta.get('seed_respected')} |",
             f"| Items x seeds | {len({r['item_id'] for r in records if r.get('warm')})} x "
             f"{len(cfg['seeds'])} = {lat['n_warm']} renders |",
             f"| Git | `{meta['git']['sha'][:10]}`{' (dirty)' if meta['git']['dirty'] else ''} on "
             f"`{meta['git']['branch']}` |",
             f"| Hardware | {meta['hardware']} |",
             f"| Started | {meta['started_at']} |", ""]

    lines += ["## Quality (mean [95% bootstrap CI] across items; seeds averaged per item)", "",
              "| Metric | All | " + " | ".join(_strata(rows)) + " |",
              "|---|---|" + "---|" * len(_strata(rows))]
    for metric, (label, higher) in METRIC_INFO.items():
        if not any(r.get(metric) is not None for r in rows):
            continue
        arrow = "↑" if higher else "↓"
        cells = [_ci(summarize(rows, metric))]
        for s in _strata(rows):
            cells.append(_ci(summarize([r for r in rows if r["stratum"] == s], metric)))
        lines.append(f"| {label} {arrow} | " + " | ".join(cells) + " |")
    if set_metrics:
        lines += ["", f"**KID** (renders vs references): {_fmt(set_metrics.get('kid_mean'), 4)} "
                      f"± {_fmt(set_metrics.get('kid_std'), 4)}; {set_metrics.get('kid_note', '')}"]
    if skipped:
        lines += ["", "**Skipped metrics:** " + "; ".join(f"`{s['metric']}`: {s['reason']}" for s in skipped)]

    lines += ["", f"## Latency (warm renders, n={lat['n_warm']})", "",
              "| Stage | p50 (s) | mean (s) | calls / image |", "|---|---|---|---|",
              f"| **Total** | {_fmt(lat['total']['p50'], 2)} | {_fmt(lat['total']['mean'], 2)} | |"]
    for name, s in sorted(lat["components"].items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(f"| {name} | {_fmt(s['p50'], 2)} | {_fmt(s['mean'], 2)} | "
                     f"{_fmt(s['calls_per_image'], 1) if s['calls_per_image'] else ''} |")
    lines += ["", f"- p95 total: {_fmt(lat['total']['p95'], 2)} s",
              f"- Engine load: {_fmt(meta.get('engine_load_s'), 1)} s · "
              f"cold first render: {_fmt(meta.get('cold_render_s'), 2)} s",
              f"- Peak RSS: {_fmt(lat['peak_rss_mb'], 0)} MB · swap used before/after run: "
              f"{_fmt(meta.get('swap_before_mb'), 0)} / {_fmt(meta.get('swap_after_mb'), 0)} MB",
              "", "## Grid", "", "Columns: control map · seeds · reference (if paired).", "",
              "![grid](grid.png)", ""]

    build_grid(run_dir, records)
    out = os.path.join(run_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def _strata(rows: List[Dict]) -> List[str]:
    return sorted({r["stratum"] for r in rows})
