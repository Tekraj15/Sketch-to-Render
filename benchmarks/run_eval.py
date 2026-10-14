"""
Sketch-to-Render evaluation harness (WS1).

  generate  render the benchmark with one engine config -> eval/runs/<stamp>_<name>/
  score     load scorers one at a time and write metrics.csv + report.md
  report    rebuild report.md / grid.png from an existing run
  add-item  validate and append a sketch to the benchmark manifest

Examples
  python benchmarks/run_eval.py generate --config configs/eval/edge_coreml_app_defaults.yaml
  python benchmarks/run_eval.py score --run eval/runs/<run_dir>
  python benchmarks/run_eval.py add-item --id scribble-bike --stratum scribble \
      --sketch eval/sketchbench/sketches/scribble-bike.png --prompt "a red motorbike" \
      --style-dna "Hyper-Gloss" --domain auto --owner user
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.evaluation.manifest import BenchItem, append_item, load_manifest, select_items
from src.evaluation.report import (build_report, merge_metric_rows, read_jsonl,
                                   read_metrics_csv, write_metrics_csv)


# ---------------------------------------------------------------- helpers
def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def hardware_info() -> str:
    import psutil
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip() or chip
    return f"{chip}, {psutil.virtual_memory().total / 1024 ** 3:.0f} GB RAM, {platform.system()} {platform.release()}"


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "diffusers", "transformers", "coremltools"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("control_mode", "auto")
    cfg.setdefault("canvas", 512)
    cfg.setdefault("seeds", [0, 1, 2, 3])
    cfg.setdefault("warmup", 1)
    cfg.setdefault("manifest", "eval/sketchbench/manifest.jsonl")
    cfg.setdefault("app_path", "app.py")
    return cfg


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


# ---------------------------------------------------------------- generate
def cmd_generate(args) -> str:
    from src.evaluation.control import load_sketch, make_control_map
    from src.evaluation.engines import build_engine
    from src.evaluation.instrument import ComponentTimer, MemorySampler, swap_used_mb
    from src.evaluation.styles import compose_prompt, load_style_table

    cfg = load_config(args.config)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in _csv(args.seeds)]
    items = select_items(load_manifest(cfg["manifest"]), ids=_csv(args.items),
                         strata=_csv(args.strata), limit=args.limit)
    styles = load_style_table(cfg["app_path"])
    unknown = sorted({i.style_dna for i in items} - set(styles))
    if unknown:
        raise SystemExit(f"Style DNA not in {cfg['app_path']}: {unknown}")

    run_name = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(os.path.join(run_dir, "images"))
    os.makedirs(os.path.join(run_dir, "controls"))
    meta = {"run_name": run_name, "config": cfg, "config_path": args.config, "git": git_info(),
            "hardware": hardware_info(), "versions": versions(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "items": [i.id for i in items], "swap_before_mb": swap_used_mb()}
    write_json(os.path.join(run_dir, "run.json"), meta)
    print(f"** Run dir: {run_dir}  ({len(items)} items x {len(cfg['seeds'])} seeds)")

    # 1. Control maps (timed separately: preprocessing is not engine latency)
    controls = {}
    for item in items:
        t0 = time.perf_counter()
        control, mode = make_control_map(load_sketch(item.sketch), cfg["control_mode"],
                                         item.input_mode, cfg["canvas"])
        rel = os.path.join("controls", f"{item.id}.png")
        control.save(os.path.join(run_dir, rel))
        controls[item.id] = (control, rel, mode, time.perf_counter() - t0)

    # 2. Engine
    t0 = time.perf_counter()
    adapter = build_engine(cfg["engine"]["type"], cfg["engine"].get("options"))
    meta["engine_load_s"] = time.perf_counter() - t0
    meta["seed_respected"] = adapter.seed_respected
    timer = ComponentTimer(adapter.device)
    if adapter.pipe is not None:
        timer.attach(adapter.pipe)
    p = cfg["params"]

    def render(item: BenchItem, seed: int):
        prompt_full = compose_prompt(item.prompt, styles[item.style_dna])
        timer.reset()
        sampler = MemorySampler(adapter.device).start()
        t = time.perf_counter()
        image = adapter.render(prompt_full, p["negative_prompt"], controls[item.id][0],
                               p["steps"], p["guidance"], p["control_scale"], seed)
        total = time.perf_counter() - t
        return image, prompt_full, total, timer.snapshot(), sampler.stop()

    try:
        # 3. Warm-up (first call pays model load / ANE compilation): timed, not scored
        for w in range(cfg["warmup"]):
            _, _, total, _, _ = render(items[0], cfg["seeds"][0])
            if w == 0:
                meta["cold_render_s"] = total
            print(f"   warm-up {w + 1}/{cfg['warmup']}: {total:.2f}s")

        # 4. Benchmark renders
        with open(os.path.join(run_dir, "records.jsonl"), "w", encoding="utf-8") as f:
            for item in items:
                control, control_rel, mode, prep_s = controls[item.id]
                for seed in cfg["seeds"]:
                    image, prompt_full, total, comps, mem = render(item, seed)
                    rel = os.path.join("images", f"{item.id}__s{seed}.png")
                    image.save(os.path.join(run_dir, rel))
                    record = {"item_id": item.id, "seed": seed, "stratum": item.stratum,
                              "domain": item.domain, "style_dna": item.style_dna,
                              "prompt_user": item.prompt, "prompt_full": prompt_full,
                              "negative_prompt": p["negative_prompt"], "params": p,
                              "control_mode": mode, "control": control_rel, "image": rel,
                              "reference": item.reference, "warm": True,
                              "timings": {"total_s": total, "preprocess_s": prep_s,
                                          "components": comps},
                              "memory": mem}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    print(f"   {item.id} s{seed}: {total:.2f}s")
    finally:
        timer.detach()
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["swap_after_mb"] = swap_used_mb()
        write_json(os.path.join(run_dir, "run.json"), meta)
    print(f"** Done. Next: python benchmarks/run_eval.py score --run {run_dir}")
    return run_dir


# ---------------------------------------------------------------- score
def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_score(args) -> None:
    from src.evaluation.metrics import DEFAULT_METRICS, get_scorer

    run_dir = args.run
    records = [r for r in read_jsonl(os.path.join(run_dir, "records.jsonl")) if r.get("warm")]
    device = args.device or default_device()
    metrics = _csv(args.metrics) or list(DEFAULT_METRICS)
    skipped_path = os.path.join(run_dir, "skipped_metrics.jsonl")
    skipped = [s for s in (read_jsonl(skipped_path) if os.path.exists(skipped_path) else [])
               if s["metric"] not in metrics]
    rows = read_metrics_csv(os.path.join(run_dir, "metrics.csv"))
    keys = [{"item_id": r["item_id"], "seed": r["seed"], "stratum": r["stratum"]} for r in records]

    for name in metrics:
        scorer = get_scorer(name)
        ok, reason = scorer.available()
        if not ok:
            skipped.append({"metric": name, "reason": reason})
            print(f"   - {name}: skipped ({reason})")
            continue
        t0 = time.perf_counter()
        try:
            scorer.load(device)
            result = scorer.score(records, run_dir)
        except Exception as e:  # one broken scorer must not lose the others' results
            skipped.append({"metric": name, "reason": f"error: {type(e).__name__}: {e}"})
            print(f"   - {name}: FAILED ({type(e).__name__}: {e})")
            continue
        finally:
            scorer.unload()
        if scorer.kind == "set":
            with open(os.path.join(run_dir, "set_metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"metric": name, **result}) + "\n")
        else:
            rows = merge_metric_rows(rows, [{**k, **v} for k, v in zip(keys, result)])
            write_metrics_csv(os.path.join(run_dir, "metrics.csv"), rows)
        print(f"   ✓ {name} ({time.perf_counter() - t0:.0f}s)")

    with open(skipped_path, "w", encoding="utf-8") as f:
        for s in skipped:
            f.write(json.dumps(s) + "\n")
    report = build_report(run_dir)
    print(f"** Report: {report}")
    if args.wandb:
        log_wandb(run_dir)


def log_wandb(run_dir: str) -> None:
    """Optional: requires WANDB_API_KEY in the user's own environment."""
    import wandb
    from src.evaluation.report import METRIC_INFO, latency_summary, summarize
    meta = json.load(open(os.path.join(run_dir, "run.json")))
    rows = read_metrics_csv(os.path.join(run_dir, "metrics.csv"))
    records = read_jsonl(os.path.join(run_dir, "records.jsonl"))
    run = wandb.init(project="sketch-to-render-eval", name=meta["run_name"], config=meta["config"])
    summary = {m: summarize(rows, m)["mean"] for m in METRIC_INFO if any(r.get(m) is not None for r in rows)}
    summary["latency_p50_s"] = latency_summary(records)["total"]["p50"]
    run.log({**summary, "grid": wandb.Image(os.path.join(run_dir, "grid.png"))})
    run.finish()


# ---------------------------------------------------------------- report / add-item
def cmd_report(args) -> None:
    print(f"** Report: {build_report(args.run)}")


def cmd_add_item(args) -> None:
    item = BenchItem(id=args.id, stratum=args.stratum, sketch=args.sketch, prompt=args.prompt,
                     style_dna=args.style_dna, domain=args.domain, owner=args.owner,
                     reference=args.reference, input_mode=args.input_mode, notes=args.notes)
    from src.evaluation.styles import load_style_table
    if item.style_dna not in load_style_table(args.app_path):
        raise SystemExit(f"Style DNA {item.style_dna!r} not in {args.app_path}")
    append_item(args.manifest, item)
    print(f"** Added {item.id} to {args.manifest}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--items", help="comma-separated item ids")
    g.add_argument("--strata", help="comma-separated strata")
    g.add_argument("--seeds", help="comma-separated seeds (overrides config)")
    g.add_argument("--limit", type=int)
    g.add_argument("--out", default="eval/runs")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--metrics", help="comma-separated; default: all available")
    s.add_argument("--device", choices=("cpu", "mps", "cuda"))
    s.add_argument("--wandb", action="store_true")
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.set_defaults(fn=cmd_report)

    a = sub.add_parser("add-item")
    for flag in ("--id", "--stratum", "--sketch", "--prompt", "--style-dna", "--domain"):
        a.add_argument(flag, required=True)
    a.add_argument("--owner", default="user")
    a.add_argument("--reference")
    a.add_argument("--input-mode")
    a.add_argument("--notes", default="")
    a.add_argument("--manifest", default="eval/sketchbench/manifest.jsonl")
    a.add_argument("--app-path", default="app.py")
    a.set_defaults(fn=cmd_add_item)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
