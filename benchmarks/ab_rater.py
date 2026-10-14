"""
Blind pairwise A/B rater for two eval runs (WS1 release gate).

For every (item, seed) present in both runs, shows the control map plus the two renders
with sides randomized, asks three questions (ties allowed), and appends answers to a CSV.
Re-launching with the same --rater resumes where that rater stopped.

  python benchmarks/ab_rater.py --run-a eval/runs/<A> --run-b eval/runs/<B> --rater alice
  python benchmarks/ab_rater.py --summarize eval/runs/ratings/<A>__vs__<B>.csv
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.pairwise import (QUESTIONS, RATING_FIELDS, bradley_terry, elo, load_ratings,
                                     to_match, win_rate)
from src.evaluation.report import read_jsonl

QUESTION_TEXT = {
    "structure": "Which follows the sketch better?",
    "quality": "Which looks more like a finished, professional render?",
    "prompt": "Which matches the description better?",
}
CHOICES = ["left", "right", "tie"]


def build_pairs(run_a: str, run_b: str, seed: int):
    def index(run):
        return {(r["item_id"], r["seed"]): r for r in read_jsonl(os.path.join(run, "records.jsonl"))
                if r.get("warm")}
    a, b = index(run_a), index(run_b)
    name_a, name_b = os.path.basename(run_a.rstrip("/")), os.path.basename(run_b.rstrip("/"))
    rng = random.Random(seed)
    pairs = []
    for key in sorted(set(a) & set(b)):
        swap = rng.random() < 0.5
        (lrun, lrec), (rrun, rrec) = ((name_b, b[key]), (name_a, a[key])) if swap else \
                                     ((name_a, a[key]), (name_b, b[key]))
        pairs.append({"item_id": key[0], "seed": key[1], "prompt": a[key]["prompt_user"],
                      "control": os.path.join(run_a, a[key]["control"]),
                      "left_run": lrun, "right_run": rrun,
                      "left_image": os.path.join(run_a if lrun == name_a else run_b, lrec["image"]),
                      "right_image": os.path.join(run_a if rrun == name_a else run_b, rrec["image"])})
    rng.shuffle(pairs)
    return pairs


def rated_keys(path: str, rater: str):
    if not os.path.exists(path):
        return set()
    return {(r["item_id"], int(r["seed"])) for r in load_ratings(path) if r["rater"] == rater}


def append_ratings(path: str, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RATING_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def launch(args):
    import gradio as gr

    out = args.out or os.path.join("eval", "runs", "ratings",
                                   f"{os.path.basename(args.run_a.rstrip('/'))}__vs__"
                                   f"{os.path.basename(args.run_b.rstrip('/'))}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = rated_keys(out, args.rater)
    queue = [p for p in build_pairs(args.run_a, args.run_b, args.seed)
             if (p["item_id"], p["seed"]) not in done]

    def view(i):
        if i >= len(queue):
            return (f"### ✅ Done: all pairs rated. Ratings saved to `{out}`", "", None, None, None,
                    *[None] * len(QUESTIONS))
        p = queue[i]
        return (f"### Pair {i + 1} / {len(queue)}  ·  rater `{args.rater}`",
                f"**Description:** {p['prompt']}", p["control"], p["left_image"], p["right_image"],
                *[None] * len(QUESTIONS))

    def submit(i, *answers):
        if i < len(queue):
            if any(a is None for a in answers):
                gr.Warning("Answer all three questions (Tie is allowed).")
                return (i, *view(i)[:5], *answers)
            p = queue[i]
            stamp = datetime.now().isoformat(timespec="seconds")
            append_ratings(out, [{"timestamp": stamp, "rater": args.rater, "item_id": p["item_id"],
                                  "seed": p["seed"], "question": q, "left_run": p["left_run"],
                                  "right_run": p["right_run"], "choice": a}
                                 for q, a in zip(QUESTIONS, answers)])
            i += 1
        return (i, *view(i))

    with gr.Blocks(title="Sketch-to-Render A/B") as demo:
        idx = gr.State(0)
        status = gr.Markdown()
        prompt = gr.Markdown()
        with gr.Row():
            control = gr.Image(label="Sketch (what the model was given)", interactive=False, height=320)
            left = gr.Image(label="Left", interactive=False, height=320)
            right = gr.Image(label="Right", interactive=False, height=320)
        radios = [gr.Radio(CHOICES, label=QUESTION_TEXT[q]) for q in QUESTIONS]
        btn = gr.Button("Submit & next", variant="primary")
        outputs = [status, prompt, control, left, right, *radios]
        demo.load(lambda: view(0), None, outputs)
        btn.click(submit, [idx, *radios], [idx, *outputs])
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


def summarize(path: str):
    ratings = load_ratings(path)
    systems = sorted({s for r in ratings for s in (r["left_run"], r["right_run"])})
    print(f"{len(ratings)} ratings, raters: {sorted({r['rater'] for r in ratings})}")
    for q in (None, *QUESTIONS):
        label = q or "all questions"
        for s in systems:
            w = win_rate(ratings, s, q)
            ci = f"[{w['lo']:.2f}, {w['hi']:.2f}]" if w["lo"] is not None else ""
            print(f"  {label:>14} | {s}: win rate {w['win_rate']:.3f} {ci} (n={w['n_ratings']})")
    matches = [to_match(r) for r in ratings]
    print("  Elo:", {k: round(v) for k, v in elo(matches).items()})
    print("  Bradley-Terry:", {k: round(v, 3) for k, v in bradley_terry(matches).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a")
    ap.add_argument("--run-b")
    ap.add_argument("--rater")
    ap.add_argument("--out", help="ratings CSV (default: eval/runs/ratings/<A>__vs__<B>.csv)")
    ap.add_argument("--seed", type=int, default=0, help="pair order / side randomization seed")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--summarize", metavar="CSV", help="print win rate / Elo / BT for a ratings file")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.run_a and args.run_b and args.rater:
        launch(args)
    else:
        ap.error("need --run-a, --run-b and --rater (or --summarize CSV)")


if __name__ == "__main__":
    main()
