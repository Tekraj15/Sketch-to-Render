# SketchBench

The benchmark set for the Sketch-to-Render evaluation harness (`benchmarks/run_eval.py`).
Current version: **v0**, 13 items. It is a seed set; the target is about 120 items.

## Data policy

Only the manifest and sketches **owned by the project author** are committed. These
live under `sketches/` with `"owner": "user"`.

Third-party images (stock photos, web images, anything watermarked) are referenced
by local path with `"owner": "third_party"` and are **never committed**. On a fresh
clone those items fail file validation until the images are placed at the same paths.
Run with `--strata scribble`, or `--items`, to restrict a run to the committed items.

## Strata

| Stratum | What it is | Control path it exercises |
|---|---|---|
| `scribble` | rough strokes drawn on the app canvas | Draw tab |
| `lineart` | clean line drawings, scans or tablet | Draw / Import "Digital Sketch" |
| `shaded_drawing` | pencil drawings with shading and hatching | Import "Paper Sketch" |
| `paper_photo` | phone photo of a sketch on paper (shadows, perspective, texture) | Import "Paper Sketch" |
| `real_photo` | a photograph used as structure input | Import "Real Photo" |

## Manifest schema (`manifest.jsonl`, one JSON object per line)

The file is JSON Lines because `.gitignore` ignores `*.json` repo-wide.

| Field | Notes |
|---|---|
| `id` | unique, no spaces |
| `stratum` | see the table above |
| `sketch` | path relative to the repo root |
| `reference` | optional paired photo. Enables DINOv2 similarity and KID |
| `prompt` | the short text a user would type (*without* the Style-DNA suffix) |
| `style_dna` | must be a key in `app.py`'s live Style-DNA table (checked on load) |
| `domain` | auto · people · aviation · urban · floral · … |
| `input_mode` | import-path hint: `Digital Sketch` · `Paper Sketch` · `Real Photo` |
| `owner` | `user` or `third_party` |
| `notes` | free text |

## Adding an item

```bash
python benchmarks/run_eval.py add-item --id scribble-bike --stratum scribble \
  --sketch eval/sketchbench/sketches/scribble-bike.png \
  --prompt "a red motorbike leaning into a corner" --style-dna "Hyper-Gloss" \
  --domain auto --owner user --input-mode "Digital Sketch"
```

The command validates the schema, the file paths and the Style DNA before appending.
To export a canvas drawing, download it from the Gradio Sketchpad, or crop it from a
screenshot as was done for the three v0 scribbles.

## Running the harness

Use the `.sketchrenderenv` environment.

```bash
# 1. Render (engine process only)
python benchmarks/run_eval.py generate --config configs/eval/edge_coreml_app_defaults.yaml
# 2. Score in a separate process: scorers load one at a time, which fits 8 GB of RAM
python benchmarks/run_eval.py score --run eval/runs/<run_dir>
# 3. Blind A/B between two runs, then summarize
python benchmarks/ab_rater.py --run-a eval/runs/<A> --run-b eval/runs/<B> --rater <name>
python benchmarks/ab_rater.py --summarize eval/runs/ratings/<A>__vs__<B>.csv
```

Each run directory contains:

| File | Contents |
|---|---|
| `run.json` | config, git SHA, versions, hardware |
| `records.jsonl` | one line per render, with timings and memory |
| `images/`, `controls/` | renders and the control maps they were made from |
| `metrics.csv` | per-render scores |
| `report.md`, `grid.png` | summary report and image grid |
