# CHG-20261013-01: WS1 Evaluation Harness, Implementation Plan

| | |
|---|---|
| **Change ID** | CHG-20261013-01 |
| **Branch** | `ft/eval-harness` (from `origin/main` @ `c32e10d`) |
| **Roadmap ref** | [sketch-render-round2-roadmap-methodology.md, WS1](sketch-render-round2-roadmap-methodology.md) |
| **Workflow** | [SKILL.md](SKILL.md): evidence-driven; this file holds the detail and chat holds summaries only |
| **Status** | Implemented and validated locally on 2026-10-14. **Awaiting the user's own validation run** (SKILL Phase 6 gate) before commit, changelog and audit |

---

## 1. Intake

**Goal:** a reproducible evaluation harness (benchmark set, two-phase generate/score runner, metrics, report, human A/B tool) and measured baselines that replace the README's unverifiable numbers.

**In scope**
- Bench manifest, loader and `add-item` CLI
- Engine adapter with external seeding and per-component timing
- Generate and score runner
- Metrics: structure, CLIP, DINOv2, PickScore, HPSv2.1, ImageReward, KID; plus latency and memory
- Report with confidence intervals
- A/B rater with win rate, Elo and Bradley-Terry
- Unit tests
- CoreML compile and edge baseline
- README §5 rewrite

**Out of scope**
- Any change to `app.py`, `src/engine/*` or `src/processing/*` (WS2)
- The import-sketch WIP (stashed)
- Frontier-API reference runs (D5, deferred)
- VQAScore (see D3 note)

---

## 2. Decisions (resolved 2026-10-14)

| ID | Decision | Resolution |
|---|---|---|
| D1 | How to isolate the import-sketch WIP | **Stash, then switch.** `stash@{0}` = "WIP feature/import-sketchs: import-sketch UI + process_uploaded_sketch (stashed before ft/eval-harness, 2026-10-14)". It holds `app.py`, `src/engine/coreml_compiler.py`, `src/processing/preprocessor.py` and the deletion of `src/__init__.py`. Untracked files (`artifacts/`, `assets/sketchs/`, `debug_temp_model/`, …) stayed in the working tree. **Restore with `git switch feature/import-sketchs && git stash pop`.** |
| D2 | Edge baseline | **Option A** (disk freed to 24 GB): compile CoreML models with `src/engine/coreml_compiler.py`, then run a true CoreML baseline |
| D3 | Scorer tier | **Option B, minus VQAScore.** VQAScore's smallest model (CLIP-FlanT5-XL, about 3B parameters) needs about 6 GB of weights in RAM *in addition to* the process, plus about 11 GB of disk. That isn't feasible on an 8 GB M2 alongside the other scorers, so it ships as a documented stub. All other scorers are implemented |
| D4 | Bench data policy | **Commit only the manifest and user-owned sketches.** Third-party images (watermarked stock photos, web images) are referenced by local path and never committed |
| D5 | Frontier reference | **Deferred** to a follow-up change |

---

## 3. Evidence ledger

| ID | Evidence | Tag |
|---|---|---|
| E1 | `origin/main` = `main` = `feature/import-sketchs` = `c32e10d` before the stash | [observed] |
| E2 | Project env `~/miniforge3/envs/.sketchrenderenv`: py3.10.19, torch 2.5.1, diffusers 0.30.2, transformers 4.44.2, coremltools 8.1, python_coreml_stable_diffusion 1.1.0, cv2 4.11, controlnet_aux 0.0.10 (HED, PiDiNet, Lineart, Anyline), skimage, scipy, psutil, pytest, gradio 4.44.1, wandb, timm 1.0.22, numpy 1.23.5, huggingface_hub 0.24.6 | [observed] |
| E3 | Apple M2, 8 GB RAM, macOS 13.3, MPS available. Disk had 8.2 GB free at intake and 24 GB after the user cleaned up | [observed] |
| E4 | `models/coreml/` was empty. `tekraj/sketch-render-coreml` and `Tekraj15/sketch-render-coreml` return 401 | [observed] |
| E5 | HF cache has complete SD1.5 (fp32 + fp16), `sd-controlnet-scribble` and LCM-LoRA | [observed] |
| E6 | Engines take no seed. diffusers without a `generator` draws from the global torch RNG, so `torch.manual_seed` from the harness controls sampling. The legacy `pipeline.py:88` hardcodes `manual_seed(42)`, so seeds are **not** respected there | [read] |
| E7 | Engine signatures differ: `steps/guidance/control_scale` (CoreML, CUDA, PyTorch) vs `num_inference_steps/guidance_scale/controlnet_conditioning_scale` (legacy MPS) | [read] |
| E8 | `.gitignore:63` ignores `*.json` (`git check-ignore` confirmed for `eval/sketchbench/manifest.json`), so the manifest is **`.jsonl`** | [observed] |
| E9 | `assets/sketchs/`: the stock photos carry shutterstock/pixers watermarks (third-party). `plane-above-cloud.jpg` is a phone photo of a pen sketch on paper. `night view dubai marin.avif` can't be read by this Pillow build | [observed] |
| E10 | The CUDA engine can't run on this machine | [observed] |
| E11 | `runwayml/stable-diffusion-v1-5` → **HTTP 307 redirect** to `stable-diffusion-v1-5/stable-diffusion-v1-5`, so online loads still work. (This corrects roadmap finding M3: not broken, but it depends on a redirect.) | [observed] |
| E12 | With `HF_HUB_OFFLINE=1`, `coreml_compiler.py:53 load_lora_weights` fails with `ValueError: When using the offline mode, you must specify a weight_name`. The compile must run online | [observed] |
| E13 | Scorer deps installed **additively** into `.sketchrenderenv`: clip (openai, git), fairscale 0.4.13, ftfy, image-reward 1.5 (`--no-deps`), lightning-utilities, open_clip_torch 3.3.0, torch_fidelity 0.4.0, torchmetrics 1.9.0, wcwidth. A `pip freeze` diff shows **no existing package changed**. Before/after freezes are in the session scratchpad | [observed] |
| E14 | `image-reward` declares `timm==0.6.13` (would downgrade 1.0.22) and needs `fairscale` and `clip` source builds. These were installed with `--no-build-isolation` (setuptools 80.9 still has `pkg_resources`) | [observed] |
| E15 | `ImageReward/__init__.py` imports `ReFL.py`, which imports `datasets` (training only, `ReFL.py:27,518`). `datasets` latest → huggingface_hub 2.2.0 (breaks pins). `datasets==2.21.0` would downgrade fsspec. **Solution:** a `sys.modules` shim active only while ImageReward is imported, after transformers is imported (otherwise transformers' `find_spec("datasets")` raises). Verified importable, with pins intact | [observed] |
| E16 | `app.py` (main) assigns `styles` twice in `process_sketch` (`:58`, `:67`); the second is live. Prompt composition is at `app.py:81-82`. The draw path is `process_sketch_input` → RGB → `ImageOps.invert` (`app.py:42-55`). Generation forces `control_scale=1.0` (`app.py:91`). Default negative prompt: `"low quality, bad anatomy, blurry"` (`app.py:185`) | [read] |

---

## 4. Design

### 4.1 Layout (all new except `.gitignore` and `README.md`)

```
src/evaluation/
  __init__.py
  manifest.py        BenchItem dataclass, load/validate/append manifest.jsonl
  styles.py          reads the live Style-DNA table from app.py via AST (single source of truth; no copy)
  control.py         control-map producers: app_draw (main) | app_import (when Preprocessor has it) | auto
  engines.py         EngineAdapter: build {coreml, cuda, pytorch, mps_legacy}, seed, render()
  instrument.py      ComponentTimer (wraps text_encoder / controlnet / unet / vae.decode with device sync), MemorySampler, swap probe
  report.py          per-item aggregation, bootstrap CI, latency percentiles, report.md, grid.png
  pairwise.py        win rate (ties = 0.5) + bootstrap CI, Elo (permutation-averaged), Bradley-Terry (MM)
  metrics/
    __init__.py      registry + availability checks
    base.py          Scorer protocol
    structure.py     edge F1 @ tolerance + symmetric Chamfer distance (pure functions) + HED extractor
    clip_score.py    CLIP ViT-L/14 (transformers): prompt-only and full-prompt cosine
    dino.py          DINOv2-base CLS cosine vs. reference (paired items only)
    pickscore.py     PickScore_v1 (transformers)
    hpsv2.py         HPS v2.1 (open_clip ViT-H-14 + xswu/HPSv2 checkpoint)
    image_reward.py  ImageReward-v1.0 (scoped `datasets` import shim, E15)
    kid.py           set-level KID (torchmetrics), skipped when n < 20
    vqascore.py      stub: unavailable on 8 GB (D3)
benchmarks/
  run_eval.py        generate | score | report | add-item
  ab_rater.py        Gradio blind pairwise rater → ratings.csv
configs/eval/
  edge_coreml_app_defaults.yaml   the app's effective defaults (steps 4, guidance 1.0, control 1.0)
  cloud_cuda_app_defaults.yaml
  mps_legacy.yaml
eval/sketchbench/
  manifest.jsonl     v0 seed set
  sketches/          user-owned sketches only (D4)
  README.md          schema, strata, data policy, how to add items
tests/evaluation/    pytest unit tests
requirements-eval.txt  scorer deps + install notes (E13–E15)
```

### 4.2 Key design choices

- **Zero edits to engines and app.** Seeding is global-RNG from outside (E6). Timing works by wrapping pipeline components and restoring them afterwards.
- **Generate and score are separate processes.** 8 GB of RAM can't hold the generator plus any scorer (E3). Scorers load **one at a time** and are freed between metrics.
- **Single source for Style DNA.** `styles.py` parses `app.py` with `ast` and takes the last `styles = {...}` in `process_sketch`. When WS2 extracts the table into a module, `styles.py` switches to importing it.
- **Control maps follow app behavior.** `app_draw` mirrors `app.py:42-55`: RGB on white, then letterbox to a square on white, then invert. The letterbox stands in for a user placing a non-square sketch on the 512² canvas. `app_import` uses `Preprocessor.process_uploaded_sketch` when it exists (WIP branch). `auto` picks import when it's available and the item has an `input_mode`. The main baseline therefore uses draw-path emulation for every stratum, and rerunning after the import feature merges measures that feature.
- **App-faithful baseline parameters.** These are what the UI actually sends, not the engine defaults: steps 4, guidance 1.0 (CFG off, so the negative prompt has no effect, roadmap E2), control_scale 1.0 (forced at `app.py:91`).
- **Structure metric without circularity.** The input side is the control map (binarize, then thin). The output side uses edges from **HED** (a different extractor family than the app's threshold/invert), then threshold, then thin. With tolerance τ = 3 px at 512 (scaled with resolution): precision = output edges within τ of the input; recall = input edges within τ of the output; F1. Chamfer = mean of the two directional mean distances, in px at 512 scale.
- **Statistics.** Average over seeds per item, then mean ± 95% bootstrap CI across items (B = 2000, fixed RNG). Latency uses warm renders only (p50 and p95); the cold first render and engine load are reported separately.
- **ZeroGPU caveat.** `@spaces.GPU` runs `generate` in a forked worker, so component timers inside it don't propagate back. On ZeroGPU only end-to-end latency is reliable; dedicated CUDA gets the full breakdown.

### 4.3 Manifest schema (`eval/sketchbench/manifest.jsonl`, one JSON object per line)

| Field | Type | Notes |
|---|---|---|
| `id` | str | unique, kebab-case |
| `stratum` | enum | `scribble` · `lineart` · `shaded_drawing` · `paper_photo` · `real_photo` |
| `sketch` | path | relative to repo root |
| `reference` | path or null | paired photo (enables DINOv2 and KID) |
| `prompt` | str | short user prompt (what a user would type) |
| `style_dna` | str | key in the app's Style-DNA table |
| `domain` | str | auto · fashion · architecture · nature · wildlife · floral · people · aviation · urban |
| `input_mode` | str or null | import-path hint: `Digital Sketch` · `Paper Sketch` · `Real Photo` |
| `owner` | enum | `user` (committable) · `third_party` (local only, never committed) |
| `notes` | str | optional |

### 4.4 v0 seed set (13 items; local paths per D4)

| id | stratum | owner | reference |
|---|---|---|---|
| scribble-thar-offroad, scribble-van-desert, scribble-girl-portrait | scribble | user (cropped from the app screenshots committed in `assets/`) | — |
| paper-plane-pen | paper_photo | unknown, treated as third_party until the user confirms | — |
| shaded-girl-sunflower | shaded_drawing | third_party | girl-in-sunflower-garden.jpeg |
| shaded-girl-garden | shaded_drawing | third_party | young-woman-standing-in-a-garden.jpeg |
| shaded-aeroplane-clouds | shaded_drawing | third_party | airplane-above-clouds.jpg |
| shaded-skyline-night | shaded_drawing | third_party | skyline-city-night-view.png |
| photo-truck-highway, photo-girl-flower-field, photo-tulip-garden, photo-dubai-skyline, photo-bnw-girl-garden | real_photo | third_party | the photo itself |

Growing to sketchbench-v1 (about 120 items) needs the user's own canvas scribbles and paper photos, each added with `run_eval.py add-item`.

---

## 5. Decomposition and status

| # | Sub-problem | Files | Risk | Verification | Status |
|---|---|---|---|---|---|
| SP1 | Manifest, loader, add-item | `src/evaluation/manifest.py`, `eval/sketchbench/*` | Low | unit tests | ✅ code + tests |
| SP2 | Structure metric | `src/evaluation/metrics/structure.py` | Low | unit tests (identity, shift, blank) | ✅ code + tests |
| SP3 | Engine adapter, seeding, instrumentation | `engines.py`, `instrument.py`, `styles.py`, `control.py` | Med | unit tests (fake modules); determinism smoke test | ✅ code + tests; smoke test pending compile |
| SP4 | `run_eval.py generate` | `benchmarks/run_eval.py`, `configs/eval/*` | Med | CoreML smoke test (2 items × 2 seeds) | ✅ code; smoke test pending compile |
| SP5 | Semantic and preference scorers | `metrics/{clip_score,dino,pickscore,hpsv2,image_reward,kid,vqascore}.py` | Med | sanity check: matching caption > mismatched caption | ✅ code; live sanity check pending |
| SP6 | `score` and `report` | `report.py`, `run_eval.py` | Low | unit tests on fixture records | ✅ code + tests |
| SP7 | A/B rater and pairwise stats | `ab_rater.py`, `pairwise.py` | Low | unit tests; launch smoke test | ✅ code + tests; launch smoke test pending |
| SP8 | Registry and requirements-eval | `metrics/__init__.py`, `requirements-eval.txt` | Low | registry test | ✅ |
| SP9 | `.gitignore` adds `eval/runs/` | `.gitignore` | Low | `git check-ignore` | ✅ verified with git check-ignore |
| SP10 | CoreML compile, edge baseline, README §5 | `models/coreml/` (ignored), `README.md` | Med | numbers trace to a run dir | ✅ README §5 rewritten from run `20261014-013359` |

---

## 6. Validation plan (filled in at Phase 6)

| Check | Command | Result |
|---|---|---|
| Unit tests | `python -m pytest tests/evaluation -q` | ✅ 42 passed (2.9 s) |
| Generate smoke test (CoreML) | `run_eval.py generate --config configs/eval/edge_coreml_app_defaults.yaml --items scribble-thar-offroad,scribble-girl-portrait --seeds 0,1 --out eval/runs/smoke` | ✅ 4 renders, exit 0. Per render: total 27–33 s = **ControlNet 24–27 s (4 calls, PyTorch fp32 CPU)** + UNet 3.3–5.5 s (4 calls, ANE) + VAE 0.3 s + text encoder < 0.1 s. Engine load 105 s, cold render 35.9 s. Swap 2.9 → 6.6 GB |
| Determinism | same item, seed 0, separate process (`20261014-012331`) | ✅ max abs pixel diff **0** vs. the first run. A different seed differs (mean abs diff 64.5) |
| Scorer sanity (matching vs. mismatched caption/reference, 4 photos × 4 captions) | `scratchpad/scorer_sanity.py mps` | ✅ all PASS: struct F1 0.385 vs 0.224, Chamfer 10.8 vs 19.4 px, CLIP 24.7 vs 6.7, DINOv2 0.966 vs 0.020, PickScore 22.6 vs 16.8, HPSv2.1 0.289 vs 0.110, ImageReward 1.32 vs −2.27 |
| A/B rater | launch on two smoke runs (port 7871), plus build_pairs → append_ratings → resume → `--summarize` | ✅ HTTP 200 with expected components (image, radio, button, state); 1 common pair; ratings CSV written; resume skips the rated pair; summary prints win rate, Elo and BT |
| Full edge baseline | `run_eval.py generate --config configs/eval/edge_coreml_app_defaults.yaml` then `score --run eval/runs/20261014-013359_edge_coreml_app_defaults` | ✅ 52/52 renders; all 7 scorers ran (KID skipped by design: 8 references < 20). Latency p50 27.8 s / p95 31.3 s (ControlNet 24.2 s ≈ 87%). Structure F1 0.285 [0.212, 0.366] overall: scribble 0.535, shaded 0.231, real photo 0.209, paper photo 0.126. Full table in README §5 and the run's `report.md` |
| Unit tests after fixes | `python -m pytest tests/evaluation -q` | ✅ 42 passed |

**Not verifiable here:** the CUDA engine path (E10). The user runs `generate --config configs/eval/cloud_cuda_app_defaults.yaml` on a GPU or the Space.

**Residual risks / notes**
- The bench is small (13 items; 3 user-owned). CIs are wide and per-stratum CIs with n = 1 aren't reported. Growing toward sketchbench-v1 (about 120 items) is the main follow-up.
- Third-party items only exist on this machine. On a fresh clone, run with `--strata scribble` until the user adds owned sketches.
- KID stays skipped until there are ≥ 20 distinct references.
- `README.md` §3 still describes 6-bit UNet quantization and cloud `torch.compile` / TAESD. Correcting those belongs to WS2 (E17, roadmap S6). This change only rewrote §5.

---

## 7. Progress log

- **2026-10-14:**
  - D1–D5 resolved; WIP stashed (`stash@{0}`); switched to `ft/eval-harness`.
  - First compile attempt failed under `HF_HUB_OFFLINE=1` (E12); re-running online.
  - Scorer deps installed additively (E13–E15).
  - Three user-owned scribbles cropped from the committed app screenshots into `eval/sketchbench/sketches/` (binarized black-on-white).
  - SP1–SP9 implemented: `src/evaluation/{manifest,styles,control,engines,instrument,report,pairwise}.py`, `src/evaluation/metrics/*` (structure, clip, dino, pickscore, hpsv2, imagereward, kid, vqascore stub), `benchmarks/run_eval.py`, `benchmarks/ab_rater.py`, `configs/eval/*.yaml`, `eval/sketchbench/{manifest.jsonl,README.md,sketches/}`, `requirements-eval.txt`, and `.gitignore` (+`eval/runs/`).
  - Unit tests: `tests/evaluation/` has **42 passing** (`python -m pytest tests/evaluation -q`, 2.9 s).
  - HPSv2.1 is loaded via open_clip directly; the `hpsv2` package pins old deps and has a known missing-vocab bug.
  - CoreML compile is running. Note: the compiler also leaves an unchunked `control-unet.mlpackage` that `rename_models()` doesn't handle. It's unused by the engine and takes extra disk; flagged as a follow-up for WS2.
  - **CoreML compile finished** (`EXIT=0`, online). Output in `models/coreml/`: ControlNet 259 MB, TextEncoder 134 MB, UnetChunk1 847 MB, UnetChunk2 794 MB, VAEDecoder 95 MB, VAEEncoder 65 MB, plus an unused unchunked `…temp_fused_model_control-unet.mlpackage` of 617 MB.
  - **E17 [observed], finding for WS2:** the compile log shows 6-bit palettization was applied to the text encoder, ControlNet and the *unchunked* `control-unet`, but the **`UnetChunk1/2` the engine loads are fp16** (1.64 GB total vs. 617 MB palettized). The quantizer looked for `…_unet.mlpackage` / `…runwayml…_control-unet.mlpackage` names that don't match the chunked outputs (`coreml_compile.log` lines 96–101, 138–143). So the README's "6-bit UNet" claim doesn't hold for this compile path. The edge baseline measures the models as the compiler produces them today.
  - Smoke, determinism, scorer-sanity and A/B checks done (see §6).
  - Fix found during the scorer sanity check: DINOv2 hits `aten::upsample_bicubic2d` with no MPS kernel (torch 2.5) → pinned to CPU (like HED and ImageReward). A follow-up `.to(device)` used the caller's device instead of `self.device` → fixed in `metrics/dino.py`.
  - **E18 [observed]:** the edge render-time breakdown confirms roadmap finding S1. **About 85% of each ~30 s render is the PyTorch fp32 ControlNet on CPU**; the ANE UNet is only 3–5 s for 4 steps. A single 4-render run grew system swap by 3.7 GB, confirming roadmap S2's memory pressure.
  - Full edge baseline generated (52 renders, 25 min) and scored (all scorers, about 4 min). README §5 rewritten with measured numbers. The previous FID, CLIP, IS and latency claims were removed as unreproducible.
  - **E19 [observed]:** on `main`, paper photos, shaded drawings and real photos go through the draw path (invert only). Their structure F1 is 0.13–0.23 vs. 0.54 for scribbles, and the paper-plane item renders no plane at all. This is the quantified "before" for the import-sketch feature (stashed WIP) and for roadmap WS3.
- **2026-10-14 (after restart):**
  - Verified the baseline survived the overnight shutdown intact: 52/52 records, images and metric rows; finished 02:00 and scored 02:05, before the shutdown. The "stopped" tasks were only the progress monitors.
  - User validation: 42 tests passed (4.89 s); README reviewed.
  - Deep-dive analysis (chance floor, real-photo ceiling, best-of-4 oracle, metric agreement, latency projection) written to [eval-harness-results.md](eval-harness-results.md).
