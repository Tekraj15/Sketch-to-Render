# Eval Harness: Baseline Results and Insights

| | |
|---|---|
| **Change** | CHG-20261013-01 (WS1 evaluation harness), see [eval-harness-implementation-plan.md](eval-harness-implementation-plan.md) |
| **Run analysed** | `eval/runs/20261014-013359_edge_coreml_app_defaults` (local only; `eval/runs/` is gitignored) |
| **System under test** | `main` @ `c32e10d`: CoreML engine, SD1.5 + LCM-LoRA (fused) + ControlNet-scribble v1.0, at the values the UI actually sends: 4 steps, guidance 1.0, control_scale 1.0, 512², negative prompt `"low quality, bad anatomy, blurry"` (inactive at CFG 1.0) |
| **Bench** | SketchBench v0: 13 items × 4 seeds = 52 renders. Strata: scribble 3, shaded drawing 4, real photo 5, paper photo 1 |
| **Hardware** | Apple M2, 8 GB RAM, macOS 13.3 |
| **Date** | 2026-10-14 |
| **User validation** | `pytest tests/evaluation`: 42 passed (user's run, 4.89 s); README §5 reviewed |

Evidence tags as in [SKILL.md](SKILL.md): `[observed]` measured · `[inferred]` reasoned or projected, not yet measured.

---

## 1. TL;DR

1. **The harness works end to end and is trustworthy enough to gate decisions:**
   - Renders are bit-exact for a given seed.
   - All 7 scorers pass a matched-vs-mismatched sanity test.
   - Latency is stable (coefficient of variation 5%), so speedups of ≥ 10% are measurable from a single run.
2. **Speed is 5.6–10× slower than this hardware can likely deliver.**
   - Measured p50 is **27.8 s**.
   - **87% of that is ControlNet running in PyTorch fp32 on the CPU**, at 6.05 s per call against 0.80 s per call for the ANE UNet.
   - Moving ControlNet onto the ANE and using batch-1 compiles projects to **≈ 2.7–5.0 s** `[inferred]`.
3. **Structure following works for canvas scribbles only.**
   - Scribbles: F1 0.535 against a chance floor of 0.200.
   - Shaded drawings (+0.05 over the floor), real photos (+0.03) and the paper photo (+0.01) are at or near chance. **On `main` the model effectively ignores uploaded sketches and photos.**
4. **Prompt alignment matches real photos, but human-preference scores fall short.**
   - On the 9 paired items, CLIP is at parity with the reference photos (24.93 vs 24.96).
   - HPSv2.1 is **−13%** (0.220 vs 0.254), ImageReward **−0.45** (−0.42 vs +0.03) and PickScore −0.87.
   - Scribble renders score lowest (ImageReward −1.57): flat, sticker-like, illustrative.
5. **Choosing the best of 4 seeds is a large free quality lever.**
   - Picking by a composite score lifts ImageReward by +0.52, structure F1 by +0.04 and HPS by +0.016 with no model change.
   - It costs 4× compute, so it's only affordable after the speed fixes.
6. **The bench is too small for fine-grained decisions.** With 13 items, the 95% CI half-width is ±0.077 on structure F1 and ±0.51 on ImageReward. About 40 items would roughly halve that.

**Next (ranked):** wire `ControlNet.mlpackage` → batch-1 and properly palettized recompile → remove the fp32 skeleton → merge the import path and re-measure the non-scribble strata → grow the bench → cloud baseline. Details in §6.

---

## 2. What we achieved

| Capability | Status | Evidence |
|---|---|---|
| Versioned benchmark (manifest, strata, owner policy, `add-item` CLI) | ✅ | `eval/sketchbench/manifest.jsonl` (13 items), schema validated in tests |
| Reproducible generation (external seeding, no engine edits) | ✅ | same seed, separate process → max pixel diff **0** `[observed]` |
| Per-component latency without touching the engines | ✅ | text encoder / ControlNet / UNet / VAE timings for all 52 renders; total CV 5% |
| Memory and swap tracking | ⚠️ partial | peak RSS + swap delta recorded, but RSS under-counts once pages swap out (see §5, L4) |
| Structure-fidelity metric (HED edge F1 + Chamfer) | ✅ | sanity: own-edges F1 0.385 vs. other-edges 0.224 |
| Semantic and preference metrics (CLIP-L/14, DINOv2, PickScore, HPSv2.1, ImageReward) | ✅ | each ranks matched above mismatched in the sanity test |
| KID | ✅ implemented, ⏸ skipped | needs ≥ 20 distinct references; v0 has 8 |
| VQAScore | ⏸ deferred | doesn't fit 8 GB RAM (D3) |
| Report (bootstrap CIs, per-stratum, latency table, image grid) | ✅ | `report.md`, `grid.png` |
| Blind A/B rater + win rate / Elo / Bradley-Terry | ✅ | launch, record, resume and summarize verified; **no human ratings collected yet** |
| README §5 replaced with measured, traceable numbers | ✅ | the previous FID / CLIP / IS / latency claims had no reproducible source |

---

## 3. Results against expectations

### 3.1 Latency: measured vs. projected vs. target

| Quantity | Value | Tag |
|---|---|---|
| Measured end-to-end p50 / p95 (warm) | **27.8 s / 31.3 s** | [observed] |
| ControlNet per call (PyTorch fp32, CPU) | 6.05 s (sd 0.31), × 4 calls = 24.2 s | [observed] |
| UNet per call (Core ML/ANE, compiled batch 2, padded at CFG-off) | 0.80 s (sd 0.05), × 4 calls = 3.2 s | [observed] |
| VAE decode / text encoder / other | 0.31 / 0.02 / 0.09 s | [observed] |
| Engine load / cold first render | 96 s / 42.5 s | [observed] |
| ControlNet efficiency gap | ControlNet has ≈ 0.42× the UNet's parameters (361M vs 860M) but runs **7.6× slower per call**, so about **18× worse per parameter** | [observed] + arithmetic |
| **Projection A:** ControlNet on ANE, both still batch 2 | 0.02 + 4 × (0.42 × 0.80) + 3.2 + 0.31 + 0.09 ≈ **5.0 s** (5.6× faster) | [inferred] |
| **Projection B:** A plus batch-1 UNet and ControlNet (no CFG padding) | 0.02 + 0.67 + 1.60 + 0.31 + 0.09 ≈ **2.7 s** (10× faster) | [inferred]; assumes ANE time scales about linearly with batch |
| Roadmap edge target | ≤ 10 s | currently **2.8× over** target; both projections land well under it |

**Reading:** the edge path isn't compute-bound. It's bound by one component running on the wrong device and precision. The 45–62 s in the old screenshots was this same bottleneck plus swap; today's warm 27.8 s excludes load and first-call compile.

### 3.2 Structure fidelity: measured vs. chance floor

**Floor** = the same renders scored against *other* items' control maps (average over all mismatched pairs). It measures how much edge overlap any SD1.5 image gets by chance.

| Stratum | Own-control F1 | Mismatched-control F1 (floor) | Lift over floor | Verdict |
|---|---|---|---|---|
| Scribble (3) | **0.535** | 0.200 | **+0.335** | Sketch clearly followed ✅ |
| Shaded drawing (4) | 0.231 | 0.179 | +0.052 | Barely above chance ❌ |
| Real photo (5) | 0.209 | 0.182 | +0.027 | ≈ chance ❌ |
| Paper photo (1) | 0.126 | 0.119 | +0.007 | Chance: sketch ignored ❌ |
| All (13) | 0.285 [0.212, 0.366] | 0.180 | +0.105 | |

No ceiling is established yet: the best achievable F1 for a perfect render of a rough scribble is unknown. The first stronger backbone in WS4 will give an empirical upper reference.

### 3.3 Aesthetics and prompt alignment: renders vs. real-photo ceiling (9 paired items)

The reference photos were scored with **the same prompts and scorers**. These photos carry stock watermarks and the prompts were written to describe them, so this is a *conservative, approximate* ceiling.

| Metric | Renders (mean of 4 seeds) | Reference photos | Gap | Best-of-4 by composite (all 13 items) |
|---|---|---|---|---|
| CLIP-L/14 ×100 vs prompt | 24.93 | 24.96 | **≈ 0 (parity)** | — |
| PickScore | 18.60 | 19.47 | −0.87 | 18.80 (+0.27 vs mean) |
| HPSv2.1 | 0.220 | 0.254 | **−0.034 (−13%)** | 0.224 (+0.016) |
| ImageReward | −0.418 | +0.027 | **−0.45** | −0.221 (+0.52) |

Over all 13 items, scribbles are the weakest on preference (ImageReward −1.57, HPS 0.186) and on CLIP-vs-prompt (18.0).

---

## 4. What the results tell us

| # | Inference | Evidence | Confidence |
|---|---|---|---|
| I1 | **The edge speed problem is one misplaced component.** Getting ControlNet off the CPU is the single highest-ROI change in the project | 87% share; 18× per-parameter inefficiency (§3.1) | High |
| I2 | **The 8 GB machine is in memory distress**, which also inflates load and cold-start times | swap +3.6 GB per run; peak RSS only 1.4 GB, so most of the working set is swapped out | High (direction) |
| I3 | **The README's "6-bit quantized UNet" isn't what runs.** The compile path palettizes an unused unchunked UNet; the engine loads fp16 chunks (1.64 GB) | compile log, file sizes (plan E17) | High |
| I4 | **Uploaded sketches and photos aren't usable on `main`.** Inverted photos or shaded drawings fed to a scribble ControlNet produce chance-level structure. The import-sketch feature and WS3 are required, not optional | §3.2: non-scribble lift +0.01 to +0.05; `paper-plane-pen` and `shaded-aeroplane-clouds` render **no airplane in any seed** | High |
| I5 | **"Pretty but unfaithful" exists and preference scorers reward it**, so structure must stay a hard gate | `shaded-girl-sunflower` has the best ImageReward (+1.49) and HPS (0.282) but F1 0.20, at the floor | High |
| I6 | **Base model + LCM at 4 steps is the aesthetic ceiling limiter, not prompt understanding.** CLIP matches real photos while preference scores trail. This supports WS4 (backbone and distillation swap) over prompt work for *aesthetics* | §3.3 parity on CLIP, −13% HPS, −0.45 ImageReward | Medium (9 items, conservative ceiling) |
| I7 | **There's a possible structure-vs-prompt tension on scribbles.** Renders that follow the sketch best match the prompt least (ρ = −0.54 between F1 and CLIP). Scribble renders drop prompt attributes ("green spoiler", "red spoiler") and turn interior strokes into decals. Plausible cause: control_scale forced to 1.0 and CFG off (guidance 1.0) leave text conditioning weak | Spearman across 52 renders; grid review | **Low:** confounded by stratum (scribble prompts are longer and attribute-heavy). Needs the sweep in A7 |
| I8 | **ImageReward and HPSv2.1 largely agree; PickScore is a weaker signal here.** Use ImageReward + HPS as primary aesthetic metrics and PickScore as secondary | ρ(IR, HPS) = 0.85; ρ(IR, Pick) = 0.44; ρ(HPS, Pick) = 0.52 | Medium |
| I9 | **Seed noise is large, so 4 seeds per item are necessary.** Best-of-N is a real lever | within-item seed SD is ≈ 30–45% of between-item SD (ImageReward 0.43 vs 0.94; F1 0.042 vs 0.144); best-of-4 ImageReward +0.52 | High |
| I10 | **At 13 items, only large effects are detectable.** CI half-widths: F1 ±0.077, ImageReward ±0.51, HPS ±0.019. At about 40 items: F1 ≈ ±0.044, ImageReward ≈ ±0.29, HPS ≈ ±0.011 (1/√n scaling) | bootstrap CIs in `report.md` | High |

---

## 5. Harness limitations found during this review

| ID | Limitation | Impact | Fix |
|---|---|---|---|
| L1 | Bench is small (13) and only 3 items are user-owned. The rest are local-only third-party images | wide CIs (I10); fresh clones can only run 3 items | grow toward v1; prioritize owned scribbles and paper photos |
| L2 | Paper-photo stratum has n = 1; per-stratum CIs need n ≥ 2 | paper-photo conclusions are anecdotal | add ≥ 10 phone photos of real paper sketches |
| L3 | KID is skipped (8 distinct references < 20); DINOv2 covers only 9 items | no distribution-level realism metric yet | add paired items |
| L4 | **Peak RSS under-counts under swap** (1.4 GB measured while swap grew 3.6 GB) | memory improvements in WS2 would be under-reported | record macOS physical footprint (`psutil` `memory_full_info().uss` or `vmmap`/`footprint`) alongside RSS |
| L5 | For `real_photo` items on `main`, the "control map" is an inverted photo, so F1 there measures edge overlap with photo texture | F1 for those items is a weaker structure signal until the import path produces real edge maps | resolves itself once A4 lands |
| L6 | The floor, ceiling, best-of-N and correlation analyses in this document were run from a session scratch script, not from `run_eval.py` | not yet one-command reproducible | productize as `run_eval.py analyze` (A9) |
| L7 | No human A/B ratings yet, and the release gate is human | metric-only conclusions so far | first A/B: baseline vs. A1/A2 candidate |
| L8 | No cloud (CUDA/ZeroGPU) numbers | the cloud half of the baseline is missing | A6 |

---

## 6. Next action items (ranked by expected reward ÷ effort)

| Rank | Action | Roadmap | Why now (evidence) | Acceptance check, using this harness |
|---|---|---|---|---|
| **A1** | **Load `ControlNet.mlpackage` in `CoreMLEngine`** (it's already compiled and palettized; map outputs to `additional_residual_*`) | WS2.1 | I1: 87% of latency | Re-run `edge_coreml_app_defaults`: p50 ≤ 8 s; structure F1 and ImageReward within the baseline CI; bit-for-bit determinism kept |
| **A2** | **Recompile: batch-1 UNet (and ControlNet) for the CFG-off path, and fix palettization of the UNet chunks** (E17) | WS2.3 + E17 | I3; projection B ≈ 2.7 s; smaller weights also ease I2 | p50 ≤ 4 s; chunk sizes ≈ 6-bit; quality within CI |
| **A3** | **Remove the fp32 PyTorch "skeleton" load** in `CoreMLEngine` (tokenizer and scheduler config only) | WS2.2 | I2: swap +3.6 GB per run; load 96 s | swap delta ≈ 0; engine load time down; footprint metric (L4) shows the drop |
| **A4** | **Merge the import-sketch branch** (`stash@{0}`), then re-run with `control_mode: auto` on the shaded, real-photo and paper strata | WS3 entry | I4: non-scribble at chance | lift over floor ≥ +0.15 F1 on shaded and paper; both airplanes appear |
| **A5** | **Grow SketchBench to ≥ 40 items** (≥ 15 owned scribbles, ≥ 10 owned paper photos, ≥ 20 paired) | WS1 follow-up | I10, L1–L3 | CI half-widths ≈ halved; KID becomes computable |
| **A6** | **Cloud baseline**: `configs/eval/cloud_cuda_app_defaults.yaml` on a CUDA box or the Space | WS1 follow-up | L8 | report with end-to-end latency (component timers don't survive ZeroGPU's forked worker) |
| **A7** | **Control/guidance sweep on scribbles**: control_scale ∈ {0.6, 0.8, 1.0} × guidance ∈ {1.0, 1.5}, with `control_guidance_end` 0.7 as an extra arm | WS2 (app fix E1/E2) + WS3 | I7 (prompt attributes dropped). Cheap once A1 makes renders ~5 s | find the Pareto point: CLIP-vs-prompt ↑ with F1 within CI; informs the UI defaults |
| **A8** | **Best-of-4 composite reranking** (ImageReward + HPS + structure F1) | WS6 | I9: oracle +0.52 ImageReward, +0.04 F1 | only after A1/A2 (4 × ~3 s); first human A/B: best-of-4 vs single seed |
| **A9** | **Harness upgrades**: `run_eval.py analyze` (floor / ceiling / best-of-N / correlations), physical-footprint memory metric, cloud-friendly timing note in report | WS1 follow-up | L4, L6 | the analysis in this doc regenerates with one command |
| **A10** | **First human A/B round** once A1+A2 produce a candidate (verifies "no quality regression" beyond metrics) | WS1 gate | L7 | ≥ 100 comparisons; candidate win rate ≥ 0.45 (no regression) |

**Recommended sequencing:** A1 → A2 → A3 as one WS2 change (all edge-engine, all measured with the existing config). A4 can run in parallel on the import branch. A5 and A6 are data/infra tasks for you, which the harness already supports. A7 and A8 follow once renders are fast.

**Out of scope for these actions (still roadmap):** WS4 backbone bake-off. I6 suggests it's the main aesthetic lever, but it should be measured on the faster, fixed pipeline from A1–A4 so the comparison is fair.

---

## 7. Appendix

### 7.1 Per-item results (mean ± seed SD over 4 seeds)

| Item | Stratum | Structure F1 | CLIP vs prompt | ImageReward | HPSv2.1 | PickScore | DINOv2 vs ref |
|---|---|---|---|---|---|---|---|
| scribble-girl-portrait | scribble | 0.55 ± 0.03 | 14.8 | −2.02 ± 0.06 | 0.173 | 17.36 | — |
| scribble-thar-offroad | scribble | 0.52 ± 0.02 | 17.0 | −1.84 ± 0.49 | 0.157 | 17.62 | — |
| scribble-van-desert | scribble | 0.53 ± 0.04 | 22.2 | −0.85 ± 0.48 | 0.229 | 19.83 | — |
| shaded-aeroplane-clouds | shaded | 0.17 ± 0.05 | 22.6 | −1.85 ± 0.23 | 0.153 | 17.94 | 0.199 |
| shaded-girl-garden | shaded | 0.26 ± 0.04 | 25.7 | −0.58 ± 0.37 | 0.227 | 17.48 | 0.173 |
| shaded-girl-sunflower | shaded | 0.20 ± 0.06 | 30.7 | +1.49 ± 0.50 | 0.282 | 19.34 | 0.573 |
| shaded-skyline-night | shaded | 0.30 ± 0.04 | 23.2 | −0.72 ± 0.10 | 0.234 | 19.70 | 0.733 |
| photo-bnw-girl-garden | real photo | 0.22 ± 0.06 | 25.5 | +0.63 ± 0.70 | 0.228 | 18.07 | 0.079 |
| photo-dubai-skyline | real photo | 0.26 ± 0.05 | 21.6 | −0.41 ± 0.24 | 0.239 | 20.14 | 0.679 |
| photo-girl-flower-field | real photo | 0.17 ± 0.02 | 29.7 | −0.51 ± 0.44 | 0.212 | 17.68 | 0.339 |
| photo-truck-highway | real photo | 0.20 ± 0.03 | 23.8 | −0.61 ± 1.11 | 0.206 | 20.38 | 0.413 |
| photo-tulip-garden | real photo | 0.19 ± 0.04 | 21.6 | −1.20 ± 0.66 | 0.198 | 16.68 | 0.394 |
| paper-plane-pen | paper photo | 0.13 ± 0.06 | 25.2 | −1.17 ± 0.18 | 0.173 | 18.66 | — |

### 7.2 Qualitative review of `grid.png`

- **Scribbles:** silhouettes are followed closely. The style is flat, decal-like and illustrative. Interior strokes (motion marks on the thar) become painted flames or decals. Colour and attribute words in the prompt ("black", "red spoiler", "green spoiler") are mostly dropped.
- **Shaded drawings:** compositions are generic. The thin airplane in `shaded-aeroplane-clouds` vanishes in all 4 seeds (clouds only). The sunflower portrait and skyline are attractive but don't follow the drawing's layout.
- **Real photos and paper photo:** the inverted-photo control maps give weak guidance. The paper-sketch airplane renders as clouds only; `photo-bnw-girl-garden` hallucinates a hedge tunnel (DINOv2 0.079); colours are oversaturated (magenta flowers).

### 7.3 Method notes

- **Statistics:** item-level means (seeds averaged first), 95% bootstrap CI over items (B = 2000), as in `report.md`.
- **Floor:** for each of the 52 renders, structure F1 against each of the other 12 items' control maps, averaged. Same HED extractor and the 3 px@512 tolerance.
- **Ceiling:** the 9 distinct reference photos (resized to 512²), scored with each item's seed-0 full prompt and the same scorers. This is conservative: the photos carry watermarks and the prompts were written to describe them, which slightly favours CLIP for the photos.
- **Best-of-4:** per item, pick the seed maximizing the selector (ImageReward; F1; or composite = mean z-score of ImageReward, HPS and F1 over all 52 renders), then average the chosen renders' metrics over items. This is an *oracle* upper bound for reranking with the same scorers; a deployed reranker would score with the same models, so the gap to a deployed reranker should be small.
- **Projections in §3.1** assume ANE ControlNet throughput per parameter equals the UNet's, and that batch-1 roughly halves per-call time. Both are to be verified by A1 and A2.
- **Reproducibility:** the analyses in §3.2, §3.3 and §4 were computed from `records.jsonl` / `metrics.csv` of the run above using session scripts. They become one command in A9.
