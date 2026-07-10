<h1 align="center">3D Print a Picture</h1>

Take a picture, get a 3D print of it!

![3D Print a Picture Screenshot](https://github.com/user-attachments/assets/ca9eb833-b6d5-43c1-9abb-16c3500fb35b)

## How it works

This app turns a 2D image into a printable height-field STL:

1. The frontend accepts an uploaded image or generates one from a prompt.
2. The FastAPI backend can optionally complete a missing left or right half for roughly symmetric subjects.
3. The FastAPI backend estimates a depth map.
4. The backend converts that depth map into a watertight STL relief mesh.
5. The frontend previews the STL and lets you download or send it onward.

The default depth backend is now local `Depth Anything V2` through Hugging Face Transformers, which can use CUDA on an NVIDIA GPU. The older remote `facebook/sapiens_depth` Gradio Space is still available as a fallback.

## Quick Links

- [Devpost Project Page](https://devpost.com/software/3d-print-a-picture) - won 2 sponsor prizes (MASV & Groq) at Hack The North 2024!
- Coming into production soon! Join the waitlist - https://3dprintpic.com/

## Getting Started

Backend
```bash
cd backend/
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --reload --port 8004
```

Video/selection planner
```bash
.\backend\.venv\Scripts\Activate.ps1
python -m uvicorn backend.video_selection_service:app --host 0.0.0.0 --reload --port 8005
```

The main backend selection endpoints can use cached SAM2 checkpoints, an opt-in cached DETR panoptic segmenter, or the deterministic click-region fallback; set `SELECTION_ALLOW_MODEL_DOWNLOAD=1` only when the backend is allowed to fetch selection checkpoints. DETR panoptic selection also supports `POST /selection/precompute` plus `POST /selection/precomputed_mask` on the main backend port, so one image segmentation map can serve repeated hover/click masks before the selected object is sent to relief STL or direct image-to-mesh STL.

The planner service exposes the model catalog and creates run plans for object selection, frame selection, camera matching, video reconstruction, direct image-to-mesh, and STL repair. It also exposes `GET /providers/image-to-mesh` plus `POST /run/image-to-mesh` for single-photo STL generation, and `GET /providers/multiview-to-mesh` plus `POST /run/multiview-to-mesh` for selected-frame/mask bundle STL generation. The image runner uses an installed provider such as TripoSR API, TripoSG, Hunyuan3D Shape, SPAR3D, or Stable Fast 3D behind the selected model id; the multiview runner currently exposes the configured `multiview-visual-hull` baseline. Both runners write `output_model.stl`, `diagnostics.json`, and `metadata.json`. Provider repositories/checkpoints still need to be installed separately and configured on the server with environment variables; request bodies cannot choose provider repo paths or Python executables.

The webapp includes printer-volume constraints for STL sizing. The default preset is `Bambu Lab P1S` with a `256 x 256 x 256 mm` build volume; custom printer dimensions, print scale, and edge clearance can be set in the output panel. For 2.5D relief STL generation, the scaled physical XY footprint is passed to the backend as `max_xy_size`, the mesh/detail sample budget is passed as `target_dimension`, and the clamped relief height is passed as `z_scale`. Smaller print scales no longer reduce the relief mesh budget in lockstep: the UI keeps the detail samples tied to the usable printer footprint, and the backend applies a size-aware detail floor before meshing so the STL is generated at the requested physical size with finer facial/object features preserved. The default relief polarity is `raised-print`, and the backend chooses the effective invert setting from the actual depth model semantics so near-high relative depth models and far-high distance models both make faces/subjects protrude; switch to `Mold` only when negative relief is intentional. Metric/distance outputs such as Apple Depth Pro are preserved without destructive min-max storage and converted to inverse-depth proximity before relief normalization, which prevents distant sky pixels from flattening the subject into a near-constant-height slab. These runs keep `output_depth_preview.png` as the raw depth diagnostic and add `output_relief_preview.png` for the transformed height signal. The relief writer also performs robust percentile normalization, local feature boosting, a gamma relief curve, lower smoothing, and an optional crisp border ring so small features such as noses and sharper rear/side walls survive the depth-to-STL conversion. Each relief job writes `output_model.stl`, `diagnostics.json`, and `metadata.json`; the response includes `diagnostics_url` and inline STL checks for watertightness, manifoldness, positive volume, connected body count, bbox health, and mesh complexity.

For an NVIDIA GPU such as a 3080 Ti, install the CUDA build instead:

```bash
cd backend/
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
uvicorn main:app --host 0.0.0.0 --reload --port 8004
```

Frontend
```bash
cd frontend/
npm install
npm run dev
```

Open http://localhost:3000.

## Configuration

Backend `.env` values:

```bash
DEPTH_PROVIDER=transformers
DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf
OUTPUT_DIR=./output
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
VIDEO_CORS_ORIGINS=http://localhost:3000,http://localhost:3001
TRIPOSG_DIR=
TRIPOSR_DIR=
HUNYUAN3D_DIR=
SPAR3D_DIR=
SF3D_DIR=
IMAGE_TO_MESH_PROVIDER_PYTHON=
IMAGE_TO_MESH_TIMEOUT_SECONDS=3600
MASV_API_KEY=
MASV_TEAM_ID=
RBC_ACCESS_TOKEN=
```

Frontend `.env.local` values:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8004
NEXT_PUBLIC_VIDEO_BACKEND_URL=http://localhost:8005
REPLICATE_API_TOKEN=
GROQ_API_KEY=
COHERE_API_KEY=
```

## Model direction

- Current default: `Depth Anything V2 Large` through the local Transformers depth pipeline because it is the best verified CUDA-backed relief-depth option in the current local workflow.
- Quality options: choose Apple Depth Pro for sharp distance-edge experiments after its weights are preloaded, Depth Anything V2 Metric Indoor/Outdoor for metric-scene alternatives, or Depth Anything V2 Base/Small when you want lighter relative-depth fallbacks. Far-high models use inverse-depth relief shaping while relative near-high models retain linear shaping. Depth Pro runs write fallback metadata when the local cache is incomplete and the backend uses the effective model semantics when choosing raised-print versus mold polarity.
- Fallback: `Sapiens Depth` for human-centric depth estimation through the remote Gradio Space.
- Current fast completion helpers: optional mirror-completion and mirror seam repair before depth estimation, useful for roughly symmetric front-facing subjects when one side is cut off.
- Current modern completion providers: optional AMUSED Inpaint, DreamShaper Inpaint, SDXL Inpaint, FLUX.1 Fill, Qwen Image Inpaint, and Qwen Image Edit paths through Diffusers. These use learned priors instead of simple mirroring; AMUSED/DreamShaper/SDXL are the practical local GPU baselines, while Qwen/FLUX are much larger and may need model access, downloads, CPU offload, and long first runs.
- Training direction: build paired examples where the input image has a masked/blank half and the target is the original full image, then train a small LoRA adapter with `backend.benchmark.train_inpainting_lora` and rank that adapter with the same benchmark objective. Exported silhouettes can be used as object-loss masks, pair export writes category-templated prompts from mesh metadata plus `pair_export_report.json` provenance, and `weight_training_pairs` can now derive per-example `sample_weight` values from benchmark object-surface/depth errors so the next adapter can focus on cases where the current completion-to-STL path fails hardest. The current path is still adapter/LoRA-style training rather than training a base diffusion model from scratch.
- STL-first mesh path: the benchmark can now compare direct image-to-mesh candidates against the depth-relief STL route. `source-mesh-oracle` and `{source_bbox_extents}` are ground-truth diagnostics for rendered mesh datasets and are never promotion eligible. The deployable `{inferred_bbox_extents}` alias instead derives target extents from the sibling depth-relief/mirror STL. `backend.benchmark.run_image_to_mesh_provider` wraps SPAR3D/SF3D/TripoSR/Hunyuan3D-style outputs, can opt into STL-space scale/bbox postprocess before export, and `external-image-to-mesh` scores the emitted STL for mesh-surface error, watertightness, body count, bounding-box health, and printable complexity.

Measured STL-first update on July 10, 2026: Colab G4 run `g4_stl_first_triposg_inferred_adaptive_s40_n10` completed all `70/70` rows (`7` methods x `10` samples) on an RTX PRO 6000 Blackwell with 95 GB VRAM. The deployable TripoSG biharmonic-prefill candidate scored `1.1038838` versus `0.1991798` for `mirror`, won all `10/10` paired comparisons, and is now `promote` with every per-sample STL gate passing. Adaptive `--mesh-max-normalized-face-density-log1p 9.95` reduced the previous five complexity failures to zero; candidate complexity peaked at `9.9499179`. See the benchmark doc and its compact evidence bundle for the result archive checksum, per-sample metrics, visual contact sheet, and diagnostic source-oracle result.

## Benchmarking Completion

See [docs/completion-benchmark.md](docs/completion-benchmark.md) for the half-image completion benchmark, rendered 3D mesh dataset generator, depth/STL artifact metrics, Colab G4 orchestrator, Kaggle kernel scaffold, and current measured findings.

Backend regression checks for the benchmark/export path:

```bash
.\backend\.venv\Scripts\python -m unittest discover -s backend/tests -v
```

Heavy benchmark/training runs should use the Colab G4 orchestrator when available. The connected notebook target for this iteration is [Colab G4](https://colab.research.google.com/drive/1SuilhFuF5L3ELkEy2rnEsTmKAL19ob60), and the script entry point is `python -m backend.benchmark.colab_g4_orchestrator`. It keeps one GPU runtime busy with batched cache, calibration, metric-weighting, LoRA training, held-out evaluation, and combined ranking stages.

Quick rendered-mesh smoke:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/rendered_procedural_v2 --source procedural --count 12 --size 256
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/rendered_procedural_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --methods mirror,biharmonic --limit 12 --skip-depth
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv
```

Real CAD assets are supported through the mesh-folder path. The current ModelNet10 smokes render 20-64 `.off` assets, rank `masked`, `mirror`, `biharmonic`, DreamShaper, AMUSED, SDXL configs, trained LoRA adapters, and direct mesh methods, emit depth-derived 3D surface metrics plus watertight STL diagnostics on depth-enabled runs, and export paired training data for inpainting adapter work. The renderer can now build category-balanced mesh samples and writes stable `asset_key`, `asset_category`, and `asset_source_split` metadata so held-out LoRA runs can audit train/eval identity overlap. Modern-provider sweeps can use `--require-modern-cache` to write a cache preflight report and fail before expensive inference if planned fp16 weights are missing; direct image-to-mesh sweeps can use `--require-image-to-mesh-providers` to do the same for provider repos, Python environments, and entrypoints before STL-quality scoring begins. The benchmark tools can also break results down by category/source split, estimate oracle or validation-style selector upside, combine repeated held-out slices, and rerank with `--score-profile object-surface` for object depth relief quality or `--score-profile stl-quality` for final STL mesh accuracy, validity, and printability; see the benchmark doc for the exact commands and measured tables.

For visual QA, generate a contact sheet from any benchmark run or sweep:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines --baseline-method masked --contact-sheet --contact-sheet-max-samples 4
.\backend\.venv\Scripts\python -m backend.benchmark.make_artifact_contact_sheet backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --max-samples 4
```

The sheet shows manifest or explicit row references, raw/completed outputs, depth previews, object-depth error heatmaps, software-rendered STL thumbnails, and key RGB/depth/surface/STL metrics side by side, so a metric-driven selection can still be audited by eye. `optimize_completion --contact-sheet` can generate the same PNG directly while writing an experiment report. The first measured deterministic hybrid, `mirror-seam-repair`, is available as an experimental provider, but the ModelNet20 image benchmark keeps plain `mirror` ahead on the stable masked-baseline objective.

The cached learned baselines are now `dreamshaper-inpaint`, `amused-inpaint`, and `sdxl-inpaint`. On the legacy four-sample ModelNet10 learned smoke, the weighted ranking is mirror `16.10`, biharmonic `11.54`, DreamShaper `10.70`, and AMUSED `3.18`, so learned completion is wired in and measurable but is not yet the default winner. On two balanced held-out-10 depth/STL slices combined into a 20-sample evidence set, the default baseline-delta ranking is mirror `11.1347`, biharmonic `9.1354`, DreamShaper `7.4938`, AMUSED `7.3424`, SDXL `4.2560`, and masked `0.0000`; the gate keeps the current `mirror` default with `20/20` paired wins over masked. Under the STL/object-surface-only score profile, mirror still leads (`2.4490`), but the learned ordering changes to SDXL `2.0278`, DreamShaper `1.9697`, AMUSED `1.8530`; SDXL remains a `hold` against mirror because it trails the current score and fails paired-current confidence.

The training loop is also wired: `export_training_pairs` creates masked/full/object-mask targets, `train_inpainting_lora` can train a DreamShaper-compatible LoRA with optional mask/seam/object-weighted loss, and `run_completion_benchmark` / `optimize_completion` can evaluate that adapter with `--start-index`, `--model-name`, `--lora-weights`, and `--lora-scale`. Pair export, LoRA training, split audits, experiment reports, and selection decisions now carry training provenance, hashes, loss recipes, prompt-family checks, and adapter hashes so LoRA candidates can be traced back to the exact recipe that produced them; older cached runs can be repaired with `backend.benchmark.backfill_lora_provenance`, and interventions can be compared with `backend.benchmark.compare_optimize_runs`. The best current learned image-only smoke is a 100-step mask/seam LoRA at scale `0.75` with rank `10.30`, close to biharmonic `11.12`; the best current learned depth/STL smoke is a 200-step mask/seam LoRA at scale `1.0` with rank `16.65`, versus base DreamShaper `8.13`, biharmonic `19.50`, and mirror `24.22`. Heavy object-only weighting, category-prompt-only training, category-prompt plus light object loss, a 48-row train / 16-row held-out larger-data pass, a category-balanced 40-train / 20-held-out low-scale sweep, category-wise selector analysis, and a prompt-matched rerun of the balanced LoRA were all useful negative results. On the balanced held-out-20 image run, the best LoRA scale was `0.5`, but it still ranked behind mirror and biharmonic and had object MAE `0.387` versus base DreamShaper `0.369`, mirror `0.160`, and biharmonic `0.127`; slice analysis showed mirror winning every category. On the balanced held-out-10 depth/STL top-candidate probe, now including the `masked` no-completion baseline, the same LoRA at scale `0.5` beats base DreamShaper downstream (`19.24` vs `14.26`) and on the stable masked-baseline score (`4.53` vs `3.90`), while mirror `24.67` / `7.69` and biharmonic `20.23` / `6.66` still lead because object RGB fidelity remains poor. `masked` ranks last in that combined probe (`10.28`, stable score `0.00`), and the separate three-method lower-bound run ranks it `7.70` versus mirror `23.33` and biharmonic `18.74`, confirming that completion helps before depth/STL even though learned completion is not yet the default. Benchmark summaries now include attempted count, error count, success rate, split audits, selected baseline deltas, paired baseline wins, paired objective confidence, candidate-normalized ranking, baseline-delta scoring, depth-derived 3D surface distance, STL watertight/positive-volume diagnostics, and an explicit promotion gate that writes `selection_decision.json` / `.md`. `backend.benchmark.ingest_stl_results` also writes promotion-gate failure and sample-hotspot tables with per-method gate names and failed sample IDs, which makes Colab tarball triage faster when a direct mesh wins visually but fails printability. Cached depth/STL runs can be upgraded with `backend.benchmark.backfill_surface_metrics` to add surface metrics without rerunning depth inference; after backfilling the held-out-10 depth/STL sweep, the surface-aware stable ranking is mirror `9.96`, biharmonic `8.42`, LoRA scale `0.5` `6.55`, base DreamShaper `5.85`, LoRA scale `0.75` `5.59`, and masked `0.00`. A prompt-matched rerun passes the prompt provenance gate but drops the LoRA score to `5.17`, so the current gate keeps `mirror` as the default. The STL writer also orients exported triangle winding against the same signed-volume convention used by those diagnostics. Learned completion is now meaningfully better than the public DreamShaper baseline in some continuity, seam, and downstream depth metrics, but it is still not the default winner.

STL-first ingest reports now also include an architecture replacement decision: the best promotion-eligible single-image or multiview STL challenger is compared against the best promotion-eligible depth-relief baseline, then reported as `promote-challenger` or `keep-depth-relief` with the score delta and any gate-blocked score leader.
