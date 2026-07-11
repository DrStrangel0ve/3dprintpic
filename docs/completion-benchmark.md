# Completion Benchmark

This benchmark measures whether a completion method helps `3dprintpic` before depth estimation and STL generation.

## Goal

Given a known full render and ground-truth depth/silhouette, hide one half of the image, complete it with a candidate method, restore the original visible pixels, then score the completed hidden half and downstream depth estimate.

The optimization target is concrete:

- maximize masked-half similarity to the original render
- avoid changing visible pixels before preservation
- minimize seam error across the mask boundary
- improve depth quality on hidden object pixels after scale/shift alignment
- preserve object silhouette/shape
- for STL-first runs, minimize final mesh-surface error while preserving watertightness, positive volume, sane body count, 3D bounding-box health, and printable complexity

Default rank scores are normalized within a single run's candidate set. Use them to choose the winner inside one table, not to compare scores across tables with different methods or metrics. For cross-sweep comparisons that include the same baseline, use `--score-mode baseline-delta --baseline-method masked`; that score is a weighted improvement over the baseline, so unchanged methods keep the same score when unrelated candidates are added.

## Quick Synthetic Smoke

Use this as the cheapest sanity check. It is deterministic and fast, but it is still a toy 2D generator.

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_synthetic_dataset --output-dir backend/output/completion-benchmark/synthetic_v2 --count 12 --size 256
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/synthetic_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/smoke_baselines_v3_maskfixed --methods mirror,biharmonic --limit 12 --skip-depth
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/smoke_baselines_v3_maskfixed/summary_metrics.csv
```

Current corrected result on 12 samples:

| method | n | rank | masked PSNR med | masked SSIM med | masked MAE med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 12 | 8.65 | 16.34 | 0.865 | 0.053 | 0.0075 |
| biharmonic | 12 | 2.00 | 6.92 | 0.625 | 0.386 | 0.0100 |

## Rendered Mesh Benchmark

This is the preferred local benchmark because it starts from actual 3D meshes and writes RGB, depth, silhouette, mask, mesh, camera metadata, and a manifest.

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/rendered_procedural_v2 --source procedural --count 12 --size 256
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/rendered_procedural_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --methods mirror,biharmonic --limit 12 --skip-depth
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv
```

Current corrected image-only result:

| method | n | rank | masked PSNR med | masked SSIM med | masked MAE med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 12 | 8.65 | 16.64 | 0.834 | 0.0509 | 0.0040 |
| biharmonic | 12 | 2.00 | 7.99 | 0.683 | 0.3060 | 0.0105 |

Depth-enabled smoke on 2 rendered mesh samples:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/rendered_procedural_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/rendered_procedural_v3_depth_corrected --methods mirror,biharmonic --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_depth_corrected/summary_metrics.csv
```

| method | n | rank | depth MAE med | object depth MAE med | object depth corr med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 2 | 16.40 | 0.126 | 0.153 | 0.725 | 0.701 |
| biharmonic | 2 | 2.00 | 0.403 | 0.316 | 0.420 | 0.426 |

Interpretation: mirror is still the baseline to beat on symmetric half-missing views. Modern diffusion completion only counts as better when it beats mirror on held-out masked pixels and downstream object-depth/silhouette metrics.

## Real Mesh Folders

The same renderer can consume downloaded mesh datasets:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/modelnet_smoke --source mesh-dir --asset-root data/modelnet10 --asset-glob "**/*.off" --count 20 --size 256 --views-per-asset 1
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/ycb_smoke --source mesh-dir --asset-root data/ycb --asset-glob "**/*.glb" --count 20 --size 256 --views-per-asset 1
```

The ModelNet10 path has now been exercised on a real 20-asset CAD batch:

```bash
.\backend\.venv\Scripts\kaggle datasets download -d balraj98/modelnet10-princeton-3d-object-dataset -p data/modelnet10 --unzip
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/modelnet10_20_s256_seed2026 --source mesh-dir --asset-root data/modelnet10/extracted/ModelNet10 --asset-glob "**/*.off" --count 20 --size 256 --views-per-asset 1 --seed 2026 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_20_baselines --methods mirror,biharmonic --limit 20 --skip-depth --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/modelnet10_20_baselines/summary_metrics.csv
```

The full archive download timed out locally after writing a usable 423 MB zip, but extraction produced 4,474 `.off` assets. The rendered 20-sample manifest completed with no render failures.

Current ModelNet10 image-only result:

| method | n | rank | masked PSNR med | masked SSIM med | masked MAE med | object PSNR med | object SSIM med | object MAE med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 20 | 9.65 | 13.86 | 0.774 | 0.0788 | 12.86 | 0.549 | 0.1247 | 0.0092 |
| biharmonic | 20 | 8.45 | 7.39 | 0.598 | 0.3415 | 17.90 | 0.580 | 0.0946 | 0.0161 |

Depth-enabled smoke on 4 ModelNet10 samples:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_4_depth_smoke --methods mirror,biharmonic --limit 4 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/modelnet10_4_depth_smoke/summary_metrics.csv
```

| method | n | rank | depth MAE med | object depth MAE med | object depth corr med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biharmonic | 4 | 13.20 | 0.312 | 0.208 | 0.365 | 0.251 |
| mirror | 4 | 12.65 | 0.131 | 0.232 | 0.188 | 0.335 |

Interpretation: on real CAD assets, mirror is still better across the whole hidden half and seam, but biharmonic can score better on object-only RGB/depth in some views. This is the first benchmark where a learned inpainting method has a real opening to beat both baselines.

Mirror seam-repair check on the same 20-sample ModelNet10 image benchmark:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_20_mirror_repair_image_v2 --methods masked,mirror,mirror-seam-repair,biharmonic --limit 20 --skip-depth --resume --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_20_mirror_repair_image_v2 --baseline-method masked --score-mode baseline-delta --contact-sheet --contact-sheet-max-samples 4 --output backend/output/completion-benchmark/runs/modelnet10_20_mirror_repair_image_v2/report.md
```

| method | stable score vs masked | masked MAE med | seam MAE med | object MAE med | object PSNR med |
| --- | ---: | ---: | ---: | ---: | ---: |
| mirror | 6.0234 | 0.07885 | 0.00923 | 0.12466 | 12.86 |
| mirror-seam-repair | 6.0218 | 0.07888 | 0.00968 | 0.12471 | 12.86 |
| biharmonic | 4.9500 | 0.34152 | 0.01613 | 0.09464 | 17.90 |
| masked | 0.0000 | 0.23016 | 0.19696 | 0.59880 | 4.09 |

Interpretation: the narrow seam-repair hybrid is not a promotion candidate. It is cheap and available as an experimental provider, but the measured default should remain plain `mirror` because the repair band slightly worsens the stable objective and seam median on this CAD slice. A feather-width sweep also kept the existing 24 px mirror blend as the best tested mirror setting.

Depth-to-STL smoke on 2 ModelNet10 samples:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_2_depth_stl_oriented_v2 --methods mirror,biharmonic --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/modelnet10_2_depth_stl_oriented_v2/summary_metrics.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_2_depth_stl_oriented_v2
```

| method | n | rank | depth MAE med | object depth MAE med | object depth corr med | silhouette IoU med | STL watertight med | STL positive volume med | STL faces med | STL z-range med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| biharmonic | 2 | 15.70 | 0.468 | 0.220 | 0.365 | 0.251 | 1.0 | 1.0 | 29580 | 49.65 |
| mirror | 2 | 15.15 | 0.131 | 0.268 | 0.241 | 0.488 | 1.0 | 1.0 | 29580 | 45.15 |

The STL writer now orients exported triangle winding to positive signed volume when possible. This keeps the downstream printable artifact score from rewarding visually plausible depth maps that export as inverted closed meshes.

STL-first direct mesh smoke on the same balanced ModelNet10 held-out slice:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_direct_mesh_s40_n2_v3 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_direct_mesh_smoke.json --start-index 40 --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,source_mesh_oracle --contact-sheet-max-samples 2 --resume --continue-on-error
```

This run compares the existing depth-to-STL relief path with a direct source-mesh diagnostic. `source-mesh-oracle` transforms the original rendered mesh into the sample camera frame and converts it to STL without depth inference; it is not a deployable image-to-3D model, but it proves the evaluator can score full mesh outputs against ground truth. On ModelNet10, the source assets can be fragmented or non-watertight, so the oracle gets near-zero mesh-surface error but still fails printability checks.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | STL watertight med | STL volume mesh med | STL bodies med | STL body excess log1p med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| biharmonic | 2 | 0.0501 | 0.1864 | 0.4595 | 1.0 | 1.0 | 1 | 0 |
| masked | 2 | 0.0000 | 0.1900 | 0.4812 | 1.0 | 1.0 | 1 | 0 |
| mirror | 2 | -0.0366 | 0.1856 | 0.4749 | 1.0 | 1.0 | 1 | 0 |
| source_mesh_oracle | 2 | -7.9074 | 0.0000 | 0.0000 | 0.0 | 0.0 | 198 | 4.0202 |

Use `external-image-to-mesh` to plug in a real image-to-3D backend. The command template receives `{input_image}`, `{input_bundle}`, `{masked_image}`, `{full_image}`, `{mask}`, `{output_mesh}`, `{output_stl}`, `{output_dir}`, `{sample_id}`, and `{method}`. The external process may write either `{output_stl}` directly or a mesh at `{output_mesh}`; the harness converts a mesh output to STL before scoring. Provider meshes can opt into STL-space postprocess before export with `--mesh-target-max-dimension`, `--mesh-min-bbox-dimension`, and `--mesh-target-faces` on `backend.benchmark.run_image_to_mesh_provider`.

When an experiment config contains `run_image_to_mesh_provider` direct-mesh commands, `optimize_completion` now writes `image_to_mesh_provider_preflight.json` before sample work starts. The report records provider readiness, wrapper/provider Python availability, configured repo env vars, expected entrypoints, and the experiments that share each provider command. Add `--require-image-to-mesh-providers` for scored G4 direct-mesh sweeps; it fails before inference when a TripoSR/TripoSG/Hunyuan/SPAR3D/SF3D repo, Python environment, or entrypoint is missing.

STL-first configs and reports now label each architecture with `stl_mode`: `depth-relief` for the existing completion/depth/STL baseline, `single-image-mesh` for direct image-to-mesh providers such as TripoSR or Hunyuan3D shape, `multiview-mesh` for video/multiview reconstruction providers, and `source-mesh-oracle` for ground-truth diagnostic runs. `optimize_completion` infers the mode when a config omits it, but explicit labels make JSON configs and aggregate reports easier to audit. STL ingest reports also emit an architecture replacement decision that compares the best promotion-eligible single-image or multiview challenger against the best promotion-eligible depth-relief baseline, with the score delta and a `promote-challenger` or `keep-depth-relief` recommendation.

`backend.benchmark.run_stl_first_smoke` now includes a cheap multiview diagnostic by default: `source_mesh_bundle_multiview_oracle`. It routes through `external-multiview-to-mesh`, reads the generated `multiview_input.json`, and emits the source mesh through `run_image_to_mesh_provider --provider source-mesh-bundle-oracle`. This is intentionally not a deployable reconstruction model; it validates the same bundle contract, STL export path, repair/scaling options, and STL-quality metrics that COLMAP/OpenMVS/VGGT/SV3D-style providers will need before spending G4 time on them. Completed STL-first smoke runs also write `stl_first_architecture_report.json` and `.md`, which identify the deployable score leader, promotion-eligible winner, source-oracle diagnostic winner, and best method in each STL lane. The promotion winner must pass hard printability gates, so a direct image-to-3D model can remain the score leader while still being blocked from promotion by non-manifold or degenerate STL geometry. Ingest and selection decisions now apply those gates to raw per-sample STL diagnostics when available, which prevents a good aggregate median from hiding one failed mesh in a 10-20 row slice. Use `run_stl_first_smoke --write-config <path> --mesh-target-bbox-source none|source|mirror|inferred|reference` to generate repeatable direct-mesh bbox-calibration configs for Colab payloads without hand-authoring JSON. Prefer `inferred` for deployable runs: `{inferred_bbox_extents}` is derived from the sibling depth-relief/mirror STL. `{source_bbox_extents}` reads hidden ground truth, is oracle-only, and is never promotion eligible.

`run_image_to_mesh_provider --provider multiview-visual-hull` is now available as the first deployable multiview baseline. It consumes `{input_bundle}`, carves a voxel visual hull from the bundle's silhouettes and orthographic cameras, emits a mesh, and then runs through the same optional `--mesh-repair`, `--mesh-target-max-dimension`, and STL export path as learned direct-mesh providers. Use `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_multiview_visual_hull_smoke.json` on a manifest with `multiview_images`/`multiview_masks` fields to compare a real multiview STL candidate against `masked`, `mirror`, `biharmonic`, and `source_mesh_oracle` without needing external COLMAP/VGGT/SV3D dependencies. Source-mesh bundle oracle rows are kept as diagnostics only, even when they exercise the same multiview bundle contract.

The companion planner service now exposes the same lane live through `GET /providers/multiview-to-mesh` and `POST /run/multiview-to-mesh`. The endpoint accepts uploaded selected frames/masks plus a `multiview_input.json`-compatible bundle, rewrites bundle paths into the job directory, invokes the builtin visual-hull provider, and returns `output_model.stl`, `diagnostics.json`, `metadata.json`, and the normalized bundle under the shared artifact contract.

Local visual-hull STL-first slice on July 10, 2026: `stl_first_visual_hull_local_s0_n10_res32` generated 10 procedural assets with three 96 px views each, then compared `masked`, `mirror`, `biharmonic`, `source_mesh_oracle`, `source_mesh_bundle_multiview_oracle`, and `visual_hull_multiview_repaired_mesh` at visual hull resolution 32 and grid extent 1.9. All 60 method/sample rows completed without logged failures. After switching the STL complexity term from coordinate-scale-dependent `stl_faces_per_bbox_volume_log1p_median` to scale-free `stl_faces_per_normalized_bbox_volume_log1p_median`, the resolution-48/40/36 probes showed visual hull was the deployable score leader but too dense for at least one per-sample complexity gate. The calibrated resolution-32 run cleared every STL gate and promoted `visual_hull_multiview_repaired_mesh`: score `1.1760` vs `masked`, score margin `+0.7215` over `mirror`, paired win rate `8/10` vs `mirror`, CI95 low `+0.2369`, median Chamfer `0.0770`, H95 `0.2148`, median faces `6508`, scale-free complexity `9.4257`, and watertight/volume/manifold/single-body medians all `1.0`. The source-mesh oracle and source-mesh bundle oracle still rank highest as diagnostics, but they are no longer eligible deployable winners.

Local STL-first multiview smoke on July 9, 2026 generated two 64 px procedural views for one synthetic asset, ran one evaluated sample through `masked`, `mirror`, `biharmonic`, `source_mesh_oracle`, and `source_mesh_bundle_multiview_oracle`, and completed with `method_failures: []`. The multiview bundle oracle tied the source-mesh oracle at `+1.1707` under `stl-quality` baseline-delta scoring, confirming the external multiview lane can receive a two-view bundle and produce a scored STL through the provider wrapper.

Provider priority for the G4 lane is now: TripoSR API as the permissive smoke-test baseline, Hunyuan3D shape-only for the stronger diffusion shape candidate once setup stabilizes, and [TripoSG](https://github.com/VAST-AI-Research/TripoSG) as the next practical single-image mesh benchmark because it exposes a GLB-writing CLI with a `--faces` control and the repo advertises 8 GB+ VRAM as its baseline. SPAR3D/SF3D/TRELLIS remain useful follow-ups if setup and license constraints are acceptable. SV3D/VGGT/COLMAP/OpenMVS-style methods belong in the `external-multiview-to-mesh` lane because they synthesize views, cameras, depths, or point clouds before any final STL-producing mesh stage. SPAR3D remains documented below as an attempted direct mesh provider, but the current Colab image hit install friction and it should not block TripoSR/Hunyuan/TripoSG iteration.

The TripoSG wrapper path is `run_image_to_mesh_provider --provider triposg`. It calls the official module entrypoint, `python -m scripts.inference_triposg --image-input ... --output-path ...`, then converts/repairs/scales the output into the benchmark's STL artifacts. Fresh Colab packages can include `--include-triposg-setup`, which clones `/content/TripoSG`, creates `/content/triposg-venv`, filters TripoSG's Python 3.12-incompatible NumPy/Torch pins out of the requirements install, and supports `TRIPOSG_SETUP_ONLY=1` for setup-only probes. Use `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_direct_mesh_smoke.json` for the first one-row G4 check.

The checked one-row TripoSG payload was generated from repo commit `d6a8e68c54dc93f737273d24bb3cb28855d19073` as `g4_stl_first_triposg_s40_n1`. It packages held-out row `40`, uses `candidate_method=triposg_masked_repaired_stl_scaled_compact_direct_mesh`, and includes `--include-triposg-setup`. The local archive `modelnet10_heldout1_stl_first_triposg_colab_inputs.tar.gz` is `133,012` bytes with SHA256 `7cecdde67bc451272fdb51ba18c599ba400b1e0b953522a89d524536f41adf28`; the inline launcher SHA256 is `f2bafe3b09230fbb3b137c2dfe6d5eb07569573685843efa0ff587f6e91d56ea`, the fetch launcher SHA256 is `3cbdb12513abddb1968e9ff34fad647d448599fe5760961cdf198ff8ec92823a`, and the fetch notebook SHA256 is `944ce04e80263630da900eda6d39e7c1fb7cf7c35de32ff5aa8708ab00eeb2cf`. The immutable archive is published from payload commit `0cac0be92c1eba77125a8e35f1bca59855287cea` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/0cac0be92c1eba77125a8e35f1bca59855287cea/colab_payloads/modelnet10_heldout1_stl_first_triposg_colab_inputs.tar.gz`, and the paste-free notebook is published from payload commit `8380ffb8975f8183846e92dad351c96ccaa99b21` at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/8380ffb8975f8183846e92dad351c96ccaa99b21/colab_payloads/run_colab_triposg_eval_fetch_notebook.ipynb`. Raw GitHub downloads were validated at `133,012` bytes / SHA256 `7cecdde67bc451272fdb51ba18c599ba400b1e0b953522a89d524536f41adf28` for the archive and `3,403` bytes / SHA256 `944ce04e80263630da900eda6d39e7c1fb7cf7c35de32ff5aa8708ab00eeb2cf` for the notebook.

The TripoSG setup-only notebook uses the same immutable archive but was packaged with `--colab-env TRIPOSG_SETUP_ONLY=1`, so it can validate provider bootstrap before spending rows on the full STL-quality benchmark. It is published from payload commit `26eb9e4bbc9231c9137e036cd6c2101bcd83026b` at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/26eb9e4bbc9231c9137e036cd6c2101bcd83026b/colab_payloads/run_colab_triposg_setup_fetch_notebook.ipynb`; the raw notebook validated at `3,625` bytes with SHA256 `b2f1b69b2edfdd6999985083330429cf4343a7713ae1f15c8f100fd2d0d718fb`, and its fetch launcher SHA256 is `14428b7a1a334892843471b60c4f3ebd8d9ab5ba3b0bd9373e6c1275ffd61e4a`. Use this notebook first on a fresh G4 runtime; if it writes a setup-only archive with `run_status=0`, rerun the normal eval notebook above without `TRIPOSG_SETUP_ONLY`.

Colab G4 TripoSG setup-only evidence on July 10, 2026 SGT: the setup notebook downloaded the immutable archive, verified `133,012` bytes / SHA256 `7cecdde67bc451272fdb51ba18c599ba400b1e0b953522a89d524536f41adf28`, injected `TRIPOSG_SETUP_ONLY=1`, and the packaged launcher returned `returncode=0` after `271.341s`. The runtime wrote `/content/g4_stl_first_triposg_s40_n1_results.tar.gz` (`57,401` bytes) and `/content/3dprintpic_colab_inputs/modelnet10_heldout1_stl_first_triposg/results_summary.json` with `run_name=g4_stl_first_triposg_s40_n1`, `run_status=0`, and `provider_setup_only=true`. The terminal log tail reached `TripoSG setup checkpoint: python deps importable` followed by `Provider setup only requested; skipping benchmark stages`, so the next run should reuse that warm runtime for the full one-row eval.

The immediate full one-row TripoSG eval on the same G4 runtime also returned `run_status=0` and wrote `/content/g4_stl_first_triposg_s40_n1_results.tar.gz` (`1,911,688` bytes), but the direct TripoSG candidate failed before mesh output. The candidate stderr showed `ModuleNotFoundError: No module named 'diso'` from TripoSG's `from diso import DiffDMC`; its aggregate row therefore had `success_rate=0.0` and `error_count=1`. The depth-relief STL baselines all succeeded with valid watertight/volume/manifold/single-component STLs on this one table row: `masked` Chamfer L1 `0.1818`, `mirror` `0.1685`, and `biharmonic` `0.1669`; `source_mesh_oracle` also succeeded with Chamfer L1 `0.2470`. The next TripoSG package should include the `diso==0.1.4` setup fix and probe `importlib.import_module('triposg.pipelines.pipeline_triposg')` so setup-only catches this dependency before eval.

Fixed TripoSG `diso` package artifacts: repo commit `608254b2f342b0ab8545263fb9c2bf8966c4090d` installs `diso==0.1.4` and turns the setup probe into real `importlib.import_module(...)` imports. The corrected archive is published from payload commit `bbad22cfff1b6e29dd5b48460e38c0429095a332` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/bbad22cfff1b6e29dd5b48460e38c0429095a332/colab_payloads/modelnet10_heldout1_stl_first_triposg_colab_inputs.tar.gz`; raw validation was `132,988` bytes with SHA256 `d6b5a7bb894a8db63c876fcb8be2ee756e02b86353d7f4a7297a02aba792b44c`. The fixed paste-free eval notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/75b53e4b4b98ec888bc98d3f1b5b9af2a378cbb5/colab_payloads/run_colab_triposg_eval_fetch_notebook.ipynb` (`3,596` raw bytes, SHA256 `fde69bbe65d1fc4702d34e87b1729975635c2e722b8596e1f6bebcefb2b9b732`), and the fixed setup-only notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/75b53e4b4b98ec888bc98d3f1b5b9af2a378cbb5/colab_payloads/run_colab_triposg_setup_fetch_notebook.ipynb` (`3,625` raw bytes, SHA256 `50c7a528b56071ebe3c8d0bb6c6e3a65e6ab0ebdadc6e9cc8e31b852c69aca1d`).

Hardened TripoSG setup artifacts: repo commit `017c18e0aa5e96716e76a1ef2bcb2d9c7ec905a5` pins upstream TripoSG to `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`, runs `python -m scripts.inference_triposg --help` during setup so CLI-only imports such as `briarmbg`, `image_process`, `pymeshlab`, and `diso` fail before eval, and adds optional `TRIPOSG_PREFETCH=1` CUDA/VRAM/model-cache checks for `VAST-AI/TripoSG` and `briaai/RMBG-1.4`. The hardened archive is published from payload commit `d83d4f568aeebdd4c77ffa1841e3f397d5b9ced9` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/d83d4f568aeebdd4c77ffa1841e3f397d5b9ced9/colab_payloads/modelnet10_heldout1_stl_first_triposg_colab_inputs.tar.gz`; raw validation was `133,397` bytes with SHA256 `8e9f78e7cf184a9d02d7884588d52b35f9bba8ca2be04c884ca8a5e7c25408a6`. The hardened paste-free eval notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/be005fbe64356ad26038b8aee98337ed7443e3cd/colab_payloads/run_colab_triposg_eval_fetch_notebook.ipynb` (`3,596` raw bytes, SHA256 `01aeda1db1f812ff7ae36385d1c92d30af6ccfd4968de8b50ac0d020bb8f4c1f`), the setup-only notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/be005fbe64356ad26038b8aee98337ed7443e3cd/colab_payloads/run_colab_triposg_setup_fetch_notebook.ipynb` (`3,654` raw bytes, SHA256 `7a50d0572027fbec2c2d9d4d82f0b1aec8dfc981ce3d79f187a8a3d663788af4`), and the setup+prefetch notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/be005fbe64356ad26038b8aee98337ed7443e3cd/colab_payloads/run_colab_triposg_setup_prefetch_fetch_notebook.ipynb` (`3,654` raw bytes, SHA256 `b5b2fa9fc18a7fea1cdeb5c24b833b446eeeaf9e0273572d0cf4ea64e016062a`).

TripoSG `diso` isolation and non-flash fallback: repo commit `4ea4ea0` split `diso==0.1.4` into a native CUDA build with `--no-build-isolation`, but Colab's Python 3.12/CUDA image still made that source-only extension path brittle. Repo commit `c94d289` now patches TripoSG after checkout so `diso` is imported lazily only on the flash decoder path and the benchmark defaults to `TRIPOSG_USE_FLASH_DECODER=0`, which uses TripoSG's hierarchical `skimage.measure.marching_cubes` extraction instead. The non-flash archive is published from payload commit `46087f816d4ec105e28da4d2cf839f1d38d64207` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/46087f816d4ec105e28da4d2cf839f1d38d64207/colab_payloads/modelnet10_heldout1_stl_first_triposg_colab_inputs.tar.gz`; raw validation was `133,930` bytes with SHA256 `42b4b57d70cd47affc79a4b03d47fd7049d1927b4f164775ead196cdb1aecc75`.

The first non-flash setup-only notebook completed on Colab G4 with `run_status=0`, `provider_setup_only=true`, and a `7,296` byte setup archive, proving the CLI import path no longer needs `diso` by default. A follow-up full eval accidentally reconnected to a CPU runtime after session cleanup; baselines completed, but TripoSG failed with `Found no NVIDIA driver on your system`. Repo commit `4a1f572` fixes the generated fetch notebooks by adding top-level Colab `"accelerator": "GPU"` metadata. The GPU-requesting notebooks are published from payload commit `89f5281`: eval `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/89f5281/colab_payloads/run_colab_triposg_eval_fetch_notebook.ipynb`, setup-only `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/89f5281/colab_payloads/run_colab_triposg_setup_fetch_notebook.ipynb`, and setup+prefetch `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/89f5281/colab_payloads/run_colab_triposg_setup_prefetch_fetch_notebook.ipynb`.

Colab GPU TripoSG eval evidence on July 10, 2026 SGT: Colab assigned a T4 runtime for the GPU-requesting eval notebook, downloaded the immutable `42b4b57d...` payload, and the launcher returned `returncode=0`. The run wrote `/content/g4_stl_first_triposg_s40_n1_results.tar.gz` (`2,905,172` bytes) and `/content/3dprintpic_colab_inputs/modelnet10_heldout1_stl_first_triposg/results_summary.json` with `run_status=0` and `provider_setup_only=false`. The direct TripoSG method no longer fails: `triposg_masked_repaired_stl_scaled_compact_direct_mesh` had `success_rate=1.0`, `error_count=0`, watertight/volume/positive-volume/manifold/single-component medians all `1.0`, Chamfer L1 `0.1666`, and Hausdorff95 `0.3674`.

On that single held-out row, however, the deployable TripoSG candidate did not beat the depth-relief baselines under the current `stl-quality` profile: `biharmonic` remained the top method at `0.1075` rank score with Chamfer L1 `0.1663` and Hausdorff95 `0.4167`; `mirror` scored `0.0008` with Chamfer L1 `0.1694`; `masked` remained the zero baseline with Chamfer L1 `0.1807`; TripoSG scored `-0.5055` despite the best Hausdorff95 because its STL-space shape diagnostics, including bounding-box aspect ratio `3.3342`, were less aligned with the target on this row. This is now a real model-quality comparison rather than a setup failure. The next TripoSG slice should run 10-20 rows on G4/A100-class memory and sweep input choice (`masked`, repaired masked, mirror prefill, biharmonic prefill), crop/mask padding, and mesh scaling/repair settings before deciding whether to fine-tune or demote the backend.

Prepared 10-row TripoSG input-sweep artifacts: repo commit `bb5167a` adds `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_prefill_candidates.json`, which compares the depth-relief baselines and `source_mesh_oracle` against three deployable TripoSG direct-mesh inputs: `masked`, `mirror` prefill, and `biharmonic` prefill. The package uses held-out ModelNet10 rows `40-49`, `stl-quality` scoring, `candidate_method=triposg_mirror_prefill_repaired_stl_scaled_compact_direct_mesh`, `current_method=mirror`, `min_paired_n=10`, `max_method_failures=3`, and the same non-flash TripoSG setup.

The immutable 10-row archive is published from payload commit `af1fb85a068f89f8aabd32b32958634ce75ea3b6` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/af1fb85a068f89f8aabd32b32958634ce75ea3b6/colab_payloads/modelnet10_heldout10_stl_first_triposg_prefill_colab_inputs.tar.gz`; raw validation was `1,623,506` bytes with SHA256 `7d2482cfa0a7ccbf91a5d6da64605019860509d50f55cd71fa9c00ae515bb8af`. The summary-printing GPU eval notebook is published from payload commit `f49e797` at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/f49e797/colab_payloads/run_colab_triposg_prefill_eval_fetch_notebook.ipynb`; raw validation was `5,786` bytes with SHA256 `0cc3ec3f9e8c831f58b808ab606f744662841e0a6ce895e01429a2dbfc72647a` and top-level Colab `"accelerator": "GPU"`. Setup-only and setup+prefetch notebooks are published in the same payload commit as `run_colab_triposg_prefill_setup_fetch_notebook.ipynb` (`5,844` bytes, SHA256 `74a93eeeee4df3eac4fbcef948e32dc54690172958af263433a6b946add0c0af`) and `run_colab_triposg_prefill_setup_prefetch_fetch_notebook.ipynb` (`5,844` bytes, SHA256 `1d3ac99a8897f95c79e46a2e1e9bf8913e6c156aeace3ece4943c4c9bbc9ea6e`). These notebooks print `---RESULTS_SUMMARY_JSON---` and `---RESULT_ARCHIVES_JSON---` after `run_colab_eval.sh`, then call `completed.check_returncode()`, so even failed Colab runs expose the result summary/archive checksum before raising.

If the Colab archive download is inconvenient, copy the notebook output containing those two marker blocks into a text file and extract the evidence locally:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.extract_colab_output_summary C:\path\to\colab_output.txt --output-dir backend\output\completion-benchmark\colab_output_extract\triposg_prefill_s40_n10
```

This writes `results_summary.json`, `result_archives.json`, `colab_output.txt`, and `colab_output_extract_report.json`. It does not replace `ingest_stl_results` when the tarball is available, but it preserves run status, setup-only state, run name, and archive size/SHA evidence from copied notebook output.

TripoSG setup+prefetch evidence on July 10, 2026 SGT: the setup+prefetch notebook above ran on a Colab T4 runtime, downloaded and SHA-verified the exact `7d2482...` payload, and launched `run_colab_eval.sh` with `TRIPOSG_SETUP_ONLY=1` and `TRIPOSG_PREFETCH=1`. The setup log reached the important checkpoints: Python deps importable, CLI imports ok, CUDA device `Tesla T4` with about `14.6 GB` VRAM, 11 TripoSG/RMBG files fetched, and `TripoSG setup checkpoint: weights prefetched`. The run then exited before benchmark rows with `Provider setup only requested; skipping benchmark stages`; `results_summary.json` reported `run_status=0`, `provider_setup_only=true`, `run_name=g4_stl_first_triposg_prefill_s40_n10`, and a small `/content/g4_stl_first_triposg_prefill_s40_n10_results.tar.gz` archive was present. The runtime was disconnected and deleted after capture to avoid leaving an idle GPU attached. Treat this as bootstrap evidence only; the full 10-row quality comparison is the G4 run below.

Colab G4 10-row TripoSG prefill eval evidence on July 10, 2026 SGT: the eval notebook above was switched from the default T4 setting to Colab `G4 GPU`, connected to `NVIDIA RTX PRO 6000 Blackwell Server Edition` with about `97,887 MiB` VRAM, downloaded and SHA-verified the exact `7d2482...` payload, and completed with `run_status=0` and `provider_setup_only=false`. The run wrote `/content/g4_stl_first_triposg_prefill_s40_n10_results.tar.gz` (`60,891,539` bytes, SHA256 `93d8fa0d58250412f8d69d091985e8efadbceb4baeae2e93b91be01b109c1655`) and printed both summary marker blocks. Local marker extraction preserved `results_summary.json`, `result_archives.json`, and `colab_output_extract_report.json`; direct browser download of the 59 MB archive was not captured in this pass, so run `ingest_stl_results` once the tarball is available locally.

| method | score | mesh Chamfer L1 med | mesh H95 med | STL aspect med | watertight/volume/manifold med | success |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| source_mesh_oracle | `0.6139` | `0.0746` | `0.2376` | `1.4834` | `1/1/1` | `1.0` |
| mirror | `0.2199` | `0.1723` | `0.5056` | `1.7935` | `1/1/1` | `1.0` |
| biharmonic | `0.1335` | `0.1885` | `0.4950` | `1.7205` | `1/1/1` | `1.0` |
| masked | `0.0000` | `0.1970` | `0.5226` | `1.7348` | `1/1/1` | `1.0` |
| `triposg_mirror_prefill_repaired_stl_scaled_compact_direct_mesh` | `-0.2636` | `0.1950` | `0.5349` | `2.2478` | `1/1/1` | `1.0` |
| `triposg_biharmonic_prefill_repaired_stl_scaled_compact_direct_mesh` | `-0.9083` | `0.1645` | `0.4010` | `4.5396` | `1/1/1` | `1.0` |
| `triposg_masked_repaired_stl_scaled_compact_direct_mesh` | `-1.1382` | `0.1829` | `0.4852` | `4.3476` | `1/1/1` | `1.0` |

All deployable methods produced watertight, volume, positive-volume, manifold, winding-consistent, and single-component STL medians of `1.0` on the 10-row slice. The selection decision was `hold` for `triposg_mirror_prefill_repaired_stl_scaled_compact_direct_mesh`, failing paired win-rate, paired CI, score-margin-vs-current, paired-win-rate-vs-current, and paired-CI-vs-current gates. The important result is that TripoSG is now a functioning direct image-to-STL backend, but under the current `stl-quality` objective it is not promotion-ready: `mirror` remains the deployable winner, while `triposg_biharmonic_prefill` is the most geometrically promising direct-mesh variant by Chamfer/H95 and should be the next tuning target for crop/mask/bbox scaling rather than replacing the default path.

The completed TripoSG calibration slice was wired as `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_bbox_tuning_candidates.json`. It kept the same STL baselines and source-mesh oracle, then compared TripoSG masked, mirror-prefill, and biharmonic-prefill direct meshes normalized to the deployable depth-relief bbox target historically named `{mirror_bbox_extents}`. It also included `triposg_biharmonic_prefill_repaired_stl_aspect_clamped2p25_direct_mesh` to separate exact bbox fitting from a softer aspect-only calibration. The measured run below used `triposg_biharmonic_prefill_repaired_stl_mirror_bbox_direct_mesh` as the main candidate and `mirror` as the current method. New configs should use the equivalent `{inferred_bbox_extents}` alias so the deployable provenance is explicit.

Prepared TripoSG bbox-tuning artifacts: repo commit `93436e18bf99caeec4e2775c9e6e0a1425ea2c2d` packages held-out rows `40-49`, `stl-quality`, `candidate_method=triposg_biharmonic_prefill_repaired_stl_mirror_bbox_direct_mesh`, `current_method=mirror`, `min_paired_n=10`, `max_method_failures=3`, and the same non-flash TripoSG setup. The immutable archive is published from payload commit `8e0aa117dc1c20a5329b0562d6bf3e7380d97da7` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/8e0aa117dc1c20a5329b0562d6bf3e7380d97da7/colab_payloads/modelnet10_heldout10_stl_first_triposg_bbox_tuning_colab_inputs.tar.gz`; raw validation was `1,623,551` bytes with SHA256 `bd3c576e692adcb4a5a8a286c0db4377ef3a182bf064629e908e21b82c77fe0e`. The summary-printing GPU eval notebook is published from payload commit `7d3cd6eba5768e4652e4a6a294bb97a3b5094628` at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/7d3cd6eba5768e4652e4a6a294bb97a3b5094628/colab_payloads/run_colab_triposg_bbox_tuning_eval_fetch_notebook.ipynb`; raw validation was `5,802` bytes with SHA256 `4ffaaf938b043f584331d808ceb541d77880e4d3ef656fb589ee2276852cfbbc`. Setup-only and setup+prefetch notebooks are published in the same payload commit as `run_colab_triposg_bbox_tuning_setup_fetch_notebook.ipynb` (`5,831` bytes, SHA256 `d4458bae97bd56698a45fe961d915b93ea756746303afa1345438da7f04062b6`) and `run_colab_triposg_bbox_tuning_setup_prefetch_fetch_notebook.ipynb` (`5,860` bytes, SHA256 `ecede69341d8592d3899879bd1bfc2cdc7d2ef5dcfc568cca9201fb5b655d44b`).

Completed G4 bbox-tuning payload: `modelnet10_heldout10_stl_first_triposg_bbox_tuning_colab_inputs.tar.gz` was regenerated from the `codex/3d-completion-benchmark-g4` branch after adding smoke-runner generated bbox configs, the in-run STL-first ingest hook, and compact GPU-preflight summaries for wrong-runtime triage. The payload used `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_mirror_bbox_generated.json`, produced by `run_stl_first_smoke --write-config ... --include-triposg --triposg-direct-input masked --triposg-direct-input mirror --triposg-direct-input biharmonic --mesh-target-bbox-source mirror --direct-mesh-reference-method mirror --mesh-target-faces 40000`, with `--require-image-to-mesh-providers`, `--colab-require-gpu-name-regex "RTX PRO 6000|Blackwell"`, and `--colab-min-gpu-memory-gb 90`. The generated `run_colab_eval.sh` writes `gpu_preflight.json` first and aborts before archive extraction, repo sync, or provider setup if Colab attaches CPU/T4/L4/A100-class hardware instead of the intended G4/Blackwell runtime. It still writes `image_to_mesh_provider_preflight.json` after setup and fails before rows if `/content/TripoSG`, `/content/triposg-venv/bin/python`, or the TripoSG entrypoint is missing. After orchestration, it runs `backend.benchmark.ingest_stl_results` against the Colab output root and adds `stl_first_ingest_report.json`, `stl_first_ingest_report.md`, and `stl_first_ingest.log` to the results archive; `results_summary.json` also includes `gpu_preflight_summary`, `failure_stage`, and a compact `stl_first_ingest_report` block. The payload is `1,625,971` bytes with SHA256 `4b46b9b238b10ff7e61d1c05674da330fa703b95e404ba7e47946c6ac94baf58`; the branch-hosted fetch notebook is `5,794` bytes with SHA256 `5d75d6c228fc629bc8a9668a849f83b79949451ffcd81dd7f37a50f595d8c2b7`; and the fetch cell is `3,897` bytes with SHA256 `4ebdba624f1cff03d8522fc7e5e19a768d04c801a57a162cf6ff12471073f874`. This is the payload used by the measured G4 run below.

Branch-notebook probe on July 10, 2026 SGT: Colab assigned `T4 (Python 3)`, so the GPU guard failed before repo setup or provider install. The cell still downloaded and SHA-verified the refreshed `dd2db2e...` payload, wrote `results_summary.json` with `run_status=1`, `output_root_exists=false`, `stl_first_ingest_report.status=skipped`, produced a tiny `/content/g4_stl_first_triposg_bbox_tuning_s40_n10_results.tar.gz` (`1,027` bytes, SHA256 `012d7479d5272251390004267e080f2c52628047cfa50473e1e0a9d5c42da097`), and the idle T4 runtime was disconnected/deleted. The current regenerated payload keeps the same early guard but now surfaces wrong-runtime failures directly as `failure_stage=gpu_preflight` and `stl_first_ingest_report.reason=gpu_preflight_failed` when `gpu_preflight.json` reports no eligible GPU.

Local relief-detail validation on July 10, 2026 SGT: the web app now computes the relief mesh sample budget from the usable printer footprint instead of the scaled print footprint, while `max_xy_size` still controls the actual STL XY size. The backend independently resolves an effective size-aware `target_dimension` from the requested budget, printer usable XY, and mesh multiplier, caps it with `RELIEF_MIN_DETAIL_DIMENSION`/`RELIEF_MAX_DETAIL_DIMENSION`, writes `requested_target_dimension`, `target_dimension`, `relief_sample_pitch_mm`, and `size_aware_detail` into `metadata.json`, and resizes near-target depth grids with NaN-aware interpolation instead of coarse stride sampling. Focused validation passed with endpoint metadata coverage and a cutout-hole resize regression: `py_compile`, `unittest backend.tests.test_main_stl_contract backend.tests.test_relief_stl_controls -v` (`30` tests), `npm run lint`, `npx tsc --noEmit --pretty false`, and `npm run build`. This keeps small physical relief prints printable at their requested size without throwing away facial/object detail before STL export.

Colab T4 setup+prefetch evidence on July 10, 2026 SGT: the setup+prefetch notebook downloaded and SHA-verified the exact `bd3c576e...` payload, launched with `TRIPOSG_SETUP_ONLY=1` and `TRIPOSG_PREFETCH=1`, and completed with `run_status=0`, `provider_setup_only=true`, and `/content/g4_stl_first_triposg_bbox_tuning_s40_n10_results.tar.gz` (`7,740` bytes, SHA256 `5de3df33a721d8624cc7f3d34692213c4aab95cbbf5a7b47bc9c742a3b731472`). Colab then kept assigning a T4 runtime for the full eval even after attempting to switch to G4. The opportunistic T4 full eval was not promotion evidence: before the browser/runtime resume failed, terminal probes showed the run root existed with `49` emitted `output_model.stl` files, all 10 baseline rows for `masked`, `mirror`, `biharmonic`, and `source_mesh_oracle`, and `9` partial STL outputs for `triposg_masked_repaired_stl_mirror_bbox_direct_mesh` with no final metrics CSV or selection decision yet. The stuck T4 runtime was disconnected and deleted after it failed to recover and the notebook returned to `Reconnect GPU`. This remains setup history only; the completed G4 evidence below supersedes it.

### Measured: G4 TripoSG Bbox Tuning

Colab run `g4_stl_first_triposg_bbox_tuning_s40_n10` completed on an NVIDIA RTX PRO 6000 Blackwell Server Edition with 95 GB VRAM. All `70/70` method/sample rows succeeded (`7` methods x `10` held-out samples), with no method errors. The input payload SHA256 was `4b46b9b238b10ff7e61d1c05674da330fa703b95e404ba7e47946c6ac94baf58`. The runtime wrote `/content/g4_stl_first_triposg_bbox_tuning_s40_n10_results.tar.gz`, `62,945,685` bytes, SHA256 `33cff6b51f0dc95e2287d71a1827f596f899ccd35ab901628b185304e22c48b7`.

| method | status | score | mesh Chamfer L1 med | mesh H95 med | bbox aspect med | scale-free complexity med |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | diagnostic/nondeployable | `2.9898396` | `0.0746460` | `0.2376423` | `1.4833750` |  |
| `triposg_biharmonic_prefill_repaired_stl_mirror_bbox_direct_mesh` | deployable candidate; hold | `0.9280740` | `0.1417979` | `0.3871739` | `1.9614717` | `9.3621272` |
| `mirror` | deployable baseline | `0.1991798` | `0.1592503` | `0.4768460` | `1.9614717` |  |

The candidate's historical `_mirror_bbox_` name means its target extents came from the sibling mirror/depth-relief STL; new configs expose the same deployable provenance as `{inferred_bbox_extents}` and `_inferred_bbox_`. The source oracle uses hidden ground truth and is never eligible for promotion, regardless of its score.

All candidate printability medians passed: watertightness, volume, manifoldness, positive volume, and single-component checks were `1.0`, while non-manifold-edge, degenerate-face, and body-excess medians were `0.0`. The decision was `hold` solely on `per_sample_stl_scale_free_complexity`: `5/10` candidate rows exceeded the maximum `10`. The outliers were `mesh_dir_757e06b221_v00`, `mesh_dir_1a200d8288_v00`, `mesh_dir_49534574e1_v00`, `mesh_dir_d6aa35e98f_v00`, and `mesh_dir_a3445448e2_v00`. Using the minimum normalized bbox volume from this slice, a fixed cap of `9,974` faces would be safe across all ten samples.

The notebook's in-run ingest report failed because nested one-method `summary_metrics.csv` files were mis-discovered as standalone runs. Result discovery now excludes nested summaries beneath aggregate optimize runs. The printed marker summary is authoritative for the measurements above. The full result archive was not captured locally because the Colab browser could not download it, so no local archive-ingest claim is made.

### Measured: Metric-Driven Inferred Bbox

The measured run uses `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_inferred_bbox_generated.json`, with `--mesh-target-bbox-source inferred` and provider option `--mesh-max-normalized-face-density-log1p 9.95`. The implemented option adaptively caps each mesh at `max(4, floor(expm1(9.95) * normalized_bbox_volume))` faces after bbox calibration, directly optimizing the gate metric `log1p(face_count / normalized_bbox_volume)`. When exact bbox extents are available, TripoSG receives that cap before inference; final postprocess verifies the actual post-decimation metric, uses `fast-simplification` when further decimation is needed, and fails closed if simplification changes the bbox enough to remain above the limit. This per-mesh limit is preferred to the conservative fixed `9,974`-face cap because it preserves more detail on shapes with larger normalized volume while retaining a small margin below the promotion maximum of `10`. Local validation passed all `159` benchmark regression tests.

Adaptive G4 artifacts use run name `g4_stl_first_triposg_inferred_adaptive_s40_n10`, immutable runtime code commit `008f45b676960c38390080bee644f18883602433`, the same held-out rows `40-49`, candidate `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh`, current method `mirror`, `min_paired_n=10`, and the existing G4/Blackwell 90 GB guard. The deterministic payload `colab_payloads/modelnet10_heldout10_stl_first_triposg_inferred_adaptive_colab_inputs.tar.gz` is `1,626,001` bytes with SHA256 `4556853d62b9150dda9ea74b684c410f2ddba2cdc99ead8946b1187d5299d3d3`. Its fetch URL is pinned to payload commit `bec81c69e8cc284082c1293b31f10682767abe43`, avoiding moving-branch cache races. The fetch cell is `3,923` bytes with SHA256 `91d1674daa392a356b9019e8779ce334767da4ddbc89631065d6384a1ef008b0`; the fetch notebook is `5,826` bytes with SHA256 `5e486dae038935faff75892eea46c961d63c457db9c0818e41d493895f6796fd`. Regenerating the archive with the same command produced the same SHA256. The immutable notebook is `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/31c89741e579c82761dc345833889938dc95a51f/colab_payloads/run_colab_triposg_inferred_adaptive_eval_fetch_notebook.ipynb`.

Colab G4 run `g4_stl_first_triposg_inferred_adaptive_s40_n10` completed all `70/70` rows on an NVIDIA RTX PRO 6000 Blackwell Server Edition with `95.5928` GB VRAM. The provider preflight reported TripoSG ready with no setup errors, the benchmark returned `run_status=0`, the in-run STL ingest returned `status=ok`, and a second local archive ingest reproduced the same architecture recommendation.

| method | status | score | mesh Chamfer L1 med | mesh H95 med | STL faces med | scale-free complexity med |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | diagnostic/nondeployable | `2.9898396` | `0.0746460` | `0.2376423` | `103` | `5.3730671` |
| `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | promoted candidate | `1.1038838` | `0.1593129` | `0.3849049` | `3,855` | `8.2388673` |
| `triposg_masked_repaired_stl_inferred_bbox_direct_mesh` | deployable challenger | `0.9674651` | `0.1890342` | `0.4861368` | `584` | `7.0139660` |
| `triposg_mirror_prefill_repaired_stl_inferred_bbox_direct_mesh` | deployable challenger | `0.5622659` | `0.1651543` | `0.4233816` | `10,677` | `9.9482615` |
| `mirror` | previous current method | `0.1991798` | `0.1592503` | `0.4768460` | `36,860` | `11.1885910` |

The selector decision is `promote` with no failed checks. The candidate won `10/10` paired comparisons against both `masked` and current `mirror`; CI95 lower bounds were `1.0655100` and `0.9277428`. All twelve per-sample STL gate families passed `10/10`. Candidate scale-free complexity ranged from `6.4386047` to `9.9499179`, so the previous five complexity failures are now zero. Relative to the earlier mirror-bbox candidate, score improved from `0.9280740` to `1.1038838` and complexity median fell from `9.3621272` to `8.2388673`; Chamfer moved from `0.1417979` to `0.1593129`, while H95 improved slightly from `0.3871739` to `0.3849049`. This is the intended trade: enough adaptive decimation to make every mesh promotion-eligible without destroying the candidate's objective advantage.

The runtime wrote `/content/g4_stl_first_triposg_inferred_adaptive_s40_n10_results.tar.gz`, `29,533,390` bytes, SHA256 `8cb4a836c3e5a1da205edb8c9dd80cf4bde26921c18110b13d5b1199ff358a78`. The archive was transferred locally, matched the Colab byte count and SHA256, and re-ingested successfully. A compact [GitHub evidence bundle](benchmark-evidence/g4_stl_first_triposg_inferred_adaptive_s40_n10/README.md) preserves the decision JSON, ranked metrics, sanitized candidate per-sample metrics, GPU/runtime preflight, ingest report, result summary, and contact sheet; the full archive remains local rather than adding a 29.5 MB binary to git.

### Measured: Hunyuan3D-2mv Octants Smoke

Run `g4_stl_first_hunyuan3d_2mv_octants8_s40_n1_s5_r256` used runtime code `8de0bc749da24ec3b2c018ff6b37d391bbc763b1`, official Hunyuan source commit `f8db63096c8282cb27354314d896feba5ba6ff8a`, and model revision `3a761b539b29fe4ff64714813aa9560fd66f5de0`. It evaluated the first asset from the exact held-out ten-asset TripoSG slice using eight consistent renders. Cardinal views at 0, 90, 180, and 270 degrees were model inputs; intermediate views at 45, 135, 225, and 315 degrees were held out for silhouette agreement. The provider did not read source geometry.

The G4 run completed all six method rows with `run_status=0`; both the generated Blackwell guard and official pipeline preflight passed. The repaired Hunyuan row led the deployable smoke score and passed every per-sample STL gate. Raw and repaired results are intentionally separate:

| method | score | Chamfer | H95 | held-out IoU | components | faces | scale-free complexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hunyuan3d_2mv_cardinal4_s5_r256_repaired_stl_inferred_bbox_direct_mesh` | `0.6308643` | `0.1641288` | `0.3861005` | `0.7914006` | `1` | `4,372` | `9.0454495` |
| `mirror` | `0.1183079` | `0.1946848` | `0.4253555` | n/a | `1` | `36,860` | `11.1772520` |
| `masked` | `0.0000000` | `0.1838510` | `0.3718838` | n/a | `1` | `36,860` | `11.1729047` |
| `hunyuan3d_2mv_cardinal4_s5_r256_raw_direct_mesh` | `-5.4496193` | `0.3767970` | `1.1326047` | `0.0101568` | `10` | `15,468` | `10.2349665` |

Printable repair converted the fragmented raw output into one watertight, manifold, winding-consistent, positive-volume body and brought complexity below the hard limit of `10`. Peak CUDA allocation/reservation were `5.5495/5.7266` GiB. Native timings separated `10.1321` seconds of initial model load from `4.0860` seconds of diffusion inference; the first complete provider invocation took `20.1984` seconds, while the repaired row reused the content-addressed raw-mesh cache and added `0.0593` seconds of repair.

The input payload was `38,240` bytes with SHA256 `48b675a222d04c1e7b3f76090b0e206e91daecd47bc7a71ab97d4a1b75c6f807`. The result archive was recovered locally at `2,497,447` bytes with SHA256 `0385ffe62c009eb2170032f048afd74b2e8769301398921c4026aeb82683a423`; local ingest reproduced the in-run `promote-challenger` architecture result. This remains smoke evidence because `n=1`. The compact [Hunyuan smoke evidence bundle](benchmark-evidence/g4_stl_first_hunyuan3d_2mv_octants8_s40_n1_s5_r256/README.md) is committed separately, and the deciding ten-row, 50-step comparison against promoted TripoSG uses the same held-out assets.

### Measured: Hunyuan3D-2mv vs TripoSG n10

Run `g4_stl_first_hunyuan3d_2mv_vs_triposg_octants8_s40_n10` completed all `70/70` rows with `run_status=0`, runnable TripoSG and Hunyuan preflights, and no provider failures. It used the same ten held-out assets, pinned runtime code `8de0bc749da24ec3b2c018ff6b37d391bbc763b1`, 50 Hunyuan diffusion steps, octree resolution 380, four cardinal model inputs, and four intermediate held-out views. All ten repaired Hunyuan rows were content-addressed cache hits on their corresponding raw inference.

| method | status | score | Chamfer median | H95 median | held-out IoU median | faces median | complexity median |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hunyuan3d_2mv_cardinal4_repaired_stl_inferred_bbox_direct_mesh` | printable challenger | `1.5784637` | `0.1741899` | `0.3504748` | `0.6551098` | `352` | `6.5212088` |
| `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | promoted incumbent | `0.2849006` | `0.1956434` | `0.4608482` | n/a | `10,710` | `9.9497980` |
| `mirror` | depth-relief control | `0.0563840` | `0.1750209` | `0.5071317` | n/a | `36,860` | `11.1802663` |
| `hunyuan3d_2mv_cardinal4_raw_direct_mesh` | diagnostic/unprintable | `-3.7386852` | `0.0488328` | `0.1558482` | `0.8998789` | `876,135` | `14.6145664` |

Repaired Hunyuan wins `10/10` paired objective comparisons against TripoSG, with paired CI95 lower bound `0.6987292`, and passes all twelve per-sample STL gate families on `10/10` assets. The generic STL architecture ingest therefore reports `promote-challenger`. The incumbent-aware selector correctly remains `hold`, however: paired Chamfer exceeds the `1.10x` ceiling on `mesh_dir_8a68107048_v00` (`1.6689293x`) and `mesh_dir_3ea1828365_v00` (`1.1347448x`), while paired H95 exceeds it on `mesh_dir_c3a9748da7_v00` (`1.1249693x`). Median ratios favor Hunyuan (`0.9042x` / `0.8174x`), so the failure is concentrated in three repair outliers rather than overall multiview reconstruction quality.

Raw Hunyuan geometry reveals the next optimization target. Its meshes range from `534,048` to `1,628,024` faces and achieve much better surface/view metrics, but only `6/10` are watertight/volumetric, `0/10` are manifold, and `3/10` are single-body. Repaired meshes pass every printability gate and complexity ranges from `5.6232598` to `9.9494379`, but face counts can collapse as low as `144`. The next iteration should perform shape-preserving dominant-component filtering and closure before adaptive face-density decimation, and optimize Chamfer/H95 while keeping the current STL gates unchanged.

The full result archive is `444,218,826` bytes with SHA256 `65ac1aba0e30c287c48de66cc5514a0796d7f67e14ed47dcf8c20afff2dd8e06`. A locally checksum-verified and re-ingested metrics archive is `1,110,155` bytes with SHA256 `09156bbe5d877ba392d5690df9f63d0e972dc041b5f8e676a9b2ecc7990c60c9`. The compact [n10 evidence bundle](benchmark-evidence/g4_stl_first_hunyuan3d_2mv_vs_triposg_octants8_s40_n10/README.md) preserves the complete decision and per-sample metric surface without committing the 444 MB mesh archive. The G4 runtime was disconnected after evidence recovery.

### Measured: Hunyuan Repair Ablation n4

Run `g4_stl_first_hunyuan3d_2mv_repair_ablation_s40_s6_n4` isolated the three Hunyuan repair outliers plus one control on runtime code `68f97d24c3ee87eb68c910efaa21d32fad4f4de3`. It compared the raw cached Hunyuan mesh, legacy printable repair, voxel closure at r192/r256 with and without 1% component-area filtering, the promoted TripoSG control, and the depth baselines. All `36/36` rows completed on the G4 Blackwell runtime. Median Hunyuan inference was `34.7425` seconds with `5.8008` GiB peak CUDA VRAM.

The configured r256 area-filtered candidate remained `hold`. It used convex hull fallback on `2/4` samples and had one `7.0766x` fill-ratio-drift outlier. Increasing voxel resolution did not remove the failure: the same bed and sofa fell back at r192 and r256. Failing post-decimation meshes collapsed from approximately `10.7k` faces to `134-210` hull faces, while successful samples retained approximately `10.7k` faces. The follow-up repair path therefore preserves an already printable voxel-decimated mesh before generic cleanup and records the exact printability predicates before and after cleanup.

This run also exposed a scoring error. The legacy hull originally ranked first at `1.5729`, but `1.2653` of that score came from reducing complexity below the printer budget. It used a hull on `4/4` samples and had median fill-ratio drift `12.2245`. Complexity is now a hard gate rather than an additive reward. Source-mesh Chamfer/H95 remain in reports only; the deployable `stl-quality` objective uses held-out camera agreement and STL validity. Single-image providers now render views held out from their primary input so TripoSG and multiview providers can be compared on the same non-oracle signal.

Repair promotion now requires hull fallback rate `<= 0.25`, median absolute fill drift `<= 0.5`, at least `75%` of samples within that drift, and no sample above `4.0`. Re-ingesting the recorded run under those gates blocks every Hunyuan repair variant and leaves TripoSG as the structurally eligible architecture. The incumbent-aware selector still returns `hold` because this four-row ablation is not a promotion run and the older TripoSG control lacks comparable held-out-view rows. Historical sections above describe the selector policy used at run time; source-geometry ratio gates are now opt-in diagnostics rather than defaults.

The compact result archive is `1,258,420` bytes with SHA256 `eb368249c84692a2c3f01938615915b543261d2e1ab9109f0d99a34e35523568`. The [repair-ablation evidence bundle](benchmark-evidence/g4_stl_first_hunyuan3d_2mv_repair_ablation_s40_s6_n4/README.md) preserves both policy versions, per-sample repair metrics, runtime provenance, corrected ingest report, and contact sheet. The next G4 action is a four-row cache-reuse smoke with the cleanup bypass and stage audits; expansion to the held-out ten is conditional on reducing fallback to at most one row without regressing held-out agreement.

### Measured: Cleanup-Bypass Audit n4

Run `g4_stl_first_hunyuan3d_2mv_cleanup_bypass_s40_s6_n4` replayed the same four assets on runtime code `d122a749b95eba893c6c881419a920e66a60860d`. All `36/36` rows completed, and depth relief plus single-image meshes now have held-out-view measurements. The configured r256 area-filtered candidate remains `hold`: hull fallback is still `2/4`, and the bathtub remains a `7.0766x` fill-drift outlier. TripoSG is the only promotion-eligible mesh provider in the slice.

The stage audit disproves generic cleanup as the sole failure. Five voxel meshes are fully printable before cleanup and now bypass it, but aggregate hull use remains unchanged. The r192 bathtub is otherwise printable and fails only because decimation leaves one degenerate face; the sofa is genuinely invalid before cleanup with `3-65` components and `6-97` nonmanifold edges. Area filtering also regresses printable chair/bed cases. The next bounded repair runs marching-cubes retriangulation only when exactly one degenerate face is the sole failed predicate, then bypasses cleanup only if the full strict audit passes. The held-out ten remains gated on the four-row result.

The run also exposed a Colab packaging defect: the benchmark changes directory inside a subshell, so post-run Python previously executed outside the repository and could not import `backend`. The generator now changes to `$REPO_DIR` before ingest/archive creation. The completed metrics were recovered without repeating inference.

The verified compact archive is `1,325,106` bytes with SHA256 `fea5c1acf27ee1bffeebaeb8abe95a2a2fb21a5c04caa54f0cba0c63fde018ac`. The [cleanup-bypass evidence bundle](benchmark-evidence/g4_stl_first_hunyuan3d_2mv_cleanup_bypass_s40_s6_n4/README.md) includes the sixteen voxel-stage audit rows, selection decision, corrected ingest report, provider/GPU provenance, and contact sheet.

One-time Colab setup:

```bash
cd /content
git clone --recurse-submodules https://github.com/Stability-AI/stable-point-aware-3d /content/stable-point-aware-3d
cd /content/stable-point-aware-3d
git submodule update --init --recursive
pip install -U setuptools==69.5.1 wheel
pip install -r requirements.txt
huggingface-cli login
export SPAR3D_DIR=/content/stable-point-aware-3d
```

SPAR3D direct-STL smoke:

```bash
python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_spar3d_direct_mesh_s40_n2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_spar3d_direct_mesh_smoke.json --start-index 40 --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,spar3d_direct_mesh --contact-sheet-max-samples 2 --resume --continue-on-error
```

The config uses `backend.benchmark.run_image_to_mesh_provider` as a thin adapter around SPAR3D's official CLI:

```bash
python -m backend.benchmark.run_image_to_mesh_provider --provider spar3d --input-image "{input_image}" --output-mesh "{output_mesh}" --output-stl "{output_stl}" --timeout 3600 --provider-device cuda --remesh-option none
```

The smoke config uses `--remesh-option none` so it works with the base SPAR3D install. Install SPAR3D's optional remesh dependencies before switching the config to `triangle` or `quad`. The same wrapper can also normalize `stable-fast-3d`, `triposr`, or shape-only `hunyuan3d-shape` outputs into the benchmark's mesh/STL artifact names. Add `--mesh-target-max-dimension 96 --mesh-min-bbox-dimension 12` when testing whether a provider's normalized model-space mesh is being unfairly punished by absolute STL scale/bounding-box diagnostics. Set `SPAR3D_DIR`, `SF3D_DIR`, or `TRIPOSR_DIR`, or pass `--provider-dir`, when the provider repo is not in a default `/content/...` location.

Direct mesh inputs can now be `masked`, `full`, `mirror`, or `biharmonic`. The `mirror` and `biharmonic` modes write `direct_mesh_input_<mode>.png` beside the provider outputs before calling the image-to-mesh backend, which makes it possible to test whether a cheap geometry prefill helps a single-image mesh model reconstruct the hidden side. After the raw masked SPAR3D smoke is healthy, run `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_spar3d_prefill_direct_mesh_smoke.json` on the same slice to compare `spar3d_masked_direct_mesh`, `spar3d_mirror_prefill_direct_mesh`, and `spar3d_biharmonic_prefill_direct_mesh` under the same STL-quality score.

Colab G4 SPAR3D install note: the July 9, 2026 Python 3.12 runtime reached repo commit `90891e8`, initialized the `texture_baker` submodule, and had `torchao 0.17.0+cu128`, but `pip install -r /content/stable-point-aware-3d/requirements.txt` still failed while building `git+https://github.com/SunzeY/AlphaCLIP.git`. The import probe showed `spar3d ok` and `gradio_app ModuleNotFoundError: gradio_litmodel3d`, so treat SPAR3D as provider-install-blocked on that runtime until the AlphaCLIP/gradio dependency path is fixed.

Use the TripoSR API venv smoke config as the no-gate direct mesh fallback. Keep TripoSR isolated from the main backend environment: TripoSR pins `transformers==4.35.0`, while the Depth Anything V2 backend path needs the newer backend requirements. The July 9, 2026 G4 probe showed that installing TripoSR into the main environment broke the depth baselines; restoring backend requirements then let `run.py --help` pass after `numpy==2.0.2`, `onnxruntime`, and `rembg` imports were healthy, but TripoSR model loading failed under the newer `transformers` stack. The wrapper now has a lean `triposr-api` provider that calls `TSR.from_pretrained` directly, composites RGBA input onto TripoSR's gray background, exports the mesh, and stubs `rembg` so the direct path does not need `rembg`, `pymatting`, `onnxruntime`, `xatlas`, or `moderngl`.

```bash
git clone https://github.com/VAST-AI-Research/TripoSR /content/TripoSR
python -m pip install -q virtualenv
python -m virtualenv --system-site-packages /content/triposr-venv
/content/triposr-venv/bin/python -m pip install -U pip setuptools wheel
/content/triposr-venv/bin/python -m pip install numpy==2.0.2 omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 trimesh==4.0.5 huggingface-hub imageio git+https://github.com/tatsy/torchmcubes.git
export TRIPOSR_DIR=/content/TripoSR
cd /content/3dprintpic
python -m pip install -r backend/requirements-cuda.txt
python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_triposr_api_venv_prefill_direct_mesh_s40_n2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposr_api_venv_prefill_direct_mesh_smoke.json --start-index 40 --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,triposr_api_masked_direct_mesh,triposr_api_masked_repaired_direct_mesh,triposr_api_mirror_prefill_direct_mesh,triposr_api_biharmonic_prefill_direct_mesh --contact-sheet-max-samples 2 --resume --continue-on-error
```

Avoid `onnxruntime-gpu` on the current G4 image: the `1.27.0` wheel probed on July 9, 2026 attempted to load CUDA 13 runtime libraries on the CUDA 12.8 Colab image. The preferred `triposr-api` path skips ONNX background removal entirely; CPU `onnxruntime` is only needed if you deliberately fall back to TripoSR's official `run.py`. Also avoid installing TripoSR requirements into the main backend interpreter; TripoSR's old `trimesh==4.0.5` breaks procedural dataset rendering under NumPy 2.0, while the backend now requires `trimesh>=4.12.2`.

The provider wrapper now accepts `--mesh-repair basic|convex-hull|printable`. `printable` preserves the raw provider mesh at `--raw-output-mesh`, runs largest-component Trimesh cleanup first, and falls back to a convex hull only when the mesh still fails watertight/volume/single-body checks. It can then scale/compact the repaired mesh with `--mesh-target-max-dimension` and `--mesh-min-bbox-dimension` before STL export; `--mesh-target-faces` attempts optional Trimesh quadric decimation when the provider environment has the simplification backend installed. The smoke config includes `triposr_api_masked_repaired_direct_mesh` so the next G4 slice can directly score raw TripoSR against a printable repaired variant instead of treating the repair as a manual postprocess.

The repo also includes a one-command launcher for this exact smoke, which is easier to rerun in Colab than pasting a long notebook cell:

```bash
python -m backend.benchmark.run_triposr_repair_smoke --repo-dir /content/3dprintpic --output-dir /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_triposr_repaired_s0_n1_v2 --triposr-python /content/triposr-venv/bin/python --provider-dir /content/TripoSR --provider-device cuda --mesh-repair printable
```

It writes `triposr_raw_vs_repaired_config.json`, `experiment/aggregate_summary.csv`, `experiment/ranked_experiments.csv`, `experiment/artifact_contact_sheet.png`, per-method metrics, and `triposr_repaired_summary.json` under the output directory. Use it after syncing the fork branch and preparing the isolated TripoSR venv; the launcher intentionally does not install provider dependencies, so dependency setup remains explicit and auditable.

Colab G4 TripoSR API smoke result: the notebook ran commit `0294097` on July 9, 2026 with backend `numpy 2.0.2`, `trimesh 4.12.2`, `transformers 5.13.0`, and provider venv `torch 2.11.0+cu128`, `transformers 4.35.0`, and `torchmcubes`. The standalone `triposr-api` provider emitted OBJ and STL from an RGBA probe image. The one-sample procedural STL-quality benchmark also completed, but direct TripoSR was not printable enough to promote:

| method | n | stl-quality score vs masked | mesh surface Chamfer med | STL watertight med | STL volume med | STL manifold med | STL components med | STL faces med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mirror` | 1 | 1.2558 | 0.1541 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `biharmonic` | 1 | 0.1515 | 0.1806 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `masked` | 1 | 0.0000 | 0.1911 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `triposr_api_masked_direct_mesh` | 1 | -14.4972 | 0.1698 | 0.0 | 0.0 | 0.0 | 2 | 36950 |

Interpretation: the direct API integration works, but raw TripoSR output needs a mesh-repair/remesh postprocess before it can beat the existing depth-to-STL relief baseline on printable STL quality. The follow-up config now includes a repaired TripoSR candidate; rerun the same smoke with `masked`, `mirror`, `biharmonic`, raw TripoSR, and repaired TripoSR, then only promote it if the STL-quality score improves without hiding excessive surface error behind hull repair.

Colab G4 repaired TripoSR smoke result: the notebook ran commit `dfff7e7` on July 9, 2026 as `g4_triposr_repaired_s0_n1_v1`, synced the fork branch, reused the existing TripoSR venv, generated one procedural render, and completed the five-candidate STL-quality smoke in `36.84s`. Backend versions were `numpy 2.0.2`, `torch 2.11.0+cu128`, `transformers 5.13.0`, and `trimesh 4.12.2`; the provider venv probed `torch 2.11.0+cu128`, `transformers 4.35.0`, `trimesh 4.12.2`, `torchmcubes`, and `tsr`.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | STL watertight med | STL volume med | STL manifold med | STL components med | STL faces med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mirror` | 1 | 1.2772 | 0.1529 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `biharmonic` | 1 | 0.2522 | 0.1777 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `masked` | 1 | 0.0000 | 0.1924 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `triposr_api_masked_repaired_direct_mesh` | 1 | -4.2972 | 0.1843 | 1.0 | 1.0 | 0.0 | 1 | 147698 |
| `triposr_api_masked_direct_mesh` | 1 | -16.1202 | 0.1861 | 0.0 | 0.0 | 0.0 | 4 | 147468 |

Interpretation: `--mesh-repair printable` materially improved the direct TripoSR artifact: raw output had `4` components, `584` non-manifold edges, and failed watertight/volume checks; the repaired output became watertight, positive-volume, winding-consistent, and single-component with slightly better surface Chamfer. It still did not promote because `stl_is_manifold=False` from remaining degenerate faces and because face density stayed very high. The repair gate now checks non-manifold and degenerate face counts before accepting a basic repair, so the next G4 smoke should rerun this same slice and verify whether repaired TripoSR becomes fully manifold or falls back to a simpler printable hull/remesh.

Colab G4 repaired TripoSR rerun after the stricter printable gate: the notebook reran the launcher on commit `f232c45` as `g4_triposr_repaired_s0_n1_v3` and completed in `35.91s`. This validated the one-command launcher plus the repaired mesh gate on the G4 runtime.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | STL watertight med | STL volume med | STL manifold med | STL components med | STL faces med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mirror` | 1 | 1.2772 | 0.1529 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `biharmonic` | 1 | 0.2522 | 0.1777 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `masked` | 1 | 0.0000 | 0.1924 | 1.0 | 1.0 | 1.0 | 1 | 29580 |
| `triposr_api_masked_repaired_direct_mesh` | 1 | -0.8642 | 0.1518 | 1.0 | 1.0 | 1.0 | 1 | 2508 |
| `triposr_api_masked_direct_mesh` | 1 | -16.1202 | 0.1861 | 0.0 | 0.0 | 0.0 | 4 | 147468 |

Interpretation: the tightened gate fixed the remaining printability failure. Raw TripoSR still emitted `584` non-manifold edges, `3` degenerate faces, `4` components, and a `147468`-face STL. The repaired candidate became watertight, volume-valid, manifold, single-component, and zero-degenerate with only `2508` faces, while improving mesh-surface Chamfer to `0.1518`. It still ranks behind `mirror` because the STL-quality profile also penalizes hull-like simplification and rewards the depth-relief baseline's strong silhouette agreement on this one procedural sample. The next direct mesh test should use the repaired path by default and compare it on a larger held-out mesh slice rather than spending more time on raw TripoSR.

Learned inpainting smoke with `dreamshaper-inpaint`:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_4_dreamshaper_s20_noempty --methods mirror,biharmonic,dreamshaper-inpaint --limit 4 --skip-depth --device auto --steps 20 --guidance 7.5 --seed 1234 --inpaint-max-dimension 256 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_4_dreamshaper_s20_noempty
```

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 15.24 | 0.0581 | 15.42 | 0.771 | 0.0236 | 0.1226 | 13.88 |
| biharmonic | 4 | 10.59 | 0.3842 | 6.78 | 0.545 | 0.0302 | 0.0989 | 17.57 |
| dreamshaper-inpaint | 4 | 5.42 | 0.1258 | 12.51 | 0.765 | 0.0389 | 0.3208 | 7.39 |

The first DreamShaper attempt at 8 steps often left the masked side blank (`object MAE` median 0.5343, rank 4.71). The stronger no-empty prompt at 20 steps fixed the chair sample visually and improved the four-sample median, but it still does not beat mirror or biharmonic on object pixels.

Local learned sweep with cached DreamShaper and AMUSED:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_4_local_learned --config backend/benchmark/experiment_configs/local_learned_smoke.json --limit 4 --skip-depth --device auto --resume --continue-on-error
```

This sweep writes `aggregate_summary.csv`, `ranked_experiments.csv`, `resolved_experiments.json`, and `experiment_report.md` in the experiment directory.

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med | object SSIM med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.10 | 0.0581 | 15.42 | 0.771 | 0.0236 | 0.1226 | 13.88 | 0.345 |
| biharmonic | 4 | 11.54 | 0.3842 | 6.78 | 0.545 | 0.0302 | 0.0989 | 17.57 | 0.427 |
| dreamshaper_s20_noempty_s256 | 4 | 10.70 | 0.1258 | 12.51 | 0.765 | 0.0389 | 0.3208 | 7.39 | 0.357 |
| amused_s12_noempty_native512 | 4 | 3.18 | 0.2310 | 10.23 | 0.599 | 0.1182 | 0.4501 | 6.75 | 0.255 |

Interpretation: DreamShaper is the best learned local provider so far, but it still loses the weighted objective to the geometric and classical baselines on this tiny ModelNet10 smoke. AMUSED is cached and fast enough to include as a contrast baseline, but its current outputs add seam/background artifacts and should not be the default completion path. The `native512` label means the AMUSED provider internally runs at its native 512-pixel minimum even when the benchmark input cap is 256.

Balanced held-out-10 depth/STL AMUSED probe:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_amused --config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_amused_depth_stl.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror --candidate-method amused_base_s12_native512 --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,amused_base_s12_native512 --resume --continue-on-error
```

This run completed `10/10` samples for every method and writes `selection_decision.json` / `.md` plus a contact sheet. AMUSED improves clearly over the `masked` lower bound and is stronger than the prompt-matched DreamShaper LoRA on the same held-out slice, but it still does not beat the incumbent mirror baseline.

| method | n | baseline-delta score | object depth MAE med | object surface Chamfer med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 9.9605 | 0.3228 | 0.1491 | 1.0 | 1.0 |
| biharmonic | 10 | 8.4180 | 0.2994 | 0.1815 | 1.0 | 1.0 |
| amused_base_s12_native512 | 10 | 6.9978 | 0.3137 | 0.1458 | 1.0 | 1.0 |
| dreamshaper_base_s20_s256 | 10 | 5.8500 | 0.4139 | 0.1780 | 1.0 | 1.0 |
| masked | 10 | 0.0000 | 0.4178 | 0.2333 | 1.0 | 1.0 |

Promotion decision: `hold`. AMUSED passes success, held-out coverage, improvement over `masked`, STL watertightness, STL positive-volume, and split-audit checks, but fails against the current mirror method: score margin `-2.9628`, paired win rate `1/10`, and current-baseline CI95 low `-4.5177`. Cross-run comparison ranks AMUSED above the prompt-matched LoRA (`6.9978` vs `5.171`), so the next learned branch should not spend more cycles on generic DreamShaper prompt/scale sweeps. The next measurable experiment should either add a stronger mask-native provider such as PowerPaint or SD3-ControlNet inpainting, or train with a depth/silhouette-aware objective that directly targets object-surface Chamfer and hidden-object depth error.

SDXL follow-up config:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_amused --config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_sdxl_depth_stl.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror --candidate-method sdxl_base_s18_s384 --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,sdxl_base_s18_s384 --resume --continue-on-error --require-modern-cache
```

This uses the existing held-out-10 AMUSED output directory intentionally: with `--resume`, cached `masked`, `mirror`, `biharmonic`, DreamShaper, and AMUSED rows are reused, while `sdxl_base_s18_s384` is the only new expensive branch. `--require-modern-cache` writes `modern_cache_preflight.json` and fails before inference if a planned modern provider cache is incomplete.

Measured SDXL result on the same held-out-10 depth/STL slice:

| method | n | baseline-delta score | paired wins vs masked | paired wins vs mirror | object MAE med | object surface Chamfer med | STL watertight med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 9.9605 |  |  | 0.1633 | 0.1491 | 1.0 |
| biharmonic | 10 | 8.4180 |  |  | 0.1266 | 0.1815 | 1.0 |
| amused_base_s12_native512 | 10 | 6.9978 | 8/10 | 1/10 | 0.2567 | 0.1458 | 1.0 |
| dreamshaper_base_s20_s256 | 10 | 5.8500 |  |  | 0.4129 | 0.1780 | 1.0 |
| sdxl_base_s18_s384 | 10 | 3.7483 | 7/10 | 0/10 | 0.5544 | 0.1683 | 1.0 |
| masked | 10 | 0.0000 |  |  | 0.6384 | 0.2333 | 1.0 |

Promotion decision: `hold`. SDXL passes cache, success-rate, split-audit, STL watertightness, and STL positive-volume checks, but fails the baseline paired-win gate (`7/10` vs required `8/10`) and all incumbent checks (`0/10` paired wins versus mirror, score margin `-6.2122`, current-baseline CI95 low `-6.9717`). Visual QA shows SDXL often inserts high-contrast colored artifacts or scene fragments into the missing half, so this SDXL baseline should remain a contrast provider rather than the next tuning target. The best measured untrained learned branch remains AMUSED, but it also stays behind mirror.

Second held-out-10 slice and combined modern evidence:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_start50 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_sdxl_depth_stl.json --start-index 50 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror --candidate-method amused_base_s12_native512 --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,amused_base_s12_native512,sdxl_base_s18_s384 --resume --continue-on-error --require-modern-cache
.\backend\.venv\Scripts\python -m backend.benchmark.combine_optimize_runs --run start40=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_amused --run start50=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_start50 --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_modern_start40_start50_combined_default --score-mode baseline-delta --baseline-method masked --score-profile default --current-method mirror --select-candidate
.\backend\.venv\Scripts\python -m backend.benchmark.combine_optimize_runs --run start40=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_amused --run start50=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_start50 --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_modern_start40_start50_combined_object_surface --score-mode baseline-delta --baseline-method masked --score-profile object-surface --current-method mirror --select-candidate
```

The second slice also completed `10/10` samples for every method. Its default baseline-delta ranking was mirror `11.8152`, biharmonic `9.4688`, DreamShaper `8.4045`, AMUSED `6.7401`, SDXL `4.5083`, and masked `0.0000`. AMUSED again held against mirror (`0/10` paired wins versus current), and SDXL held with a negative incumbent margin.

Combining the start-40 and start-50 slices into a single 20-sample evidence set gives the current best model-selection signal:

| score profile | method | n | baseline-delta score | object surface Chamfer med | object surface RMSE med | object surface Hausdorff95 med |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| default | mirror | 20 | 11.1347 | 0.1293 | 0.1602 | 0.3829 |
| default | biharmonic | 20 | 9.1354 | 0.1825 | 0.2190 | 0.4263 |
| default | dreamshaper_base_s20_s256 | 20 | 7.4938 | 0.1647 | 0.2048 | 0.3970 |
| default | amused_base_s12_native512 | 20 | 7.3424 | 0.1651 | 0.2147 | 0.4392 |
| default | sdxl_base_s18_s384 | 20 | 4.2560 | 0.1683 | 0.1966 | 0.4329 |
| object-surface | mirror | 20 | 2.4490 | 0.1293 | 0.1602 | 0.3829 |
| object-surface | sdxl_base_s18_s384 | 20 | 2.0278 | 0.1683 | 0.1966 | 0.4329 |
| object-surface | dreamshaper_base_s20_s256 | 20 | 1.9697 | 0.1647 | 0.2048 | 0.3970 |
| object-surface | amused_base_s12_native512 | 20 | 1.8530 | 0.1651 | 0.2147 | 0.4392 |

Interpretation: the standard objective keeps `mirror` as the default with `20/20` paired wins over the masked lower bound and CI95 low `8.2126`. The object-surface profile is useful because it changes the learned-method ordering: SDXL becomes the best learned provider for printable-surface metrics, while DreamShaper briefly edges mirror on one 10-sample aggregate slice. Neither finding passes the paired incumbent gate on the combined set, so the correct action is to keep mirror as the product default and use the object-surface profile to guide the next training/provider search rather than adding a runtime selector.

Depth-to-STL DreamShaper smoke:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_2_dreamshaper_depth_stl_s20_noempty --methods mirror,biharmonic,dreamshaper-inpaint --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --steps 20 --guidance 7.5 --seed 1234 --inpaint-max-dimension 256 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_2_dreamshaper_depth_stl_s20_noempty
```

| method | n | rank | depth MAE med | object depth MAE med | silhouette IoU med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 2 | 25.07 | 0.131 | 0.268 | 0.488 | 1.0 | 1.0 |
| biharmonic | 2 | 17.62 | 0.468 | 0.220 | 0.251 | 1.0 | 1.0 |
| dreamshaper-inpaint | 2 | 9.43 | 0.287 | 0.509 | 0.245 | 1.0 | 1.0 |

The mesh-folder path was smoke-tested locally using generated `.ply` assets:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/mesh_dir_smoke --source mesh-dir --asset-root backend/output/completion-benchmark/rendered_procedural_v2 --asset-glob *.ply --count 3 --size 128
```

## Methods

- `masked`: no-completion lower bound; leaves the half-blank benchmark input unchanged.
- `mirror`: geometric symmetry baseline.
- `mirror-seam-repair`: mirror baseline plus a narrow classical biharmonic repair band on the generated side of the seam; cheap enough for the app and ranked as a deterministic hybrid candidate.
- `biharmonic`: classical non-generative inpainting baseline.
- `sdxl-inpaint`: public SDXL inpainting checkpoint through Diffusers; practical middle tier for a 12 GB GPU at 384-512 px.
- `dreamshaper-inpaint`: public SD1.5-style DreamShaper inpainting checkpoint through Diffusers; smaller fp16 footprint and the first learned baseline to try locally.
- `amused-inpaint`: public AMUSED masked-token inpainting checkpoint through Diffusers; small/fast learned baseline with a different failure profile from diffusion.
- `flux-fill`: FLUX.1 Fill through Diffusers.
- `qwen-image-inpaint`: Qwen Image Inpaint through Diffusers.
- `qwen-image-edit`: official Qwen Image Edit through Diffusers, prompted to fill the white half.

The current local import-tested stack is:

- `torch==2.11.0+cu128`
- `diffusers==0.39.0`
- `transformers==5.13.0`
- `huggingface-hub==1.22.0`
- `accelerate==1.14.0`
- `peft==0.19.1`

FLUX/Qwen inference is intentionally benchmark-driven: run them on 10-20 rendered samples and accept them only if they beat mirror by the rank objective.

Provider preflight:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.preflight_modern_providers --output backend/output/completion-benchmark/modern_provider_preflight_live_readiness.jsonl
```

Current preflight results now separate local pipeline importability from model-card access and advertised pipeline class. `pipeline-override` means the Diffusers class imports locally, but the model card or `model_index.json` advertises a different pipeline; keep those candidates benchmark-gated and do not promote them from metadata alone.

| provider | model | readiness | gated | license | approx size |
| --- | --- | --- | --- | --- | ---: |
| `sdxl-inpaint` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | `ready` | false | openrail++ | 19.39 GB total; fp16 cache plan is 6.47 GB |
| `dreamshaper-inpaint` | `Lykon/dreamshaper-8-inpainting` | `ready` | false | creativeml-openrail-m | 7.66 GB total; fp16 cache plan is 1.99 GB and is cached locally |
| `amused-inpaint` | `amused/amused-512` | `pipeline-override` | false | openrail++ | 4.93 GB total; fp16 cache plan is 1.64 GB and is cached locally |
| `flux-fill` | `black-forest-labs/FLUX.1-Fill-dev` | `auth-required` | `auto`; model index gated without auth | other | 54.07 GB |
| `qwen-image-inpaint` | `Qwen/Qwen-Image-Edit` | `pipeline-override` | false | Apache-2.0 | 53.76 GB |
| `qwen-image-edit` | `Qwen/Qwen-Image-Edit` | `ready` | false | Apache-2.0 | 53.76 GB |

Large-provider runtime notes:

- The Qwen Image Edit model card uses BF16 for Diffusers CUDA inference (`https://huggingface.co/Qwen/Qwen-Image-Edit`), so the harness loads `qwen-image-edit` and `qwen-image-inpaint` as `torch.bfloat16` on CUDA.
- The FLUX.1 Fill model card uses BF16, `guidance_scale=30`, `num_inference_steps=50`, and `max_sequence_length=512` (`https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev`). The harness now loads `flux-fill` as BF16 on CUDA and passes `max_sequence_length=512`.
- For these large providers, the loader enables attention slicing, VAE slicing, and VAE tiling when the pipeline exposes those hooks. The Colab G4 runtime has enough VRAM for the first scored pass, but the memory savers make the same config more resilient across runtime images.

Because these modern providers are large, cache weights separately before a scored benchmark. For SDXL, the provider cache utility plans a smaller fp16 download:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider sdxl-inpaint --output backend/output/completion-benchmark/sdxl_cache_plan.json
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider sdxl-inpaint --download
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider dreamshaper-inpaint --output backend/output/completion-benchmark/dreamshaper_cache_plan.json
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider dreamshaper-inpaint --download
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider amused-inpaint --output backend/output/completion-benchmark/amused_cache_plan.json
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider amused-inpaint --download
```

Use `--model-name` with `cache_provider` when an experiment overrides the provider's default Hugging Face model id. For scored sweeps, add `--require-modern-cache` to `optimize_completion`; it checks every modern provider in the config, records selected/cached/missing files in `modern_cache_preflight.json`, and stops before any sample work when the cache is incomplete. This prevents partial SDXL-style downloads from producing a misleading `n=0` benchmark. The local SDXL fp16 cache is now complete (`18 / 18` selected files, `6.465 GB`, `0` missing); the final UNet file was recovered with a resumable curl download and verified against SHA256 `6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f`.

For the next G4 pass, keep Qwen Image Edit and FLUX Fill in separate configs so a gated FLUX auth issue cannot block the ready Qwen run:

```bash
python -m backend.benchmark.cache_provider qwen-image-edit --full --download --download-mode snapshot --max-workers 8 --output backend/output/completion-benchmark/qwen_edit_cache_plan.json
python -m backend.benchmark.cache_provider flux-fill --full --download --download-mode snapshot --max-workers 8 --output backend/output/completion-benchmark/flux_fill_cache_plan.json
```

Use `backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json` first. Use `backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_flux_fill_g4_depth_stl.json` only after Hugging Face login and FLUX access acceptance; the preflight currently reports `auth-required` for that model without auth.

The first local SDXL smoke reached the model-fetch stage and stalled with only about 397 MB cached after a bounded 30-minute attempt. The blocked state was captured as a benchmark failure report:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_1_sdxl_offline_partial_failure --methods sdxl-inpaint --limit 1 --skip-depth --device auto --steps 2 --guidance 7.5 --seed 1234 --inpaint-max-dimension 256 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_1_sdxl_offline_partial_failure
```

Once the cache step completes, run modern providers through `optimize_completion` with `--resume --continue-on-error` and an initial `inpaint_max_dimension` of 384.

DreamShaper cache status after the local run:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider dreamshaper-inpaint --output backend/output/completion-benchmark/dreamshaper_cache_plan.json
```

The current plan reports `14 / 14` selected fp16 files cached, `1.99 GB` cached, and `0` missing files.

AMUSED cache status after the local run:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.cache_provider amused-inpaint --output backend/output/completion-benchmark/amused_cache_plan.json
```

The current plan reports `12 / 12` selected fp16 files cached, `1.64 GB` cached, and `0` missing files. Diffusers warns that AMUSED is deprecated after `0.33.1`, so keep it as a small comparison baseline rather than the long-term flagship provider.

Lighter modern candidates checked for future integration:

| candidate | model(s) | preflight size/access | integration note |
| --- | --- | --- | --- |
| PowerPaint v2.1 | `JunhaoZhuang/PowerPaint-v2-1` | ~8.87 GB, Apache-2.0, no top-level `model_index.json` | Mask-native and practical, but needs PowerPaint-specific glue. |
| SD3 ControlNet Inpainting | `alimama-creative/SD3-Controlnet-Inpainting` + `stabilityai/stable-diffusion-3-medium-diffusers` | ControlNet ~3.92 GB open; SD3 Medium base ~28.88 GB gated auto-access | Diffusers exposes `StableDiffusion3ControlNetInpaintingPipeline`; best clean pipeline after access. |
| BrushNet / BrushNetX | `TencentARC/BrushEdit` or `TencentARC/BrushNet` assets | BrushEdit snapshot ~61.9 GB; selective BrushNet assets should be smaller | Modern mask-native method, but custom integration. |
| ControlNet++ SDXL | `xinsir/controlnet-union-sdxl-1.0` + SDXL base | ControlNet repo ~4.72 GB, Apache-2.0; needs SDXL base | Possible SDXL path, less direct than SD3 ControlNet. |

## Metrics

Per sample/method:

- `masked_psnr`, `masked_ssim`, `masked_mae`
- `object_masked_psnr`, `object_masked_ssim`, `object_masked_mae`
- `visible_mae_pre_preserve`, `visible_mae`
- `seam_mae_pre_preserve`, `seam_mae`, `object_seam_mae`
- optional `depth_mae`, `depth_rmse`, `depth_corr`
- optional `object_depth_mae`, `object_depth_rmse`, `object_depth_corr`
- optional `surface_chamfer_l1`, `surface_chamfer_rmse`, `surface_rmse`, `surface_hausdorff95`, `surface_point_count`
- optional `object_surface_chamfer_l1`, `object_surface_chamfer_rmse`, `object_surface_rmse`, `object_surface_hausdorff95`, `object_surface_point_count`
- optional `silhouette_iou_masked`
- optional `stl_exists`, `stl_is_watertight`, `stl_is_volume`, `stl_is_manifold`, `stl_nonmanifold_edge_count`, `stl_nonmanifold_edge_count_log1p`, `stl_degenerate_face_count`, `stl_degenerate_face_ratio`, `stl_winding_consistent`, `stl_positive_volume`, `stl_single_component`, `stl_component_count`, `stl_component_excess_log1p`, `stl_bbox_has_volume`, `stl_bbox_aspect_ratio`, `stl_faces_per_bbox_volume_log1p`, `stl_faces_per_normalized_bbox_volume_log1p`, `stl_faces`, `stl_z_range`
- optional `mesh_surface_chamfer_l1`, `mesh_surface_chamfer_rmse`, `mesh_surface_hausdorff95`, `mesh_surface_point_count`

Depth metrics align predicted relative depth to ground truth using scale + shift fit on the visible half. Object-depth metrics restrict fit/evaluation to saved silhouette pixels, which prevents background from dominating the score.
Surface metrics convert aligned predicted and ground-truth depth into normalized camera-view `(x, y, z)` point clouds over the hidden half. `surface_chamfer_l1` is symmetric nearest-neighbor distance, `surface_chamfer_rmse` is the bidirectional nearest-neighbor RMS distance, `surface_rmse` is same-pixel depth-surface RMSE, and `surface_hausdorff95` is a robust 95th-percentile Hausdorff distance. The `object_surface_*` variants restrict the comparison to hidden pixels inside the saved object silhouette, so the rank objective can penalize a completion that looks acceptable in 2D but produces the wrong 3D relief surface.
Mesh-surface metrics compare the emitted STL to the original source mesh when the manifest contains a mesh path. If the manifest includes `camera`, the reference mesh is first transformed into the rendered camera frame; both meshes are then normalized before seeded area-weighted surface sampling. This makes the metric a shape accuracy signal for rendered datasets rather than an absolute print-size check. `stl_bbox_aspect_ratio` remains an STL-quality shape signal, while `stl_faces_per_normalized_bbox_volume_log1p` measures mesh complexity after normalizing bbox volume by the largest bbox side cubed so direct meshes are not rewarded or punished merely for coordinate scale. The legacy `stl_faces_per_bbox_volume_log1p` field is still emitted for backwards compatibility, but ranking and promotion use scale-free complexity when available. The `stl-quality` profile uses these `mesh_surface_*` fields when available and pairs them with STL validity/printability diagnostics.

Cached depth/STL runs from before the surface metrics existed can be upgraded without rerunning depth inference:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.backfill_surface_metrics backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --dry-run
.\backend\.venv\Scripts\python -m backend.benchmark.backfill_surface_metrics backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2
```

The backfill command reads `per_sample_metrics.csv`, resolves `split_audit.json` -> `manifest.jsonl` for `gt_depth` / `mask` / silhouette references, infers cached `output_depth_data.npy` beside completed/STL artifacts when the older CSV lacks an explicit `depth_data` column, updates method summaries, and rewrites `aggregate_summary.csv` for sweep directories.

## Regression Checks

Run the focused backend regression suite before trusting new benchmark/export changes:

```bash
.\backend\.venv\Scripts\python -m unittest discover -s backend/tests -v
```

Current coverage includes minimal height-field STL export, positive Trimesh signed-volume orientation, no-valid-cell failure behavior, mesh-to-mesh surface distance, camera-framed mesh-surface references, transform-aware GLB/scene loading, different-triangulation tolerance, direct source-mesh STL export without depth, external command image-to-mesh export, and `optimize_completion` per-sample CSV experiment labeling/idempotency. This suite is intentionally small; use it as a fast guardrail before heavier image/depth/STL benchmarks.

## Optimization Loop

Rank a run:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --baseline-method masked
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv --score-mode baseline-delta --baseline-method masked --output backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/ranked_methods_masked_delta.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --baseline-method masked --score-mode baseline-delta --output backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/benchmark_report_masked_delta.md
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv --score-mode baseline-delta --baseline-method masked --score-profile object-surface
```

The default rank objective rewards masked PSNR/SSIM, object-depth correlation, silhouette IoU, STL existence, STL watertightness, and positive STL volume; it penalizes masked MAE, seam MAE, visible drift before preservation, object-depth error, and depth-derived 3D surface distance. Use `--score-profile object-surface` when the question is specifically printable object relief geometry: it ignores RGB-only completion metrics and weights object depth, object surface Chamfer/RMSE/Hausdorff95, silhouette IoU, and STL validity. Use `--score-profile stl-quality` when comparing final STL-producing methods, including direct image-to-mesh candidates: it weights `mesh_surface_*` error against the source mesh when available and penalizes non-watertight, non-volume, multi-body, degenerate, or overly dense STL outputs. `rank_methods`, `report_run`, `optimize_completion`, `select_completion_candidate`, `compare_optimize_runs`, and `analyze_experiment_slices` all accept the same profile name, with explicit `--weight metric=value` overrides applied on top. Summaries also include `attempted_n`, `error_count`, `logged_failure_count`, and `success_rate` so rankings produced with `--continue-on-error` reveal whether methods were scored on equal coverage. `report_run` writes a Markdown audit trail with rankings, coverage, metric medians, selected baseline deltas, paired baseline wins, paired objective confidence, failures, and example artifacts. The default `--score-mode normalized` uses min/max normalization within the candidate set, which is useful for ordering one run but makes numeric scores move when candidates are added or removed. Use `--score-mode baseline-delta --baseline-method masked` for a candidate-set-stable score: the baseline gets `0`, positive scores are weighted sign-normalized metric improvements, and negative scores regress the objective. The baseline-delta section still reports absolute movement in selected diagnostic medians. `Delta` is raw `method - baseline`; `Improvement` is sign-normalized so positive means better, even when the raw metric is lower-is-better. The paired-wins section compares only shared finite `sample_id` rows, reports `Baseline n`, `Method n`, `Common n`, finite `Paired n`, and skipped non-finite pairs; `Win Rate` is strict wins divided by finite paired n, with ties excluded from wins. The paired-objective confidence section applies the same metric weights to raw per-sample improvements over the baseline and bootstraps the mean improvement; this is intentionally not the same as candidate-normalized aggregate rank. Treat a CI entirely above `0` as a stronger promotion signal than an aggregate rank alone.

After backfilling the held-out-10 depth/STL top sweep, the surface-aware baseline-delta ranking is mirror `9.96`, biharmonic `8.42`, LoRA scale `0.5` `6.55`, base DreamShaper `5.85`, LoRA scale `0.75` `5.59`, and masked `0.00`. The LoRA scale `0.5` candidate has better object-surface Chamfer than `masked` (`0.167` vs `0.233`) but still trails mirror (`0.149`) and fails the incumbent promotion gate with only `2/10` paired wins versus mirror and a paired-current CI below zero. A follow-up AMUSED held-out-10 depth/STL run raises the best learned baseline-delta score to `6.9978` and object-surface Chamfer to `0.1458`, but still fails the current-method gate with only `1/10` paired wins versus mirror. The follow-up SDXL held-out-10 depth/STL run succeeds after cache completion and progress-bar suppression, but scores only `3.7483` and wins `0/10` against mirror.

Render a visual artifact sheet when a metric result needs inspection:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --baseline-method masked --contact-sheet --contact-sheet-max-samples 6
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_depth_stl_top_sweep.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error --score-mode baseline-delta --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 --contact-sheet-max-samples 4
.\backend\.venv\Scripts\python -m backend.benchmark.make_artifact_contact_sheet backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --methods masked,mirror,biharmonic,dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 --max-samples 4 --output backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/artifact_contact_sheet_top4.png
```

`report_run --contact-sheet` and `optimize_completion --contact-sheet` call the same `make_artifact_contact_sheet` helper and link the PNG from the Markdown report. The standalone helper still accepts either a single `run_completion_benchmark` output directory or an `optimize_completion` sweep directory, and method filters can match the displayed method, experiment name, or `base_method`. It reads `per_sample_metrics.csv`, uses explicit `full_image`, `masked_image`, `mask`, `depth_data`, `depth_preview`, and `stl_model` columns from new runs, falls back to `split_audit.json` -> `manifest.jsonl` for older runs or stale row paths, infers `output_depth_preview.png` or `output_depth_data.npy` beside historical completed image/STL artifacts, and writes a PNG with raw/completed/depth/object-depth-error/STL thumbnails plus the key RGB, depth, surface, and STL metrics. The object-depth-error thumbnail aligns predicted depth to the reference and colors hidden-object error; it uses per-row scaling, so compare the location and pattern of errors more than absolute color intensity across rows. The STL thumbnail is a small software-rendered isometric mesh preview. Together they make inverted, collapsed, or oddly shaped relief exports easier to catch by eye. Use this as a sanity check after ranking: the promotion gate stays metric-driven, but the sheet makes blank completions, seam artifacts, depth failures, and bad printable geometry obvious before spending more GPU time.

Run an experiment sweep:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/rendered_procedural_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/rendered_smoke --limit 2 --skip-depth
```

Each sweep now writes `resolved_experiments.json` and `experiment_report.md` next to `aggregate_summary.csv` and `ranked_experiments.csv`, so runs are self-documenting. For LoRA candidates, the resolved config and report expand `training_report.json` into a Training Recipes section with loss recipe, train rows/assets, steps, final/best loss, prompt family, metadata hash, and adapter hash. Add `--emit-stl` without `--skip-depth` to rank the full completion-to-depth-to-STL path from the same experiment config. Direct mesh experiments such as `source-mesh-oracle` and `external-image-to-mesh` may set `skip_depth` with `emit_stl` because they produce the STL directly. Add `--direct-mesh-command` when a config includes `external-image-to-mesh`; quote placeholders inside the command template if the paths may contain spaces. Direct image-to-mesh configs also write `image_to_mesh_provider_preflight.json`; add `--require-image-to-mesh-providers` to stop before sample work when provider setup is incomplete. Add `--score-mode baseline-delta --baseline-method masked` when a sweep includes `masked` and you want stable optimization scores across experiments. Add `--select-candidate --current-method <method>` to make the sweep also write `selection_decision.json` and `selection_decision.md`.

Turn a sweep into an explicit promotion decision:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_depth_stl_top_sweep.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror
.\backend\.venv\Scripts\python -m backend.benchmark.select_completion_candidate backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --baseline-method masked --current-method mirror
.\backend\.venv\Scripts\python -m backend.benchmark.select_completion_candidate backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --baseline-method masked --current-method mirror --candidate-method dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 --output-json backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/selection_decision_lora_scale050_vs_mirror.json --output-md backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/selection_decision_lora_scale050_vs_mirror.md
```

The selector writes strict `selection_decision.json` and `selection_decision.md`, either from `optimize_completion --select-candidate` or from the standalone `select_completion_candidate` command. A candidate must pass success-rate, paired sample count, paired win-rate, positive bootstrap CI, STL printability gates, train/eval overlap, and score-margin gates before it is promoted. The STL hard gates require watertightness, volume-valid geometry, manifoldness, consistent winding, positive volume, a single connected component, nonzero bbox volume, no non-manifold edges, no degenerate faces, no extra components, bounded bbox aspect ratio, and bounded face density by default. LoRA candidates must also carry a `training_report.json`; when split-audit provenance includes a prompt-family mismatch, the selector fails closed instead of promoting an adapter that was trained and evaluated under different prompt assumptions. When `--current-method` is supplied and the candidate is not already current, the selector also recomputes paired objective confidence with the current method as the baseline; this prevents a method that only beats the lower-bound `masked` baseline from replacing a stronger incumbent. If `--current-method` is supplied but missing from the summary, the selector fails closed with `hold`; if the top candidate is already the current method and all gates pass, the decision is `keep_current`. `split_audit.json` is required by default because promotion should come from a held-out check; pass `--allow-missing-split-audit` only for quick smoke runs that are not making a default-selection claim.

Slice and selector analysis:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.analyze_experiment_slices backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout20_train40_lora_s200_low_scale_sweep --selector-field asset_category --selector-train-filter asset_source_split=train --selector-eval-filter asset_source_split=test
```

This reads the existing `per_sample_metrics.csv` files from an `optimize_completion` directory and writes `slice_analysis/coverage.csv`, `slice_summary.csv`, `slice_winners.csv`, `oracle_selection.csv`, `oracle_ranked_summary.csv`, `selector_choices.csv`, `selector_eval_summary.csv`, and `slice_analysis_report.md`. By default it restricts comparisons to sample IDs that succeeded for every method; pass `--allow-unequal-coverage` only for diagnostic failure analysis. Slice winners are descriptive. The oracle is an upper bound that chooses the best method per sample using ground-truth metrics. The validation-style selector chooses methods on one filtered subset and evaluates on another, which is safer but still only as strong as the chosen split and sample count.

Example JSON config for modern methods:

```json
[
  {
    "name": "mirror",
    "method": "mirror"
  },
  {
    "name": "biharmonic",
    "method": "biharmonic"
  },
  {
    "name": "sdxl_inpaint_seed1234_steps18_s384",
    "method": "sdxl-inpaint",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 18,
    "guidance": 7.5,
    "seed": 1234,
    "inpaint_max_dimension": 384
  },
  {
    "name": "dreamshaper_inpaint_seed1234_steps20_s384",
    "method": "dreamshaper-inpaint",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 20,
    "guidance": 7.5,
    "seed": 1234,
    "inpaint_max_dimension": 384
  },
  {
    "name": "amused_inpaint_seed1234_steps12_s384",
    "method": "amused-inpaint",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 12,
    "guidance": 10.0,
    "seed": 1234,
    "inpaint_max_dimension": 384
  },
  {
    "name": "flux_fill_seed1234_steps16_s384",
    "method": "flux-fill",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 16,
    "guidance": 30.0,
    "seed": 1234,
    "inpaint_max_dimension": 384
  },
  {
    "name": "qwen_inpaint_seed1234_steps16_s384",
    "method": "qwen-image-inpaint",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 16,
    "guidance": 4.0,
    "seed": 1234,
    "inpaint_max_dimension": 384
  },
  {
    "name": "qwen_edit_seed1234_steps16_s384",
    "method": "qwen-image-edit",
    "prompt": "Complete the missing half of the same {category} with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing {category}, not an empty background.",
    "steps": 16,
    "guidance": 4.0,
    "seed": 1234,
    "inpaint_max_dimension": 384
  }
]
```

Run it:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_modern_s384 --config backend/benchmark/experiment_configs/modern_inpainting.example.json --limit 12 --skip-depth --resume --continue-on-error
```

For modern diffusion methods:

1. Generate 10-20 rendered mesh samples.
2. Run `masked,mirror,mirror-seam-repair,biharmonic,amused-inpaint,dreamshaper-inpaint,sdxl-inpaint,flux-fill,qwen-image-inpaint,qwen-image-edit` with fixed seeds.
3. Choose the best provider by `rank_methods`, not visual preference alone.
4. Sweep prompt, guidance, steps, seed, and mask feathering.
5. Export pairs for adapter/LoRA training if prompt/model selection cannot beat mirror.
6. Re-run the same held-out benchmark and require improvement.

Training pair export:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/rendered_procedural_v2/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/rendered_12 --limit 12
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_20 --limit 20
```

Each pair contains a masked input, mask, full target, prompt metadata, and `object_mask` when the source manifest has a silhouette. Mesh renders also carry `asset_category`, stable relative `asset_key`, and `asset_source_split` when the source folder exposes `train`/`test`/`val`. Pair export writes `pair_export_report.json` with the source manifest hash, metadata hash, prompt mode, manifest indices, object-mask coverage, category/source-split counts, and asset keys; `train_inpainting_lora` consumes that report and copies the relevant hashes into its own training provenance. The default prompt is now category-templated from `asset_id` metadata, for example "same chair" or "same bathtub", so training prompts can match the evaluation prompt family. Use `--literal-prompt` to write `--prompt` exactly as provided. The intended training setup is adapter/LoRA-style inpainting fine-tuning against the fixed benchmark objective, not training a base diffusion model from scratch.

Metric-trained LoRA loop:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_16_train --start-index 0 --limit 16
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_16_train/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_dreamshaper --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 200 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_lora_eval --config backend/benchmark/experiment_configs/trained_lora.example.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

Use `--dry-run` on `train_inpainting_lora` to validate exported pairs without loading model weights:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_16_train/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_dreamshaper_dryrun --resolution 256 --limit 4 --train-batch-size 2 --dry-run
```

For metric-aligned training, `train_inpainting_lora` also accepts:

- `--mask-loss-weight`: extra latent MSE weight inside the hidden/inpaint mask.
- `--seam-loss-weight`: extra latent MSE weight on mask-boundary latents.
- `--object-loss-weight`: extra latent MSE weight on pixels that are both hidden and inside the exported silhouette.
- `--sample-weight-field`: per-row metadata field, default `sample_weight`, used to weight whole training examples.

`--object-loss-weight` is loss-only. It does not add a silhouette channel at inference, so the 9-channel inpainting UNet contract remains unchanged. Dry-run reports split-wide object coverage (`object_inpaint_fraction_mean`, `object_inpaint_fraction_min`, and `object_inpaint_nonzero_rows`) so zero-object rows do not silently dilute a run.

Metric-derived training weights:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_60_balanced_train10_mirror_depth_stl_surface_calibration --methods mirror --start-index 0 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.weight_training_pairs --metadata backend/output/completion-benchmark/training_pairs/modelnet10_60_balanced_train40_s0_generic/metadata.jsonl --benchmark-dir backend/output/completion-benchmark/runs/modelnet10_60_balanced_train10_mirror_depth_stl_surface_calibration --method mirror --score-profile object-surface --base-weight 1 --scale 2 --max-weight 4 --output backend/output/completion-benchmark/training_pairs/modelnet10_60_balanced_train40_s0_generic/metadata_weighted_mirror_object_surface_train10.jsonl
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_60_balanced_train40_s0_generic/metadata_weighted_mirror_object_surface_train10.jsonl --output-dir backend/output/completion-benchmark/lora_dry_runs/modelnet10_train40_weighted_mirror_object_surface_train10 --resolution 256 --train-batch-size 4 --mask-loss-weight 3 --seam-loss-weight 4 --object-loss-weight 1 --dry-run
```

`weight_training_pairs` reads benchmark per-sample metrics for a selected method, converts the selected score profile into normalized hard-sample scores, and writes a weighted metadata JSONL. Missing benchmark rows keep `sample_weight=1.0`, so a small calibration benchmark can safely weight a larger training-pair file. The first train-split calibration above matched `10/40` rows, produced weights from `1.0` to `2.8922` with mean `1.1831`, and dry-run validated all 40 rows with `sample_weight_non_default_rows=10`. A tiny 20-step local smoke wrote `backend/output/completion-benchmark/lora/modelnet10_train40_weighted_surface_s20/training_report.json`, final loss `0.03185`, best loss `0.00276`, and adapter SHA256 `b20234189c59f94c9e7d139f437b0b6b259acd417bfb94f6a6f73d7cf38a6b10`. This is a training-data optimization method, not a promotion claim; the resulting adapter still needs a held-out `optimize_completion` run and selector gate before becoming a candidate default.

The benchmark now accepts adapter-aware modern completion flags:

- `--start-index`: zero-based manifest row offset, used for held-out slices.
- `--model-name`: base Diffusers model override, for example `Lykon/dreamshaper-8-inpainting`.
- `--lora-weights`: local Diffusers LoRA adapter directory or safetensors file.
- `--lora-scale`: optional adapter weight; omitted means Diffusers default scale.
- `--edit-mask-fill`: for edit-only providers, replace the masked region with `white`, `gray`, `checker`, `mirror`, or `biharmonic` before generation; `mirror` and `biharmonic` are geometry-prefill cues for refinement sweeps.

Every benchmark run writes `split_audit.json`. For LoRA runs with a `training_report.json`, it records eval/train category coverage, eval/train source-split coverage, train/eval asset counts, `train_eval_asset_overlap_count`, loss recipe, train metadata/pair-export hashes, adapter hash, prompt family, and whether the training prompt family matches the evaluation prompt template. Treat `train_eval_asset_overlap_count == 0` and `prompt_family_matches_eval == true` as required gates for held-out LoRA promotion claims.

Cached LoRA runs created before the provenance fields existed can be upgraded without rerunning training:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.backfill_lora_provenance --output backend/output/completion-benchmark/lora_provenance_backfill_summary.json
```

The backfill command updates old `training_report.json` files with metadata hashes, pair-export summaries, train loss extrema, prompt family, loss recipe, and adapter hash. It also refreshes existing sweep `split_audit.json` files so `select_completion_candidate` can enforce the same training-report and prompt-family gates on cached results. On the cached balanced held-out-10 depth/STL probe, the stricter regenerated decision still holds the best LoRA against `mirror`; the failed checks are now incumbent score margin, paired win rate versus `mirror`, paired CI versus `mirror`, and `prompt_family_matches_eval`.

To compare two sweep directories after an intervention, use `compare_optimize_runs`. It ranks each run independently with baseline-delta scoring, evaluates the requested candidate with the normal selector, and writes side-by-side CSV/JSON/Markdown rows with scores, paired confidence, failed checks, and prompt-family provenance.

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.compare_optimize_runs --run old=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --candidate old=dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 --run prompt_matched=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_prompt_matched --candidate prompt_matched=dreamshaper_lora_balanced_train40_promptmatched_scale050_s256 --baseline-method masked --current-method mirror --output-md backend/output/completion-benchmark/experiments/prompt_matched_vs_mismatched_lora_comparison.md
```

To merge repeated held-out slices into one promotion-sized evidence set, use `combine_optimize_runs`. It prefixes sample IDs with the run label, recomputes `aggregate_summary.csv`, `ranked_experiments.csv`, paired objective confidence, and an optional `selection_decision.json` / `.md`.

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.combine_optimize_runs --run start40=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_amused --run start50=backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_modern_start50 --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_modern_start40_start50_combined_object_surface --score-mode baseline-delta --baseline-method masked --score-profile object-surface --current-method mirror --select-candidate
```

Smoke evidence from the local 3080 Ti:

| check | result |
| --- | --- |
| pair export | `4` ModelNet10 pairs written to `training_pairs/modelnet10_4_smoke` |
| trainer dry-run | validated `3x256x256` target/masked tensors and `1x256x256` mask tensors; mask fraction `0.5` |
| one-step LoRA train | wrote `pytorch_lora_weights.safetensors`, `train_metrics.csv`, and `training_report.json`; step-1 loss `0.09096` |
| adapter benchmark load | `run_completion_benchmark` loaded the local adapter offline and produced one-sample image metrics |

The one-step adapter is only a plumbing proof, not a quality claim. The next real run should train on a train split, evaluate on held-out assets, and accept the adapter only if its `rank_methods` score beats the public DreamShaper run and at least one non-learned baseline.

Held-out LoRA smoke:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0 --start-index 0 --limit 16
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_s20 --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 20 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_s20 --config backend/benchmark/experiment_configs/modelnet10_train16_lora_s20.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

The 20-step adapter trained on manifest rows `0-15` and evaluated on rows `16-19`. Training logged 20 optimizer steps; step-1 loss was `0.06385` and step-20 loss was `0.02574`.

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med | object SSIM med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.01 | 0.0836 | 13.60 | 0.781 | 0.0022 | 0.1243 | 14.13 | 0.572 |
| biharmonic | 4 | 11.12 | 0.3056 | 8.21 | 0.630 | 0.0107 | 0.0900 | 17.90 | 0.668 |
| dreamshaper_lora_train16_s20_s256 | 4 | 6.81 | 0.1389 | 11.15 | 0.749 | 0.0598 | 0.3674 | 7.46 | 0.413 |
| dreamshaper_base_s20_s256 | 4 | 4.35 | 0.1768 | 10.99 | 0.750 | 0.1284 | 0.4027 | 6.93 | 0.445 |

Interpretation: even this tiny 20-step adapter improves the learned baseline on the held-out slice, mainly by reducing hidden-half MAE and seam error. It still loses to mirror and biharmonic, so the next optimization target is not "LoRA works" but "LoRA beats at least one non-learned baseline on held-out object metrics."

Masked/seam/object loss iteration on the same split:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_mask3_seam4_s20 --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 20 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0_object --start-index 0 --limit 16
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0_object/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_obj8_mask2_seam2_s20 --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 20 --object-loss-weight 8 --mask-loss-weight 2 --seam-loss-weight 2 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_object_s20 --config backend/benchmark/experiment_configs/modelnet10_train16_lora_s20_object.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med | object SSIM med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.10 | 0.0836 | 13.60 | 0.781 | 0.0022 | 0.1243 | 14.13 | 0.572 |
| biharmonic | 4 | 11.12 | 0.3056 | 8.21 | 0.630 | 0.0107 | 0.0900 | 17.90 | 0.668 |
| dreamshaper_lora_mask3_seam4_s20_s256 | 4 | 7.30 | 0.1348 | 11.62 | 0.749 | 0.0602 | 0.3650 | 7.12 | 0.410 |
| dreamshaper_lora_plain_s20_s256 | 4 | 7.27 | 0.1389 | 11.15 | 0.749 | 0.0598 | 0.3674 | 7.46 | 0.413 |
| dreamshaper_base_s20_s256 | 4 | 4.80 | 0.1768 | 10.99 | 0.750 | 0.1284 | 0.4027 | 6.93 | 0.445 |
| dreamshaper_lora_obj8_mask2_seam2_s20_s256 | 4 | 4.69 | 0.1763 | 10.67 | 0.715 | 0.1023 | 0.3849 | 7.01 | 0.358 |

Interpretation: mask/seam weighting is a small positive move over the plain 20-step adapter. Heavy object weighting is a negative result: it improves object MAE versus base DreamShaper, but it hurts the learned adapter's overall rank and object SSIM, so it is not the current path forward.

LoRA scale sweep:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_scale_sweep_s20 --config backend/benchmark/experiment_configs/modelnet10_lora_scale_sweep_s20.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | object MAE med | object PSNR med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.09 | 0.0836 | 13.60 | 0.781 | 0.1243 | 14.13 |
| biharmonic | 4 | 11.13 | 0.3056 | 8.21 | 0.630 | 0.0900 | 17.90 |
| dreamshaper_lora_mask3_seam4_scale125_s20_s256 | 4 | 7.48 | 0.1277 | 11.99 | 0.754 | 0.3489 | 7.19 |
| dreamshaper_lora_plain_scale125_s20_s256 | 4 | 7.37 | 0.1288 | 11.76 | 0.753 | 0.3586 | 7.04 |
| dreamshaper_lora_mask3_seam4_scale100_s20_s256 | 4 | 7.30 | 0.1348 | 11.62 | 0.749 | 0.3650 | 7.12 |
| dreamshaper_lora_plain_scale100_s20_s256 | 4 | 7.28 | 0.1389 | 11.15 | 0.749 | 0.3674 | 7.46 |

Interpretation: `mask3/seam4` at `lora_scale=1.25` is the current best learned inpainting setting on this held-out smoke. It is still not good enough to replace mirror/biharmonic, but it is the strongest learned checkpoint for the next longer training run.

Longer mask/seam training:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_mask3_seam4_s100 --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 100 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_mask3_seam4_s200 --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 200 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_mask3_seam4_s100_sweep --config backend/benchmark/experiment_configs/modelnet10_lora_mask3_seam4_s100_sweep.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_mask3_seam4_s200_sweep --config backend/benchmark/experiment_configs/modelnet10_lora_mask3_seam4_s200_sweep.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

Image-only held-out results:

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med | object SSIM med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.08 | 0.0836 | 13.60 | 0.781 | 0.0022 | 0.1243 | 14.13 | 0.572 |
| biharmonic | 4 | 11.12 | 0.3056 | 8.21 | 0.630 | 0.0107 | 0.0900 | 17.90 | 0.668 |
| dreamshaper_lora_mask3_seam4_s100_scale075_s256 | 4 | 10.30 | 0.1429 | 11.42 | 0.781 | 0.0157 | 0.3178 | 7.70 | 0.447 |
| dreamshaper_lora_mask3_seam4_s200_scale100_s256 | 4 | 9.84 | 0.1009 | 13.14 | 0.768 | 0.0166 | 0.3056 | 8.37 | 0.407 |
| dreamshaper_base_s20_s256 | 4 | 5.92 | 0.1768 | 10.99 | 0.750 | 0.1284 | 0.4027 | 6.93 | 0.445 |

Interpretation: 100 training steps with mask/seam weighting almost catches biharmonic in the image-only weighted objective. The 200-step adapter improves raw masked MAE/PSNR and object MAE, but its image-only rank is slightly lower because object SSIM and normalized tradeoffs move against it.

Depth/STL check for the longer adapters:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_depth_stl_s100_s200 --config backend/benchmark/experiment_configs/modelnet10_lora_depth_stl_s100_s200.json --start-index 16 --limit 4 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error
```

| method | n | rank | masked MAE med | object MAE med | object depth MAE med | object depth corr med | silhouette IoU med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 24.22 | 0.0836 | 0.1243 | 0.2626 | 0.1345 | 0.2971 | 1.0 | 1.0 |
| biharmonic | 4 | 19.50 | 0.3056 | 0.0900 | 0.2475 | 0.3051 | 0.2971 | 1.0 | 1.0 |
| dreamshaper_lora_mask3_seam4_s200_scale100_s256 | 4 | 16.65 | 0.1009 | 0.3056 | 0.3493 | 0.3773 | 0.4459 | 1.0 | 1.0 |
| dreamshaper_lora_mask3_seam4_s100_scale075_s256 | 4 | 16.11 | 0.1429 | 0.3178 | 0.3055 | -0.0976 | 0.3072 | 1.0 | 1.0 |
| dreamshaper_base_s20_s256 | 4 | 8.13 | 0.1768 | 0.4027 | 0.3838 | 0.1714 | 0.2971 | 1.0 | 1.0 |

Interpretation: the learned completion still does not beat mirror or biharmonic overall, mostly because object RGB MAE remains too high. The 200-step adapter is the current best learned full-pipeline checkpoint: it roughly doubles base DreamShaper's full-pipeline rank, improves hidden-half RGB strongly, keeps STL exports watertight/positive-volume, and produces the best silhouette IoU in this run. The next training target should directly improve object-region RGB fidelity without sacrificing the silhouette/depth gains.

Prompt-aligned and gentle object-loss follow-up:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0_category --start-index 0 --limit 16
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0_category/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_mask3_seam4_s100_catprompt --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 100 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_train16_s0_category/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_train16_dreamshaper_mask3_seam4_obj1_s100_catprompt --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 16 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 100 --mask-loss-weight 3 --seam-loss-weight 4 --object-loss-weight 1 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_catprompt_s100_sweep --config backend/benchmark/experiment_configs/modelnet10_lora_catprompt_s100_sweep.json --start-index 16 --limit 4 --skip-depth --device auto --resume --continue-on-error
```

Image-only held-out results from the combined prompt/object sweep:

| method | n | rank | masked MAE med | masked PSNR med | masked SSIM med | seam MAE med | object MAE med | object PSNR med | object SSIM med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 16.07 | 0.0836 | 13.60 | 0.781 | 0.0022 | 0.1243 | 14.13 | 0.572 |
| biharmonic | 4 | 11.12 | 0.3056 | 8.21 | 0.630 | 0.0107 | 0.0900 | 17.90 | 0.668 |
| dreamshaper_lora_mask3_seam4_s200_scale100_s256 | 4 | 10.14 | 0.1009 | 13.14 | 0.768 | 0.0166 | 0.3056 | 8.37 | 0.407 |
| dreamshaper_lora_mask3_seam4_s100_scale075_s256 | 4 | 10.06 | 0.1429 | 11.42 | 0.781 | 0.0157 | 0.3178 | 7.70 | 0.447 |
| dreamshaper_lora_catprompt_obj1_s100_scale100_s256 | 4 | 8.59 | 0.1447 | 10.98 | 0.776 | 0.0242 | 0.3439 | 7.12 | 0.431 |
| dreamshaper_lora_catprompt_s100_scale100_s256 | 4 | 8.42 | 0.1415 | 11.04 | 0.777 | 0.0324 | 0.3424 | 7.16 | 0.433 |

Depth/STL check for the best prompt/object candidate:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_20_s256_seed2026/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_heldout4_lora_catprompt_obj1_depth_stl --config backend/benchmark/experiment_configs/modelnet10_lora_catprompt_obj1_depth_stl.json --start-index 16 --limit 4 --device auto --emit-stl --resume --continue-on-error
```

| method | n | rank | masked MAE med | object MAE med | object depth MAE med | object depth corr med | silhouette IoU med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 23.16 | 0.0836 | 0.1243 | 0.2626 | 0.1345 | 0.2971 | 1.0 | 1.0 |
| biharmonic | 4 | 17.45 | 0.3056 | 0.0900 | 0.2475 | 0.3051 | 0.2971 | 1.0 | 1.0 |
| dreamshaper_lora_mask3_seam4_s200_scale100_s256 | 4 | 13.92 | 0.1009 | 0.3056 | 0.3493 | 0.3773 | 0.4459 | 1.0 | 1.0 |
| dreamshaper_lora_catprompt_obj1_s100_scale100_s256 | 4 | 11.66 | 0.1447 | 0.3439 | 0.2914 | 0.2295 | 0.2971 | 1.0 | 1.0 |

Interpretation: prompt alignment by itself was a negative result on this tiny split, and gentle object loss (`--object-loss-weight 1`) only nudged the category-prompt adapter inside its own family. It did not beat the generic-prompt s100/s200 LoRAs, nor did it recover object RGB fidelity. The next training target should be more data and/or stronger geometry-aware supervision, not simply more prompt templating or a larger object-loss scalar.

Larger 64-sample data check:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/modelnet10_64_s256_seed3030 --source mesh-dir --asset-root data/modelnet10/extracted/ModelNet10 --asset-glob "**/*.off" --count 64 --size 256 --views-per-asset 1 --seed 3030 --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_64_s256_seed3030/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_64_train48_s0_generic --start-index 0 --limit 48 --prompt "Complete the missing half naturally while preserving the same subject, lighting, perspective, and background." --literal-prompt
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_64_train48_s0_generic/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_64_train48_dreamshaper_mask3_seam4_s200_generic --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 48 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 200 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_64_s256_seed3030/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_64_heldout16_train48_lora_s200_sweep --config backend/benchmark/experiment_configs/modelnet10_64_lora_train48_s200_sweep.json --start-index 48 --limit 16 --skip-depth --device auto --resume --continue-on-error
```

The 64-row render produced all `64` samples with no render failures. The train split used rows `0-47`; held-out evaluation used rows `48-63`. The 48-row dry-run reported object masks on all rows, `object_inpaint_nonzero_rows=48`, and `object_inpaint_fraction_mean=0.1317`.

Image-only held-out-16 results:

| method | n | attempted | success | rank | masked MAE med | object MAE med | object SSIM med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 16 | 16 | 1.00 | 15.71 | 0.0725 | 0.1541 | 0.442 | 0.0115 |
| biharmonic | 16 | 16 | 1.00 | 10.88 | 0.3616 | 0.0960 | 0.466 | 0.0188 |
| dreamshaper_lora_train16_mask3_seam4_s200_scale100_s256 | 16 | 16 | 1.00 | 8.19 | 0.1080 | 0.3153 | 0.352 | 0.0472 |
| dreamshaper_base_s20_s256 | 16 | 16 | 1.00 | 7.54 | 0.1212 | 0.3296 | 0.290 | 0.0651 |
| dreamshaper_lora_train48_mask3_seam4_s200_scale075_s256 | 16 | 16 | 1.00 | 7.15 | 0.1158 | 0.3414 | 0.323 | 0.0724 |
| dreamshaper_lora_train48_mask3_seam4_s200_scale100_s256 | 16 | 16 | 1.00 | 6.63 | 0.1120 | 0.3576 | 0.337 | 0.0770 |
| dreamshaper_lora_train48_mask3_seam4_s200_scale125_s256 | 16 | 16 | 1.00 | 5.12 | 0.1345 | 0.3577 | 0.325 | 0.0848 |

Interpretation: simply tripling training data from 16 to 48 rows did not improve this LoRA recipe. The old 16-row adapter still beats the 48-row adapter on the fresh held-out-16 split, and the 48-row adapter regresses object RGB MAE and seam quality. This points away from "just add more samples" and toward better data diversity/control, different loss shaping, or explicit geometry/depth conditioning.

Category-balanced 60-sample audit:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.generate_rendered_dataset --output-dir backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040 --source mesh-dir --asset-root data/modelnet10/extracted/ModelNet10 --asset-glob "**/*.off" --count 60 --size 256 --views-per-asset 1 --seed 4040 --mesh-sample-strategy balanced --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.export_training_pairs --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/training_pairs/modelnet10_60_balanced_train40_s0_generic --start-index 0 --limit 40 --prompt "Complete the missing half naturally while preserving the same subject, lighting, perspective, and background." --literal-prompt
.\backend\.venv\Scripts\python -m backend.benchmark.train_inpainting_lora --metadata backend/output/completion-benchmark/training_pairs/modelnet10_60_balanced_train40_s0_generic/metadata.jsonl --output-dir backend/output/completion-benchmark/lora/modelnet10_60_balanced_train40_dreamshaper_mask3_seam4_s200_generic --base-model Lykon/dreamshaper-8-inpainting --variant fp16 --resolution 256 --limit 40 --train-batch-size 1 --gradient-accumulation-steps 4 --max-train-steps 200 --mask-loss-weight 3 --seam-loss-weight 4 --mixed-precision fp16 --allow-tf32
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout20_train40_lora_s200_sweep --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_train40_s200_sweep.json --start-index 40 --limit 20 --skip-depth --device auto --resume --continue-on-error
```

The balanced renderer produced `60` samples: `6` each for bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, and toilet. The train slice uses rows `0-39` with `4/category`; held-out eval uses rows `40-59` with `2/category`. Source-folder provenance is mixed because this is a local benchmark slice over ModelNet10, not the official ModelNet train/test protocol: the full manifest has `train:41, test:19`, train has `train:26, test:14`, and eval has `train:15, test:5`. The identity gate is clean: `40` train assets, `20` eval assets, `train_eval_asset_overlap_count=0`.

Balanced held-out-20 image-only results:

| method | n | rank | masked MAE med | object MAE med | object SSIM med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 20 | 15.60 | 0.0980 | 0.1604 | 0.4085 | 0.0171 |
| biharmonic | 20 | 10.68 | 0.3217 | 0.1266 | 0.4878 | 0.0262 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale075_s256 | 20 | 8.60 | 0.1378 | 0.4083 | 0.3826 | 0.0373 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale100_s256 | 20 | 7.73 | 0.1423 | 0.4152 | 0.3838 | 0.0404 |
| dreamshaper_base_s20_s256 | 20 | 7.37 | 0.1511 | 0.3692 | 0.3500 | 0.0626 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale125_s256 | 20 | 6.29 | 0.1499 | 0.4495 | 0.3952 | 0.0493 |

Low LoRA-scale follow-up:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout20_train40_lora_s200_low_scale_sweep --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_train40_s200_low_scale_sweep.json --start-index 40 --limit 20 --skip-depth --device auto --resume --continue-on-error
```

| method | n | rank | masked MAE med | object MAE med | object PSNR med | object SSIM med | seam MAE med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 20 | 15.60 | 0.0980 | 0.1604 | 11.86 | 0.4085 | 0.0171 |
| biharmonic | 20 | 10.68 | 0.3217 | 0.1266 | 16.19 | 0.4878 | 0.0262 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 | 20 | 7.76 | 0.1390 | 0.3870 | 6.47 | 0.3572 | 0.0431 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale075_s256 | 20 | 7.43 | 0.1378 | 0.4083 | 6.30 | 0.3826 | 0.0373 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale0625_s256 | 20 | 7.41 | 0.1355 | 0.4184 | 5.87 | 0.3840 | 0.0390 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale025_s256 | 20 | 6.57 | 0.1484 | 0.4067 | 6.60 | 0.3387 | 0.0497 |
| dreamshaper_base_s20_s256 | 20 | 6.25 | 0.1511 | 0.3692 | 6.94 | 0.3500 | 0.0626 |

Interpretation: category balancing and source/asset auditing make the held-out claim cleaner, but they do not fix the current LoRA recipe. The balanced adapter improves hidden-half continuity and seam quality versus base DreamShaper, especially near `lora_scale=0.5-0.75`, but object-region RGB fidelity is still poor. Lowering scale recovers a little object MAE at `0.5` (`0.387`), yet it remains worse than base DreamShaper (`0.369`), mirror (`0.160`), and biharmonic (`0.127`). Also note that this adapter was trained with a generic literal prompt and evaluated with category-templated prompts, so this run should not be described as category-prompt training. The next useful training change is likely geometry-conditioned supervision or a stronger mask-native model, not another scalar sweep of this adapter.

Slice/selector follow-up on the same balanced held-out run:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.analyze_experiment_slices backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout20_train40_lora_s200_low_scale_sweep --selector-field asset_category --selector-train-filter asset_source_split=train --selector-eval-filter asset_source_split=test
```

Findings:

- `mirror` wins every `asset_category` slice and both `asset_source_split` slices under the current weighted objective.
- The per-sample oracle picks `mirror` for `16/20` samples, with one sample each going to `biharmonic`, base DreamShaper, LoRA scale `0.5`, and LoRA scale `0.75`. The oracle rank score is only slightly above mirror, so even perfect per-sample routing has limited upside on this split.
- The category selector trained on source-split `train` rows and evaluated on source-split `test` rows ranks below plain mirror on the five-row test subset: selector score `14.05` vs mirror `14.27`.

Interpretation: category-aware routing is not the missing ingredient for this balanced ModelNet10 slice. The path forward should be a genuinely better completion model or geometry-conditioned training target rather than a selector over the current methods.

Balanced depth/STL top-candidate probe:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_depth_stl_top_sweep.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error
```

This full-pipeline run evaluates one held-out sample per ModelNet10 category from rows `40-49` of the balanced manifest. It is a smaller probe than the image-only held-out-20 sweep, and its source-folder provenance is `train:9, test:1`, so treat it as a downstream check for the top candidates rather than a final model-selection proof.

| method | n | rank | masked MAE med | object MAE med | object depth MAE med | object depth corr med | silhouette IoU med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 24.67 | 0.1122 | 0.1633 | 0.3228 | -0.0744 | 0.3839 | 1.0 | 1.0 |
| biharmonic | 10 | 20.23 | 0.3487 | 0.1266 | 0.2994 | -0.0829 | 0.3269 | 1.0 | 1.0 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 | 10 | 19.24 | 0.1706 | 0.4127 | 0.3003 | 0.0665 | 0.3469 | 1.0 | 1.0 |
| dreamshaper_base_s20_s256 | 10 | 14.26 | 0.1798 | 0.4129 | 0.4139 | 0.0344 | 0.2720 | 1.0 | 1.0 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale075_s256 | 10 | 13.97 | 0.1694 | 0.4380 | 0.4453 | -0.0503 | 0.2813 | 1.0 | 1.0 |
| masked | 10 | 10.28 | 0.2649 | 0.6384 | 0.4178 | 0.1906 | 0.3269 | 1.0 | 1.0 |

Stable masked-baseline score for the same aggregate summary:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/aggregate_summary.csv --output backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/ranked_experiments_masked_delta.csv --score-mode baseline-delta --baseline-method masked
```

| method | n | baseline-delta score vs masked | masked MAE med | object MAE med | object depth MAE med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 7.69 | 0.1122 | 0.1633 | 0.3228 | 0.3839 |
| biharmonic | 10 | 6.66 | 0.3487 | 0.1266 | 0.2994 | 0.3269 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 | 10 | 4.53 | 0.1706 | 0.4127 | 0.3003 | 0.3469 |
| dreamshaper_base_s20_s256 | 10 | 3.90 | 0.1798 | 0.4129 | 0.4139 | 0.2720 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale075_s256 | 10 | 3.63 | 0.1694 | 0.4380 | 0.4453 | 0.2813 |
| masked | 10 | 0.00 | 0.2649 | 0.6384 | 0.4178 | 0.3269 |

Paired objective confidence against `masked`:

| method | paired n | wins | mean objective improvement | mean CI95 | median improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 10/10 | 7.24 | [5.49, 8.99] | 7.88 |
| biharmonic | 10 | 10/10 | 6.30 | [4.54, 8.03] | 6.35 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 | 10 | 10/10 | 4.24 | [2.61, 6.07] | 4.11 |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale075_s256 | 10 | 8/10 | 3.48 | [1.04, 5.79] | 4.10 |
| dreamshaper_base_s20_s256 | 10 | 9/10 | 3.19 | [1.44, 5.09] | 2.86 |

Interpretation: mirror still wins the full completion-to-depth-to-STL objective, and biharmonic remains the strongest object-RGB baseline. The balanced LoRA at scale `0.5` is useful evidence, though: it beats base DreamShaper on both the normalized downstream rank (`19.24` vs `14.26`) and the stable masked-baseline score (`4.53` vs `3.90`), improves object-depth MAE (`0.3003` vs `0.4139`), improves silhouette IoU (`0.3469` vs `0.2720`), and has a paired objective CI entirely above `0` versus `masked` (`[2.61, 6.07]`). It still has poor object RGB MAE (`0.4127`), so it should not become the default; LoRA scale `0.75` now ranks just below base DreamShaper in the combined objective. The `masked` baseline ranks last in this combined top-candidate probe, and the report now includes aggregate median deltas, paired per-sample wins, and paired objective confidence against `masked` so rank changes can be audited without confusing weighted score movement with raw metric improvement. The first version of this run exposed a signed-volume convention mismatch in the STL exporter; the writer now orients triangle winding using the same Trimesh volume convention as the benchmark diagnostics, and the repeated v2 run exported `10/10` watertight, positive-volume STLs for every candidate.

Promotion gate result on this run:

| candidate | current | decision | reason |
| --- | --- | --- | --- |
| mirror | mirror | keep_current | top method already equals current default; all quality gates pass |
| dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 | mirror | hold | paired confidence vs `masked` passes, but it loses to `mirror`: stable score margin `-3.165`, paired wins `2/10`, paired incumbent CI `[-4.67, -1.18]` |

Prompt-matched follow-up:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_prompt_matched --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_depth_stl_prompt_matched.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror --candidate-method dreamshaper_lora_balanced_train40_promptmatched_scale050_s256 --resume --continue-on-error
```

This controlled rerun evaluates the same balanced LoRA with the exact generic literal prompt used for training. The provenance gate now passes (`prompt_family_matches_eval=true`), but the candidate gets worse: the surface-aware baseline-delta score drops from `6.5525` in the category-prompt run to `5.1710`, paired mean improvement versus `masked` drops from `6.1637` to `4.8684`, and paired mean improvement versus `mirror` drops from `-3.1836` to `-4.4789`. The decision remains `hold`; failed checks are now only incumbent quality checks (`score_margin_vs_current`, `paired_win_rate_vs_current`, `paired_ci95_low_vs_current`). Prompt matching cleans up the experimental claim, but it is a negative model-quality result for this adapter.

No-completion lower-bound probe on the same held-out-10 slice:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.run_completion_benchmark --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines --methods masked,mirror,biharmonic --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines/summary_metrics.csv --output backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines/ranked_methods.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines --output backend/output/completion-benchmark/runs/modelnet10_60_balanced_heldout10_masked_depth_stl_baselines/run_report.md --baseline-method masked
```

This is a standalone three-method run; its rank scores are normalized over `masked`, `mirror`, and `biharmonic` only, so use it for lower-bound deltas rather than direct numeric comparison with the combined top-candidate sweep above.

| method | n | rank | masked MAE med | object MAE med | depth MAE med | object depth MAE med | seam MAE med | STL watertight med | STL positive volume med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mirror | 10 | 23.33 | 0.1122 | 0.1633 | 0.1647 | 0.3228 | 0.0188 | 1.0 | 1.0 |
| biharmonic | 10 | 18.74 | 0.3487 | 0.1266 | 0.2941 | 0.2994 | 0.0278 | 1.0 | 1.0 |
| masked | 10 | 7.70 | 0.2649 | 0.6384 | 1.0495 | 0.4178 | 0.2024 | 1.0 | 1.0 |

Interpretation: leaving the missing half blank is a much weaker full-pipeline lower bound. It can look deceptively acceptable on broad masked SSIM because the white background is easy, but object RGB MAE, seam MAE, and aligned depth MAE are all much worse. The regenerated report's baseline-delta table shows mirror improving masked MAE by `0.1527`, seam MAE by `0.1836`, and depth MAE by `0.8848` versus `masked`; biharmonic improves object MAE by `0.5118`. The paired table confirms this is not only a median artifact: mirror beats `masked` on the same sample for depth MAE `10/10`, object MAE `10/10`, seam MAE `10/10`, and masked MAE `9/10`; biharmonic beats `masked` on object MAE `10/10` and seam MAE `10/10`. The paired objective confidence table is also entirely positive: mirror mean improvement `7.24` with CI `[5.49, 8.99]`, and biharmonic `6.30` with CI `[4.54, 8.03]`. This justifies keeping completion enabled for half-missing benchmark cases even when the best current learned inpainting model is not the winner.

## Colab G4 Orchestrator

The preferred heavy-compute path for the next iteration is a single connected Colab G4 notebook, not competing local GPU processes. The current notebook target is `https://colab.research.google.com/drive/1SuilhFuF5L3ELkEy2rnEsTmKAL19ob60`; its probed runtime is an NVIDIA RTX PRO 6000 Blackwell Server Edition with about 95 GB VRAM, CUDA available, PyTorch `2.11.0+cu128`, and Python `3.12.13`. Do not use unrelated ARC or competition-only Blackwell resources for this project.

The orchestrator script is `backend/benchmark/colab_g4_orchestrator.py`. It is a normal Python entry point and also uses cell-friendly structure, so it can be pasted into Colab or run from a cloned checkout:

```bash
python -m backend.benchmark.colab_g4_orchestrator --repo-url https://github.com/jennyzzt/3dprintpic.git --run-name g4_object_surface_modelnet60 --dataset-source procedural --dataset-count 60 --train-limit 40 --calibration-limit 10 --eval-start 40 --eval-start 50 --eval-limit 10 --score-profile object-surface --train-steps 200
```

If a rendered ModelNet10/YCB/Objaverse-style manifest has already been uploaded or generated in Drive, use it directly:

```bash
python -m backend.benchmark.colab_g4_orchestrator --use-current-repo --manifest /content/drive/MyDrive/3dprintpic/manifest.jsonl --dataset-source existing --run-name g4_uploaded_meshes --cache-download-mode snapshot --train-steps 400 --eval-start 40 --eval-start 50
```

The stages are resume-friendly:

- `setup`: clone/fetch the repo and install `backend/requirements-cuda.txt`.
- `dataset`: generate a rendered procedural or mesh-folder manifest when no uploaded manifest is supplied.
- `cache`: cache `sdxl-inpaint`, `dreamshaper-inpaint`, and `amused-inpaint` once on the Colab VM.
- `calibrate`: run mirror on a train slice with depth/STL metrics.
- `weight`: convert object-surface/depth failures into per-row `sample_weight`.
- `train`: train a DreamShaper-compatible weighted LoRA. This adapter is SD1.5 inpaint compatible; do not load it into SDXL/AMUSED/FLUX/Qwen.
- `eval`: rank modern providers plus the weighted LoRA on held-out slices with `optimize_completion`.
- `combine`: combine repeated held-out slices and run the selector gate against the current `mirror` method.

Use `--stage <name>` to resume a subset after an interruption and `--dry-run` to print/log the planned commands without doing downloads or training. Each run writes `gpu_probe.json`, `orchestrator_config.json`, `command_log.jsonl`, generated configs, calibration reports, LoRA reports, held-out experiment reports, contact sheets, and `orchestrator_result.json` under `backend/output/completion-benchmark/colab_g4/<run-name>`. The intended parallelism is within one orchestrator run: keep model caches warm, batch samples, let CPU-side rendering/masking/data prep feed GPU inference, and batch metrics before ranking.

First G4 smoke on the fork branch `codex/3d-completion-benchmark-g4` completed the `setup,dataset,calibrate,weight,train` stages after adding `torchao>=0.16.0` to avoid the Colab image's incompatible preinstalled `torchao 0.10.0` with `peft 0.19.1`. The smoke used 20 procedural rendered samples, 12 train pairs, four mirror calibration samples, object-surface sample weighting, and 20 weighted LoRA steps. It wrote `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_weighted_train_smoke/lora/weighted_object-surface_s20/training_report.json` with final step `20`, final loss `0.020325`, best loss `0.008421`, final unweighted loss `0.020235`, `3/12` non-default sample weights, weight range `1.0` to `2.9933`, mean weight `1.2765`, and adapter SHA256 `701be19b845be5c3d51005afa5d52baec99851d1838555c860bb3ca1fbfcc2af`. This proves the G4 training lane works; the next heavier Colab pass should add the `eval,combine` stages with cached SDXL/DreamShaper/AMUSED providers.

Resume that exact smoke into held-out evaluation without retraining:

```bash
python -m backend.benchmark.colab_g4_orchestrator --use-current-repo --run-name g4_weighted_train_smoke --stage cache --stage eval --stage combine --manifest /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_weighted_train_smoke/dataset/manifest.jsonl --existing-lora-weights /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_weighted_train_smoke/lora/weighted_object-surface_s20 --train-steps 20 --eval-start 12 --eval-start 16 --eval-limit 4 --score-profile object-surface --require-modern-cache
```

The resumed `cache,eval,combine` pass completed on the same G4 runtime for held-out procedural rows `12-19` (`8` samples total). Under the object-surface baseline-delta score, the weighted LoRA ranked first but did not pass the incumbent gate:

| method | n | score | object surface Chamfer L1 med | object depth MAE med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: |
| dreamshaper_weighted_lora_object-surface_s20_scale0.75 | 8 | 10.1810 | 0.0700 | 0.1666 | 0.6627 |
| dreamshaper_base_s20_s256 | 8 | 9.8270 | 0.0797 | 0.1679 | 0.4890 |
| mirror | 8 | 9.4968 | 0.0877 | 0.2172 | 0.4929 |
| biharmonic | 8 | 9.0549 | 0.1179 | 0.1712 | 0.3555 |
| sdxl_base_s18_s384 | 8 | 8.2870 | 0.1477 | 0.2597 | 0.3513 |
| amused_base_s12_native512 | 8 | 7.9494 | 0.1459 | 0.3586 | 0.3706 |
| masked | 8 | 0.0000 | 0.7736 | 1.2914 | 0.3513 |

Selector decision: `hold` for `dreamshaper_weighted_lora_object-surface_s20_scale0.75` against current `mirror`. Failed checks were paired win rate versus current (`0.625`, threshold `>= 0.8`) and paired CI95 low versus current (`-1.1505`, threshold `> 0`). This is positive evidence that metric-weighted training can beat mirror on a small procedural object-surface aggregate, but it is not a default-promotion result.

Local 3080 Ti ModelNet10 held-out probe:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.colab_g4_orchestrator --use-current-repo --run-name local_3080ti_dreamshaper_probe_s48_s50_n2 --stage eval --stage combine --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --existing-lora-weights backend\output\completion-benchmark\lora\modelnet10_train40_weighted_surface_s20 --modern-config backend\benchmark\experiment_configs\local_dreamshaper_weighted_probe.json --eval-start 48 --eval-start 50 --eval-limit 2 --score-profile object-surface --train-steps 20 --eval-steps 12 --eval-inpaint-max-dimension 256 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --min-paired-n 2 --allow-missing-split-audit --contact-sheet-max-samples 4
```

This reuses the Colab G4 orchestration path on a 12 GB RTX 3080 Ti while the Colab G4 runtime is disconnected. The local config keeps the base DreamShaper branch at `12` steps and `256` pixels, then compares it with `masked`, `mirror`, `biharmonic`, and the already trained weighted DreamShaper LoRA on two held-out slices (`4` total ModelNet10 rows). The run completed with `0` failures and kept the current `mirror` default:

| method | n | score | object surface Chamfer L1 med | object depth MAE med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: |
| mirror | 4 | 3.8332 | 0.0777 | 0.1476 | 0.3055 |
| biharmonic | 4 | 2.7402 | 0.1529 | 0.2317 | 0.2685 |
| dreamshaper_weighted_lora_modelnet10_train40_weighted_surface_s20_scale0.75 | 4 | 2.5992 | 0.1391 | 0.2513 | 0.2885 |
| dreamshaper_base_s12_s256 | 4 | 1.4371 | 0.2247 | 0.4099 | 0.2903 |
| masked | 4 | 0.0000 | 0.3624 | 0.5607 | 0.2677 |

Interpretation: this is a directional smoke, not a robust held-out claim: it covers two `eval-limit=2` slices, and the weighted LoRA's paired objective against `masked` wins only `2/4` with CI95 `[-0.2637, 3.7965]`. The aggregate medians still show useful signal: the weighted LoRA beats base DreamShaper on object-surface and object-depth metrics, so metric-weighted training is improving downstream geometry. It still trails `mirror` on the held-out objective and has weak object RGB fidelity (`object MAE` median `0.4462`), so this is not a promotion result. The run emitted depth and STL artifacts, but the combined object-surface score above is driven by depth, surface, and silhouette metrics; STL validity is audited in per-sample rows rather than contributing to this score. Audit outputs are under `backend/output/completion-benchmark/colab_g4/local_3080ti_dreamshaper_probe_s48_s50_n2/combined/modern_weighted_object-surface_n4/`, with contact sheets in each `experiments/modern_weighted_eval_*_n2/` directory.

The same local 3080 Ti lane was then expanded to the full held-out `20` rows (`eval-start 40`, `eval-start 50`, `eval-limit 10`) while the Colab G4 runtime was disconnected:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.colab_g4_orchestrator --use-current-repo --run-name local_3080ti_dreamshaper_full_heldout20_s40_s50 --stage eval --stage combine --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --existing-lora-weights backend\output\completion-benchmark\lora\modelnet10_train40_weighted_surface_s20 --modern-config backend\benchmark\experiment_configs\local_dreamshaper_weighted_probe.json --eval-start 40 --eval-start 50 --eval-limit 10 --score-profile object-surface --train-steps 20 --eval-steps 12 --eval-inpaint-max-dimension 256 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --min-paired-n 10 --allow-missing-split-audit --contact-sheet-max-samples 6
```

Full held-out result:

| method | n | score | object surface Chamfer L1 med | object depth MAE med | silhouette IoU med |
| --- | ---: | ---: | ---: | ---: | ---: |
| mirror | 20 | 2.4490 | 0.1293 | 0.2768 | 0.3839 |
| biharmonic | 20 | 1.8350 | 0.1825 | 0.2968 | 0.3158 |
| dreamshaper_base_s12_s256 | 20 | 1.7280 | 0.1793 | 0.3959 | 0.3049 |
| dreamshaper_weighted_lora_modelnet10_train40_weighted_surface_s20_scale0.75 | 20 | 1.6666 | 0.1701 | 0.3576 | 0.3208 |
| masked | 20 | 0.0000 | 0.3325 | 0.5413 | 0.3158 |

Selector decision: `hold` and keep current `mirror`. The full run completed `100/100` method-sample rows with `0` failures. `mirror` led the object-surface aggregate and had paired objective CI95 `[0.8513, 3.3124]` against `masked`, but only `15/20` wins (`0.75`, below the `0.8` gate). The weighted LoRA still improves some geometry medians versus base DreamShaper, including object-surface Chamfer (`0.1701` vs `0.1793`) and object-depth MAE (`0.3576` vs `0.3959`), but its combined score is lower and its paired objective against `masked` is not robust (`12/20`, CI95 `[-0.3314, 2.8452]`). The next useful training change is geometry-conditioned supervision or a stronger mask-native provider such as Qwen Image Edit / FLUX Fill on a large-memory runtime, not another small DreamShaper prompt or scale sweep.

Local regression coverage for the G4 resume lane:

```powershell
.\backend\.venv\Scripts\python -m unittest backend.tests.test_benchmark_regressions.ColabG4OrchestratorRegressionTests -v
```

These tests dry-run `eval,combine` with an existing LoRA, combine-only reuse of completed eval directories, and fail-fast missing manifest/adapter validation. This keeps the notebook workflow recoverable after interruptions without accidentally retraining, changing existing-LoRA labels, or silently combining incomplete slices.

To move local rendered ModelNet assets and a trained adapter onto Colab without committing generated artifacts, package a Colab-ready bundle:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --lora-weights backend\output\completion-benchmark\lora\modelnet10_train40_weighted_surface_s20 --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20 --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_weighted_surface_s20_run_colab_eval.sh --colab-archive-path /content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz --run-name g4_modelnet10_weighted_surface_eval_s20 --eval-start 40 --eval-start 50 --eval-limit 10 --score-profile object-surface --train-steps 20 --max-method-failures 2
```

The bundle rewrites manifest paths to the chosen extraction root and includes the adapter directory plus training report. The current package report is `modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz.report.json`: `60` rows, `420` referenced dataset files, `423` payload files plus the rewritten manifest and `run_colab_eval.sh`, and SHA256 `60504cdda38ef121ebd74fdf3470d65e22bf46ab487c55b2e0eb0d25d43243fe`.

The packager can now generate a no-LoRA launcher for a large-provider slice. This is the preferred Qwen smoke because it avoids appending the DreamShaper weighted-LoRA branch to the experiment config:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_qwen_edit_sanity_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_60_qwen_edit_sanity --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_qwen_edit_sanity_run_colab_eval.sh --colab-archive-path /content/modelnet10_60_qwen_edit_sanity_colab_inputs.tar.gz --run-name g4_modelnet10_qwen_edit_s20_s512_sanity --modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json --cache-provider qwen-image-edit --cache-full --eval-start 40 --eval-start 50 --eval-limit 2 --score-profile object-surface --eval-steps 20 --eval-inpaint-max-dimension 512 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --min-paired-n 2 --max-method-failures 2 --allow-missing-split-audit --contact-sheet-methods masked,mirror,biharmonic,qwen_edit_s20_s512
```

After uploading the tarball to `/content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz` in Colab, extract the launcher and run the real ModelNet10 weighted-LoRA held-out gate with:

```bash
mkdir -p /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20
tar -xzf /content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz -C /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20 run_colab_eval.sh
EXPECTED_SHA256=60504cdda38ef121ebd74fdf3470d65e22bf46ab487c55b2e0eb0d25d43243fe bash /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20/run_colab_eval.sh
```

The launcher verifies `EXPECTED_SHA256` when set, validates the rewritten manifest, adapter weights, and training report, clones `REPO_DIR` if a fresh Colab runtime does not already have the repo, then writes `launch_preflight.json` with the archive SHA, resolved git commit, manifest row count, and adapter path before model cache/eval work begins. It also tees the G4 run to `run_colab_eval.log` and always writes `/content/g4_modelnet10_weighted_surface_eval_s20_results.tar.gz` containing the log, preflight, `results_summary.json`, and the orchestrator output directory when present. `results_summary.json` now includes compact per-eval and combined-run digests with artifact presence, top method, candidate decision, failed checks, per-method coverage/error counts, `stl_mode`, key mesh-surface metrics, and STL printability gates, so partial Colab archives can be triaged without unpacking every CSV manually. Newly generated launchers also run the STL-first ingest helper before archiving and include `stl_first_ingest_report.json`, `stl_first_ingest_report.md`, and `stl_first_ingest.log` when output CSVs are available. They now emit `/content/<run-name>_results_compact.tar.gz` before the full archive as well; it preserves every CSV/JSON/JSONL/log/Markdown result plus the contact sheet, but excludes GLB/STL/PLY/NPY and per-sample render binaries, so it remains locally re-ingestable and practical to checksum-transfer from Colab. Use `--max-method-failures 2` for large-provider prompt sweeps; one broken provider/config is then recorded and skipped instead of repeating the same expensive failure across every sample.

After downloading an older Colab results archive, or when the in-run ingest report is missing or failed, run the STL-first ingest helper locally to re-rank the full CSVs from the archive and separate deployable STL architectures from `source_mesh_oracle` diagnostics:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.ingest_stl_results g4=C:\path\to\g4_run_results.tar.gz --output-dir backend\output\completion-benchmark\ingested\g4_run --score-profile stl-quality --score-mode baseline-delta --baseline-method masked
```

This writes `stl_first_ingest_report.json` and `stl_first_ingest_report.md`. The report picks the best deployable method among `depth-relief`, `single-image-mesh`, and `multiview-mesh`, while keeping `source-mesh-oracle` as a calibration row so ground-truth diagnostics do not accidentally become the product winner.

Next large-provider G4 slice, using the same packaged ModelNet10 manifest after the archive has been extracted:

```bash
cd /content/3dprintpic
python -m backend.benchmark.colab_g4_orchestrator --use-current-repo --run-name g4_modelnet10_qwen_edit_s20_s512 --stage cache --stage eval --stage combine --manifest /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20/inputs/manifest.jsonl --modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json --cache-provider qwen-image-edit --cache-full --cache-download-mode snapshot --cache-max-workers 8 --eval-start 40 --eval-start 50 --eval-limit 10 --score-profile object-surface --eval-steps 20 --eval-inpaint-max-dimension 512 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --min-paired-n 10 --allow-missing-split-audit --require-modern-cache --contact-sheet-methods masked,mirror,biharmonic,qwen_edit_s20_s512
```

For a cheap live sanity check before the full held-out `20`, change both `--eval-limit 10` and `--min-paired-n 10` to `2`. After Qwen finishes, repeat the same command with `--run-name g4_modelnet10_flux_fill_s20_s512`, `--modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_flux_fill_g4_depth_stl.json`, `--cache-provider flux-fill`, and contact-sheet method `flux_fill_s20_s512` if FLUX auth/access is ready.

Colab G4 Qwen procedural sanity:

The notebook `https://colab.research.google.com/drive/1SuilhFuF5L3ELkEy2rnEsTmKAL19ob60` ran commit `2ccf589` on the G4 runtime with `Qwen/Qwen-Image-Edit` cached and evaluated a tiny procedural slice (`dataset-count=4`, `eval-start=0`, `eval-limit=2`, `eval-steps=20`, `inpaint_max_dimension=512`, object-surface score). The run completed and wrote `/content/g4_qwen_procedural_sanity_s0_n2_results.tar.gz` (`4.2 MB`) plus the orchestrator output directory `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_qwen_procedural_sanity_s0_n2`; the idle runtime was then disconnected/deleted, so rerun the launcher when runtime-local artifacts are needed for contact-sheet inspection.

| method | score |
| --- | ---: |
| `mirror` | 3.3505 |
| `biharmonic` | 1.4383 |
| `masked` | 0.0000 |
| `qwen_edit_s20_s512` | -2.3662 |

Interpretation: this is an integration sanity result, not a model-quality conclusion. It proves the G4 path can cache and run Qwen Image Edit end-to-end through completion, depth, STL artifact generation, ranking, and tarball packaging. On this small procedural slice, the generic Qwen edit prompt underperformed the deterministic mirror baseline badly enough that the next Qwen iteration should inspect the contact sheet/depth outputs and adjust the edit prompt or mask presentation before spending a full held-out ModelNet20 run.

Next Qwen prompt sweep:

`backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_prompt_sweep_g4_depth_stl.json` keeps `masked`, `mirror`, and `biharmonic`, then compares eight Qwen Image Edit prompt/guidance/mask-presentation variants on the same cached model:

- `qwen_edit_baseline_s20_s512`: the original generic white-region prompt.
- `qwen_edit_strict_white_region_s20_s512`: explicitly treats pure white pixels as the only edit region and forbids duplicate objects or blank output.
- `qwen_edit_mirror_prefill_refine_s20_s512`: gives Qwen a mirrored geometry prefill in the missing half and asks it to refine that side.
- `qwen_edit_biharmonic_prefill_refine_s20_s512`: gives Qwen a smooth biharmonic prefill in the missing half and asks it to refine that side.
- `qwen_edit_symmetry_depth_s20_s512`: asks for a seam-continuous symmetric outline and smooth depth surface for downstream 3D reconstruction.
- `qwen_edit_lowcfg_shape_s20_s512`: lowers true CFG to `2.0` and emphasizes one continuous 3D object.
- `qwen_edit_center_seam_single_object_s20_s512`: gives the center seam as the edit boundary and forbids duplicate objects.
- `qwen_edit_checker_region_s20_s512`: replaces the white missing region with a checkerboard edit cue before Qwen Image Edit so the model is less likely to treat pure white as background.

Package a no-LoRA prompt-sweep launcher with:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_qwen_edit_prompt_sweep_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_60_qwen_edit_prompt_sweep --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_qwen_edit_prompt_sweep_run_colab_eval.sh --colab-archive-path /content/modelnet10_60_qwen_edit_prompt_sweep_colab_inputs.tar.gz --run-name g4_modelnet10_qwen_edit_prompt_sweep_s20_s512 --modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_prompt_sweep_g4_depth_stl.json --cache-provider qwen-image-edit --cache-full --eval-start 40 --eval-start 50 --eval-limit 2 --score-profile object-surface --eval-steps 20 --eval-inpaint-max-dimension 512 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --min-paired-n 2 --max-method-failures 2 --allow-missing-split-audit --contact-sheet-methods masked,mirror,biharmonic,qwen_edit_baseline_s20_s512,qwen_edit_strict_white_region_s20_s512,qwen_edit_mirror_prefill_refine_s20_s512,qwen_edit_biharmonic_prefill_refine_s20_s512,qwen_edit_symmetry_depth_s20_s512,qwen_edit_lowcfg_shape_s20_s512,qwen_edit_center_seam_single_object_s20_s512,qwen_edit_checker_region_s20_s512
```

The procedural G4 sanity version of this sweep ran in the notebook on commit `61ad905` as `g4_qwen_prompt_sweep_procedural_s0_n2` (`dataset-count=4`, `eval-start=0`, `eval-limit=2`, full Qwen snapshot cache, `20` steps, `512` max dimension). It completed in about `11m42s` and wrote `/content/g4_qwen_prompt_sweep_procedural_s0_n2_results.tar.gz` (`9.1 MB`) inside the Colab runtime. The object-surface baseline-delta ranking was:

| method | score |
| --- | ---: |
| `mirror` | 3.3505 |
| `biharmonic` | 1.4383 |
| `masked` | 0.0000 |
| `qwen_edit_baseline_s20_s512` | -2.3662 |
| `qwen_edit_lowcfg_shape_s20_s512` | -2.4132 |
| `qwen_edit_center_seam_single_object_s20_s512` | -2.4916 |
| `qwen_edit_strict_white_region_s20_s512` | -3.8060 |
| `qwen_edit_symmetry_depth_s20_s512` | -5.0894 |
| `qwen_edit_checker_region_s20_s512` | -5.2982 |

Interpretation: prompt-only and mask-presentation-only Qwen edits did not improve the tiny procedural objective. The baseline Qwen prompt remained the least bad measured Qwen setting, but still trailed deterministic `mirror` by `5.7167` score points and `biharmonic` by `3.8045`, so the next check changed the actual edit input with geometry-prefill cues instead of only changing prompt text.

The geometry-prefill follow-up ran in the same notebook on commit `cef3154` as `g4_qwen_geometry_prefill_procedural_s0_n2` (`dataset-count=4`, `eval-start=0`, `eval-limit=2`, full Qwen snapshot cache, `20` steps, `512` max dimension, `--max-method-failures 2`). It completed in about `14m`, wrote `/content/g4_qwen_geometry_prefill_procedural_s0_n2_results.tar.gz` (`12 MB`) inside the runtime, and the runtime was then disconnected/deleted. The object-surface baseline-delta ranking was:

| method | score |
| --- | ---: |
| `mirror` | 3.3505 |
| `biharmonic` | 1.4383 |
| `masked` | 0.0000 |
| `qwen_edit_baseline_s20_s512` | -2.3662 |
| `qwen_edit_lowcfg_shape_s20_s512` | -2.4132 |
| `qwen_edit_center_seam_single_object_s20_s512` | -2.4916 |
| `qwen_edit_strict_white_region_s20_s512` | -3.8060 |
| `qwen_edit_mirror_prefill_refine_s20_s512` | -3.8660 |
| `qwen_edit_checker_region_s20_s512` | -4.2327 |
| `qwen_edit_symmetry_depth_s20_s512` | -5.0894 |
| `qwen_edit_biharmonic_prefill_refine_s20_s512` | -8.4653 |

Interpretation: geometry-prefilling the Qwen edit input did not rescue this provider on the tiny procedural objective. Mirror-prefill was slightly worse than strict-white prompting and `1.4998` score points below the baseline Qwen prompt; biharmonic-prefill was the worst tested variant. Do not spend the packaged held-out ModelNet prompt sweep on these Qwen variants without first inspecting artifacts and identifying a specific edit-conditioning fix. The next learned-method lane should pivot to a stronger mask-native provider or to training/evaluating geometry-conditioned supervision that directly targets object-surface and hidden-object depth metrics.

STL-first next phase:

The end product is a printable STL, not a better-looking 3D preview. Keep the current depth/inpaint benchmark as the fast baseline, but compare it against STL-native branches by emitting STLs and ranking or optimizing with `--score-profile stl-quality`. This profile ranks final STL validity, mesh quality, complexity, and available surface/depth agreement metrics. Each emitted STL now records `stl_is_volume`, `stl_winding_consistent`, `stl_single_component`, `stl_component_excess`, `stl_bbox_has_volume`, bounding-box dimensions/aspect, and log-scaled face-density proxies in addition to watertightness, positive volume, face count, surface area, and volume.

Use `stl_mode` in configs and reports to keep comparisons honest: `depth-relief` for the current image completion plus depth relief STL, `single-image-mesh` for direct image-to-mesh providers, `multiview-mesh` for video/multiview reconstruction providers, and `source-mesh-oracle` only for ground-truth diagnostics. The runner infers these labels for old configs, but new STL-first configs should set them explicitly.

Next experiment lanes:

- Fast 2.5D relief: current completion/depth/STL path with `--emit-stl`, then rank or optimize the run with `--score-profile stl-quality`; this remains the baseline and should stay cheap.
- Single-image mesh: add direct image-to-3D candidates that emit mesh/STL artifacts, such as Hunyuan3D/TripoSR/SV3D-style paths where practical, then rank them with the same STL diagnostics rather than visual preview quality. Compare raw provider meshes against repaired/remeshed variants because STL printability can fail even when the preview mesh looks plausible.
- Video or multiview mesh: reconstruct from selected or every frames with camera/keypoint matching and object masks/crops; Gaussian splatting or NeRF should be optional intermediate backends only when the final extracted mesh/STL improves.

Promotion should require STL-facing evidence: watertightness, manifold/volume status, winding consistency, body count, printable thickness or bounding-box sanity, hole/degeneracy checks when available, mesh complexity, and surface Chamfer/visual-depth agreement when ground truth exists. Candidate selection now hard-gates available STL fields, including `stl_is_volume`, `stl_is_manifold`, `stl_winding_consistent`, `stl_single_component`, `stl_bbox_has_volume`, nonmanifold-edge count, degenerate-face ratio, component excess, bbox aspect, and face density; a direct mesh cannot promote only by winning Chamfer while failing printability.

The STL-first harness now has a one-command launcher:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.run_stl_first_smoke --repo-dir . --output-dir backend/output/completion-benchmark/experiments/stl_first_reconstruction_smoke_local_s0_n1 --dataset-count 1 --limit 1 --size 128 --stl-target-dimension 64 --contact-sheet-max-samples 1 --no-include-triposr-api --no-include-hunyuan3d-shape
```

By default, it writes `stl_first_reconstruction_config.json`, generates a tiny rendered dataset when `--manifest` is not supplied, runs `optimize_completion`, ranks with `--score-mode baseline-delta --score-profile stl-quality --baseline-method masked`, writes a contact sheet, and summarizes the run in `stl_first_summary.json`. The default local candidate set is `masked`, `mirror`, `biharmonic`, and `source_mesh_oracle`; the source oracle is now repair-aware and uses the smoke's `--mesh-repair` mode before STL export so raw dataset mesh defects do not become a misleading negative control. Add `--include-triposr-api`, `--include-hunyuan3d-shape`, or `--multiview-command '...'` when the corresponding provider environment is ready. Add `--mesh-target-max-dimension <n>` and `--mesh-min-bbox-dimension <n>` to generated provider commands when the experiment should compare model-space meshes against STL-space scaled/compact variants. Add `--select-candidate --candidate-method <method> --current-method mirror --min-paired-n <n>` to emit `selection_decision.json/.md` and include the promotion decision in `stl_first_summary.json`; the smoke runner forwards the same STL hard-gate threshold flags as `optimize_completion` for promotion-grade local checks.

For manual or Colab runs, `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_reconstruction_candidates.json` keeps the same relief baselines, adds `source_mesh_oracle`, and includes repaired TripoSR API plus an STL-scaled compact repaired TripoSR sibling, plus repaired Hunyuan3D shape candidates. Run it with the same STL-quality command shape:

```bash
python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_reconstruction_candidates_s40_n2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_reconstruction_candidates.json --start-index 40 --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,source_mesh_oracle,triposr_api_masked_repaired_direct_mesh,triposr_api_masked_repaired_stl_scaled_compact_direct_mesh,hunyuan3d_shape_masked_repaired_direct_mesh --contact-sheet-max-samples 2 --resume --continue-on-error
```

For the first Hunyuan3D shape-only G4 run, use the smaller dedicated config `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_hunyuan3d_shape_candidate.json`. It compares relief baselines, the repaired source-mesh oracle, raw repaired Hunyuan shape, STL-scaled compact repaired Hunyuan shape, and a mirror-bbox calibrated Hunyuan shape. The generated Colab script can now bootstrap `/content/Hunyuan3D-2.1` and `/content/hunyuan3d-venv` with `--include-hunyuan3d-setup`; this follows the official shape pipeline API while skipping the heavier paint/PBR stack because textures do not affect printable STL quality. The setup uses a shallow sparse clone of the `hy3dshape` runtime package/config directories so fresh runtimes do not spend time checking out the repo's large demo and training assets. It also probes PyMeshLab because Hunyuan's official import path can require it even for shape-only mesh postprocessing; the Colab launcher pins `pymeshlab==2023.12.post3` because that release has a CPython 3.12 `manylinux_2_31_x86_64` wheel, while newer PyMeshLab releases pushed the G4 image toward a source metadata/build path.

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_heldout10_stl_first_hunyuan3d_shape_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_heldout10_stl_first_hunyuan3d_shape --start-index 40 --limit 10 --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\run_colab_hunyuan3d_shape_eval.sh --colab-archive-path /content/modelnet10_heldout10_stl_first_hunyuan3d_shape_colab_inputs.tar.gz --repo-remote https://github.com/DrStrangel0ve/3dprintpic.git --repo-ref codex/3d-completion-benchmark-g4 --run-name g4_stl_first_hunyuan3d_shape_s40_n10 --modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_hunyuan3d_shape_candidate.json --eval-start 0 --eval-limit 10 --score-profile stl-quality --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --candidate-method hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh --current-method mirror --min-paired-n 10 --allow-missing-split-audit --contact-sheet-methods masked,mirror,biharmonic,source_mesh_oracle,hunyuan3d_shape_masked_repaired_direct_mesh,hunyuan3d_shape_masked_repaired_stl_scaled_compact_direct_mesh,hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh --contact-sheet-max-samples 4 --no-require-modern-cache --skip-cache --max-method-failures 2 --include-hunyuan3d-setup
```

When regenerating the checked package artifact, also pass `--report C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_heldout10_stl_first_hunyuan3d_shape_colab_inputs_report.json`, `--inline-colab-launcher C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\run_colab_hunyuan3d_shape_eval_inline_cell.py`, `--fetch-colab-launcher C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\run_colab_hunyuan3d_shape_eval_fetch_cell.py`, `--fetch-colab-notebook C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\run_colab_hunyuan3d_shape_eval_fetch_notebook.ipynb`, and `--fetch-colab-payload-url https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/b6fce644326e611feb130dcdbb5d723e084dd04f/colab_payloads/modelnet10_heldout10_stl_first_hunyuan3d_shape_colab_inputs.tar.gz` so the SHA, packaged-row report, pasteable Colab bootstrap cells, and paste-free Colab notebook stay beside the archive. The package writer emits deterministic gzip/tar metadata, LF-only generated launchers, a setup-wide `run_colab_eval.log`, and installs backend requirements before provider setup, so archive and launcher SHA256 values are stable across Windows, GitHub raw downloads, and Colab while setup-stage failures are still diagnosable. The inline cell embeds the tarball as base64 chunks, writes it to `/content/modelnet10_heldout10_stl_first_hunyuan3d_shape_colab_inputs.tar.gz`, verifies size/SHA256, extracts only `run_colab_eval.sh`, and launches it with `EXPECTED_SHA256` set.

The inline launcher is exact-artifact glue: regenerating the tarball invalidates the old cell because both the embedded bytes and SHA change. Base64 expands the payload by roughly one third, so this is convenient for the current 1.6 MB Hunyuan slice but manual upload/extract/run remains the fallback for larger packages or browser-paste failures.

If the Colab UI has trouble accepting the 2 MB inline paste, use the smaller fetch cell `run_colab_hunyuan3d_shape_eval_fetch_cell.py` from the local `outputs` directory or open the paste-free Colab notebook at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/c0c7365413855590398fca0ca305417cbd976fbb/colab_payloads/run_colab_hunyuan3d_shape_eval_fetch_notebook.ipynb`. The notebook is published from payload commit `c0c7365413855590398fca0ca305417cbd976fbb`, has raw SHA256 `a7070790ef345fc4cbd1338dd28c87076fb44b195ad96eb076b5bcd6924e685d`, and downloads the immutable tarball from archive commit `b6fce644326e611feb130dcdbb5d723e084dd04f`. The tarball raw download was validated at `1,621,939` bytes with SHA256 `e60cfc13b2d3018d60d9f17d5cd9288cf5040daabec9a2e9745ac0eeb9d5205c`; the fetch launcher verifies that exact size/SHA, extracts `run_colab_eval.sh`, and launches the same Hunyuan setup/eval path.

Hunyuan G4 launch history is now captured as three concrete failure/success checkpoints. The first fetch notebook (`0276b6df...`) entered `run_colab_eval.sh` but exited before writing preflight/log/archive evidence. The second setup-logged notebook (`aa8d4add...`) did write a failure archive and traced setup failure to `backend/pic_to_3d.py` importing `from stl import mesh` before `numpy-stl` was installed. The current pre-provider-install notebook (`c0c736...`) installed backend requirements before Hunyuan setup, completed with `run_status=0`, and wrote `/content/g4_stl_first_hunyuan3d_shape_s40_n10_results.tar.gz` plus `results_summary.json`.

The full Colab archive download did not land through the browser, so a compact evidence bundle was recovered from the live notebook output and ingested locally from `backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_shape_s40_n10_recovered`:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.ingest_stl_results hunyuan=backend\output\completion-benchmark\colab_g4\g4_stl_first_hunyuan3d_shape_s40_n10_recovered --output-dir backend\output\completion-benchmark\ingested\g4_stl_first_hunyuan3d_shape_s40_n10 --score-profile stl-quality --score-mode baseline-delta --baseline-method masked
```

Colab G4 Hunyuan result (`g4_stl_first_hunyuan3d_shape_s40_n10`, recovered summary, July 9, 2026): setup and orchestration succeeded, but all three Hunyuan direct-mesh candidates failed before producing a mesh. The first two non-skipped Hunyuan samples exited with status `137`, then the benchmark skipped the remaining rows after `--max-method-failures=2`. The depth-relief baselines still completed and remain the deployable winner for this run.

| method | n | attempted | stl-quality score vs masked | success rate | mesh surface Chamfer med | H95 med | STL watertight med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 10 | 10 | 0.6122 | 1.0 | 0.0746 | 0.2376 | 1.0 |
| `mirror` | 10 | 10 | 0.2111 | 1.0 | 0.1722 | 0.5090 | 1.0 |
| `biharmonic` | 10 | 10 | 0.1332 | 1.0 | 0.1892 | 0.4924 | 1.0 |
| `masked` | 10 | 10 | 0.0000 | 1.0 | 0.1966 | 0.5226 | 1.0 |
| `hunyuan3d_shape_masked_repaired_direct_mesh` | 0 | 10 | 0.0000 | 0.0 |  |  |  |
| `hunyuan3d_shape_masked_repaired_stl_scaled_compact_direct_mesh` | 0 | 10 | 0.0000 | 0.0 |  |  |  |
| `hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh` | 0 | 10 | 0.0000 | 0.0 |  |  |  |

Promotion decision: `hold` for `hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh`. It failed the success-rate and paired-evidence gates (`paired_n=0`) and scored below current `mirror` by `-0.2111`. Treat this as a provider-runtime failure, not a Hunyuan quality conclusion.

The next Hunyuan retry should be a low-memory single-candidate smoke before another 10-row comparison. The provider wrapper now exposes Hunyuan shape inference knobs and the candidate config passes `--num-inference-steps 30 --guidance-scale 5.0 --octree-resolution 256 --num-chunks 8000 --disable-progress --low-vram`. Hunyuan provider calls also emit JSON resource markers to stderr before/after import, model load, offload, inference, and export; these include `/proc/meminfo`, `ru_maxrss`, and CUDA memory when available. External image-to-mesh commands write `external_command.stdout.log` and `external_command.stderr.log` beside each attempted sample and include their tails in the raised benchmark error, so a repeat `137` should leave better evidence in the result archive.

For the immediate retry, use `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_hunyuan3d_shape_debug.json` with `--eval-limit 1 --max-method-failures 1`. It keeps `masked`, `mirror`, `biharmonic`, and `source_mesh_oracle` for STL-quality context, but launches only `hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh` in the single-image mesh lane. The generated Colab run script now exports `PYTHONUNBUFFERED=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_MODULE_LOADING=LAZY`, and `MALLOC_ARENA_MAX=2`, and pins the Hunyuan runtime checkout with `HUNYUAN3D_REF=82920d643c0dc2f7bfd7255f45f62d386edfe60c` unless overridden.

The checked one-row debug package for that retry was regenerated from repo commit `ba8f9657e00015c31ecce881aead301acddf054e` as `g4_stl_first_hunyuan3d_shape_debug_s40_n1`. It packaged one held-out row, uses `candidate_method=hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh`, keeps `current_method=mirror`, and requires `min_paired_n=1`. The local artifact `modelnet10_heldout1_stl_first_hunyuan3d_shape_debug_colab_inputs.tar.gz` is `133,462` bytes with SHA256 `2d06006ff003def7381fef075baa869bfe01f8799a1d7c02739061be47b15214`; the inline launcher SHA256 is `41788e108fc38ed7176632c08e64b156ef762b1e1ba0e6dfba66564212f642fd`, the fetch launcher SHA256 is `e01af0dada75f3fe4c52564351a4d5009b5159d6b53cb9959caf8c4d99ca378d`, and the fetch notebook SHA256 is `f7f72822e65743066bdca963e240dfc4ce5a8d10f4d6b22cdae01c47db3b492f`.

The immutable debug payload archive is published from payload commit `f77075fe7ec5f7a2c15d0d78d43e41f55217ed9d` at `https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/f77075fe7ec5f7a2c15d0d78d43e41f55217ed9d/colab_payloads/modelnet10_heldout1_stl_first_hunyuan3d_shape_debug_colab_inputs.tar.gz`. The paste-free notebook is published from payload commit `ca9276200c78c9df14a46d6dcb15f0d482fff12e` at `https://colab.research.google.com/github/DrStrangel0ve/3dprintpic/blob/ca9276200c78c9df14a46d6dcb15f0d482fff12e/colab_payloads/run_colab_hunyuan3d_shape_debug_eval_fetch_notebook.ipynb`; its raw notebook URL validated at `3,459` bytes with SHA256 `f7f72822e65743066bdca963e240dfc4ce5a8d10f4d6b22cdae01c47db3b492f`, and the raw archive validated at `133,462` bytes with SHA256 `2d06006ff003def7381fef075baa869bfe01f8799a1d7c02739061be47b15214`.

Follow-up G4 terminal retry evidence on July 10, 2026 SGT: the GitHub-hosted Colab notebook opened on a CPU runtime first, and `nvidia-smi` was unavailable, so that run was stopped before spending setup time on the wrong machine. The original Drive notebook then connected to `G4 (Python 3)` and probed as an NVIDIA RTX PRO 6000 Blackwell Server Edition with about `95 GB` VRAM, CUDA available, PyTorch `2.11.0+cu128`, and Python `3.12.13`. Terminal-driven launches hit two real runtime resets before benchmark preflight: first during backend requirements install, then after a manual backend repair (`trimesh>=4.12.2`, `numpy-stl`, `scikit-image`, `matplotlib`) and a successful `backend_imports_ok 2.11.0+cu128` probe, again during the Hunyuan venv dependency setup after the `pymeshlab` probe. Treat this as Colab setup fragility, not Hunyuan model-quality evidence.

The package launcher was hardened from that evidence. Generated `run_colab_eval.sh` now honors `COLAB_SKIP_BACKEND_INSTALL=1` for a pre-warmed runtime, while keeping the full backend install path as the default for fresh runtimes. The Hunyuan setup probe now checks missing Python deps with `importlib.util.find_spec`, catches missing parent packages, prints one compact `missing Hunyuan3D Python deps: ...` line, and splits Hunyuan venv installs into labeled chunks (`core diffusers stack`, `geometry and image stack`, `model helpers`, `pymeshlab`) so a reset leaves a clear last completed setup stage.

A hardened-package rerun on the same G4 notebook then repaired the backend geometry stack, verified `backend_imports_ok 2.11.0+cu128`, and launched the one-row package with `COLAB_SKIP_BACKEND_INSTALL=1`. That avoided the backend reinstall path and reached the Hunyuan setup, but the runtime disconnected again immediately after the `python -m virtualenv --system-site-packages /content/hunyuan3d-venv` creation output; a fresh terminal showed `/content/3dprintpic`, `/content/Hunyuan3D-2.1`, `/content/hunyuan3d-venv`, the extracted package, and the result archive were all gone. The following stdlib-venv package moved the failure forward without wiping the runtime: it wrote `/content/g4_stl_first_hunyuan3d_shape_debug_s40_n1_results.tar.gz` with `run_colab_eval.log` and `results_summary.json`, `run_status=1`, and the log tail showed `Creating Hunyuan3D Python env with stdlib venv` followed by `Error: Command '['/content/hunyuan3d-venv/bin/python3', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.` The next package therefore defaults to `HUNYUAN3D_VENV_BACKEND=target`: it keeps the configured `/content/hunyuan3d-venv/bin/python` path as a wrapper, installs Hunyuan-only packages into `/content/hunyuan3d-deps` with `pip --target`, and retains `stdlib` and `virtualenv` as override backends.

Target-backend G4 follow-up evidence: the first target-wrapper package reached setup and wrote a failure archive with `run_status=1`; the log failed at `cat > "$HUNYUAN3D_VENV/bin/python"` with `/content/hunyuan3d-venv/bin/python: Text file busy`, because target mode tried to overwrite an existing executable left by an earlier stdlib/virtualenv attempt. Commit `5c2fd7a` now writes the wrapper to a unique temp path and atomically `mv -f`s it into place. That cleared the file-busy stop, but target-mode pip then tried to resolve heavyweight dependencies including `torch-2.8.0` and the terminal disconnected during install, so commit `36c762e` added `--no-deps` for target-mode Hunyuan installs to reuse Colab's existing `torch 2.11.0+cu128`. The next run avoided the Torch reinstall and reached PyMeshLab, but newer `pymeshlab` selected a source tarball and disconnected during `Preparing metadata (setup.py): started`; commit `fd525ef` pins `pymeshlab==2023.12.post3` for a binary CPython 3.12 wheel. The pinned-wheel package was raw-validated, but the first G4 run from that package disconnected/reset around `Creating Hunyuan3D Python env wrapper with target deps`; reopening `/content` showed no result archive, no extracted `run_colab_eval.log`, no `/content/hunyuan3d-deps`, and no wrapper. Treat the latest failure as Colab runtime instability during provider bootstrap.

The launcher now supports staged provider setup before benchmark rows are consumed. Set `COLAB_PROVIDER_SETUP_ONLY=1` for any provider setup-only run, or `HUNYUAN3D_SETUP_ONLY=1` for the Hunyuan-specific case. The launcher still extracts the package, syncs the repo, optionally installs backend requirements, runs provider setup, writes `launch_preflight.json`, exits with `run_status=0`, sets `provider_setup_only=true` in `results_summary.json`, and writes the normal results archive. Hunyuan setup also logs checkpoint lines after the target wrapper is ready, after Python deps are importable, and after optional weight prefetch. The next clean G4 retry should first run the current one-row package with `COLAB_SKIP_BACKEND_INSTALL=1 HUNYUAN3D_SETUP_ONLY=1 HUNYUAN3D_PREFETCH=0`; if that survives, rerun setup-only with `HUNYUAN3D_PREFETCH=1`, then run the full one-row benchmark without setup-only.

Generated Colab packages can now gate the actual runtime before provider setup with `--colab-require-gpu-name-regex` and `--colab-min-gpu-memory-gb`. The guard records `nvidia-smi` name/VRAM evidence in `gpu_preflight.json`, adds it to the results archive, and fails fast when Colab silently assigns CPU/T4 or another undersized GPU to a G4-only payload.

Regression coverage for that hardening:

```powershell
.\backend\.venv\Scripts\python.exe -m py_compile backend\benchmark\package_colab_inputs.py backend\tests\test_benchmark_regressions.py
.\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_benchmark_regressions.ColabInputPackageRegressionTests backend.tests.test_benchmark_regressions.StlExportRegressionTests backend.tests.test_benchmark_regressions.StlResultIngestRegressionTests
git diff --check
```

Local Hunyuan setup smoke on the 3080 Ti validated the narrow sparse checkout and `hy3dshape.pipelines` import path after installing the same shape-runtime dependency family (`einops`, `omegaconf`, `opencv-python`, `pymeshlab`, `timm`, `torchdiffeq`). The actual unauthenticated checkpoint download stalled locally at `model.fp16.ckpt` with a 0-byte incomplete file, so the first full Hunyuan quality comparison is still assigned to G4. The provider wrapper now cleans that exact interrupted Hunyuan cache shape and retries once when `from_pretrained` finds `config.yaml` but no checkpoint file.

The Hunyuan setup preflight now runs `run_image_to_mesh_provider --provider hunyuan3d-shape --prefetch-only` after dependency probes. That downloads/checks the shape checkpoint before `colab_g4_orchestrator` starts consuming benchmark rows, so download/auth failures are visible as setup failures instead of repeated per-sample provider failures. Set `HUNYUAN3D_PREFETCH=0` only when deliberately testing the setup without downloading weights.

The Hunyuan candidate config now includes `hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh`, which uses the mirror depth-relief baseline's STL bounding-box extents via `{mirror_bbox_extents}`. This tests whether Hunyuan's shape prior is being penalized mainly by arbitrary model-space scale before comparing it against the compact fixed-scale variant.

Promotion-sized runs should use held-out slices large enough for paired evidence and should explicitly name the direct-mesh candidate being challenged:

```bash
python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_reconstruction_candidates_s40_n10 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_reconstruction_candidates.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --select-candidate --candidate-method triposr_api_masked_repaired_stl_scaled_compact_direct_mesh --current-method mirror --min-paired-n 10 --allow-missing-split-audit --contact-sheet --contact-sheet-methods masked,mirror,biharmonic,source_mesh_oracle,triposr_api_masked_repaired_direct_mesh,triposr_api_masked_repaired_stl_scaled_compact_direct_mesh,hunyuan3d_shape_masked_repaired_direct_mesh --contact-sheet-max-samples 4 --resume --continue-on-error
```

Colab G4 promotion package for the same held-out rows `40-49`:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_heldout10_stl_first_triposr_scaled_compact_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_heldout10_stl_first_triposr_scaled_compact --start-index 40 --limit 10 --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_heldout10_stl_first_triposr_scaled_compact_run_colab_eval.sh --colab-archive-path /content/modelnet10_heldout10_stl_first_triposr_scaled_compact_colab_inputs.tar.gz --repo-remote https://github.com/DrStrangel0ve/3dprintpic.git --repo-ref codex/3d-completion-benchmark-g4 --run-name g4_stl_first_triposr_scaled_compact_s40_n10 --modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_reconstruction_candidates.json --eval-start 0 --eval-limit 10 --score-profile stl-quality --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --stl-target-dimension 96 --candidate-method triposr_api_masked_repaired_stl_scaled_compact_direct_mesh --current-method mirror --min-paired-n 10 --allow-missing-split-audit --contact-sheet-methods masked,mirror,biharmonic,source_mesh_oracle,triposr_api_masked_repaired_direct_mesh,triposr_api_masked_repaired_stl_scaled_compact_direct_mesh,hunyuan3d_shape_masked_repaired_direct_mesh --contact-sheet-max-samples 4 --no-require-modern-cache --skip-cache --max-method-failures 2
```

The package command slices the local manifest to rows `40-49`, then rewrites that smaller manifest into the tarball. Because the Colab launcher evaluates the packaged manifest, use `--eval-start 0 --eval-limit 10` inside the package rather than `--eval-start 40`. Direct mesh STL-first configs do not need the modern inpainting cache stage; pass `--skip-cache` so a launcher installs backend requirements but does not default to caching unrelated inpainting providers before eval/combine. The scaled/compact bundle packaged `10` rows, `70` referenced files, and wrote SHA256 `263dc8add527dbcc9ba567f554a7ccb08663e8145f06e91a864c059c59880446` with explicit `candidate_method=triposr_api_masked_repaired_stl_scaled_compact_direct_mesh` and `current_method=mirror`.

In Colab, prepare `/content/TripoSR` and `/content/triposr-venv/bin/python` with the TripoSR API setup above before running the generated launcher. Then upload the tarball to `/content/modelnet10_heldout10_stl_first_triposr_scaled_compact_colab_inputs.tar.gz`, extract `run_colab_eval.sh`, and run it to produce `/content/g4_stl_first_triposr_scaled_compact_s40_n10_results.tar.gz` plus `results_summary.json`.

Colab G4 STL-first TripoSR promotion result: the notebook ran commit `2e69185` on July 9, 2026 as `g4_stl_first_triposr_promotion_s40_n10`, verified archive SHA256 `ebf1050a8878e45c442ccb72c2bb2f64092659d5c4c318dff9b57a1f1f4712d1`, reused the isolated TripoSR API venv (`torch 2.11.0+cu128`, `transformers 4.35.0`, `trimesh 4.12.2`), and completed with `run_status=0`.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | STL watertight med | STL manifold med | STL faces med | success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 10 | 0.6139 | 0.0746 | 0.2376 | 1.0 | 1.0 | 103 | 1.0 |
| `mirror` | 10 | 0.2199 | 0.1723 | 0.5056 | 1.0 | 1.0 | 29580 | 1.0 |
| `biharmonic` | 10 | 0.1335 | 0.1885 | 0.4950 | 1.0 | 1.0 | 29580 | 1.0 |
| `masked` | 10 | 0.0000 | 0.1970 | 0.5226 | 1.0 | 1.0 | 29580 | 1.0 |
| `triposr_api_masked_repaired_direct_mesh` | 10 | -1.9375 | 0.1524 | 0.4067 | 1.0 | 1.0 | 2312 | 1.0 |
| `hunyuan3d_shape_masked_repaired_direct_mesh` | 0 | 0.0000 |  |  |  |  |  | 0.0 |

Promotion decision: `hold`. Repaired TripoSR passed the STL printability gates on all 10 samples: watertight, volume-valid, manifold, winding-consistent, positive-volume, single-component, no non-manifold edges, no degenerate faces, acceptable bbox aspect, and acceptable face density. It still failed objective promotion with `0/10` paired wins, paired CI95 low `-2.6950`, score margin versus mirror `-2.1573`, `0/10` paired wins versus mirror, and paired-current CI95 low `-2.8486`. The useful signal is that repaired TripoSR is now printable and lower-complexity than the relief STL baseline while beating mirror on median mesh-surface Chamfer, but the full STL-quality objective still prefers the depth-relief path. The next direct mesh iteration should inspect the contact sheet and per-sample score components, then try mirror/biharmonic prefilled TripoSR or a properly installed Hunyuan3D/SV3D-style provider before changing the app default.

Colab G4 repaired TripoSR prefill follow-up: the notebook ran commit `39f8d6c` on July 9, 2026 as `g4_stl_first_triposr_prefill_s40_n10`, verified archive SHA256 `3f6d5175260dfa5b2236298abb23992f1120fa29f884c5caae2f6356d3a6d38d`, and completed with `run_status=0`. This reran the same held-out rows with masked, mirror-prefilled, and biharmonic-prefilled repaired TripoSR candidates.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | STL watertight med | STL manifold med | STL faces med | success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 10 | 0.6139 | 0.0746 | 0.2376 | 1.0 | 1.0 | 103 | 1.0 |
| `mirror` | 10 | 0.2199 | 0.1723 | 0.5056 | 1.0 | 1.0 | 29580 | 1.0 |
| `biharmonic` | 10 | 0.1335 | 0.1885 | 0.4950 | 1.0 | 1.0 | 29580 | 1.0 |
| `masked` | 10 | 0.0000 | 0.1970 | 0.5226 | 1.0 | 1.0 | 29580 | 1.0 |
| `triposr_api_masked_repaired_direct_mesh` | 10 | -1.9375 | 0.1524 | 0.4067 | 1.0 | 1.0 | 2312 | 1.0 |
| `triposr_api_mirror_prefill_repaired_direct_mesh` | 10 | -2.2439 | 0.1501 | 0.4013 | 1.0 | 1.0 | 1841 | 1.0 |
| `triposr_api_biharmonic_prefill_repaired_direct_mesh` | 10 | -2.3817 | 0.1412 | 0.3995 | 1.0 | 1.0 | 2883 | 1.0 |

Promotion decision: `hold` for `triposr_api_mirror_prefill_repaired_direct_mesh`. All repaired TripoSR variants passed the printability gates on all 10 samples. Prefilling improved direct mesh surface metrics versus masked TripoSR, with biharmonic prefill giving the best median Chamfer and H95, but the full STL-quality objective still favored the depth-relief `mirror` baseline. The mirror-prefill candidate failed with `0/10` paired wins, paired CI95 low `-3.0241`, score margin versus mirror `-2.4638`, `0/10` paired wins versus mirror, and paired-current CI95 low `-3.1712`. The next useful iteration is to inspect per-sample score components and contact sheets to identify which non-Chamfer terms pull direct meshes below the relief baseline, then either tune the score profile for true full-mesh STL preference or try a stronger modern direct mesh backend.

The rank-score explainer now decomposes `rank_score` into metric contributions and writes `rank_score_explanation.md/json/csv` beside future `optimize_completion` runs. The current STL-quality profile uses `stl_faces_per_normalized_bbox_volume_log1p_median` for the mesh-complexity contribution so direct mesh candidates in model units and depth-relief STLs in print units remain comparable. Older aggregate CSVs are upgraded in memory when they contain `stl_faces_median`, `stl_bbox_volume_median`, and `stl_bbox_max_dimension_median`.

Colab G4 STL-scale follow-up: the notebook reran the same held-out rows on commit `392c112` as `g4_stl_first_triposr_scaled_compact_reuse_s40_n10`, reusing `/content/3dprintpic_colab_inputs/modelnet10_heldout10_stl_first_triposr_prefill/inputs/manifest.jsonl`. It completed with `run_status=0` in `175.1s` and produced `/content/g4_stl_first_triposr_scaled_compact_reuse_s40_n10_results.tar.gz` (`46,520,157` bytes). This run compared the existing repaired TripoSR output against a scaled/compact sibling using `--mesh-target-max-dimension 96 --mesh-min-bbox-dimension 12`.

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | bbox aspect med | face-density log1p med | STL faces med | success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 10 | 0.6139 | 0.0746 | 0.2376 | 1.4834 | 4.1921 | 103 | 1.0 |
| `triposr_api_masked_repaired_stl_scaled_compact_direct_mesh` | 10 | 0.2791 | 0.1524 | 0.4067 | 2.3694 | 0.0066 | 2312 | 1.0 |
| `mirror` | 10 | 0.2199 | 0.1723 | 0.5056 | 1.7935 | 0.0829 | 29580 | 1.0 |
| `biharmonic` | 10 | 0.1335 | 0.1885 | 0.4950 | 1.7205 | 0.0796 | 29580 | 1.0 |
| `masked` | 10 | 0.0000 | 0.1970 | 0.5226 | 1.7348 | 0.0803 | 29580 | 1.0 |
| `triposr_api_masked_repaired_direct_mesh` | 10 | -1.9375 | 0.1524 | 0.4067 | 2.3694 | 8.8730 | 2312 | 1.0 |

Promotion decision: `hold` for `triposr_api_masked_repaired_stl_scaled_compact_direct_mesh`. The scale/compact postprocess preserved TripoSR's normalized mesh-surface gains and removed the face-density penalty, lifting the aggregate score above `mirror` by `+0.0593`. It still failed promotion gates with only `6/10` paired wins, CI95 low `-0.3477` versus masked, `6/10` paired wins versus mirror, and CI95 low `-0.5331` versus mirror. The next direct-mesh iteration should inspect the four losing samples and target per-sample consistency rather than basic STL scale: likely input selection/prefill choice, orientation/axis normalization, or a stronger direct mesh backend.

The paired objective loss explainer added in commit `1f0caf1` was run in the same Colab notebook against the scaled/compact result after fetching `codex/3d-completion-benchmark-g4`. It writes `paired_objective_explanation.md/json/csv` beside the experiment and sorts the lowest per-sample score deltas first:

```bash
python -m backend.benchmark.explain_paired_objective /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_triposr_scaled_compact_reuse_s40_n10/experiment --candidate-method triposr_api_masked_repaired_stl_scaled_compact_direct_mesh --baseline-method masked --current-method mirror --score-profile stl-quality
```

The four direct-mesh losses versus `mirror` are dominated by `stl_bbox_aspect_ratio`, not topology invalidity:

| sample | score vs mirror | top hurts | top helps | candidate Chamfer | mirror Chamfer | candidate bbox aspect | mirror bbox aspect |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `mesh_dir_92752051d0_v00` | -1.4547 | bbox aspect -1.2984; H95 -0.1434 | face density +0.0183 | 0.1734 | 0.1704 | 4.3165 | 1.7196 |
| `mesh_dir_49534574e1_v00` | -0.9965 | bbox aspect -0.7481; Chamfer -0.2586 | H95 +0.1328 | 0.2316 | 0.1669 | 3.3230 | 1.8269 |
| `mesh_dir_757e06b221_v00` | -0.6695 | bbox aspect -0.7829; H95 -0.0515 | Chamfer +0.0848; RMSE +0.0588 | 0.1472 | 0.1684 | 3.6206 | 2.0547 |
| `mesh_dir_1a13fc2246_v00` | -0.3397 | bbox aspect -0.2619; H95 -0.0877 | face density +0.0189 | 0.1691 | 0.1690 | 2.2616 | 1.7378 |

Historical interpretation: STL scaling fixed the earlier mesh-complexity penalty, but it did not fix occasional elongated or axis-misaligned TripoSR shapes. This motivated the aspect/orientation calibration candidates below, including postprocess alignment to camera-frame oracle or deployable mirror-relief extents before STL export.

The first calibration candidate is now wired through `backend.benchmark.run_image_to_mesh_provider` as `--mesh-max-bbox-aspect-ratio`. It keeps the existing scale/compact export, then thickens only the shorter STL-space axes until the final bbox aspect is below the requested ratio. The reconstruction config includes `triposr_api_masked_repaired_stl_aspect_clamped2p25_direct_mesh`, which uses:

```bash
--mesh-target-max-dimension 96 --mesh-min-bbox-dimension 12 --mesh-max-bbox-aspect-ratio 2.25
```

Use that method as the next `--candidate-method` against `mirror` on the same held-out rows. This is a calibration probe, not a default: it may improve the four elongated losses, but it can distort true thin objects, so promotion still has to pass paired STL-quality gates.

Colab G4 aspect-clamp result: the notebook terminal ran commit `76a08a1` as `g4_stl_first_triposr_aspect_clamp_s40_n10`, reusing the same packaged held-out manifest and the isolated TripoSR API venv. It completed successfully and wrote `/content/g4_stl_first_triposr_aspect_clamp_s40_n10_results.tar.gz` (`25,172,098` bytes).

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | bbox aspect med | success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `triposr_api_masked_repaired_stl_aspect_clamped2p25_direct_mesh` | 10 | 0.4597 | 0.1446 | 0.3782 | 2.2500 | 1.0 |
| `mirror` | 10 | 0.2199 | 0.1723 | 0.5056 | 1.7935 | 1.0 |
| `masked` | 10 | 0.0000 | 0.1970 | 0.5226 | 1.7348 | 1.0 |

Promotion decision: `hold`. The aspect clamp is a real improvement over the scaled/compact direct mesh: aggregate score rose from `0.2791` to `0.4597`, paired wins versus `mirror` improved from `6/10` to `7/10`, and the previous `mesh_dir_757e06b221_v00` loss became a win. It still fails promotion because the paired win-rate gate requires `>= 0.8` and the paired CI95 low versus `mirror` is slightly negative at `-0.0055`.

Remaining losses versus `mirror`:

| sample | score vs mirror | candidate/mirror bbox aspect | candidate/mirror Chamfer | candidate/mirror H95 |
| --- | ---: | ---: | ---: | ---: |
| `mesh_dir_49534574e1_v00` | -0.4474 | 2.2500 / 1.8269 | 0.2249 / 0.1669 | 0.5879 / 0.6401 |
| `mesh_dir_1a13fc2246_v00` | -0.3360 | 2.2500 / 1.7378 | 0.1692 / 0.1690 | 0.5735 / 0.5291 |
| `mesh_dir_92752051d0_v00` | -0.1578 | 2.2500 / 1.7196 | 0.1632 / 0.1704 | 0.4391 / 0.4438 |

Historical interpretation: clamping fixed the worst elongation outlier without breaking STL validity, but a fixed `2.25` ratio over-penalized relative bbox aspect against mirror on all three remaining losses. This motivated the softer clamp and per-sample bbox probes that followed.

Follow-up fixed-ratio check: commit `bf6e688` added and ran `triposr_api_masked_repaired_stl_aspect_clamped3p0_direct_mesh` on the same Colab G4 runtime as `g4_stl_first_triposr_aspect_clamp3_s40_n10`. It completed and wrote `/content/g4_stl_first_triposr_aspect_clamp3_s40_n10_results.tar.gz` (`25,171,727` bytes), but was worse than the `2.25` cap:

| method | n | stl-quality score vs masked | paired wins vs mirror | Chamfer med | H95 med | bbox aspect med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `triposr_api_masked_repaired_stl_aspect_clamped2p25_direct_mesh` | 10 | 0.4597 | 7/10 | 0.1446 | 0.3782 | 2.2500 |
| `triposr_api_masked_repaired_stl_aspect_clamped3p0_direct_mesh` | 10 | 0.3353 | 6/10 | 0.1485 | 0.3939 | 2.3694 |
| `mirror` | 10 | 0.2199 |  | 0.1723 | 0.5056 | 1.7935 |

The looser cap reintroduced four losses: `mesh_dir_49534574e1_v00`, `mesh_dir_92752051d0_v00`, `mesh_dir_1a13fc2246_v00`, and `mesh_dir_757e06b221_v00`. This made a plain fixed-ratio sweep less promising and led to the per-sample mirror-relief and source-oracle bbox probes below.

The subsequent calibration probe used per-sample bbox matching. External direct-mesh command templates expose `{source_bbox_extents}` and `{mirror_bbox_extents}` plus per-axis `{source_bbox_x}`/`y`/`z` and `{mirror_bbox_x}`/`y`/`z` placeholders. `{source_bbox_extents}` is derived from the manifest mesh in the rendered camera frame and scaled to `--stl-target-dimension`; it is oracle-only. `{mirror_bbox_extents}` is read from an already-emitted sibling mirror STL when available and is now also exposed through the clearer deployable alias `{inferred_bbox_extents}`. The provider wrapper accepts `--mesh-target-bbox-extents X,Y,Z` and anisotropically scales the repaired provider mesh to those final STL-space extents before STL export.

`backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposr_source_bbox_candidate.json` is an oracle calibration probe for the same held-out 10 rows:

```bash
--mesh-repair printable --mesh-target-bbox-extents "{source_bbox_extents}"
```

This is not deployable because it uses the hidden source mesh. Its purpose is to answer whether per-sample geometry calibration can clear the remaining direct-mesh losses. If it passes, replace the oracle target with a deployable estimate from mirror/depth bbox statistics or multiview/video reconstruction metadata.

Colab G4 source-bbox oracle calibration result: the notebook terminal ran commit `e3b0941` as `g4_stl_first_triposr_source_bbox_s40_n10`, using the same packaged held-out manifest and isolated TripoSR API venv. It completed successfully and wrote `/content/g4_stl_first_triposr_source_bbox_s40_n10_results.tar.gz` (`26,375,237` bytes). The historical selector printed `promote` for the oracle-calibrated candidate, but that output predates the explicit oracle guard and must not be interpreted as a deployable promotion:

| method | n | stl-quality score vs masked | mesh surface Chamfer med | mesh surface H95 med | bbox aspect med | success rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `triposr_api_masked_repaired_stl_source_bbox_direct_mesh` | 10 | 0.9007 | 0.1483 | 0.3655 | 1.3638 | 1.0 |
| `mirror` | 10 | 0.2199 | 0.1723 | 0.5056 | 1.7935 | 1.0 |
| `masked` | 10 | 0.0000 | 0.1970 | 0.5226 | 1.7484 | 1.0 |

The historical metric gates reported `10/10` wins versus `masked` and `10/10` wins versus current `mirror`; paired CI95 low versus mirror was `0.5299`. This confirms that scale/proportion calibration explains a meaningful part of the direct-mesh gap, but the source-bbox probe uses hidden ground truth and is nondeployable by definition. Current selection logic marks it diagnostic and would hold an explicitly requested source-bbox candidate or skip it during automatic candidate selection. The deployable replacement is `{inferred_bbox_extents}`, derived from mirror/depth-relief STL geometry rather than the source mesh.

Historical deployable mirror-bbox calibration was wired for optimize sweeps. `optimize_completion` passes the sweep root to direct-mesh runs as `--direct-mesh-reference-output-dir`, so a candidate can resolve the previously emitted mirror STL at `<sweep>/mirror/<sample_id>/mirror/output_model.stl`. The config `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposr_mirror_bbox_candidate.json` uses:

```bash
--mesh-repair printable --mesh-target-bbox-extents "{mirror_bbox_extents}"
```

That config remains reproducible as the historical TripoSR follow-up to the source-bbox upper bound. For current TripoSG work, use `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_inferred_bbox_generated.json`; its `inferred` naming makes the deployable depth-relief provenance explicit and cannot be confused with the source oracle.

For fresh Colab runtimes, package TripoSR runs with `package_colab_inputs --include-run-script --include-triposr-setup` and Hunyuan3D shape runs with `--include-hunyuan3d-setup`. The generated `run_colab_eval.sh` embeds the provider repo clone, isolated venv creation, dependency install/probe, and `PYTHONPATH` setup before launching the benchmark, so deleting an idle G4 runtime no longer leaves the next direct-mesh run missing provider dependencies.

Multiview/video backends should use the new `external-multiview-to-mesh` method. The benchmark writes `{input_bundle}` as `multiview_input.json` beside the provider outputs, with `primary_image`, `masked_image`, `full_image`, `mask`, `camera`, optional `video_path`/`frames_dir`, and a `views` list containing sibling images, masks, cameras, and view ids. `generate_rendered_dataset --views-per-asset N` now annotates rows that share an `asset_key` with `multiview_images`, `multiview_masks`, `multiview_cameras`, `multiview_view_ids`, and `multiview_primary_index`; `package_colab_inputs` carries those list-valued paths into Colab bundles.

Next autonomous STL-first branch target: keep `mirror` depth-relief as the fast printable baseline, but compare it against three final-STL lanes instead of optimizing preview quality. Lane 1 is fast 2.5D relief STL from the current completed/depth image. Lane 2 is single-image mesh generation through repaired direct providers (`triposr-api`, Hunyuan3D shape, then another modern image-to-3D provider only if setup/licensing is clean). Lane 3 is video/multiview reconstruction through `external-multiview-to-mesh`, where selected frames, masks, cameras, and crops flow through the bundle contract before STL export. Gaussian splatting or NeRF-style reconstruction should only enter as an internal backend if it improves the final mesh/STL; promotion is still based on watertightness, manifoldness, positive volume, thickness/holes where measurable, surface Chamfer/Hausdorff against ground truth, mesh complexity, and visual/depth agreement.

Local STL-first launcher smoke (`stl_first_reconstruction_smoke_local_s0_n1`, `dataset-count=1`, `size=128`, `stl-target-dimension=64`) completed successfully in `82.91s` for the benchmark stage. Ranking:

| method | n | stl-quality score vs masked |
| --- | ---: | ---: |
| `source_mesh_oracle` | 1 | 1.4656 |
| `mirror` | 1 | 1.4008 |
| `biharmonic` | 1 | 0.0691 |
| `masked` | 1 | 0.0000 |

This is a launcher integration smoke, not a model-quality conclusion. It confirms that the STL-first orchestration path can generate data, emit relief STLs, evaluate direct mesh outputs, rank final STL artifacts, and write the expected summary/report files.

Repair-aware selector smoke (`stl_first_reconstruction_smoke_repaired_oracle_selection_local_s0_n1`) also completed locally with `--select-candidate --candidate-method mirror --current-method mirror --min-paired-n 1`. The benchmark stage took `80.89s`; `stl_first_summary.json` embedded `selection_decision`, and the decision was `keep_current` for `mirror` with no failed STL gates. The generated config recorded `source_mesh_oracle.source_mesh_repair=printable`, and the ranked output kept the smoke ordering: `source_mesh_oracle=1.4656`, `mirror=1.4008`, `biharmonic=0.0691`, `masked=0.0000`.

Cheap baseline command for the next STL-quality run:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_stl_quality_baseline_s40_n2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_baselines.json --start-index 40 --limit 2 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --score-mode baseline-delta --score-profile stl-quality --baseline-method masked --contact-sheet --contact-sheet-methods masked,mirror,biharmonic --contact-sheet-max-samples 2 --resume --continue-on-error
```

Local 3080 Ti smoke result (`modelnet10_60_balanced_stl_quality_baseline_s40_n2`, `start-index=40`, `limit=2`): `mirror` ranked first at `0.7766`, `biharmonic` second at `0.2333`, and `masked` stayed at `0.0000`. All three methods completed `2/2` samples and emitted STL meshes with median `stl_is_watertight=1`, `stl_is_volume=1`, `stl_winding_consistent=1`, `stl_single_component=1`, `stl_bbox_has_volume=1`, and `stl_positive_volume=1`. This confirms the fixed 2.5D relief exporter is now a valid STL baseline for the next direct image-to-mesh comparison.

### Pixal3D G4 STL sanity result

The first official Pixal3D STL-first comparison ran on the Colab G4 RTX PRO 6000 Blackwell runtime against the promoted TripoSG path and the existing depth-relief baselines. The final successful run was `g4_stl_first_pixal3d_vs_triposg_s40_n1_r4` on commit `c18fbde`, using packaged ModelNet row `40` (`mesh_dir_757e06b221_v00`), Pixal3D resolution `1024`, seed `42`, inferred deployable bbox calibration, and the `stl-quality` profile. The full structured record is [pixal3d-g4-s40-n1-r4.json](benchmark-evidence/pixal3d-g4-s40-n1-r4.json).

The setup uses the pinned official Pixal3D source commit and replaces the gated `briaai/RMBG-2.0` checkpoint from the published pipeline config with public `ZhengPeng7/BiRefNet`. Both Pixal methods consume the same content-addressed raw mesh cache entry (`1a1d35d...`), so raw-versus-repaired comparisons do not rerun diffusion or compare different stochastic meshes.

| method | n | STL-quality score vs masked | mesh Chamfer | mesh H95 | normalized face-density log1p | watertight / manifold / volume / one component |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pixal3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 1 | 1.4021 | 0.1822 | 0.4241 | 5.2001 | yes / yes / yes / yes |
| `source_mesh_oracle` | 1 | 1.0043 | 0.2464 | 0.4186 | 5.6857 | yes / yes / yes / yes |
| `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 1 | 0.8420 | 0.1264 | 0.3145 | 9.9499 | yes / yes / yes / yes |
| `biharmonic` | 1 | 0.0964 | 0.1604 | 0.4017 | 11.1779 | yes / yes / yes / yes |
| `mirror` | 1 | 0.0834 | 0.1617 | 0.4066 | 11.1662 | yes / yes / yes / yes |
| `masked` | 1 | 0.0000 | 0.1677 | 0.4153 | 11.1885 | yes / yes / yes / yes |
| `pixal3d_biharmonic_prefill_raw_direct_mesh` | 1 | -18.5190 | 0.1521 | 0.3655 | 14.7317 | no / no / no / no |

Failure-to-fix sequence:

- `r1` reached both provider preflights but Pixal3D failed on the gated background-removal model.
- `r2` generated the first real Pixal mesh with public BiRefNet. The raw output had `993,925` faces, `437` components, and `1,404` nonmanifold edges; printable repair was killed with exit `137`.
- `r3` reused the exact cached mesh and reduced it to `40,478` faces before topology work. The first safety guard rejected that bounded residual because it required the final `10,923`-face budget too early.
- `r4` accepted the residual under a `50,000`-face topology ceiling, selected the largest repaired component, applied printable fallback and final complexity processing, and emitted a watertight, manifold, positive-volume, single-component STL. The result archive was `121,229,396` bytes with SHA256 `2116aeace7c7ce8597d4c1193f2bd6be320dafda8b2506c7516d92a5f59dbc30`.

The original selector reported `promote` with no failed STL gates, but this is a one-row sanity result, not a model-level promotion. TripoSG still has substantially better surface accuracy on this sample (`0.1264` versus `0.1822` Chamfer and `0.3145` versus `0.4241` H95). The follow-up selector now applies a `1.10` worst-paired-sample ratio ceiling for both metrics. Replaying the recorded values produces `hold`: Pixal3D is `1.4418x` TripoSG's Chamfer and `1.3485x` its H95. This prevents a coarse printable hull from winning mainly because it is cheap.

The next-run setup also converts the four observed Hugging Face revisions into executable pins. It resolves Pixal3D, MoGe-2, DINOv3, and BiRefNet at their exact recorded SHAs, validates their required snapshot files, passes local paths into the patched official launcher (`model.pt` for MoGe), switches Hugging Face loading offline after resolution, and includes canonical repo IDs plus revisions in provider preflight, package reports, and raw-mesh cache identity. Changing any one revision invalidates the cache key.

The TripoSG control arm is immutable in the same run: `VAST-AI/TripoSG` is pinned to `2c1c516d22d58db486a058d98d31bb6177344e06` and `briaai/RMBG-1.4` to `2ceba5a5efaec153162aedea169f76caf9b46cf8`. Setup prefetches those revisions into the official local directories, and the patched launcher requires the same revisions at inference time. This prevents a moving incumbent checkpoint from contaminating the Pixal3D comparison.

Runtime retention note: the notebook preserved the complete result summary and archive digest, but the idle-cost automation disconnected the runtime before the archive/contact sheet could be copied locally. That automation is paused for the next measured batch so compact evidence can be captured before deliberate disconnect.

### Pixal3D G4 five-sample result

The pinned follow-up completed on the same G4 as `g4_stl_first_pixal3d_vs_triposg_s40_n5_r5` at commit `a19def7`. It evaluated five packaged ModelNet rows, seeded both direct providers with `42`, reused content-addressed Pixal3D and TripoSG geometry caches, and restored the depth baselines after scoping Hugging Face offline mode to Pixal-only processes. The complete machine-readable trail is [pixal3d-g4-s40-n5-r1-r5.json](benchmark-evidence/pixal3d-g4-s40-n5-r1-r5.json).

| method | n | STL-quality score vs masked | mesh Chamfer med | mesh H95 med | bbox aspect med | complexity log1p med | printable samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 5 | 2.8629 | 0.0510 | 0.2687 | 1.4115 | 5.6857 | 5/5 (diagnostic only) |
| `pixal3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 5 | 1.2783 | 0.1822 | 0.4822 | 2.0006 | 5.4596 | 5/5 |
| `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 5 | 0.9521 | 0.1264 | 0.3492 | 2.0006 | 9.5266 | 5/5 |
| `biharmonic` | 5 | 0.1674 | 0.1604 | 0.4186 | 1.9407 | 11.1779 | 5/5 |
| `mirror` | 5 | 0.0394 | 0.1568 | 0.4698 | 2.0006 | 11.2083 | 5/5 |
| `masked` | 5 | 0.0000 | 0.1748 | 0.4505 | 1.9613 | 11.1885 | 5/5 |
| `pixal3d_biharmonic_prefill_raw_direct_mesh` | 5 | -17.9783 | 0.1232 | 0.3276 | 1.5988 | 14.4826 | 0/5 |

The final decision is `hold`, despite Pixal3D's higher composite score and complete printability. Against the TripoSG incumbent, Pixal3D won only `3/5` paired objectives, its CI95 lower bound was `-0.2112`, and every paired surface-regression gate failed. Median paired Chamfer and H95 ratios were `1.4415x` and `1.3600x`; worst-sample ratios were `1.5485x` and `2.4093x`, above the `1.10x` ceiling. The selector is doing the intended job: low polygon count and watertightness cannot compensate for losing the hidden-side surface.

The repair investigation used measured limits rather than repeatedly raising an arbitrary ceiling:

| run | change | Pixal repaired coverage | measured outcome |
| --- | --- | ---: | --- |
| `r1` | initial five-row package | 5/5 | invalid comparison because Pixal offline mode poisoned depth baselines; old repair also produced distorted proportions |
| `r2` | provider-scoped offline mode, 50k ceiling | 1/5 | one residual stopped at `51,234` faces, then the one-failure budget skipped three rows |
| `r3` | 64k ceiling | 2/5 | next residual stopped at `64,704`; two rows remained unobserved |
| `r4` | failure budget raised to five, no repair change | 2/5 | all blocked residuals measured at `64,704`, `68,336`, and `103,036` faces |
| `r5` | bounded 131,072-face pre-repair ceiling | 5/5 | no OOM; all final STLs passed hard printability gates |

The 131,072-face ceiling still reduces Pixal's `964,623-998,737` raw faces by at least `7.6x` before topology work. It is a resource guard, not a quality target. The printable fallback emits only `40-128` final faces on these samples, which explains why the raw Pixal meshes have competitive surface metrics while the repaired meshes regress. The next Pixal experiment should keep these exact cached raw meshes and compare shape-preserving component filtering and hole closing against the current coarse fallback; promotion thresholds should not be relaxed.

All five compact evidence bundles were copied locally before disconnect, checksum-verified, and contain the result summary, selection decision, per-sample metrics, provider preflight, contact sheet, failure logs, and run-log tail. The final full archive was `655,742,061` bytes with SHA256 `c75c59822fbcecce3224e20420d1ef655f917bf6457f52e311d80d4e59512e09`; the final compact bundle was `1,254,954` bytes with SHA256 `8bac48828f71d49be6b48e15dfa6380a4801b3a5d5a42f1c2dae8593f3f317de`.

### TRELLIS.2 G4 five-sample result

The pinned TRELLIS.2 lane completed on the same five held-out ModelNet rows as `g4_stl_first_trellis2_vs_triposg_s40_n5_r3` at commit `6acebcd`. It used the exact source revision `75fbf0183001ed9876c8dbb35de6b68552ee08bd`, the pinned `microsoft/TRELLIS.2-4B` and sparse-structure snapshots, public exact-revision DINOv3 and BiRefNet substitutes for gated references, and a validated `xformers` attention backend. This is a compatibility-adapted pinned run, not an unmodified official pipeline. The complete machine-readable trail is [trellis2-g4-s40-n1-r3-n5-r1-r3.json](benchmark-evidence/trellis2-g4-s40-n1-r3-n5-r1-r3.json).

| method | n | STL-quality score vs masked | mesh Chamfer med | mesh H95 med | complexity log1p med | printable samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_mesh_oracle` | 5 | 2.8629 | 0.0510 | 0.2687 | 5.6857 | 5/5 (diagnostic only) |
| `trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 5 | 1.3779 | 0.1676 | 0.3595 | 6.5909 | 5/5 |
| `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` | 5 | 0.9521 | 0.1264 | 0.3492 | 9.5266 | 5/5 |
| `biharmonic` | 5 | 0.1674 | 0.1604 | 0.4186 | 11.1779 | 5/5 |
| `mirror` | 5 | 0.0394 | 0.1568 | 0.4698 | 11.2083 | 5/5 |
| `masked` | 5 | 0.0000 | 0.1748 | 0.4505 | 11.1885 | 5/5 |
| `trellis2_biharmonic_prefill_raw_direct_mesh` | 5 | -26.4387 | 0.2097 | 0.4543 | 16.4081 | 0/5 |

The reliability work was measured in three five-row passes:

| run | change | repaired TRELLIS coverage | measured outcome |
| --- | --- | ---: | --- |
| `n5_r1` | first pinned batch | 3/5 | one empty sparse structure and one `2,768,786`-face mesh above the safe repair limit |
| `n5_r2` | one deterministic empty-structure retry plus bounded fast component filtering | 5/5 | retry seed `43` recovered the empty sample; all five STLs passed hard gates |
| `n5_r3` | volume-first fast component ranking and timing/sidecar integrity fixes | 5/5 | corrected the destructive filter policy and kept legacy cached inference timing unknown |

The final decision remains `hold`, despite the higher composite score and `4/5` paired objective wins against TripoSG. The paired CI95 lower bound was positive (`0.1693`), but the selector also protects worst-sample surface quality: TRELLIS-to-TripoSG worst ratios were `1.3447x` for Chamfer and `1.2580x` for H95, both above the `1.10x` ceiling. Median paired ratios were `1.0583x` and `0.9814x`, so the issue is localized rather than a blanket failure; the next repair experiment should target those outliers rather than weaken the gate.

Raw TRELLIS meshes ranged from `550,804` to `2,768,786` faces and `319` to `133,533` components, and none were printable. Volume-first repair reduced the dense desk sample to a `48`-face printable body, but its H95 rose to `0.5487`, illustrating the same tradeoff seen with Pixal3D: closing a fragmented generative mesh can discard hidden-side shape. The next TRELLIS iteration should compare shape-preserving component closure against this fallback on the exact cached raw meshes before expanding the sample count.

Runtime evidence is intentionally qualified. TRELLIS generation was measured on the cache-producing pass at a median `55.5545s`; final-run repair was `3.5903s` median. TripoSG used a legacy cache whose metadata did not contain inference duration, so its final-run inference field is blank rather than the cache lookup time. Peak child-process VRAM and self-intersection counts are also explicitly unsupported, not zero. This dataset has one view per asset, so held-out view agreement is unavailable; Chamfer/H95 against the hidden source mesh remain the relevant surface checks, with the source geometry kept diagnostic-only.

The final package was `649,643` bytes with SHA256 `f541a97f0fddab2b70cdf26b1321e8698cf9e089a6fa01718bd741f2b252575d`. The full result archive was `290,340,713` bytes with SHA256 `61ba9e27746543e7cd0d2b7d182746b2980cd02de9ecef359f8889f470cb70ef`; the locally verified 150-file compact bundle was `1,905,521` bytes with SHA256 `a87d95b0408ea140154493c819c40100eb0f079ec067fcfc1fdc6af2708129a6`.

## Kaggle

A GPU-enabled Kaggle kernel scaffold lives in `backend/benchmark/kaggle`.

```bash
.\backend\.venv\Scripts\kaggle kernels push -p backend/benchmark/kaggle
```

Useful environment variables:

- `THREEDPRINTPIC_REPO`: Git URL or Kaggle input checkout path.
- `THREEDPRINTPIC_REF`: branch, tag, or SHA to fetch and check out.
- `COMPLETION_DATASET`: `synthetic`, `rendered-procedural`, or `mesh-dir`.
- `COMPLETION_METHODS`: for example `mirror,biharmonic,amused-inpaint,dreamshaper-inpaint`.
- `COMPLETION_BENCHMARK_COUNT`: sample count.
- `COMPLETION_START_INDEX`: zero-based manifest row offset for benchmark/eval.
- `COMPLETION_SKIP_DEPTH`: `1` for cheap image-only runs, `0` for depth metrics.
- `COMPLETION_ASSET_ROOT`: required for `mesh-dir`.
- `COMPLETION_ASSET_GLOB`: for example `**/*.glb` or `**/*.off`.
- `COMPLETION_VIEWS_PER_ASSET`: views per asset for `mesh-dir`.
- `COMPLETION_TRAIN_LORA`: `1` to export pairs, train a LoRA adapter, and evaluate it with `optimize_completion`.
- `COMPLETION_TRAIN_LORA_BASE_MODEL`: default `Lykon/dreamshaper-8-inpainting`.
- `COMPLETION_TRAIN_LORA_METHOD`: default `dreamshaper-inpaint`.
- `COMPLETION_TRAIN_LORA_STEPS`: training steps for the adapter.
- `COMPLETION_TRAIN_LORA_LIMIT`: pair count used for LoRA training; defaults to `COMPLETION_BENCHMARK_COUNT`.
- `COMPLETION_TRAIN_LORA_START_INDEX`: zero-based train slice start; default `0`.
- `COMPLETION_TRAIN_LORA_EVAL_START_INDEX`: zero-based held-out eval slice start; default `COMPLETION_START_INDEX`.
- `COMPLETION_TRAIN_LORA_RESOLUTION`: default `256`.
- `COMPLETION_TRAIN_LORA_BATCH_SIZE`: default `1`.
- `COMPLETION_TRAIN_LORA_GRAD_ACCUM`: default `4`.

Examples:

```bash
THREEDPRINTPIC_REPO=https://github.com/<you>/3dprintpic.git THREEDPRINTPIC_REF=<branch-or-sha> python run_kaggle_benchmark.py
COMPLETION_DATASET=rendered-procedural python run_kaggle_benchmark.py
COMPLETION_DATASET=mesh-dir COMPLETION_ASSET_ROOT=/kaggle/input/meshes COMPLETION_ASSET_GLOB="**/*.glb" COMPLETION_VIEWS_PER_ASSET=3 python run_kaggle_benchmark.py
COMPLETION_SKIP_DEPTH=0 python run_kaggle_benchmark.py
COMPLETION_METHODS=amused-inpaint,dreamshaper-inpaint python run_kaggle_benchmark.py
COMPLETION_DATASET=mesh-dir COMPLETION_ASSET_ROOT=/kaggle/input/modelnet10 COMPLETION_ASSET_GLOB="**/*.off" COMPLETION_TRAIN_LORA=1 COMPLETION_TRAIN_LORA_STEPS=200 python run_kaggle_benchmark.py
```

Default behavior stays cheap: synthetic dataset, `mirror,biharmonic`, and depth skipped.

## Dataset Direction

Recommended asset sources:

| dataset | why | command / note |
| --- | --- | --- |
| AI Habitat YCB | small real-object default; optimized GLBs, CC-BY-4.0 | `hf download ai-habitat/ycb --repo-type dataset --local-dir data/ycb` |
| Google Scanned Objects | realistic scanned household objects, CC-BY-4.0 | larger; use after the renderer path is stable |
| Objaverse filtered subset | diverse assets; per-object licenses | filter to permissive licenses and selected UIDs |
| Thingi10K | 3D-print-relevant STL stress test | filter hard for manifold/permissive objects |
| ModelNet10 on Kaggle | easy CAD `.off` sanity set | `kaggle datasets download -d balraj98/modelnet10-princeton-3d-object-dataset -p data/modelnet10 --unzip` |

Kaggle search found `sanvik74/shapenet-core-v2`, but it is about 27.7 GB and has low usability metadata. Treat it as a later dataset, not the first smoke test.
