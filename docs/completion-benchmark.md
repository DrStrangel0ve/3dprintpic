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
.\backend\.venv\Scripts\python -m backend.benchmark.preflight_modern_providers --output backend/output/completion-benchmark/modern_provider_preflight_v4.jsonl
```

Current preflight results:

| provider | model | gated | license | approx size |
| --- | --- | --- | --- | ---: |
| `sdxl-inpaint` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | false | openrail++ | 19.39 GB total; fp16 cache plan is 6.47 GB |
| `dreamshaper-inpaint` | `Lykon/dreamshaper-8-inpainting` | false | creativeml-openrail-m | 7.66 GB total; fp16 cache plan is 1.99 GB and is cached locally |
| `amused-inpaint` | `amused/amused-512` | false | openrail++ | 4.93 GB total; fp16 cache plan is 1.64 GB and is cached locally |
| `flux-fill` | `black-forest-labs/FLUX.1-Fill-dev` | `auto`; model index gated without auth | other | 54.07 GB |
| `qwen-image-inpaint` | `Qwen/Qwen-Image-Edit` | false; uses non-advertised inpaint pipeline override | Apache-2.0 | 53.76 GB |
| `qwen-image-edit` | `Qwen/Qwen-Image-Edit` | false; advertised `QwenImageEditPipeline` | Apache-2.0 | 53.76 GB |

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
- optional `stl_exists`, `stl_is_watertight`, `stl_positive_volume`, `stl_faces`, `stl_z_range`

Depth metrics align predicted relative depth to ground truth using scale + shift fit on the visible half. Object-depth metrics restrict fit/evaluation to saved silhouette pixels, which prevents background from dominating the score.
Surface metrics convert aligned predicted and ground-truth depth into normalized camera-view `(x, y, z)` point clouds over the hidden half. `surface_chamfer_l1` is symmetric nearest-neighbor distance, `surface_chamfer_rmse` is the bidirectional nearest-neighbor RMS distance, `surface_rmse` is same-pixel depth-surface RMSE, and `surface_hausdorff95` is a robust 95th-percentile Hausdorff distance. The `object_surface_*` variants restrict the comparison to hidden pixels inside the saved object silhouette, so the rank objective can penalize a completion that looks acceptable in 2D but produces the wrong 3D relief surface.

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

Current coverage includes minimal height-field STL export, positive Trimesh signed-volume orientation, no-valid-cell failure behavior, and `optimize_completion` per-sample CSV experiment labeling/idempotency. This suite is intentionally small; use it as a fast guardrail before heavier image/depth/STL benchmarks.

## Optimization Loop

Rank a run:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --baseline-method masked
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv --score-mode baseline-delta --baseline-method masked --output backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/ranked_methods_masked_delta.csv
.\backend\.venv\Scripts\python -m backend.benchmark.report_run backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected --baseline-method masked --score-mode baseline-delta --output backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/benchmark_report_masked_delta.md
.\backend\.venv\Scripts\python -m backend.benchmark.rank_methods --summary backend/output/completion-benchmark/runs/rendered_procedural_v3_corrected/summary_metrics.csv --score-mode baseline-delta --baseline-method masked --score-profile object-surface
```

The default rank objective rewards masked PSNR/SSIM, object-depth correlation, silhouette IoU, STL existence, STL watertightness, and positive STL volume; it penalizes masked MAE, seam MAE, visible drift before preservation, object-depth error, and depth-derived 3D surface distance. Use `--score-profile object-surface` when the question is specifically printable object geometry: it ignores RGB-only completion metrics and weights object depth, object surface Chamfer/RMSE/Hausdorff95, silhouette IoU, and STL validity. `rank_methods`, `report_run`, `optimize_completion`, `select_completion_candidate`, `compare_optimize_runs`, and `analyze_experiment_slices` all accept the same profile name, with explicit `--weight metric=value` overrides applied on top. Summaries also include `attempted_n`, `error_count`, `logged_failure_count`, and `success_rate` so rankings produced with `--continue-on-error` reveal whether methods were scored on equal coverage. `report_run` writes a Markdown audit trail with rankings, coverage, metric medians, selected baseline deltas, paired baseline wins, paired objective confidence, failures, and example artifacts. The default `--score-mode normalized` uses min/max normalization within the candidate set, which is useful for ordering one run but makes numeric scores move when candidates are added or removed. Use `--score-mode baseline-delta --baseline-method masked` for a candidate-set-stable score: the baseline gets `0`, positive scores are weighted sign-normalized metric improvements, and negative scores regress the objective. The baseline-delta section still reports absolute movement in selected diagnostic medians. `Delta` is raw `method - baseline`; `Improvement` is sign-normalized so positive means better, even when the raw metric is lower-is-better. The paired-wins section compares only shared finite `sample_id` rows, reports `Baseline n`, `Method n`, `Common n`, finite `Paired n`, and skipped non-finite pairs; `Win Rate` is strict wins divided by finite paired n, with ties excluded from wins. The paired-objective confidence section applies the same metric weights to raw per-sample improvements over the baseline and bootstraps the mean improvement; this is intentionally not the same as candidate-normalized aggregate rank. Treat a CI entirely above `0` as a stronger promotion signal than an aggregate rank alone.

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

Each sweep now writes `resolved_experiments.json` and `experiment_report.md` next to `aggregate_summary.csv` and `ranked_experiments.csv`, so runs are self-documenting. For LoRA candidates, the resolved config and report expand `training_report.json` into a Training Recipes section with loss recipe, train rows/assets, steps, final/best loss, prompt family, metadata hash, and adapter hash. Add `--emit-stl` without `--skip-depth` to rank the full completion-to-depth-to-STL path from the same experiment config. Add `--score-mode baseline-delta --baseline-method masked` when a sweep includes `masked` and you want stable optimization scores across experiments. Add `--select-candidate --current-method <method>` to make the sweep also write `selection_decision.json` and `selection_decision.md`.

Turn a sweep into an explicit promotion decision:

```bash
.\backend\.venv\Scripts\python -m backend.benchmark.optimize_completion --manifest backend/output/completion-benchmark/modelnet10_60_balanced_s256_seed4040/manifest.jsonl --output-dir backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --config backend/benchmark/experiment_configs/modelnet10_60_balanced_lora_depth_stl_top_sweep.json --start-index 40 --limit 10 --depth-provider depth-anything-v2 --depth-model depth-anything/Depth-Anything-V2-Small-hf --device auto --emit-stl --stl-target-dimension 96 --resume --continue-on-error --score-mode baseline-delta --baseline-method masked --select-candidate --current-method mirror
.\backend\.venv\Scripts\python -m backend.benchmark.select_completion_candidate backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --baseline-method masked --current-method mirror
.\backend\.venv\Scripts\python -m backend.benchmark.select_completion_candidate backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2 --baseline-method masked --current-method mirror --candidate-method dreamshaper_lora_balanced_train40_mask3_seam4_s200_scale050_s256 --output-json backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/selection_decision_lora_scale050_vs_mirror.json --output-md backend/output/completion-benchmark/experiments/modelnet10_60_balanced_heldout10_depth_stl_top_sweep_v2/selection_decision_lora_scale050_vs_mirror.md
```

The selector writes strict `selection_decision.json` and `selection_decision.md`, either from `optimize_completion --select-candidate` or from the standalone `select_completion_candidate` command. A candidate must pass success-rate, paired sample count, paired win-rate, positive bootstrap CI, STL watertight/positive-volume, train/eval overlap, and score-margin gates before it is promoted. LoRA candidates must also carry a `training_report.json`; when split-audit provenance includes a prompt-family mismatch, the selector fails closed instead of promoting an adapter that was trained and evaluated under different prompt assumptions. When `--current-method` is supplied and the candidate is not already current, the selector also recomputes paired objective confidence with the current method as the baseline; this prevents a method that only beats the lower-bound `masked` baseline from replacing a stronger incumbent. If `--current-method` is supplied but missing from the summary, the selector fails closed with `hold`; if the top candidate is already the current method and all gates pass, the decision is `keep_current`. `split_audit.json` is required by default because promotion should come from a held-out check; pass `--allow-missing-split-audit` only for quick smoke runs that are not making a default-selection claim.

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

Local regression coverage for the G4 resume lane:

```powershell
.\backend\.venv\Scripts\python -m unittest backend.tests.test_benchmark_regressions.ColabG4OrchestratorRegressionTests -v
```

These tests dry-run `eval,combine` with an existing LoRA, combine-only reuse of completed eval directories, and fail-fast missing manifest/adapter validation. This keeps the notebook workflow recoverable after interruptions without accidentally retraining, changing existing-LoRA labels, or silently combining incomplete slices.

To move local rendered ModelNet assets and a trained adapter onto Colab without committing generated artifacts, package a Colab-ready bundle:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.package_colab_inputs --manifest backend\output\completion-benchmark\modelnet10_60_balanced_s256_seed4040\manifest.jsonl --lora-weights backend\output\completion-benchmark\lora\modelnet10_train40_weighted_surface_s20 --output C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz --extract-root /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20 --include-run-script --run-script C:\Users\arnav\Documents\Codex\2026-07-09\jennyzzt-3dprintpic-https-github-com-jennyzzt\outputs\modelnet10_60_weighted_surface_s20_run_colab_eval.sh --colab-archive-path /content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz --run-name g4_modelnet10_weighted_surface_eval_s20 --eval-start 40 --eval-start 50 --eval-limit 10 --score-profile object-surface --train-steps 20
```

The bundle rewrites manifest paths to the chosen extraction root and includes the adapter directory plus training report. The current package report is `modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz.report.json`: `60` rows, `420` referenced dataset files, `423` payload files plus the rewritten manifest and `run_colab_eval.sh`, and SHA256 `ff0528099fe0ec8c1d1f0f7801a30a8a7a24d7af9557c1fc5203f43453ec0e93`.

After uploading the tarball to `/content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz` in Colab, extract the launcher and run the real ModelNet10 weighted-LoRA held-out gate with:

```bash
mkdir -p /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20
tar -xzf /content/modelnet10_60_weighted_surface_s20_colab_inputs.tar.gz -C /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20 run_colab_eval.sh
EXPECTED_SHA256=ff0528099fe0ec8c1d1f0f7801a30a8a7a24d7af9557c1fc5203f43453ec0e93 bash /content/3dprintpic_colab_inputs/modelnet10_60_weighted_surface_s20/run_colab_eval.sh
```

The launcher verifies `EXPECTED_SHA256` when set, validates the rewritten manifest, adapter weights, and training report, clones `REPO_DIR` if a fresh Colab runtime does not already have the repo, then writes `launch_preflight.json` with the archive SHA, resolved git commit, manifest row count, and adapter path before model cache/eval work begins.

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
