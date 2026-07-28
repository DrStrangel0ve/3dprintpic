# Experiment Report: modern_weighted_eval_s0_n1

- Generated: `2026-07-12T20:11:38+00:00`
- Manifest: `/content/3dprintpic_colab_inputs/inputs/manifest.jsonl`
- Limit: `1`
- Aggregate summary: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_trellis2_component_close_guard_s40_n1/experiments/modern_weighted_eval_s0_n1/aggregate_summary.csv`
- Ranked output: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_trellis2_component_close_guard_s40_n1/experiments/modern_weighted_eval_s0_n1/ranked_experiments.csv`
- Resolved config: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_trellis2_component_close_guard_s40_n1/experiments/modern_weighted_eval_s0_n1/resolved_experiments.json`
- Contact sheet: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_trellis2_component_close_guard_s40_n1/experiments/modern_weighted_eval_s0_n1/artifact_contact_sheet.png`

## Ranking

_Score profile: `stl-quality` (final STL mesh accuracy, validity, and printable complexity metrics)._

_Score mode: weighted sign-normalized metric improvement against `masked`. The baseline scores `0`; positive scores improve the objective and negative scores regress it. Scores are stable when unrelated candidate methods are added._

| Rank | Method | Score | n | Attempted | Success Rate | Errors | Masked MAE | Object MAE | Masked PSNR | Object PSNR | Masked SSIM | Object SSIM | Seam MAE | Object Depth MAE | Object Depth Corr | Object Surface Chamfer | Object Surface Chamfer RMSE | Object Surface Hausdorff95 | Mesh Surface Chamfer | Mesh Surface Chamfer RMSE | Mesh Surface Hausdorff95 | Inferred BBox Shape log-MAE | Inferred BBox Shape Rel-MAE | Inferred BBox Centered IoU | Silhouette IoU | STL Watertight | STL Volume Mesh | STL Winding | STL Positive Volume | STL Single Body | STL Bodies | STL Body Excess | STL Body Excess log1p | STL Min Dimension | STL 3D BBox | STL Aspect | STL Scale-Free Complexity log1p | STL Face Density log1p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mirror | 0.0217 | 1 | 1 | 1 | 0 | 0.1243 | 0.2252 | 11.5526 | 9.0123 | 0.7213 | 0.571 | 0.0152 | 0.3863 | -0.2241 | 0.1564 | 0.169 | 0.2743 | 0.1617 | 0.1996 | 0.4066 |  |  |  | 0.5099 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 49.5304 | 1 | 1.918 | 11.1662 | 0.0792 |
| 2 | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 | 1 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.1264 | 0.1565 | 0.3145 | 0.2198 | 0.1722 | 0.5328 |  | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 49.5304 | 1 | 1.918 | 9.9499 | 0.0241 |
| 3 | trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 | 1 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.1679 | 0.1961 | 0.3944 | 0.2198 | 0.1722 | 0.5328 |  | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 49.5304 | 1 | 1.918 | 5.553 | 3.00e-04 |
| 4 | masked | 0 | 1 | 1 | 1 | 0 | 0.3578 | 0.6363 | 6.6704 | 3.7954 | 0.7017 | 0.5092 | 0.2663 | 0.1746 | -0.0168 | 0.1203 | 0.1511 | 0.3701 | 0.1677 | 0.2044 | 0.4153 |  |  |  | 0.5099 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 48.4369 | 1 | 1.9613 | 11.1885 | 0.081 |
| 5 | trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | 0 | 0 | 1 | 0 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 6 | trellis2_biharmonic_prefill_raw_direct_mesh | -23.0239 | 1 | 1 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.2094 | 0.2342 | 0.4305 | 0.2198 | 0.1722 | 0.5328 |  | 0 | 0 | 0 | 0 | 0 | 632 | 631 | 6.4489 | 0.2674 | 1 | 3.7396 | 15.897 | 15.897 |

_Coverage warning: ranked methods have unequal success rates; compare medians with care._

## Selected Baseline Deltas: `masked`

Raw `Delta` is `method - baseline`; `Improvement` is sign-normalized so positive means better. These are aggregate median deltas for selected diagnostic metrics, not weighted rank points or paired per-sample wins.

| Method | Metric | Value | masked | Delta | Improvement |
| --- | --- | --- | --- | --- | --- |
| mirror | Depth Corr | 0.5915 | -0.5052 | 1.0966 | 1.0966 |
| mirror | Depth MAE | 0.2542 | 0.4441 | -0.1899 | 0.1899 |
| mirror | Masked MAE | 0.1243 | 0.3578 | -0.2335 | 0.2335 |
| mirror | Masked PSNR | 11.5526 | 6.6704 | 4.8821 | 4.8821 |
| mirror | Masked SSIM | 0.7213 | 0.7017 | 0.0196 | 0.0196 |
| mirror | Mesh Surface Chamfer | 0.1617 | 0.1677 | -0.006 | 0.006 |
| mirror | Mesh Surface Chamfer RMSE | 0.1996 | 0.2044 | -0.0049 | 0.0049 |
| mirror | Mesh Surface Hausdorff95 | 0.4066 | 0.4153 | -0.0088 | 0.0088 |
| mirror | Object Depth Corr | -0.2241 | -0.0168 | -0.2073 | -0.2073 |
| mirror | Object Depth MAE | 0.3863 | 0.1746 | 0.2118 | -0.2118 |
| mirror | Object MAE | 0.2252 | 0.6363 | -0.4111 | 0.4111 |
| mirror | Object PSNR | 9.0123 | 3.7954 | 5.2169 | 5.2169 |
| mirror | Object SSIM | 0.571 | 0.5092 | 0.0619 | 0.0619 |
| mirror | Object Surface Chamfer | 0.1564 | 0.1203 | 0.0361 | -0.0361 |
| mirror | Object Surface Chamfer RMSE | 0.169 | 0.1511 | 0.0179 | -0.0179 |
| mirror | Object Surface Hausdorff95 | 0.2743 | 0.3701 | -0.0958 | 0.0958 |
| mirror | Seam MAE | 0.0152 | 0.2663 | -0.2511 | 0.2511 |
| mirror | Silhouette IoU | 0.5099 | 0.5099 | 0 | 0 |
| mirror | Surface Chamfer | 0.1304 | 0.2638 | -0.1334 | 0.1334 |
| mirror | Surface Chamfer RMSE | 0.1913 | 0.3475 | -0.1562 | 0.1562 |
| mirror | Surface Hausdorff95 | 0.5078 | 0.6838 | -0.1761 | 0.1761 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Chamfer | 0.2094 | 0.1677 | 0.0418 | -0.0418 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Chamfer RMSE | 0.2342 | 0.2044 | 0.0298 | -0.0298 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Hausdorff95 | 0.4305 | 0.4153 | 0.0152 | -0.0152 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 0.1679 | 0.1677 | 1.87e-04 | -1.87e-04 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 0.1961 | 0.2044 | -0.0083 | 0.0083 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 0.3944 | 0.4153 | -0.0209 | 0.0209 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 0.1264 | 0.1677 | -0.0413 | 0.0413 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 0.1565 | 0.2044 | -0.0479 | 0.0479 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 0.3145 | 0.4153 | -0.1008 | 0.1008 |

## Paired Baseline Wins: `masked`

Each row compares a method against the baseline on the same `sample_id`. `Wins` count strictly positive sign-normalized improvements; ties are reported separately.

| Method | Metric | Baseline n | Method n | Common n | Paired n | Skipped | Wins | Win Rate | Ties | Median Delta | Median Improvement | Mean Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mirror | Depth Corr | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 1.0966 | 1.0966 | 1.0966 |
| mirror | Depth MAE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.1899 | 0.1899 | 0.1899 |
| mirror | Masked MAE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.2335 | 0.2335 | 0.2335 |
| mirror | Masked PSNR | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 4.8821 | 4.8821 | 4.8821 |
| mirror | Masked SSIM | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 0.0196 | 0.0196 | 0.0196 |
| mirror | Mesh Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.006 | 0.006 | 0.006 |
| mirror | Mesh Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0049 | 0.0049 | 0.0049 |
| mirror | Mesh Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0088 | 0.0088 | 0.0088 |
| mirror | Object Depth Corr | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | -0.2073 | -0.2073 | -0.2073 |
| mirror | Object Depth MAE | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.2118 | -0.2118 | -0.2118 |
| mirror | Object MAE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.4111 | 0.4111 | 0.4111 |
| mirror | Object PSNR | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 5.2169 | 5.2169 | 5.2169 |
| mirror | Object SSIM | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 0.0619 | 0.0619 | 0.0619 |
| mirror | Object Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.0361 | -0.0361 | -0.0361 |
| mirror | Object Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.0179 | -0.0179 | -0.0179 |
| mirror | Object Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0958 | 0.0958 | 0.0958 |
| mirror | Seam MAE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.2511 | 0.2511 | 0.2511 |
| mirror | Silhouette IoU | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 1 | 0 | 0 | 0 |
| mirror | Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.1334 | 0.1334 | 0.1334 |
| mirror | Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.1562 | 0.1562 | 0.1562 |
| mirror | Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.1761 | 0.1761 | 0.1761 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.0418 | -0.0418 | -0.0418 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.0298 | -0.0298 | -0.0298 |
| trellis2_biharmonic_prefill_raw_direct_mesh | Mesh Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 0.0152 | -0.0152 | -0.0152 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | 1.87e-04 | -1.87e-04 | -1.87e-04 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0083 | 0.0083 | 0.0083 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0209 | 0.0209 | 0.0209 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0413 | 0.0413 | 0.0413 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.0479 | 0.0479 | 0.0479 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | -0.1008 | 0.1008 | 0.1008 |

## Paired Objective Confidence: `masked`

Each row applies the same metric weights used for ranking to each shared `sample_id`, but as raw sign-normalized improvement over the baseline rather than candidate-normalized aggregate medians. The confidence interval bootstraps paired sample improvements with a fixed seed; if the CI is entirely above `0`, the method has a stronger promotion signal than an aggregate median alone.

| Method | Baseline n | Method n | Common n | Paired n | Skipped | Wins | Win Rate | Ties | Median Objective Improvement | Mean Objective Improvement | Mean CI95 Low | Mean CI95 High | Mean Metrics/Sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mirror | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 0.0217 | 0.0217 | 0.0217 | 0.0217 | 13 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 0.0217 | 0.0217 | 0.0217 | 0.0217 | 12 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 0.0217 | 0.0217 | 0.0217 | 0.0217 | 12 |
| trellis2_biharmonic_prefill_raw_direct_mesh | 1 | 1 | 1 | 1 | 0 | 0/1 | 0 | 0 | -23.0239 | -23.0239 | -23.0239 | -23.0239 | 12 |

## Experiment Config

| Name | Base Method | STL Mode | Steps | Guidance | Seed | Max Dim | Edit Fill | Model | LoRA | LoRA Scale | Start | Skip Depth | Emit STL | Mesh Samples | Direct Input | Direct Ext | Direct Timeout | Direct Ref Root | Direct BBox Source | Oracle Diagnostic | Direct Ref Method | Source Repair | Direct Command | Prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked | masked | depth-relief | 24 |  | 1234 | 768 | input |  |  |  | 0 | False | True | 4096 | masked | glb | 1800 |  | none | False | mirror | none |  | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| mirror | mirror | depth-relief | 24 |  | 1234 | 768 | input |  |  |  | 0 | False | True | 4096 | masked | glb | 1800 |  | none | False | mirror | none |  |  |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | external-image-to-mesh | single-image-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | biharmonic | glb | 3600 |  | inferred | False | mirror | none | /content/triposg-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider triposg --input-image "{inp... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| trellis2_biharmonic_prefill_raw_direct_mesh | external-image-to-mesh | single-image-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | biharmonic | glb | 3600 |  | none | False | mirror | none | /content/trellis2-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider trellis2 --input-image "{i... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | external-image-to-mesh | single-image-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | biharmonic | glb | 3600 |  | inferred | False | mirror | none | /content/trellis2-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider trellis2 --input-image "{i... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | external-image-to-mesh | single-image-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | biharmonic | glb | 3600 |  | inferred | False | mirror | none | /content/trellis2-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider trellis2 --input-image "{i... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |

## Training Recipes

LoRA candidates with `training_report.json` are expanded here so benchmark scores can be traced back to the actual training recipe instead of only the adapter path.

_None._

## Failures

| Method | Failures | Last Error Type | Last Error |
| --- | --- | --- | --- |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | 1 | RuntimeError | External image-to-mesh command failed with exit status 1. Command: /content/trellis2-venv/bin/python -m backend.benchma... |

## Split Audit

| Method | Eval n | Eval Assets | Train Assets | Train/Eval Overlap | Eval Categories | Train Categories | Eval Source Splits | Train Source Splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
| mirror | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | 1 | 1 | 0 | 0 | table:1 |  | train:1 |  |
