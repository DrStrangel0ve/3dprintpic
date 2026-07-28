# Experiment Report: modern_weighted_eval_s0_n10

- Generated: `2026-07-12T16:51:12+00:00`
- Manifest: `/content/3dprintpic_colab_inputs/modelnet10_hunyuan3d_2mv_optimal_d985_s40_n10/inputs/manifest.jsonl`
- Limit: `10`
- Aggregate summary: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10/aggregate_summary.csv`
- Ranked output: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10/ranked_experiments.csv`
- Resolved config: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10/resolved_experiments.json`
- Contact sheet: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10/artifact_contact_sheet.png`

## Ranking

_Score profile: `stl-quality` (final STL mesh accuracy, validity, and printable complexity metrics)._

_Score mode: weighted sign-normalized metric improvement against `masked`. The baseline scores `0`; positive scores improve the objective and negative scores regress it. Scores are stable when unrelated candidate methods are added._

| Rank | Method | Score | n | Attempted | Success Rate | Errors | Masked MAE | Object MAE | Masked PSNR | Object PSNR | Masked SSIM | Object SSIM | Seam MAE | Object Depth MAE | Object Depth Corr | Object Surface Chamfer | Object Surface Chamfer RMSE | Object Surface Hausdorff95 | Mesh Surface Chamfer | Mesh Surface Chamfer RMSE | Mesh Surface Hausdorff95 | Inferred BBox Shape log-MAE | Inferred BBox Shape Rel-MAE | Inferred BBox Centered IoU | Silhouette IoU | STL Watertight | STL Volume Mesh | STL Winding | STL Positive Volume | STL Single Body | STL Bodies | STL Body Excess | STL Body Excess log1p | STL Min Dimension | STL 3D BBox | STL Aspect | STL Scale-Free Complexity log1p | STL Face Density log1p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 0.9742 | 10 | 10 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.1548 | 0.1796 | 0.3523 | 0.2975 | 0.3889 | 0.435 |  | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 48.838 | 1 | 1.9452 | 6.5704 | 9.71e-04 |
| 2 | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.1266 | 10 | 10 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.1956 | 0.2441 | 0.4608 | 0.2975 | 0.3889 | 0.435 |  | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 48.838 | 1 | 1.9452 | 9.9498 | 0.0241 |
| 3 | mirror | 0.0372 | 10 | 10 | 1 | 0 | 0.0226 | 0.0343 | 22.022 | 19.2956 | 0.9197 | 0.7271 | 0.0018 | 0.1529 | 0.252 | 0.0756 | 0.1299 | 0.2357 | 0.175 | 0.2195 | 0.5071 |  |  |  | 0.2557 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 48.838 | 1 | 1.9452 | 11.1803 | 0.0803 |
| 4 | masked | 0 | 10 | 10 | 1 | 0 | 0.2054 | 0.5201 | 10.3517 | 5.4492 | 0.8374 | 0.5136 | 0.1756 | 0.1401 | -0.0155 | 0.0955 | 0.1304 | 0.3154 | 0.1877 | 0.2251 | 0.5037 |  |  |  | 0.2404 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 49.6372 | 1 | 1.9139 | 11.164 | 0.0791 |
| 5 | hunyuan3d_2mv_cardinal4_raw_direct_mesh | -2.5078 | 10 | 10 | 1 | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 0.0488 | 0.0635 | 0.1558 | 0.2975 | 0.3889 | 0.435 |  | 1 | 1 | 1 | 1 | 0 | 9.5 | 8.5 | 2.1784 | 1.068 | 1 | 1.8704 | 14.6146 | 12.5431 |

## Selected Baseline Deltas: `masked`

Raw `Delta` is `method - baseline`; `Improvement` is sign-normalized so positive means better. These are aggregate median deltas for selected diagnostic metrics, not weighted rank points or paired per-sample wins.

| Method | Metric | Value | masked | Delta | Improvement |
| --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Chamfer | 0.0488 | 0.1877 | -0.1389 | 0.1389 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Chamfer RMSE | 0.0635 | 0.2251 | -0.1616 | 0.1616 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Hausdorff95 | 0.1558 | 0.5037 | -0.3479 | 0.3479 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 0.1548 | 0.1877 | -0.0329 | 0.0329 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 0.1796 | 0.2251 | -0.0455 | 0.0455 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 0.3523 | 0.5037 | -0.1514 | 0.1514 |
| mirror | Depth Corr | 0.2874 | -0.1942 | 0.4817 | 0.4817 |
| mirror | Depth MAE | 0.1904 | 0.4045 | -0.2141 | 0.2141 |
| mirror | Masked MAE | 0.0226 | 0.2054 | -0.1828 | 0.1828 |
| mirror | Masked PSNR | 22.022 | 10.3517 | 11.6704 | 11.6704 |
| mirror | Masked SSIM | 0.9197 | 0.8374 | 0.0823 | 0.0823 |
| mirror | Mesh Surface Chamfer | 0.175 | 0.1877 | -0.0127 | 0.0127 |
| mirror | Mesh Surface Chamfer RMSE | 0.2195 | 0.2251 | -0.0056 | 0.0056 |
| mirror | Mesh Surface Hausdorff95 | 0.5071 | 0.5037 | 0.0034 | -0.0034 |
| mirror | Object Depth Corr | 0.252 | -0.0155 | 0.2675 | 0.2675 |
| mirror | Object Depth MAE | 0.1529 | 0.1401 | 0.0128 | -0.0128 |
| mirror | Object MAE | 0.0343 | 0.5201 | -0.4858 | 0.4858 |
| mirror | Object PSNR | 19.2956 | 5.4492 | 13.8464 | 13.8464 |
| mirror | Object SSIM | 0.7271 | 0.5136 | 0.2135 | 0.2135 |
| mirror | Object Surface Chamfer | 0.0756 | 0.0955 | -0.02 | 0.02 |
| mirror | Object Surface Chamfer RMSE | 0.1299 | 0.1304 | -5.50e-04 | 5.50e-04 |
| mirror | Object Surface Hausdorff95 | 0.2357 | 0.3154 | -0.0798 | 0.0798 |
| mirror | Seam MAE | 0.0018 | 0.1756 | -0.1737 | 0.1737 |
| mirror | Silhouette IoU | 0.2557 | 0.2404 | 0.0153 | 0.0153 |
| mirror | Surface Chamfer | 0.1244 | 0.2642 | -0.1398 | 0.1398 |
| mirror | Surface Chamfer RMSE | 0.1838 | 0.3129 | -0.1291 | 0.1291 |
| mirror | Surface Hausdorff95 | 0.5672 | 0.6128 | -0.0456 | 0.0456 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 0.1956 | 0.1877 | 0.0079 | -0.0079 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 0.2441 | 0.2251 | 0.019 | -0.019 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 0.4608 | 0.5037 | -0.0429 | 0.0429 |

## Paired Baseline Wins: `masked`

Each row compares a method against the baseline on the same `sample_id`. `Wins` count strictly positive sign-normalized improvements; ties are reported separately.

| Method | Metric | Baseline n | Method n | Common n | Paired n | Skipped | Wins | Win Rate | Ties | Median Delta | Median Improvement | Mean Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | -0.1413 | 0.1413 | 0.1268 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 9/10 | 0.9 | 0 | -0.1733 | 0.1733 | 0.1441 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | Mesh Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 8/10 | 0.8 | 0 | -0.3825 | 0.3825 | 0.2808 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 7/10 | 0.7 | 0 | -0.0245 | 0.0245 | 0.0259 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 8/10 | 0.8 | 0 | -0.042 | 0.042 | 0.0367 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 9/10 | 0.9 | 0 | -0.1464 | 0.1464 | 0.1058 |
| mirror | Depth Corr | 10 | 10 | 10 | 10 | 0 | 6/10 | 0.6 | 0 | 0.3098 | 0.3098 | 0.3696 |
| mirror | Depth MAE | 10 | 10 | 10 | 10 | 0 | 6/10 | 0.6 | 0 | -0.065 | 0.065 | 0.2276 |
| mirror | Masked MAE | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | -0.1629 | 0.1629 | 0.1628 |
| mirror | Masked PSNR | 10 | 10 | 10 | 10 | 0 | 9/10 | 0.9 | 0 | 12.1099 | 12.1099 | 12.1623 |
| mirror | Masked SSIM | 10 | 10 | 10 | 10 | 0 | 8/10 | 0.8 | 0 | 0.0724 | 0.0724 | 0.0587 |
| mirror | Mesh Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 7/10 | 0.7 | 0 | -0.0036 | 0.0036 | 0.009 |
| mirror | Mesh Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 7/10 | 0.7 | 0 | -4.14e-04 | 4.14e-04 | 0.0051 |
| mirror | Mesh Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 4/10 | 0.4 | 0 | 0.0055 | -0.0055 | -0.0034 |
| mirror | Object Depth Corr | 10 | 10 | 10 | 8 | 2 | 5/8 | 0.625 | 0 | 0.0438 | 0.0438 | 0.286 |
| mirror | Object Depth MAE | 10 | 10 | 10 | 10 | 0 | 5/10 | 0.5 | 0 | 0.0015 | -0.0015 | 0.1454 |
| mirror | Object MAE | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | -0.4726 | 0.4726 | 0.4377 |
| mirror | Object PSNR | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | 13.4741 | 13.4741 | 13.6541 |
| mirror | Object SSIM | 10 | 10 | 10 | 10 | 0 | 9/10 | 0.9 | 0 | 0.1462 | 0.1462 | 0.1458 |
| mirror | Object Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 6/10 | 0.6 | 0 | -0.0159 | 0.0159 | 0.1101 |
| mirror | Object Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 6/10 | 0.6 | 0 | -0.0022 | 0.0022 | 0.1066 |
| mirror | Object Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 5/10 | 0.5 | 0 | -0.0174 | 0.0174 | 0.1606 |
| mirror | Seam MAE | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | -0.1558 | 0.1558 | 0.1502 |
| mirror | Silhouette IoU | 10 | 10 | 10 | 10 | 0 | 3/10 | 0.3 | 6 | 0 | 0 | 0.0533 |
| mirror | Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 5/10 | 0.5 | 0 | -0.0138 | 0.0138 | 0.1005 |
| mirror | Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 5/10 | 0.5 | 0 | -0.0048 | 0.0048 | 0.1054 |
| mirror | Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 7/10 | 0.7 | 0 | -0.0099 | 0.0099 | 0.1177 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer | 10 | 10 | 10 | 10 | 0 | 6/10 | 0.6 | 0 | -0.0014 | 0.0014 | -0.0108 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Chamfer RMSE | 10 | 10 | 10 | 10 | 0 | 3/10 | 0.3 | 0 | 0.0057 | -0.0057 | -0.0094 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | Mesh Surface Hausdorff95 | 10 | 10 | 10 | 10 | 0 | 8/10 | 0.8 | 0 | -0.0195 | 0.0195 | 0.0161 |

## Paired Objective Confidence: `masked`

Each row applies the same metric weights used for ranking to each shared `sample_id`, but as raw sign-normalized improvement over the baseline rather than candidate-normalized aggregate medians. The confidence interval bootstraps paired sample improvements with a fixed seed; if the CI is entirely above `0`, the method has a stronger promotion signal than an aggregate median alone.

| Method | Baseline n | Method n | Common n | Paired n | Skipped | Wins | Win Rate | Ties | Median Objective Improvement | Mean Objective Improvement | Mean CI95 Low | Mean CI95 High | Mean Metrics/Sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 10 | 10 | 10 | 10 | 0 | 10/10 | 1 | 0 | 0.8735 | 0.9969 | 0.602 | 1.4246 | 14 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 10 | 10 | 10 | 10 | 0 | 9/10 | 0.9 | 0 | 0.0979 | 0.1084 | 0.0275 | 0.1839 | 14 |
| mirror | 10 | 10 | 10 | 10 | 0 | 7/10 | 0.7 | 0 | 0.0418 | 0.0705 | -0.0556 | 0.1987 | 15 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | 10 | 10 | 10 | 10 | 0 | 0/10 | 0 | 0 | -3.7209 | -4.979 | -7.411 | -2.4654 | 14 |

## Experiment Config

| Name | Base Method | STL Mode | Steps | Guidance | Seed | Max Dim | Edit Fill | Model | LoRA | LoRA Scale | Start | Skip Depth | Emit STL | Mesh Samples | Direct Input | Direct Ext | Direct Timeout | Direct Ref Root | Direct BBox Source | Oracle Diagnostic | Direct Ref Method | Source Repair | Direct Command | Prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked | masked | depth-relief | 24 |  | 1234 | 768 | input |  |  |  | 0 | False | True | 4096 | masked | glb | 1800 |  | none | False | mirror | none |  | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| mirror | mirror | depth-relief | 24 |  | 1234 | 768 | input |  |  |  | 0 | False | True | 4096 | masked | glb | 1800 |  | none | False | mirror | none |  |  |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | external-image-to-mesh | single-image-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | biharmonic | glb | 3600 |  | inferred | False | mirror | none | /content/triposg-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider triposg --input-image "{inp... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | external-multiview-to-mesh | multiview-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | full | glb | 3600 |  | none | False | mirror | none | /content/hunyuan3d-2mv-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider hunyuan3d-2mv --provi... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | external-multiview-to-mesh | multiview-mesh | 24 |  | 1234 | 768 | input |  |  |  | 0 | True | True | 4096 | full | glb | 3600 |  | inferred | False | mirror | none | /content/hunyuan3d-2mv-venv/bin/python -m backend.benchmark.run_image_to_mesh_provider --provider hunyuan3d-2mv --provi... | Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background. |

## Training Recipes

LoRA candidates with `training_report.json` are expanded here so benchmark scores can be traced back to the actual training recipe instead of only the adapter path.

_None._

## Failures

_None._

## Split Audit

| Method | Eval n | Eval Assets | Train Assets | Train/Eval Overlap | Eval Categories | Train Categories | Eval Source Splits | Train Source Splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked | 10 | 10 | 0 | 0 | bathtub:1, bed:1, chair:1, desk:1, dresser:1, monitor:1, night_stand:1, sofa:1, table:1, toilet:1 |  | test:1, train:9 |  |
| mirror | 10 | 10 | 0 | 0 | bathtub:1, bed:1, chair:1, desk:1, dresser:1, monitor:1, night_stand:1, sofa:1, table:1, toilet:1 |  | test:1, train:9 |  |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 10 | 10 | 0 | 0 | bathtub:1, bed:1, chair:1, desk:1, dresser:1, monitor:1, night_stand:1, sofa:1, table:1, toilet:1 |  | test:1, train:9 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | 10 | 10 | 0 | 0 | bathtub:1, bed:1, chair:1, desk:1, dresser:1, monitor:1, night_stand:1, sofa:1, table:1, toilet:1 |  | test:1, train:9 |  |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 10 | 10 | 0 | 0 | bathtub:1, bed:1, chair:1, desk:1, dresser:1, monitor:1, night_stand:1, sofa:1, table:1, toilet:1 |  | test:1, train:9 |  |
