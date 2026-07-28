# Completion Candidate Decision: hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh

- Input: `backend\output\completion-benchmark\ingested\g4_stl_first_hunyuan3d_2mv_decimator_ablation_s40_s6_n4\extracted\decimator\output\g4_stl_first_hunyuan3d_2mv_decimator_ablation_s40_s6_n4\experiments\modern_weighted_eval_s0_n4`
- Decision: `hold`
- Candidate: `hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh`
- Current: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh`
- Baseline: `mirror`
- Score profile: `stl-quality`
- Candidate score: `0.3516`
- Current score: `-0.174`

## Gate Checks

| Check | Passed | Value | Threshold | Detail |
| --- | --- | --- | --- | --- |
| deployable_candidate | yes |  | not a hidden-source/oracle diagnostic |  |
| success_rate | yes | 1 | >= 1.0 |  |
| paired_n | yes | 4 | >= 4 |  |
| paired_win_rate | yes | 1 | >= 0.8 |  |
| paired_ci95_low | yes | 0.3058 | > 0.0 |  |
| score_margin_vs_current | yes | 0.5256 | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_n_vs_current | yes | 4 | >= 4 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_win_rate_vs_current | no | 0.75 | >= 0.8 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_ci95_low_vs_current | yes | 0.0375 | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_heldout_view_silhouette_iou_mean_ratio_vs_current | yes | 1.0857 | <= 1.1 | paired_n=4; required_n=4; median_ratio=0.8264; worst_sample=mesh_dir_8a68107048_v00; candidate=0.5874; current=0.6377 |
| stl_watertight | yes | 1 | >= 1.0 |  |
| stl_volume | yes | 1 | >= 1.0 |  |
| stl_manifold | yes | 1 | >= 1.0 |  |
| stl_winding_consistent | yes | 1 | >= 1.0 |  |
| stl_positive_volume | yes | 1 | >= 1.0 |  |
| stl_single_component | yes | 1 | >= 1.0 |  |
| stl_bbox_has_volume | yes | 1 | >= 1.0 |  |
| stl_nonmanifold_edges | yes | 0 | <= 0.0 |  |
| stl_degenerate_face_ratio | yes | 0 | <= 0.0 |  |
| stl_component_excess | yes | 0 | <= 0.0 |  |
| stl_bbox_aspect_ratio | yes | 1.9452 | <= 10.0 |  |
| stl_scale_free_complexity | yes | 9.8499 | <= 10.0 |  |
| repair_convex_hull_fallback_rate | yes | 0.25 | <= 0.25 |  |
| repair_volume_fill_ratio_relative_change_abs | yes | 0.4693 | <= 0.5 |  |
| per_sample_stl_watertight | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_volume | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_manifold | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_winding_consistent | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_positive_volume | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_single_component | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_bbox_has_volume | yes | 4 | 4/4 samples >= 1.0 |  |
| per_sample_stl_nonmanifold_edges | yes | 4 | 4/4 samples <= 0.0 |  |
| per_sample_stl_degenerate_face_ratio | yes | 4 | 4/4 samples <= 0.0 |  |
| per_sample_stl_component_excess | yes | 4 | 4/4 samples <= 0.0 |  |
| per_sample_stl_bbox_aspect_ratio | yes | 4 | 4/4 samples <= 10.0 |  |
| per_sample_stl_scale_free_complexity | yes | 4 | 4/4 samples <= 10.0 |  |
| per_sample_repair_volume_fill_ratio_relative_change_abs | yes | 4 | 4/4 samples <= 4.0 |  |
| repair_volume_fill_ratio_relative_change_abs_coverage | yes | 0.75 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_c3a9748da7_v00 |
| train_eval_asset_overlap | yes | 0 | <= 0 |  |

## Paired Objective

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 4 | 4/4 | 1 | 0.4362 | 0.3058 | 0.6052 |

## Paired Objective vs Current

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 4 | 3/4 | 0.75 | 0.4099 | 0.0375 | 0.6149 |

## Ranked Methods

| Method | Baseline-Delta Score | Success Rate |
| --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 0.3516 | 1 |
| hunyuan3d_2mv_voxel_r192_endpoint_d985_repaired_stl_inferred_bbox_direct_mesh | 0.1876 | 1 |
| hunyuan3d_2mv_voxel_r192_endpoint_d990_repaired_stl_inferred_bbox_direct_mesh | 0.1727 | 1 |
| hunyuan3d_2mv_voxel_r192_optimal_d990_repaired_stl_inferred_bbox_direct_mesh | 0.171 | 1 |
| hunyuan3d_2mv_voxel_r192_optimal_d995_repaired_stl_inferred_bbox_direct_mesh | 0.1688 | 1 |
| hunyuan3d_2mv_voxel_r192_endpoint_d995_repaired_stl_inferred_bbox_direct_mesh | 0.1645 | 1 |
| mirror | 0 | 1 |
| masked | -0.1056 | 1 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | -0.174 | 1 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | -6.0675 | 1 |
