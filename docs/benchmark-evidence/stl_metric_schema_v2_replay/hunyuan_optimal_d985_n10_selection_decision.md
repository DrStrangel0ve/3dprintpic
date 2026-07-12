# Completion Candidate Decision: hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh

- Input: `backend\output\completion-benchmark\stl_first_ingest\g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10\extracted\g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10\output\g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10\experiments\modern_weighted_eval_s0_n10`
- Decision: `hold`
- Candidate: `hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh`
- Current: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh`
- Baseline: `masked`
- Score profile: `stl-quality`
- Candidate score: `0.9742`
- Current score: `0.1266`

## Gate Checks

| Check | Passed | Value | Threshold | Detail |
| --- | --- | --- | --- | --- |
| deployable_candidate | yes |  | not a hidden-source/oracle diagnostic |  |
| success_rate | yes | 1 | >= 1.0 |  |
| paired_n | yes | 10 | >= 10 |  |
| paired_win_rate | yes | 1 | >= 0.8 |  |
| paired_ci95_low | yes | 0.602 | > 0.0 |  |
| score_margin_vs_current | yes | 0.8476 | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_n_vs_current | yes | 10 | >= 10 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_win_rate_vs_current | yes | 0.9 | >= 0.8 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_ci95_low_vs_current | yes | 0.498 | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_heldout_view_silhouette_iou_mean_ratio_vs_current | yes | 1.0857 | <= 1.1 | paired_n=10; required_n=10; median_ratio=0.7881; worst_sample=mesh_dir_8a68107048_v00; candidate=0.5874; current=0.6377 |
| paired_mesh_surface_chamfer_ratio_vs_current | no | 1.6838 | <= 1.1 | paired_n=10; required_n=10; median_ratio=0.8429; worst_sample=mesh_dir_8a68107048_v00; candidate=0.1782; current=0.1058 |
| paired_mesh_surface_hausdorff95_ratio_vs_current | no | 1.6083 | <= 1.1 | paired_n=10; required_n=10; median_ratio=0.8175; worst_sample=mesh_dir_c3a9748da7_v00; candidate=0.5201; current=0.3234 |
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
| stl_scale_free_complexity | yes | 6.5704 | <= 10.0 |  |
| repair_convex_hull_fallback_rate | no | 0.6 | <= 0.25 |  |
| repair_volume_fill_ratio_relative_change_abs | yes | 0.4693 | <= 0.5 |  |
| per_sample_stl_watertight | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_volume | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_manifold | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_winding_consistent | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_positive_volume | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_single_component | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_bbox_has_volume | yes | 10 | 10/10 samples >= 1.0 |  |
| per_sample_stl_nonmanifold_edges | yes | 10 | 10/10 samples <= 0.0 |  |
| per_sample_stl_degenerate_face_ratio | yes | 10 | 10/10 samples <= 0.0 |  |
| per_sample_stl_component_excess | yes | 10 | 10/10 samples <= 0.0 |  |
| per_sample_stl_bbox_aspect_ratio | yes | 10 | 10/10 samples <= 10.0 |  |
| per_sample_stl_scale_free_complexity | yes | 10 | 10/10 samples <= 10.0 |  |
| per_sample_repair_volume_fill_ratio_relative_change_abs | no | 8 | 10/10 samples <= 4.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_abf713d7a4_v00 |
| repair_volume_fill_ratio_relative_change_abs_coverage | no | 0.6 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_abf713d7a4_v00,mesh_dir_509850f403_v00,mesh_dir_c3a9748da7_v00 |
| split_audit_present | yes | 1 | present |  |
| train_eval_asset_overlap | yes | 0 | <= 0 |  |

## Paired Objective

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 10 | 10/10 | 1 | 0.9969 | 0.602 | 1.4246 |

## Paired Objective vs Current

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 10 | 9/10 | 0.9 | 0.8885 | 0.498 | 1.28 |

## Ranked Methods

| Method | Baseline-Delta Score | Success Rate |
| --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 0.9742 | 1 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.1266 | 1 |
| mirror | 0.0372 | 1 |
| masked | 0 | 1 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | -2.5078 | 1 |
