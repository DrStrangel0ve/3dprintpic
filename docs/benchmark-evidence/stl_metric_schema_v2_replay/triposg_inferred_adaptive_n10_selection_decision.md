# Completion Candidate Decision: triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh

- Input: `backend\output\completion-benchmark\colab_evidence\g4_stl_first_triposg_inferred_adaptive_s40_n10\extracted\output\g4_stl_first_triposg_inferred_adaptive_s40_n10\experiments\modern_weighted_eval_s0_n10`
- Decision: `hold`
- Candidate: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh`
- Current: `mirror`
- Baseline: `masked`
- Score profile: `stl-quality`
- Candidate score: `-0.0077`
- Current score: `0.0493`

## Gate Checks

| Check | Passed | Value | Threshold | Detail |
| --- | --- | --- | --- | --- |
| deployable_candidate | yes |  | not a hidden-source/oracle diagnostic |  |
| success_rate | yes | 1 | >= 1.0 |  |
| paired_n | yes | 10 | >= 10 |  |
| paired_win_rate | no | 0.4 | >= 0.8 |  |
| paired_ci95_low | no | -0.0341 | > 0.0 |  |
| score_margin_vs_current | no | -0.057 | > 0.0 | current_method=mirror |
| paired_n_vs_current | yes | 10 | >= 10 | current_method=mirror |
| paired_win_rate_vs_current | no | 0 | >= 0.8 | current_method=mirror |
| paired_ci95_low_vs_current | no | 0 | > 0.0 | current_method=mirror |
| paired_mesh_surface_chamfer_ratio_vs_current | no | 1.1587 | <= 1.1 | paired_n=10; required_n=10; median_ratio=0.9083; worst_sample=mesh_dir_49534574e1_v00; candidate=0.1743; current=0.1504 |
| paired_mesh_surface_hausdorff95_ratio_vs_current | yes | 1.0214 | <= 1.1 | paired_n=10; required_n=10; median_ratio=0.8442; worst_sample=mesh_dir_49534574e1_v00; candidate=0.6048; current=0.5922 |
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
| stl_bbox_aspect_ratio | yes | 1.9615 | <= 10.0 |  |
| stl_scale_free_complexity | yes | 8.2389 | <= 10.0 |  |
| repair_convex_hull_fallback_rate | no |  | <= 0.25 |  |
| repair_volume_fill_ratio_relative_change_abs | no |  | <= 0.5 |  |
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
| per_sample_repair_volume_fill_ratio_relative_change_abs | no | 0 | 10/10 samples <= 4.0 | failed_samples=mesh_dir_757e06b221_v00,mesh_dir_9f32f114c4_v00,mesh_dir_1a200d8288_v00,mesh_dir_92752051d0_v00,mesh_dir_49534574e1_v00,mesh... |
| repair_volume_fill_ratio_relative_change_abs_coverage | no | 0 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_757e06b221_v00,mesh_dir_9f32f114c4_v00,mesh_dir_1a200d8288_v00,mesh_dir_92752051d0_v00,mesh_dir_49534574e1_v00,mesh... |
| split_audit_present | yes | 1 | present |  |
| train_eval_asset_overlap | yes | 0 | <= 0 |  |

## Paired Objective

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 10 | 4/10 | 0.4 | -0.0142 | -0.0341 | 0.0025 |

## Paired Objective vs Current

| Method | Paired n | Wins | Win Rate | Mean Improvement | CI95 Low | CI95 High |
| --- | --- | --- | --- | --- | --- | --- |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 10 | 0/10 | 0 | 0 | 0 | 0 |

## Ranked Methods

| Method | Baseline-Delta Score | Success Rate |
| --- | --- | --- |
| source_mesh_oracle | 0.2313 | 1 |
| mirror | 0.0493 | 1 |
| masked | 0 | 1 |
| biharmonic | -0.0026 | 1 |
| triposg_masked_repaired_stl_inferred_bbox_direct_mesh | -0.0077 | 1 |
| triposg_mirror_prefill_repaired_stl_inferred_bbox_direct_mesh | -0.0077 | 1 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | -0.0077 | 1 |
