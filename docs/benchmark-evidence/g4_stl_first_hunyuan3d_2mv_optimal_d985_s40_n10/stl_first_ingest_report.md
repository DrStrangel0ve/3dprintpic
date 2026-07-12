# STL-First Result Ingest

- Generated: `2026-07-12T16:51:13+00:00`
- Score profile: `stl-quality`
- Score mode: `baseline-delta`
- Baseline: `masked`
- Result directories: `1`

## Inputs

| Label | Source | Materialized Root |
| --- | --- | --- |
| g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10 | /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10 | /content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10 |

Source-mesh oracle rows, including source-mesh bundle oracles, are kept as diagnostics. Only non-oracle `depth-relief`, `single-image-mesh`, and `multiview-mesh` rows are treated as deployable STL architectures.

## Run: g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10

- Run directory: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10`
- Summary: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/experiments/modern_weighted_eval_s0_n10/aggregate_summary.csv`
- Summary rows: `5`
- Per-sample rows: `50`
- Deployable score leader: `hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh` (multiview-mesh) score `0.9742`
- Promotion-eligible winner: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` (single-image-mesh) score `0.1266`

### Architecture Decision

| Decision | Recommended | Mode | Depth-Relief Baseline | Promotion Challenger | Score-Leading Challenger | Delta vs Depth-Relief | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| promote-challenger | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh |  | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh |  | A full-mesh or multiview STL challenger cleared promotion gates and no promotion-eligible depth-relief baseline was found. |

### Architecture Leaders

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth-relief | mirror | 0.0372 | 10 | 1 | 0.175 | 0.5071 | 1 | 1 | 1 | 1 | 11.1803 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.1266 | 10 | 1 | 0.1956 | 0.4608 | 1 | 1 | 1 | 1 | 9.9498 | yes |  |
| multiview-mesh | hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 0.9742 | 10 | 1 | 0.1548 | 0.3523 | 1 | 1 | 1 | 1 | 6.5704 | no | Convex Hull Fallback, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |

### Top Methods

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiview-mesh | hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | 0.9742 | 10 | 1 | 0.1548 | 0.3523 | 1 | 1 | 1 | 1 | 6.5704 | no | Convex Hull Fallback, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.1266 | 10 | 1 | 0.1956 | 0.4608 | 1 | 1 | 1 | 1 | 9.9498 | yes |  |
| depth-relief | mirror | 0.0372 | 10 | 1 | 0.175 | 0.5071 | 1 | 1 | 1 | 1 | 11.1803 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| depth-relief | masked | 0 | 10 | 1 | 0.1877 | 0.5037 | 1 | 1 | 1 | 1 | 11.164 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| multiview-mesh | hunyuan3d_2mv_cardinal4_raw_direct_mesh | -2.5078 | 10 | 1 | 0.0488 | 0.1558 | 1 | 1 | 0 | 0 | 14.6146 | no | Manifold, Single Body, Degenerate Face Ratio, Body Excess log1p, Scale-Free Complexity log1p, Sample Watertight, Sample Volume Mesh, Sample... |

### Promotion Gate Failures

| Method | STL Mode | Score | Gate | Field | Value | Threshold | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.9742 | Convex Hull Fallback | repair_convex_hull_used_mean | 0.6 | <= 0.25 |  |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.9742 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 8 | 10/10 samples <= 4.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_abf713d7a4_v00 |
| hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.9742 | Sample Repair Fill-Ratio Drift Coverage | repair_volume_fill_ratio_relative_change_abs | 0.6 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_abf713d7a4_v00,mesh_dir_509850f403_v00,mesh_dir_c3a9748da7_v00 |
| mirror | depth-relief | 0.0372 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.1803 | <= 10.0 |  |
| mirror | depth-relief | 0.0372 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 10/10 samples <= 10.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_014d7f1847_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh... |
| masked | depth-relief | 0 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.164 | <= 10.0 |  |
| masked | depth-relief | 0 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 10/10 samples <= 10.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_014d7f1847_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh... |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Manifold | stl_is_manifold_median | 0 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Single Body | stl_single_component_median | 0 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Degenerate Face Ratio | stl_degenerate_face_ratio_median | 1.45e-05 | <= 0.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Body Excess log1p | stl_component_excess_log1p_median | 2.1784 | <= 0.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 14.6146 | <= 10.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Watertight | stl_is_watertight | 6 | 10/10 samples >= 1.0 | failed_samples=mesh_dir_4af07fffff_v00,mesh_dir_abf713d7a4_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Volume Mesh | stl_is_volume | 6 | 10/10 samples >= 1.0 | failed_samples=mesh_dir_4af07fffff_v00,mesh_dir_abf713d7a4_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Manifold | stl_is_manifold | 0 | 10/10 samples >= 1.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_014d7f1847_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh... |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Single Body | stl_single_component | 3 | 10/10 samples >= 1.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh_dir_c3a9748da7_v00,mesh... |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p | 6 | 10/10 samples <= 0.0 | failed_samples=mesh_dir_4af07fffff_v00,mesh_dir_abf713d7a4_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Degenerate Face Ratio | stl_degenerate_face_ratio | 0 | 10/10 samples <= 0.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_014d7f1847_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh... |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Body Excess log1p | stl_component_excess_log1p | 3 | 10/10 samples <= 0.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh_dir_c3a9748da7_v00,mesh... |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -2.5078 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 10/10 samples <= 10.0 | failed_samples=mesh_dir_e49bbc93b7_v00,mesh_dir_4af07fffff_v00,mesh_dir_014d7f1847_v00,mesh_dir_4f704c790f_v00,mesh_dir_abf713d7a4_v00,mesh... |

### Sample Failure Hotspots

| Sample | Methods | Gates | Method Names | STL Modes | Gate Names |
| --- | --- | --- | --- | --- | --- |
| mesh_dir_abf713d7a4_v00 | 4 | 12 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Repair Fill-Ratio Drift Cov... |
| mesh_dir_e49bbc93b7_v00 | 4 | 9 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Repair Fill-Ratio Drift Coverage, Sample Repair Fill-Ratio ... |
| mesh_dir_c3a9748da7_v00 | 4 | 8 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Repair Fill-Ratio Drift Coverage, Sample Scale-Free Complex... |
| mesh_dir_509850f403_v00 | 4 | 6 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_optimal_d985_repaired_stl_inferred_bbox_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Degenerate Face Ratio, Sample Manifold, Sample Repair Fill-Ratio Drift Coverage, Sample Scale-Free Complexity log1p |
| mesh_dir_001bfe840e_v00 | 3 | 10 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Scale-Free Complexity log1p... |
| mesh_dir_3ea1828365_v00 | 3 | 10 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Scale-Free Complexity log1p... |
| mesh_dir_4af07fffff_v00 | 3 | 10 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Scale-Free Complexity log1p... |
| mesh_dir_4f704c790f_v00 | 3 | 7 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Scale-Free Complexity log1p, Sample Single Body |
| mesh_dir_014d7f1847_v00 | 3 | 5 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Degenerate Face Ratio, Sample Manifold, Sample Scale-Free Complexity log1p |
| mesh_dir_8a68107048_v00 | 3 | 5 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Degenerate Face Ratio, Sample Manifold, Sample Scale-Free Complexity log1p |
