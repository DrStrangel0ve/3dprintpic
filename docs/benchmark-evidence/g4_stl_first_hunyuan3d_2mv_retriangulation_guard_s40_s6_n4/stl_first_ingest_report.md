# STL-First Result Ingest

- Generated: `2026-07-11T20:30:20+00:00`
- Score profile: `stl-quality`
- Score mode: `baseline-delta`
- Baseline: `masked`
- Result directories: `1`

## Inputs

| Label | Source | Materialized Root |
| --- | --- | --- |
| g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4 | backend\output\completion-benchmark\colab_evidence\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4_results_compact.tar.gz | backend\output\completion-benchmark\ingested\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\extracted\g4_stl_first_hunyuan3d_2m... |

Source-mesh oracle rows, including source-mesh bundle oracles, are kept as diagnostics. Only non-oracle `depth-relief`, `single-image-mesh`, and `multiview-mesh` rows are treated as deployable STL architectures.

## Run: g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4

- Run directory: `backend\output\completion-benchmark\ingested\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\extracted\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\output\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\experiments\modern_weighted_eval_s0_n4`
- Summary: `backend\output\completion-benchmark\ingested\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\extracted\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\output\g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4\experiments\modern_weighted_eval_s0_n4\aggregate_summary.csv`
- Summary rows: `9`
- Per-sample rows: `36`
- Deployable score leader: `hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh` (multiview-mesh) score `0.3681`
- Promotion-eligible winner: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` (single-image-mesh) score `-0.0756`

### Architecture Decision

| Decision | Recommended | Mode | Depth-Relief Baseline | Promotion Challenger | Score-Leading Challenger | Delta vs Depth-Relief | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| promote-challenger | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh |  | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh |  | A full-mesh or multiview STL challenger cleared promotion gates and no promotion-eligible depth-relief baseline was found. |

### Architecture Leaders

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth-relief | mirror | 0.1056 | 4 | 1 | 0.1683 | 0.434 | 1 | 1 | 1 | 1 | 11.1803 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | -0.0756 | 4 | 1 | 0.1847 | 0.3617 | 1 | 1 | 1 | 1 | 9.9493 | yes |  |
| multiview-mesh | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | 0.3681 | 4 | 1 | 0.1844 | 0.3499 | 1 | 1 | 1 | 1 | 6.101 | no | Convex Hull Fallback, Repair Fill-Ratio Drift, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |

### Top Methods

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiview-mesh | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | 0.3681 | 4 | 1 | 0.1844 | 0.3499 | 1 | 1 | 1 | 1 | 6.101 | no | Convex Hull Fallback, Repair Fill-Ratio Drift, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |
| multiview-mesh | hunyuan3d_2mv_voxel_r192_a001_repaired_stl_inferred_bbox_direct_mesh | 0.2777 | 4 | 1 | 0.2072 | 0.3577 | 1 | 1 | 1 | 1 | 6.1906 | no | Convex Hull Fallback, Repair Fill-Ratio Drift, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |
| multiview-mesh | hunyuan3d_2mv_voxel_r192_all_repaired_stl_inferred_bbox_direct_mesh | 0.2672 | 4 | 1 | 0.1864 | 0.3539 | 1 | 1 | 1 | 1 | 7.8935 | no | Convex Hull Fallback, Repair Fill-Ratio Drift, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |
| multiview-mesh | hunyuan3d_2mv_voxel_r256_a001_repaired_stl_inferred_bbox_direct_mesh | 0.1997 | 4 | 1 | 0.1779 | 0.3682 | 1 | 1 | 1 | 1 | 7.981 | no | Convex Hull Fallback, Sample Repair Fill-Ratio Drift Maximum |
| multiview-mesh | hunyuan3d_2mv_voxel_r256_all_repaired_stl_inferred_bbox_direct_mesh | 0.1993 | 4 | 1 | 0.1779 | 0.3682 | 1 | 1 | 1 | 1 | 7.9815 | no | Convex Hull Fallback, Sample Repair Fill-Ratio Drift Maximum |
| depth-relief | mirror | 0.1056 | 4 | 1 | 0.1683 | 0.434 | 1 | 1 | 1 | 1 | 11.1803 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| depth-relief | masked | 0 | 4 | 1 | 0.1816 | 0.4959 | 1 | 1 | 1 | 1 | 11.1622 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | -0.0756 | 4 | 1 | 0.1847 | 0.3617 | 1 | 1 | 1 | 1 | 9.9493 | yes |  |
| multiview-mesh | hunyuan3d_2mv_cardinal4_raw_direct_mesh | -5.9691 | 4 | 1 | 0.047 | 0.1479 | 0.5 | 0.5 | 0 | 0 | 14.6146 | no | Watertight, Volume Mesh, Manifold, Single Body, Nonmanifold Edges log1p, Degenerate Face Ratio, Body Excess log1p, Scale-Free Complexity lo... |

### Promotion Gate Failures

| Method | STL Mode | Score | Gate | Field | Value | Threshold | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.3681 | Convex Hull Fallback | repair_convex_hull_used_mean | 1 | <= 0.25 |  |
| hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.3681 | Repair Fill-Ratio Drift | repair_volume_fill_ratio_relative_change_abs_median | 12.2245 | <= 0.5 |  |
| hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.3681 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 1 | 4/4 samples <= 4.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.3681 | Sample Repair Fill-Ratio Drift Coverage | repair_volume_fill_ratio_relative_change_abs | 0.25 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_voxel_r192_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2777 | Convex Hull Fallback | repair_convex_hull_used_mean | 0.75 | <= 0.25 |  |
| hunyuan3d_2mv_voxel_r192_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2777 | Repair Fill-Ratio Drift | repair_volume_fill_ratio_relative_change_abs_median | 3.8409 | <= 0.5 |  |
| hunyuan3d_2mv_voxel_r192_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2777 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 2 | 4/4 samples <= 4.0 | failed_samples=mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_voxel_r192_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2777 | Sample Repair Fill-Ratio Drift Coverage | repair_volume_fill_ratio_relative_change_abs | 0.25 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_voxel_r192_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2672 | Convex Hull Fallback | repair_convex_hull_used_mean | 0.5 | <= 0.25 |  |
| hunyuan3d_2mv_voxel_r192_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2672 | Repair Fill-Ratio Drift | repair_volume_fill_ratio_relative_change_abs_median | 0.5621 | <= 0.5 |  |
| hunyuan3d_2mv_voxel_r192_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2672 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 3 | 4/4 samples <= 4.0 | failed_samples=mesh_dir_001bfe840e_v00 |
| hunyuan3d_2mv_voxel_r192_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.2672 | Sample Repair Fill-Ratio Drift Coverage | repair_volume_fill_ratio_relative_change_abs | 0.5 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00 |
| hunyuan3d_2mv_voxel_r256_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.1997 | Convex Hull Fallback | repair_convex_hull_used_mean | 0.5 | <= 0.25 |  |
| hunyuan3d_2mv_voxel_r256_a001_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.1997 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 3 | 4/4 samples <= 4.0 | failed_samples=mesh_dir_001bfe840e_v00 |
| hunyuan3d_2mv_voxel_r256_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.1993 | Convex Hull Fallback | repair_convex_hull_used_mean | 0.5 | <= 0.25 |  |
| hunyuan3d_2mv_voxel_r256_all_repaired_stl_inferred_bbox_direct_mesh | multiview-mesh | 0.1993 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 3 | 4/4 samples <= 4.0 | failed_samples=mesh_dir_001bfe840e_v00 |
| mirror | depth-relief | 0.1056 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.1803 | <= 10.0 |  |
| mirror | depth-relief | 0.1056 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 4/4 samples <= 10.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00,mesh_dir_8a68107048_v00 |
| masked | depth-relief | 0 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.1622 | <= 10.0 |  |
| masked | depth-relief | 0 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 4/4 samples <= 10.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00,mesh_dir_8a68107048_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Watertight | stl_is_watertight_median | 0.5 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Volume Mesh | stl_is_volume_median | 0.5 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Manifold | stl_is_manifold_median | 0 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Single Body | stl_single_component_median | 0 | >= 1.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p_median | 0.5493 | <= 0.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Degenerate Face Ratio | stl_degenerate_face_ratio_median | 2.36e-05 | <= 0.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Body Excess log1p | stl_component_excess_log1p_median | 2.2499 | <= 0.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 14.6146 | <= 10.0 |  |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Watertight | stl_is_watertight | 2 | 4/4 samples >= 1.0 | failed_samples=mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Volume Mesh | stl_is_volume | 2 | 4/4 samples >= 1.0 | failed_samples=mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Manifold | stl_is_manifold | 0 | 4/4 samples >= 1.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00,mesh_dir_8a68107048_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Single Body | stl_single_component | 1 | 4/4 samples >= 1.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p | 2 | 4/4 samples <= 0.0 | failed_samples=mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Degenerate Face Ratio | stl_degenerate_face_ratio | 0 | 4/4 samples <= 0.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00,mesh_dir_8a68107048_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Body Excess log1p | stl_component_excess_log1p | 1 | 4/4 samples <= 0.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00 |
| hunyuan3d_2mv_cardinal4_raw_direct_mesh | multiview-mesh | -5.9691 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 4/4 samples <= 10.0 | failed_samples=mesh_dir_c3a9748da7_v00,mesh_dir_001bfe840e_v00,mesh_dir_3ea1828365_v00,mesh_dir_8a68107048_v00 |

### Sample Failure Hotspots

| Sample | Methods | Gates | Method Names | STL Modes | Gate Names |
| --- | --- | --- | --- | --- | --- |
| mesh_dir_001bfe840e_v00 | 8 | 18 | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh, hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_a0... | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Repair Fill-Ratio Drift Cov... |
| mesh_dir_c3a9748da7_v00 | 6 | 11 | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh, hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_a0... | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Repair Fill-Ratio Drift Coverage, Sample Repair Fill-Ratio ... |
| mesh_dir_3ea1828365_v00 | 5 | 14 | hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh, hunyuan3d_2mv_cardinal4_raw_direct_mesh, hunyuan3d_2mv_voxel_r192_a0... | depth-relief, multiview-mesh | Sample Body Excess log1p, Sample Degenerate Face Ratio, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Repair Fill-Ratio Drift Cov... |
| mesh_dir_8a68107048_v00 | 3 | 5 | hunyuan3d_2mv_cardinal4_raw_direct_mesh, masked, mirror | depth-relief, multiview-mesh | Sample Degenerate Face Ratio, Sample Manifold, Sample Scale-Free Complexity log1p |
