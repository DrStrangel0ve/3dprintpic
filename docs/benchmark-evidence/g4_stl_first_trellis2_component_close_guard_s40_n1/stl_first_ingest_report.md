# STL-First Result Ingest

- Generated: `2026-07-12T21:11:32+00:00`
- Score profile: `stl-quality`
- Score mode: `baseline-delta`
- Baseline: `masked`
- Result directories: `1`

## Inputs

| Label | Source | Materialized Root |
| --- | --- | --- |
| g4_stl_first_trellis2_component_close_guard_s40_n1 | backend\output\completion-benchmark\colab_evidence\g4_stl_first_trellis2_component_close_guard_s40_n1_results_compact.tar.gz | backend\output\completion-benchmark\stl_first_ingest\g4_stl_first_trellis2_component_close_guard_s40_n1\extracted\g4_stl_first_trellis2_com... |

Source-mesh oracle rows, including source-mesh bundle oracles, are kept as diagnostics. Only non-oracle `depth-relief`, `single-image-mesh`, and `multiview-mesh` rows are treated as deployable STL architectures.

## Run: g4_stl_first_trellis2_component_close_guard_s40_n1

- Run directory: `backend\output\completion-benchmark\stl_first_ingest\g4_stl_first_trellis2_component_close_guard_s40_n1\extracted\g4_stl_first_trellis2_component_close_guard_s40_n1\output\g4_stl_first_trellis2_component_close_guard_s40_n1\experiments\modern_weighted_eval_s0_n1`
- Summary: `backend\output\completion-benchmark\stl_first_ingest\g4_stl_first_trellis2_component_close_guard_s40_n1\extracted\g4_stl_first_trellis2_component_close_guard_s40_n1\output\g4_stl_first_trellis2_component_close_guard_s40_n1\experiments\modern_weighted_eval_s0_n1\aggregate_summary.csv`
- Summary rows: `6`
- Per-sample rows: `5`
- Deployable score leader: `mirror` (depth-relief) score `0.0217`
- Promotion-eligible winner: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` (single-image-mesh) score `0.0217`

### Architecture Decision

| Decision | Recommended | Mode | Depth-Relief Baseline | Promotion Challenger | Score-Leading Challenger | Delta vs Depth-Relief | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| promote-challenger | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh |  | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |  | A full-mesh or multiview STL challenger cleared promotion gates and no promotion-eligible depth-relief baseline was found. |

### Architecture Leaders

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth-relief | mirror | 0.0217 | 1 | 1 | 0.1617 | 0.4066 | 1 | 1 | 1 | 1 | 11.1662 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 | 1 | 0.1264 | 0.3145 | 1 | 1 | 1 | 1 | 9.9499 | yes |  |

### Top Methods

| STL Mode | Method | Score | n | Success | Chamfer | H95 | Watertight | Volume | Manifold | Single Body | Scale-Free Complexity | Promote | Gate Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth-relief | mirror | 0.0217 | 1 | 1 | 0.1617 | 0.4066 | 1 | 1 | 1 | 1 | 11.1662 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 | 1 | 0.1264 | 0.3145 | 1 | 1 | 1 | 1 | 9.9499 | yes |  |
| single-image-mesh | trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 | 1 | 0.1679 | 0.3944 | 1 | 1 | 1 | 1 | 5.553 | no | Convex Hull Fallback, Repair Fill-Ratio Drift, Sample Repair Fill-Ratio Drift Maximum, Sample Repair Fill-Ratio Drift Coverage |
| depth-relief | masked | 0 | 1 | 1 | 0.1677 | 0.4153 | 1 | 1 | 1 | 1 | 11.1885 | no | Scale-Free Complexity log1p, Sample Scale-Free Complexity log1p |
| single-image-mesh | trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | 0 | 0 | 0 |  |  |  |  |  |  |  | no | Success, STL Exists, Watertight, Volume Mesh, Manifold, Winding, Positive Volume, Single Body, 3D BBox, Nonmanifold Edges log1p, Degenerate... |
| single-image-mesh | trellis2_biharmonic_prefill_raw_direct_mesh | -23.0239 | 1 | 1 | 0.2094 | 0.4305 | 0 | 0 | 0 | 0 | 15.897 | no | Watertight, Volume Mesh, Manifold, Winding, Positive Volume, Single Body, Nonmanifold Edges log1p, Body Excess log1p, Scale-Free Complexity... |

### Failed Provider Diagnostics

| Method | Sample | Error Type | Target Faces | Strict Faces | Topology-Relaxed Faces | Boundary-Relaxed Faces | Self-Intersections | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | mesh_dir_757e06b221_v00 | RuntimeError | 10,704 | 20,834 | 18,862 | 10,704 | 2,943 | Component-close mesh repair produced self-intersecting faces: count=2943 |

### Promotion Gate Failures

| Method | STL Mode | Score | Gate | Field | Value | Threshold | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mirror | depth-relief | 0.0217 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.1662 | <= 10.0 |  |
| mirror | depth-relief | 0.0217 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 1/1 samples <= 10.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh | 0.0217 | Convex Hull Fallback | repair_convex_hull_used_mean | 1 | <= 0.25 |  |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh | 0.0217 | Repair Fill-Ratio Drift | repair_volume_fill_ratio_relative_change_abs_median | 271.476 | <= 0.5 |  |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh | 0.0217 | Sample Repair Fill-Ratio Drift Maximum | repair_volume_fill_ratio_relative_change_abs | 0 | 1/1 samples <= 4.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | single-image-mesh | 0.0217 | Sample Repair Fill-Ratio Drift Coverage | repair_volume_fill_ratio_relative_change_abs | 0 | >= 0.75 samples <= 0.5 | failed_samples=mesh_dir_757e06b221_v00 |
| masked | depth-relief | 0 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 11.1885 | <= 10.0 |  |
| masked | depth-relief | 0 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 1/1 samples <= 10.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Success | success_rate | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | STL Exists | stl_exists_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Watertight | stl_is_watertight_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Volume Mesh | stl_is_volume_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Manifold | stl_is_manifold_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Winding | stl_winding_consistent_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Positive Volume | stl_positive_volume_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Single Body | stl_single_component_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | 3D BBox | stl_bbox_has_volume_median |  | >= 1.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p_median |  | <= 0.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Degenerate Face Ratio | stl_degenerate_face_ratio_median |  | <= 0.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Body Excess log1p | stl_component_excess_log1p_median |  | <= 0.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | BBox Aspect | stl_bbox_aspect_ratio_median |  | <= 10.0 |  |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | single-image-mesh | 0 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median |  | <= 10.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Watertight | stl_is_watertight_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Volume Mesh | stl_is_volume_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Manifold | stl_is_manifold_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Winding | stl_winding_consistent_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Positive Volume | stl_positive_volume_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Single Body | stl_single_component_median | 0 | >= 1.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p_median | 9.7308 | <= 0.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Body Excess log1p | stl_component_excess_log1p_median | 6.4489 | <= 0.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p_median | 15.897 | <= 10.0 |  |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Watertight | stl_is_watertight | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Volume Mesh | stl_is_volume | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Manifold | stl_is_manifold | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Winding | stl_winding_consistent | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Positive Volume | stl_positive_volume | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Single Body | stl_single_component | 0 | 1/1 samples >= 1.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Nonmanifold Edges log1p | stl_nonmanifold_edge_count_log1p | 0 | 1/1 samples <= 0.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Body Excess log1p | stl_component_excess_log1p | 0 | 1/1 samples <= 0.0 | failed_samples=mesh_dir_757e06b221_v00 |
| trellis2_biharmonic_prefill_raw_direct_mesh | single-image-mesh | -23.0239 | Sample Scale-Free Complexity log1p | stl_faces_per_normalized_bbox_volume_log1p | 0 | 1/1 samples <= 10.0 | failed_samples=mesh_dir_757e06b221_v00 |

### Sample Failure Hotspots

| Sample | Methods | Gates | Method Names | STL Modes | Gate Names |
| --- | --- | --- | --- | --- | --- |
| mesh_dir_757e06b221_v00 | 4 | 13 | masked, mirror, trellis2_biharmonic_prefill_raw_direct_mesh, trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | depth-relief, single-image-mesh | Sample Body Excess log1p, Sample Manifold, Sample Nonmanifold Edges log1p, Sample Positive Volume, Sample Repair Fill-Ratio Drift Coverage,... |
