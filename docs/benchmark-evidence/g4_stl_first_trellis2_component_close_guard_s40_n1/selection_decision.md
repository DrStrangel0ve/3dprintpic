# Completion Candidate Decision: trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh

- Input: `/content/3dprintpic/backend/output/completion-benchmark/colab_g4/g4_stl_first_trellis2_component_close_guard_s40_n1/experiments/modern_weighted_eval_s0_n1`
- Decision: `hold`
- Candidate: `trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh`
- Current: `triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh`
- Baseline: `masked`
- Score profile: `stl-quality`
- Candidate score: `0`
- Current score: `0.0217`

## Gate Checks

| Check | Passed | Value | Threshold | Detail |
| --- | --- | --- | --- | --- |
| deployable_candidate | yes |  | not a hidden-source/oracle diagnostic |  |
| success_rate | no | 0 | >= 1.0 |  |
| paired_n | no | 0 | >= 1 |  |
| paired_win_rate | no |  | >= 0.8 |  |
| paired_ci95_low | no |  | > 0.0 |  |
| score_margin_vs_current | no | -0.0217 | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_n_vs_current | no | 0 | >= 1 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_win_rate_vs_current | no |  | >= 0.8 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_ci95_low_vs_current | no |  | > 0.0 | current_method=triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh |
| paired_heldout_view_silhouette_iou_mean_ratio_vs_current | no |  | <= 1.1 | paired_n=0; required_n=1; median_ratio=; missing_or_nonfinite=mesh_dir_757e06b221_v00 |
| train_eval_asset_overlap | yes | 0 | <= 0 |  |

## Paired Objective

_None._

## Ranked Methods

| Method | Baseline-Delta Score | Success Rate |
| --- | --- | --- |
| mirror | 0.0217 | 1 |
| triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 |
| trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh | 0.0217 | 1 |
| masked | 0 | 1 |
| trellis2_biharmonic_prefill_component_close_d995_no_hull_stl_inferred_bbox_direct_mesh | 0 | 0 |
| trellis2_biharmonic_prefill_raw_direct_mesh | -23.0239 | 1 |
