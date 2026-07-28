# Hunyuan3D-2mv Decimator Ablation, Held-Out Four

This bundle records
`g4_stl_first_hunyuan3d_2mv_decimator_ablation_s40_s6_n4`, completed on
exact runtime commit `9e37d1c026236211f29052a990cb80aa8d3e55b4`.
All `40/40` rows completed on an NVIDIA RTX PRO 6000 Blackwell Server
Edition. The run reused the exact four held-out assets from the previous
repair audits and compared optimal versus endpoint quadric-collapse placement
at adaptive scale-free density caps `9.95`, `9.90`, and `9.85`.

## Decision

The `optimal` placement at cap `9.85` is the only Hunyuan variant that clears
the bounded expansion gates:

- all four pre-clean meshes have zero degenerate faces;
- exactly one of four rows uses a convex hull;
- median absolute fill-ratio drift is `0.4693`, three of four rows are at or
  below `0.5`, and the worst row is `0.6190`;
- every final scale-free complexity is at or below `9.8499933`;
- median held-out silhouette IoU rises to `0.5263`, from `0.4840` for the
  prior optimal `9.95` budget and `0.4378` for TripoSG;
- median mesh Chamfer/H95 are `0.1509`/`0.3495`, versus
  `0.1847`/`0.3617` for TripoSG.

The strict selector still returns `hold`, not promotion. The candidate wins
`3/4` paired objectives against TripoSG (`0.75`) while policy requires
`0.8`. Every printability, complexity, fill-drift, split, surface, and
held-out degradation check passes. This is enough to justify the prescribed
held-out-ten confirmation, but not enough to replace the promoted TripoSG
incumbent.

## Placement Finding

Endpoint placement is decisively worse on this slice. Across all three
budgets it leaves mean pre-clean degenerates between `0.5` and `0.75`, uses a
hull on half the rows, and produces median fill drift between `3.8211` and
`3.8398`. It also has worse mesh Chamfer than the matching optimal variants.
No endpoint row is eligible for expansion.

Lowering only the optimal budget from `9.95` to `9.85` crosses a useful
topology threshold. The adaptive median target drops from `10,770` to
`9,745.5` faces, pre-clean degenerates fall from `0.25` to `0`, and hull use
drops from `0.5` to `0.25`. The held-out result improves on three of four
samples; the fourth changes by `-0.0077` versus optimal `9.95`.

## Geometry And Runtime Audit

The winning variant has complete simplification-audit coverage (`4/4`). Its
median simplification drift is:

| metric | median |
| --- | ---: |
| absolute volume change | `0.002300` |
| maximum bbox-extent change | `0.002033` |
| normalized bounds-center shift | `0.000774` |
| normalized surface Chamfer | `0.008525` |
| normalized surface H95 | `0.016281` |

The diagnostic costs a median `0.2019s` and is reported separately. Median
repair time excluding that diagnostic is `2.8748s`; total repair time is
`3.0767s`. Raw Hunyuan inference is `34.6409s` median at `5.8008` GiB peak
CUDA VRAM. Cached repair variants reduce median provider invocation to
`0.0107s` before repair.

## Provenance

- Runtime: `935.85s` for orchestration and evaluation.
- Rewritten four-row manifest SHA256:
  `5c8d61d39e840c003bc36d683188bb94c219fb96b7cc03e59bb1ffcab9852c6f`.
- Input payload: `806,667` bytes, SHA256
  `df95f584fb6ade9126baee5cbd193fa88df94fec74dcca86a5719e3a4b748303`.
- Compact result archive: `1,513,003` bytes, `247` entries, SHA256
  `0c77ea4d78af8ce52c25ab6cc0548d51e8380be5e5abf18faf75c92edf7597ac`.
- Full result archive: `336,432,138` bytes, SHA256
  `1f3740b5db4cff5a9da1d42c0ad55ed480f086a06d0251b4150d9300f3cf9279`;
  intentionally not committed.

`decimator_comparison.csv` contains the six aggregate cells,
`repair_stage_per_sample.csv` contains all twenty-four repair rows, and
`candidate_pairwise_per_sample.csv` preserves the candidate's paired held-out
comparison. The two selector files distinguish the configured endpoint probe
from the measured optimal `9.85` candidate. Source geometry remains
diagnostic-only and no oracle method is present.

Validation: full backend suite `312` passing; complete benchmark/provider
regression slice `206` passing.

## Measured Follow-Up

The prescribed held-out-ten confirmation completed `50/50` rows. Optimal
d9.85 won `9/10` paired objectives against TripoSG but remained `hold`: hull
fallback rose to six of ten, only six rows stayed within fill drift `0.5`, and
two rows exceeded the hard drift limit `4.0`. The
[held-out-ten evidence bundle](../g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10/README.md)
records the result. Hunyuan decimator tuning, endpoint placement, and post-hoc
retriangulation are now closed.
