# Hunyuan3D-2mv Optimal d9.85, Held-Out Ten

This bundle records `g4_stl_first_hunyuan3d_2mv_optimal_d985_s40_n10`,
completed on exact runtime commit
`a674560f7a537c45477c9c028cdfa61831ae52d9`. All `50/50` rows completed on
an NVIDIA RTX PRO 6000 Blackwell Server Edition. The five arms were masked,
mirror, the promoted TripoSG incumbent, raw Hunyuan3D-2mv, and the optimal
placement Hunyuan repair at adaptive density cap `9.85`.

## Decision

The strict decision is `hold`; TripoSG remains the production incumbent.
Hunyuan d9.85 wins `9/10` paired objectives against TripoSG (`0.90`) with
CI95 low `0.4980`, and its within-run score is `0.9742` versus `0.1266` for
TripoSG. It nevertheless fails three repair-safety gates:

- six of ten outputs use a convex hull, above the `0.25` maximum rate;
- only six of ten outputs keep absolute fill drift at or below `0.5`, below
  the required `0.75` coverage;
- two outputs exceed the per-sample maximum drift of `4.0`, at `9.2483` and
  `13.9271`.

All ten final candidate meshes are watertight, volume-valid, manifold,
consistently wound, single-component, and free of degenerate faces. Every
complexity is at or below `9.8499933`. Those final printability checks are not
enough to waive the hull/fill failures: only four pre-clean meshes are already
printable, and only seven have zero pre-clean degenerates.

## Quality Finding

The repaired candidate has strong geometric and view agreement despite the
failed safety gates:

| metric | Hunyuan d9.85 | TripoSG |
| --- | ---: | ---: |
| median held-out silhouette IoU | `0.6748` | `0.5491` |
| median mesh Chamfer | `0.1548` | `0.1956` |
| median mesh H95 | `0.3523` | `0.4608` |

Nine samples improve held-out IoU. The only IoU regression is
`mesh_dir_8a68107048_v00`, at `0.5874` versus TripoSG's `0.6377`; the strict
held-out degradation guard still passes. Raw Hunyuan remains a useful shape
diagnostic at median Chamfer/H95 `0.0488`/`0.1558`, but it is not printable:
the median output is non-manifold, multi-component, and has scale-free
complexity `14.6146`.

The contact sheet is complete and was visually checked. It also makes the
mechanical failure understandable: hull-repaired rows are visibly much
simpler and bulkier than their dense raw meshes, even when silhouette scores
remain good.

## Geometry And Runtime Audit

Simplification-audit coverage is `10/10`. Median drift introduced by the
topology-preserving simplifier is:

| metric | median |
| --- | ---: |
| absolute volume change | `0.000058` |
| maximum bbox-extent change | `0.001273` |
| normalized bounds-center shift | `0.000470` |
| normalized surface Chamfer | `0.009008` |
| normalized surface H95 | `0.017130` |

The simplifier itself therefore remains geometry-preserving; the expanded
slice fails when downstream repair falls back to a hull. Median raw Hunyuan
inference is `34.8155s` at `5.8008` GiB peak CUDA VRAM. The repaired arm
reuses the raw cache, reducing median provider invocation to `0.0116s`.
Median repair time is `3.2609s`, plus `0.2173s` diagnostic time, for `3.4790s`
total.

## Provenance

- Orchestration and evaluation: `2388.395s`.
- Rewritten ten-row manifest SHA256:
  `617576b7ee9a426dcef3e5edb3b94f7e8c44094a3d04ec78eaab1ad2d367e743`.
- Input payload: `1,069,454` bytes, SHA256
  `6a93c55dd97990c1627abd0ae7090304f017408c7a9d4ac3f5a8fe9b31d91af6`.
- Compact result archive: `1,905,135` bytes, `230` entries, SHA256
  `f1396c15cfbfae03bdebec23db3524268f1d51f8e20b479043ffbd026124f4f9`.
- Full result archive: `440,792,618` bytes, SHA256
  `8c6ba52b47e440699765706ea1a2dbb90c8a4ef83cb628753dc2f1263fe5642a`;
  intentionally not committed.

`heldout_ten_comparison.csv` preserves the five aggregate arms,
`repair_stage_per_sample.csv` contains all ten candidate repair rows, and
`candidate_pairwise_per_sample.csv` records direct candidate/TripoSG deltas.
The strict selector, ingest report, provider/GPU preflights, resolved config,
and contact sheet are included. Source geometry remains diagnostic-only and
no oracle method is present.

Validation: full backend suite `313` passing with `6` passing subtests; all
`12` JSON artifacts parse, and the compact aggregate/comparison/repair/paired
tables contain the expected `5`/`5`/`10`/`10` rows.

## Next Experiment

The Hunyuan decimator search is closed. Optimal d9.85 generalized on the
objective metrics but not on printable repair safety, while endpoint and
post-hoc retriangulation were already rejected. The next bounded lane is a
cached TRELLIS.2 repair experiment on its exact five-row slice: retain
components above one percent of area, apply bounded hole closure, use optimal
simplification at cap `9.95`, infer the bbox, and disallow convex-hull
fallback. Compare raw TRELLIS, the existing repaired control, the new repair,
and TripoSG under the same final-STL gates.
