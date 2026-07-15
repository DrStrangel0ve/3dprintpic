# Relief framing, boundary, and selected-surface v6

This pass targets three visible failures in the 30 mm relief path: faces clipped
by the image frame were flattened at the base border, concave selection corners
could create one-pixel attachment cliffs, and the exact llama/group candidate
was discarded because an unchanged source discontinuity exceeded a global edge
maximum.

The implementation is intentionally local. Positive-context selections preserve
face pixels that touch the frame, concave attachment conflicts receive a one-pixel
projection followed by the unchanged physical cap, and selected-object gradient
compression may defer edge-only failures to an exact post-blend audit. That audit
measures only edges touched by the candidate while retaining the existing detail,
height-span, correction-span, cardinal-edge, and diagonal-edge limits. A failed or
missing audit still routes to the prior bounded fallback. Finite coverage must
match exactly, changed-edge excess is measured independently of source-relative
ratios, and signed direction reversals are recorded. A provisional candidate is
rejected when reversals form at least four cardinal-plus-diagonal edges or one
sparse reversal exceeds four physical steps; the exact llama candidate remains
below those limits at three total reversals and a `1.951`-step maximum.

This follows the relief literature's emphasis on compressing depth while retaining
small discontinuities and perceptual gradients, and the newer normal-integration
work's explicit separation of discontinuities from smooth-domain reconstruction:

- [Digital Bas-Relief from 3D Scenes](https://gfx.cs.princeton.edu/pubs/Weyrich_2007_DBF/index.php)
- [Discontinuity-preserving Normal Integration with Auxiliary Edges](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_Discontinuity-preserving_Normal_Integration_with_Auxiliary_Edges_CVPR_2024_paper.html)
- [Photo2Relief](https://arxiv.org/abs/2307.11364)

## Measured result

The clean ten-scene framing matrix covers small, near-full-frame, left-clipped,
and right-clipped faces plus six earlier scene archetypes. All rows pass at exact
revision `1a3cfc9efd8649e460c4aeba1a93f46fa2214297`. The minimum face correlation
is `0.8691`, all background depth and gradient correlations are effectively
`1.0`, the largest boundary p99/max consumes only `72.95%`/`47.40%` of its
limit, and every mesh is one printable watertight volume.

The clean 12-row 20/30/40 mm topology sweep also passes. Its minimum face and
selected-surface lighting correlations are `0.9896` and `0.9228`; minimum
background depth correlation is `0.999967`; maximum selected-solver cardinal
p99/cardinal max/diagonal max ratios are `1.4851`/`2.1609`/`2.2787` against
`12`/`24`/`24` limits.

The paired exact cached-depth llama replay isolates the newly measured failure.
The old fallback has only `0.1201` selected-surface lighting correlation and
`80.79` degrees p95 normal error. The audited candidate reaches `0.9507` and
`10.73` degrees; its weakest meaningful component reaches `0.8848`. Its exact
changed-edge p99/max ratios are only `1.553`/`4.906`. Background centered RMS
and p02-p98 span are `7.8597` and `22.0333` mm, background depth/gradient
correlations are `0.999921`/`0.988976`, and the actual subject-boundary max is
`2.1217` mm, below the `4.0` mm hard ceiling.

The exact portrait is unchanged by the selected-surface acceptance change. Its
two face correlations remain `0.9524`/`0.9712`, RMS retentions remain
`0.9499`/`0.9735`, and final per-face lighting correlations are
`0.8668`/`0.9665`. Background RMS/span remain `6.8158`/`19.4998` mm.

Both exact STLs are single-component, watertight, manifold,
winding-consistent positive volumes with zero degenerate faces. Mutually
incompatible one-pixel attachment constraints remain reported separately: 13
for the portrait and 41 for the more detailed llama candidate. Every satisfiable
attachment step is at most `0.8000002` mm, and neither output is mislabeled as a
strict attachment success.

Only aggregate exact-input telemetry is tracked. Private images, masks, source
hashes, depth arrays, API responses, previews, and meshes remain local and
ignored.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_scene_regression `
  --output-dir backend/output/relief-framing-boundary-v6-n10

.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief-visual-boundary-v6-n12

.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Clean summary checksums and all aggregate exact-input values are in
`summary.json`.
