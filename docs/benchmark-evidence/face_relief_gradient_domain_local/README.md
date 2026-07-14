# Face Relief Gradient-Domain Height Sweep

Date: 2026-07-13

This deterministic benchmark isolates portrait geometry from learned depth. One
analytic face/head/neck/torso height field is evaluated at 12, 20, 30, 40, and
50 mm with:

- `legacy_quadratic`: short head mask, one-sided slope shaving, and quadratic
  boundary shift;
- `screened_poisson_neck`: the former 12 mm rigid face core with tapered neck
  attachment;
- `screened_gradient_reconstruction`: soft large-gradient attenuation followed
  by a screened sparse reconstruction of the complete requested relief.

All three variants receive the same `2.0 mm/mm` slope budget. The benchmark
fails its process if the reconstructed 30, 40, or 50 mm rows miss curvature,
cardinal/diagonal edge, or solver quality gates.

Run it from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_face_relief_height_sweep
```

## Result

The rigid-core metric is intentionally favorable to
`screened_poisson_neck`: it compares against a fixed 12 mm face. The new method
instead preserves the expression requested at each output height, so the
relevant signal is curvature correlation and RMS curvature retention versus
that requested surface.

| Height | Method | Requested curvature correlation | Curvature RMS retained | Jaw/neck p95 | Surface step p99 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 30 | legacy quadratic | 0.9495 | 0.4308 | 0.6902 mm | 0.8000 mm |
| 30 | rigid screened Poisson | 1.0000 | 0.4000 | 0.4743 mm | 0.8000 mm |
| 30 | screened gradient reconstruction | **0.9966** | **0.9727** | 1.3745 mm | 2.9858 mm |
| 50 | legacy quadratic | 0.9868 | 0.2438 | 0.6870 mm | 0.8000 mm |
| 50 | rigid screened Poisson | 1.0000 | 0.2400 | 0.5082 mm | 0.8000 mm |
| 50 | screened gradient reconstruction | **0.9949** | **0.9440** | 1.8406 mm | 3.6918 mm |

The fixed-core path preserves the *12 mm* curvature, which is why its
correlation remains 1 while its requested-height RMS retention falls to 0.24 at
50 mm. The reconstructed path preserves 94.4% of requested curvature energy at
50 mm without introducing a rigid face island.

The new response is a compression scale, not a hard neighbor-step cap. On the
50 mm fixture, input edge p99 is 14.103 mm, the attenuated target is 3.254 mm,
and the integrable output is 3.692 mm. Audited structural silhouettes may be
steeper. This semantic change is explicit in telemetry as
`hard_slope_limit_enforced: false`.

The acceptance layer is deliberately separate from the soft compression
response. It rejects non-finite or nonconverged solves, missing face-detail
coverage, face-curvature correlation below `0.80`, curvature RMS retention
outside `0.25-2.0`, cardinal or diagonal edge ratios above the declared
`12x` p99 / `24x` maximum bounds, height-span ratios outside `0.5-1.15`, and
corrections larger than `0.9` of the input height span. A rejected candidate
falls back to the prior face-attachment path and remains visible in telemetry.

## Real Checks

The sibling `face_relief_expression_gradient_local` replay covers two
expression-heavy frontal portraits and one oblique portrait at 30 and 50 mm.
All six solvers converged and all six STLs passed watertightness, volume,
component, winding, and degenerate-face checks. The source images and generated
meshes are not copied into tracked evidence.

`summary.json` is the machine-readable analytic output.
`height_sweep_shaded.png` is a generated three-method, two-light visual audit.
