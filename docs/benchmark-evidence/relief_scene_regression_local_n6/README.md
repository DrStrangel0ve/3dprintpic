# Privacy-safe 30 mm relief scene regression

This bounded CPU replay tests the production background-context and face-detail
guards on six deterministic analytic scenes. It contains no private photo,
selection mask, depth capture, or user mesh.

The matrix varies portrait placement, expression, yaw, foreground-to-background
contrast, and broad background structure: a planar room, framed wall, shelving,
soft outdoor forms, a split-depth wall, and a deliberately close low-contrast
background. Every scene uses a 30 mm relief, 0.4 mm sample pitch, 2.0 mm/mm slope
budget, and the production 45% background-depth budget.

## Algorithm under test

`full_scene_depth_with_bounded_background_context_v3` makes three changes that
matter for scene reliefs:

- Source context is mapped into the physically recoverable capacity above the
  required support ramp instead of being suppressed by that ramp.
- Face/head solver updates are clipped to the user's selected subject. The
  physical cap, rather than the face solver, owns the printable attachment to
  the surrounding scene.
- Background retention compares the final processed foreground plus original
  background against the emitted result, so required support-ramp changes are
  held constant. Overlapping 6 mm windows reject localized structure loss that
  global correlation can hide, and unavailable telemetry fails closed.

## Result

The clean run passed every gate on implementation revision
`7ff402422511f5dcaa83ad16af4eae10990fca4c`.

- 6/6 scene gates passed with no fallback.
- Minimum normalized source-context correlation: `1.000000`; minimum physically
  recoverable context coverage: `0.712951`.
- Minimum face detail correlation: `0.954464`; minimum face RMS retention:
  `1.016996`.
- Minimum final background correlation: `0.999981`; minimum gradient
  correlation: `0.999497`.
- Minimum background RMS/span retention: `0.999788` / `1.000000`.
- Minimum local-window correlation: `0.999933`; maximum local shape-error ratio:
  `0.011637`.
- Maximum far-background height: `13.5100002 mm`; maximum satisfiable attachment
  jump: `0.8000002 mm`.
- All six outputs are one-component, watertight, winding-consistent printable
  volumes with zero nonmanifold edges and zero degenerate faces.
- Thirteen mutually incompatible one-pixel boundary constraints were reported;
  they are excluded from satisfiable-constraint certification and are not
  mislabeled as strict cap passes.

The adversarial controls all pass. Whole-background flattening is rejected; a
localized object deletion is rejected in five windows even though its global
depth/gradient correlations remain `0.9831` / `0.8251`; and insufficient
background coverage is reported as unavailable instead of passing silently.

## Exact-input replay

The two local user-input replays are intentionally not included in this evidence
directory. Aggregate current-code results are recorded here without source
images, masks, depth arrays, renders, meshes, or job identifiers.

- Portrait: final background correlation/RMS/gradient correlation are
  `0.999993` / `0.999933` / `0.999313`; minimum local correlation is `0.997440`.
  Final face correlation/RMS retention are `0.964652` / `0.935924`, with both
  face components above `0.960153` / `0.919891`.
- Llama/group: normalized source-context correlation is `1.000000` over `83.18%`
  physically recoverable context. Final background depth, RMS, span, gradient,
  and local-window retention are `1.000000` relative to the identically capped
  final-foreground reference.
- Both emit one watertight manifold component with zero degenerate faces, far
  background at or below `13.5100002 mm`, and every satisfiable attachment at or
  below `0.8000002 mm`.

Run the privacy-safe matrix from the repository root:

```powershell
python -m backend.benchmark.run_relief_scene_regression `
  --output-dir backend/output/relief-scene-regression-local-n6 `
  --summary-path docs/benchmark-evidence/relief_scene_regression_local_n6/summary.json
```

The compact per-scene telemetry is in `summary.json`. Generated NPY and STL
artifacts remain under the ignored `backend/output` directory.
