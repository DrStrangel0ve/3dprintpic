# Named face-part relief evidence

This evidence validates facial-feature stability independently at 20, 30, and
40 mm relief heights. The privacy-safe matrix uses analytic inputs only. The
exact-input file contains aggregate measurements only; no private image, mask,
depth map, surface, render, or mesh is tracked.

## Reproduction

Implementation revision: `f2c6cb8644a2f4d1d5736972a855023f2698c432`

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief_visual_sweep_named_parts_clean_n12
```

The provenance audit covers the exporter, face-mask producer, named-part
metrics, relief regression harnesses, and mesh renderer. It was clean for the
recorded run.

## Result

- All 12 analytic rows pass all appearance, physical-cap, complete-shell, and
  printability gates.
- All 12 cross-height topology comparisons pass whole-face and named-part
  gates.
- Worst named-part shape correlation: `0.9999505343`.
- Worst named-part face-normalized RMSE: `0.0089420188`.
- Worst named-part gradient correlation: `0.9923251807`.
- Mouth flattening and nose oversharpening negative controls are both rejected.
- Every STL is a single watertight, consistently wound positive volume with no
  nonmanifold edges or degenerate faces and exactly matches its complete
  measured shell.

The full machine-readable run is `summary.json` (636,204 bytes, SHA256
`c14769ceb78b48be65cd8a685668835668a31ac568423dfb7dfe80951382fb4d`).
`exact-private-summary.json` records only aggregate measurements from the fixed
two-face replay.

## Decision

On an accepted screened high-relief face reconstruction, the later RGB-derived
feature emboss is suppressed while the bounded feature bridge remains active.
The emboss duplicated and distorted geometry already reconstructed by the face
solver. Lower-relief and rejected-screening paths retain their previous
behavior.
