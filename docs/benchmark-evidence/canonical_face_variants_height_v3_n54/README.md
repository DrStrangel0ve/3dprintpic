# Varied canonical face relief at 20, 30, and 40 mm

This privacy-safe regression uses six deterministic, topology-preserving
deformations of the checksum-pinned Apache-2.0 MediaPipe canonical face. It
measures yaw `-45`, `0`, and `+45` degrees at relief heights `20`, `30`, and
`40` mm on clean implementation revision
`c0bc7ccb81ac724e6cdb3a99b7d406f9dbb2d36a`.

## Result

- All 54 rows pass every source-context, face, named-part, background,
  physical-cap, attachment, shell, and printability gate.
- All 54 STLs are one watertight, winding-consistent, positive-volume
  component with no nonmanifold edges or degenerate faces.
- Minimum named-part shape correlation is `0.9745589`.
- Minimum physical-scale gradient correlation is `0.9697441`. The separately
  retained raw-raster floor reaches `0.6719382`, above its `0.25` anti-collapse
  gate.
- Worst named-part p95 absolute error is `1.2892024` mm and worst RMSE is
  `0.5090800` mm after one face-wide affine fit.
- Minimum face relighting correlation is `0.9961028`; maximum p95 normal angle
  is `1.4522514` degrees.
- Minimum recoverable background-context coverage is `0.6295445` with the new
  `0.50` depth ratio. Final background depth, RMS, span, and gradient retention
  are all effectively `1.0`.
- Far background is at most `20.0100002` mm in the 40 mm rows. Every feasible
  subject attachment remains at or below `0.8000002` mm per neighboring sample.
- Every 40 mm row uses the screened face reconstruction; no destructive
  frequency-separated fallback is emitted.

## High-relief behavior

The initial 40 mm profile matrix rejected the `0.25` screened candidate and
fell back to a surface with 13-14 mm nose RMSE. The selected path retries with
screening `8.0` only when detail loss is paired exclusively with edge-gate
failures. That candidate is provisional: it can be emitted only after the exact
post-blend surface passes a baseline-aware cardinal/diagonal edge audit, detail
correlation and RMS gates, correction/span limits, physical attachment checks,
and complete STL topology checks. A projected-nose case may first try the
existing `0.10` cardinal-p99 retry; a rejected low-detail result can then enter
the same guarded high-detail path.

Face-part metric schema v3 keeps raw-raster gradient correlation as a separate
anti-collapse floor while applying the strict `0.75` correlation gate at the
0.8 mm physical smoothing radius. Raw slope, curvature, and distribution gates
remain active. Thin profile parts retain the full visible support when a
one-pixel erosion would keep less than 35%, and both the attempted and selected
support ratios are recorded. The same support rule applies to millimeter-error
metrics, whose affine-fit schema is independently versioned as v2.

## Evidence

- `canonical-variant-summary.json`: 1,107,212 bytes, SHA256
  `9a3502fee5af8a7cff55891d21c88e8ed514967a21aab8bc60720c908fc8e2fc`.
- `face-40mm-contact-sheet.png`: 461,668 bytes, SHA256
  `da0fc7c08c22f20888effd23d9160288e62eab37220a1488bccf24c69fb917dc`.
- Clean matrix wall time: 274.7 seconds; summed row runtime: 269.5 seconds.
- Backend validation: 467 tests and 44 subtests passed; two existing
  dependency/runtime warnings remain.

The contact sheet is derived from emitted 40 mm surfaces. The full generated
STLs remain local because they total 627,268,536 bytes. The canonical source
surface is diagnostic-only; every scored artifact is the complete emitted STL
shell.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_canonical_face_variant_smoke `
  --output-dir backend/output/canonical_face_variants_height_v3_n54 `
  --relief-heights-mm "20,30,40"
```
