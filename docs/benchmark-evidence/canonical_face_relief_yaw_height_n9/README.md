# Canonical face relief fidelity at 20, 30, and 40 mm

This privacy-safe regression measures absolute face-surface fidelity, background
depth retention, and STL emission on implementation revision
`f6a473c3f9f25b0f6e58e36c2574663aefd687f6`.

The face oracle is MediaPipe's 468-vertex canonical face model at source commit
`a908d668c730da128dfa8d9f6bd25d519d006692`. The Apache-2.0 asset and license
are checksum-pinned in `backend/benchmark/assets/mediapipe_canonical_face/`.
The open canonical surface is used only as a diagnostic geometry oracle; the
exporter still emits the complete printable relief shell.

## Method

The harness renders yaw `0`, `-30`, and `+30` degrees at 20, 30, and 40 mm.
Face-part labels share the primary triangle z-buffer, so occluded landmarks
cannot leak through the visible surface. One shared face-wide affine fit removes
only global height scale and offset before millimeter errors are reported for
the eyes, eyebrows, nose, and mouth. Independent gates also cover normals,
relighting, shape, gradients, slope, curvature, background depth and gradients,
physical caps, attachment, topology, and exact STL shell agreement.

The face solver first uses screening weight `0.25`. A `0.10` retry is eligible
only when the complete primary failure set is exactly `cardinal_edge_p99`; the
retry must pass every unchanged compression gate. The final face blend is
clamped to the selected subject, then the exact bounded surface is audited
again. Updated edges use the existing baseline-aware rule while raw physical
ratios are retained separately. Flat, missing, nonfinite, or incomplete detail
telemetry fails closed.

## Results

- All 9 canonical rows pass every gate on a clean implementation revision.
- Minimum face relighting correlation: `0.9993846094`.
- Minimum mean-normal cosine: `0.9999540248`; maximum p95 normal angle:
  `1.2414781` degrees.
- Worst named-part p95 absolute error after the shared affine fit: `0.1106409`
  mm.
- Minimum named-part shape/gradient correlation: `0.9995394` / `0.8932212`.
- Minimum background depth/gradient correlation: effectively `1.0`.
- Every output is one watertight, consistently wound, positive-volume manifold
  with zero degenerates and an exact 232,320-facet shell.

The only retry is yaw `-30` at 40 mm. The primary cardinal p99 ratio was
`13.2044481`; the `0.10` candidate reduced it to `10.9943959` under the same
`12` gate. Before bounding, the face blend would have changed the background by
up to `1.3162689` mm. The emitted candidate changes it by exactly `0` mm. Its
post-blend baseline-aware p99/max ratios are `1.0360189` / `2.8971316`; raw
physical p99/max remain recorded at `2.1826894` / `9.4849420`.

The companion deterministic 12-scene matrix also passes every row, cross-height
face comparison, negative control, background/selection appearance gate,
selection-solver gate, physical cap, exact shell check, and printability check.
Its minimum face and background relighting correlations are `0.9900043` and
`0.9999999999999998`; minimum background depth correlation is `1.0`.

## Evidence

- `canonical-summary.json`: 118,623 bytes, SHA256
  `5b0d1eaf0a7d33f275da59344240e79f1ecd92ae1346c7a88469f1e1b56b4f9f`.
- `analytic-summary.json`: 635,019 bytes, SHA256
  `872c212c07bc545439e0852bfdd7db85fe137c8d0377667e75e4c9ac2aa4369a`.
- Full backend validation: 459 tests and 32 subtests passed; two existing
  dependency/runtime warnings remain.
- Independent final review found no remaining P1 or P2 issues.

Historical exact-input portrait and llama measurements remain documented
separately. Their original private selection masks were intentionally not
tracked, so this revision does not claim a fresh exact-input replay from the
cached depth arrays alone.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_canonical_face_relief_smoke `
  --output-dir backend/output/canonical_face_relief_yaw_height_n9 `
  --yaws-deg "0,-30,30" `
  --relief-heights-mm "20,30,40"

.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief_visual_sweep_adaptive_retry_clean_n12
```
