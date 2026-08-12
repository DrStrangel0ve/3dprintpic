# Scale-Independent Relief Sampling

Status: production behavior, August 2026

## Problem

The original printability pass used the requested output footprint for both
geometry processing and STL coordinates. At a 76 mm output size with a
`0.8 mm` minimum feature, it reduced a `384 x 512` relief to `143 x 191`.
Faces that were recognizable at the full 256 mm footprint could therefore
lose most of their samples before the STL was written.

## Production policy

Relief calculation and physical export now use two explicit dimensions:

| Field | Meaning |
| --- | --- |
| `detail_basis_mm` | Full usable printer footprint used for image-detail filtering, minimum-feature resampling, background context, face attachment, and geometry guards. |
| `max_xy_size` | Requested longest X/Y edge of the emitted STL. |
| `target_dimension` | Upper bound on depth and relief samples. |
| `processing_mesh_sample_pitch_mm` | Sample pitch used while shaping the height field. |
| `mesh_sample_pitch_mm` | Actual sample pitch in the emitted STL. This preserves the historical metric contract. |

The frontend sends the printer's full usable footprint as `detail_basis_mm`
for every relief request. The backend calculates the approved Z field at that
basis, crops it if required, and changes only X/Y coordinates when applying
the selected print scale. Legacy callers that omit `detail_basis_mm` retain
the old single-footprint behavior.

`detail_basis_mm` requires `max_xy_size`, and it is clamped so it can never be
smaller than the requested output. This prevents mixed-unit direct API runs.

## Post-scale audit

Recognition-first scaling intentionally does not flatten the retained Z field
after X/Y compression. A smaller print can consequently contain slopes or
feature spacing that a configured nozzle cannot reproduce. The
`emitted_printability` block reports:

- actual emitted sample pitch;
- slope p95, p99, maximum, and violating-edge counts;
- the processing minimum feature scaled into output coordinates;
- whether the run is recognition-first oversampling.

These are diagnostics, not silent geometry edits. Watertightness,
manifoldness, winding, components, and degenerates remain hard STL checks.

## Validation

The exact private group-photo replay was run through the live FastAPI route on
an NVIDIA GeForce RTX 3080 Ti with CUDA 12.8. No private pixels, masks, depth,
or meshes are committed.

| Measurement | 256 mm accepted run | 76 mm scaled run |
| --- | ---: | ---: |
| Relief grid | `384 x 512` | `384 x 512` |
| Processing pitch | `0.500978 mm` | `0.500978 mm` |
| Emitted pitch | `0.500978 mm` | `0.148728 mm` |
| Final X/Y scale | `1.0` | `0.296875` |
| Z-field maximum delta | reference | `0.0 mm` |
| Z-field RMSE | reference | `0.0 mm` |
| Z-field correlation | reference | `1.0` |
| Refined faces | 3 | 3 |
| Worst face-detail correlation | - | `0.990573` |
| Worst face-detail RMS retention | - | `0.854630` |
| Watertight / manifold | yes / yes | yes / yes |
| Components / degenerates | 1 / 0 | 1 / 0 |

The generated 76 mm STL contains 786k-class height-field faces before
silhouette removal and is substantially larger than the former downsampled
file. That size increase is deliberate: the output retains the accepted
height field instead of discarding recognition detail based on print scale.

## Regression coverage

Backend tests assert that:

- a smaller X/Y output and full-size output produce exactly equal Z arrays;
- selection masks and subject-lock behavior preserve the same invariant;
- X/Y bounds change while Z extent does not;
- emitted and processing pitches remain distinct and correctly named;
- post-scale printability telemetry is finite and explicit;
- a detail basis without an output size fails closed.

The Playwright contract verifies that relief requests send the full printer
footprint as `detail_basis_mm`. Lint, TypeScript, focused browser tests, and the
complete backend suite are required before release.
