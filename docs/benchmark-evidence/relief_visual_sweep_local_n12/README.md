# Relief appearance sweep at 20, 30, and 40 mm

This directory contains a deterministic, privacy-safe regression for face
naturalness and retained background depth. All inputs are analytic arrays. No
private photo, mask, depth capture, render, or mesh is tracked.

## Coverage

The matrix has 12 rows: four mask topologies at 20, 30, and 40 mm relief.

- Centered subject.
- Off-axis subject clipped by the left frame edge.
- Near-full-frame subject with 63.84% mask coverage.
- Two disconnected subject components with one internal mask hole.

Each row checks per-face physical normals and four deterministic Lambertian
lights, background surface appearance, global and localized background depth,
the physical far-background cap, subject attachment, watertight printability,
and exact agreement between the measured heightfield and the reopened STL.
Pairwise 20/30/40 mm comparisons also reject faces whose normalized shape
changes as relief height increases.

Negative controls reject a flattened face, heavy smoothing, missing candidate
pixels, damage to only one face component, cross-height local face damage, and
localized background deletion.

## Measured failure and solver ablation

The first exact two-face replay exposed a failure hidden by the aggregate face
score. With the old high-relief face solver screening weight of `0.01`, the
larger face had a worst-light correlation of `0.773947` and maximum lighting RMS
retention of `1.405562`; the component gate requires at least `0.80` and at most
`1.40`. Coverage was 100%, so this was geometry drift rather than omitted data.

Nine cached-depth variants isolated the cause:

| Variant | Worst-light correlation | Maximum RMS retention | Component passes |
|---|---:|---:|:---:|
| Control | 0.773947 | 1.405562 | No |
| Face gradient retention 0.90 | 0.777642 | 1.400625 | No |
| Face gradient retention 1.00 | 0.785752 | 1.390144 | No |
| Screening weight 0.03 | 0.799281 | 1.361748 | No |
| Screening weight 0.05 | 0.819375 | 1.319847 | Yes |
| Restoration cap 0.45 mm | 0.776759 | 1.419409 | No |
| Restoration cap 0.30 mm | 0.779186 | 1.442238 | No |
| Face restoration disabled | 0.789453 | 1.476874 | No |
| Retention 1.00 plus screening 0.03 | 0.810070 | 1.349277 | Yes |

The selected change raises only the face-aware high-relief solver screening
weight to `0.05`. It provides the larger appearance margin while keeping the
cardinal edge p99/max ratios at `4.4358` / `10.2297`, below their `12` / `24`
limits. Background depth, the physical cap, topology, and zero-degenerate output
all passed for every ablation row.

On the exact current-code portrait replay, both face components now pass:

- Larger face: worst-light correlation `0.819375`, normal mean cosine `0.978746`,
  p95 normal angle `23.9395` degrees, and maximum lighting RMS retention
  `1.319847`.
- Smaller face: worst-light correlation `0.915401`, normal mean cosine
  `0.981443`, p95 normal angle `22.0691` degrees, and maximum lighting RMS
  retention `1.182159`.
- Final background depth correlation/RMS retention/gradient correlation:
  `0.999996` / `0.999941` / `0.999465` with 100% candidate coverage.
- The result is one watertight manifold component with zero degenerate faces;
  the far-background and satisfiable attachment caps pass.

The llama/group replay has no detected face region, so the face-only solver
change does not alter its surface. Its current final background depth and
appearance comparisons remain `1.0`, with one watertight manifold component and
zero degenerates.

## Clean matrix result

The clean run passed every gate on implementation revision
`b2c82eb99e2a5b18afd7573823ad9f0f47e200c6` in `10.706` seconds.

- 12/12 rows passed; all reopened STLs exactly matched their measured surface.
- Minimum face normal mean cosine: `0.991248`.
- Maximum face p95 normal angle: `15.3139` degrees.
- Minimum face lighting correlation: `0.971095`; maximum lighting MAE:
  `0.031201`.
- Minimum background depth correlation: `0.999774`.
- Cross-height minimum face shape/gradient correlations: `0.997841` /
  `0.993255`; maximum normalized shape RMSE: `0.023164`.
- Every STL is a one-component, watertight, winding-consistent positive volume
  with zero nonmanifold edges and zero degenerate faces.

`summary.json` is 135,590 bytes with SHA256
`7336b7e83b06d9240cddaaf2b77b9b36752f96f6e59ed8e7241fd767d39f7200`.
Generated NPY and STL artifacts remain under ignored `backend/output` paths;
their sizes and SHA256 hashes are recorded in the summary.

Run the matrix from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief-visual-sweep-local-n12-clean `
  --summary-path docs/benchmark-evidence/relief_visual_sweep_local_n12/summary.json
```
