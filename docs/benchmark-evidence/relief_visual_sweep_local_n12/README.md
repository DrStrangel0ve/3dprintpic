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

Each row checks per-component physical normals and four deterministic
Lambertian lights for faces, selected non-face surfaces, and background regions.
It also checks global and localized background depth, the physical
far-background cap, subject attachment, watertight printability, and exact
agreement between the measured heightfield and every reopened STL facet:
top, bottom, and exposed walls. Pairwise 20/30/40 mm comparisons reject faces
whose normalized shape changes as relief height increases.

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

The llama/group replay has no detected face region, so this face-only ablation
does not alter its surface. The independent selected-surface ablation below does
exercise that input. Its background depth and appearance comparisons remain
`1.0`, with one watertight manifold component and zero degenerates.

## Selected-surface failure and solver ablation

Independent selected-surface scoring exposed a second blind spot. The face and
background passed, but the unprotected selected surface was over-compressed as
relief height increased. Six of 12 rows failed. The near-full-frame 40 mm row
fell to `0.493615` worst-light correlation, `0.895348` normal mean cosine,
`0.561853` normal p05 cosine, and `55.8160` degrees p95 normal error.

A bounded one-row ablation varied only the selected-object screened data term
and recoverable-gradient retention. The same near-full-frame 40 mm row was used
for every variant:

| Screening / retention | Worst-light correlation | Normal p05 | p95 angle | Pass |
|---|---:|---:|---:|:---:|
| 0.01 / 0.50 control | 0.493615 | 0.561853 | 55.8160 | No |
| 0.02 / 0.90 | 0.557070 | 0.628113 | 51.0890 | No |
| 0.05 / 0.90 | 0.653150 | 0.664202 | 48.3789 | No |
| 0.10 / 0.90 | 0.717714 | 0.695961 | 45.8962 | No |
| 0.20 / 0.90 | 0.778278 | 0.757221 | 40.7802 | No |
| 0.30 / 0.90 | 0.812140 | 0.801346 | 36.7411 | No |
| 0.50 / 0.90 | 0.850441 | 0.849354 | 31.8585 | No |
| 0.75 / 0.90 | 0.872536 | 0.872687 | 29.2276 | Yes |
| 1.00 / 0.90 | 0.883616 | 0.882618 | 28.0402 | Yes |

Both passing variants cleared the first synthetic matrix, but the exact
llama/group replay exposed a weaker fourth component that the aggregate still
hid. At screening `1.0`, that component had only `0.745409` worst-light
correlation, `0.779480` normal p05 cosine, and `38.7870` degrees p95 normal
error. A second bounded ablation therefore tested the exact cached input:

| Screening / retention | Worst component correlation | Normal p05 | p95 angle | Edge max | Result |
|---|---:|---:|---:|---:|:---:|
| 1.0 / 0.9 | 0.745409 | 0.779480 | 38.7870 | 18.5282 | Fail |
| 1.5 / 0.9 | 0.792347 | 0.898020 | 26.1005 | 20.4689 | Fail |
| 2.0 / 0.9 | 0.843994 | 0.957626 | 16.7388 | 21.9396 | Pass |
| 2.0 / 1.0 | 0.843828 | 0.957545 | 16.7548 | 21.9397 | Pass |
| 3.0 / 0.9 | rejected before emission | - | - | 24.0606 | Fail closed |
| 4.0 / 0.9 | rejected before emission | - | - | 25.5308 | Fail closed |

The selected setting is therefore `2.0 / 0.90`. It is the first candidate that
passes every measurable exact component, while `3.0` crosses the existing
cardinal edge limit of `24`. The exact portrait replay also passes at this
setting: selected-surface aggregate correlation is `0.939763`, its two
measurable components are `0.894192` and `0.938591`, and cardinal edge p99/max
ratios are `4.6718` / `23.3351`. Both face components, background depth, the
physical caps, and one-component watertight topology remain valid.

The exact llama/group confirmation records full appearance, background-depth,
cap, and mesh telemetry on commit `9467322`. Its four measurable selected
components are `0.966550`, `0.951022`, `0.988312`, and `0.844021`; one isolated
one-pixel selection fragment is explicitly excluded as unmeasurable. The main
background component and all global/local background-depth checks pass. One
enclosed 144-pixel background region has no 64-sample interior after the strict
1.5 mm boundary exclusion, so it remains explicitly unmeasured rather than
being counted as a component pass. The final STL is still one watertight,
manifold, consistently wound positive volume with zero degenerates.

A separate end-to-end regression exercises 0.4 and 0.2 mm physical sample
pitches; the finer path fails closed to its protected baseline if its candidate
exceeds an edge guard, while both outputs clear the appearance and
physical-emission gates.

The disconnected topology now measures two selected components and two
background components independently. At 40 mm its aggregate selected and
background worst-light correlations are `0.919790` and effectively `1.0`.
Focused negative controls retain identical sampled heights while changing the
top diagonal, removing side walls, or reversing one wall; all three are
rejected by the facet/shell contract.

## Clean matrix result

The clean schema-v2 run passed every gate on implementation revision
`9467322e59f681f449e4b86358105d2b51bb5e80` in `14.773` seconds.

- 12/12 rows passed; all reopened STLs exactly matched their complete expected
  shells.
- Minimum face normal mean cosine: `0.991248`.
- Maximum face p95 normal angle: `15.3139` degrees.
- Minimum face lighting correlation: `0.971095`; maximum lighting MAE:
  `0.031201`.
- Minimum aggregate selected normal mean cosine and lighting correlation:
  `0.980935` / `0.895048`. Across individual selected components, the minimum
  lighting correlation and normal p05 cosine are `0.847740` and `0.888539`;
  maximum p95 normal error is `27.3097` degrees.
- Minimum background depth and lighting correlations: `0.999774` / effectively
  `1.0`.
- Maximum selected-solver cardinal p99/cardinal max/diagonal max ratios are
  `10.3559` / `17.1018` / `12.0929`, below their `12` / `24` / `24` limits.
- Cross-height minimum face shape/gradient correlations: `0.997841` /
  `0.993255`; maximum normalized shape RMSE: `0.023164`.
- Every STL is a one-component, watertight, winding-consistent positive volume
  with zero nonmanifold edges and zero degenerate faces.
- Every row has `58,560` expected shell facets. Minimum oriented shell-normal
  cosine is `0.999999999989`; maximum shell-coordinate error is
  `0.000001526` mm.

`summary.json` is 236,630 bytes with SHA256
`22b7fcf7820968a538e5f2de6720036217655098d696e9cdd01e71ce34962969`.
Generated NPY and STL artifacts remain under ignored `backend/output` paths;
their sizes and SHA256 hashes are recorded in the summary.

Run the matrix from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief-visual-sweep-local-n12-clean `
  --summary-path docs/benchmark-evidence/relief_visual_sweep_local_n12/summary.json
```
