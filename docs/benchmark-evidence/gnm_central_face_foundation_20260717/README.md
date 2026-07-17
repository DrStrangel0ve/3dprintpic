# Central part-aware GNM face foundation

Status: **relative improvement pass; absolute face-quality hold**.

This iteration addresses a measured failure in the previous GNM mean-face
foundation: full-face correction improved broad shape but pulled expressive
faces toward a neutral template and reduced local gradient correlation. The
published evidence is privacy-safe scalar telemetry and hashes. Source images,
exact geometry, depth arrays, and STL files remain local and ignored.

## Research and ablations

The official [Microsoft DAViD](https://github.com/microsoft/DAViD) Base ONNX
model was integrated as an isolated research provider, pinned at source
revision `20a3eb66f61d0e1caff42489c6775e142179bd46`. CUDA preflight succeeded on
the local RTX 3080 Ti with mean eight-row inference time of `0.2232 s`. Its
isolated session-load-through-first-inference footprint was approximately
`1.031 GiB`. The model is MIT licensed.

DAViD passed its one-row provider smoke after the central fusion was selected,
so it expanded to the same eight-row selector. It tied the current path at 90
named-part failures and slightly improved median shape correlation and RMSE,
but median gradient correlation moved from `0.443874` to `0.443673`. The
eight-row strict paired gradient gate therefore failed. DAViD remains a
challenger, not a production dependency.

The local ablation then compared:

- A scalar high-confidence strength sweep from `0.00` to `1.00`.
- Gaussian smoothing of the whole-face correction.
- Part-restricted nose, eye, eyebrow, and mouth masks with bounded dilation and
  feathering.

No non-zero scalar strength passed every expression/occlusion gate. The
cross-gate winner keeps the mouth hard core unchanged and applies the GNM
correction only around the detected nose and both eyes. That core is dilated by
3 px and feathered with Gaussian sigma `0.5 px`. The existing one-pixel face
boundary remains exactly zero, and no background value is modified by this
stage.

## Identity-disjoint small-face expansion

The final production-faithful expansion contains 20 validation rows across 8
identities, 8 expressions, yaw `-38/-30/+38`, and clear or eye-band occlusion.
All identities are disjoint from the GNM training corpus.

| Metric | Paired control | Central GNM |
| --- | ---: | ---: |
| Combined six-part failures | 232 | 229 |
| Median shape correlation | 0.808345 | 0.809049 |
| Median gradient correlation | 0.376706 | 0.377193 |
| Median normalized RMSE | 0.188878 | 0.188527 |

GNM applied on 18 rows and reliability-gated itself off on 2. No row added a
named-part failure. The worst individual gradient delta was `-0.006410`, inside
the new hard `-0.01` bound. This bound is now part of the standard corpus
selector so a future aggregate win cannot hide one badly smoothed face.

This is not acceptable absolute face quality: 229 of 240 named-part metric
opportunities still fail (`95.4%`). The selector now requires an absolute
failure rate at or below `50%`, so the overall gate correctly remains hold even
though every relative comparison passes.

## Existing regression gates

The sealed three-row exact gate remains green: failures stay at `19 -> 17`, and
the hard 76 px row stays at `10 -> 8`. On that row, shape correlation is
`0.804892 -> 0.806602`, gradient correlation is `0.623512 -> 0.626152`, and
normalized RMSE is `0.175959 -> 0.175270`. Larger faces remain byte-identical
to control.

The 15-row varied-scene gate moves from `84 -> 82` failures. Two of three small
identities improve, none regress, and all 12 medium/close rows remain exact
control. Median gradient correlation stays improved at
`0.703735 -> 0.707898`.

## 30 mm STL gate

| Metric | Result |
| --- | ---: |
| Named-part failures | 9 -> 8 |
| Background p02-p98 span | 15.9630 mm |
| Paired background correlation | 0.999997 |
| Background gradient correlation | 0.998655 |
| Maximum relief height | 29.9908 mm |
| Far background / ceiling | 18.6174 / 19.5100 mm |
| Feasible attachment maximum / limit | 0.8000002 / 0.8 mm |
| Watertight components | 1 |
| Non-manifold edges / degenerates | 0 / 0 |

The STL has 232,320 faces and 116,162 vertices, consistent winding, positive
volume, and exact shell agreement. It was regenerated with the candidate's own
feature-weight mask; the baseline and candidate mask hashes differ as expected.
Eight mutually incompatible one-pixel attachment constraints remain explicitly
reported; every satisfiable constraint passes.

## Decision

Retain the central part-aware restriction as a safer incremental replacement
for whole-face GNM correction, but do not promote the current face pipeline as
production-quality. Keep DAViD research-only. Absolute failures remain high on
tiny turned expressive faces, and GNM is still a static mean head. The next
useful lane is a maintained, license-compatible camera-aligned face geometry
provider or a larger supervised corpus that can train expression-aware
geometry directly.

## Reproduce

```powershell
backend\.venv\Scripts\python.exe -m backend.benchmark.evaluate_gnm_face_foundation_corpus_gate `
  --corpus-root backend\output\research\gnm_corpus\validation_size_balanced_n32 `
  --cache-root backend\output\research\gnm_corpus\validation_size_balanced_n32_production_cache `
  --reference-corpus-summary backend\output\research\gnm_corpus\identity_stratified_n80\summary.json `
  --output-dir backend\output\research\gnm_corpus\gnm_mean_face_foundation_central_parts_s05_all_small_n20 `
  --row-set all-small --device cuda
```

The command intentionally exits `2` after writing evidence because the
absolute-quality floor remains unmet.

Compact metrics and immutable raw evidence hashes are in `results.json`.
