# Landmark-aligned SMIRK face geometry

This experiment tests whether a modern monocular 3D face model can improve
small faces before the depth map is converted into a 30 mm relief STL.
It is a research lane, not a production default.

## Method

The implementation uses the official SMIRK geometry path at source commit
`c7de404c4389f073906a6db1adabf62efcea3f35`. It loads the encoder checkpoint
strictly, emits the native 5,023-vertex FLAME mesh, applies SMIRK's
weak-perspective camera, and rasterizes the front surface without requiring
PyTorch3D.

Small faces are registered with MediaPipe's 478 landmarks. If full-frame
detection misses, the existing production face box guides an upscaled
landmark pass. The crop follows the official SMIRK demo: scale `1.4`, integer
side length, and a 224 by 224 network input.

The rasterized mesh contributes only low-frequency face shape. Existing local
depth retains fine image detail, and the selection taper keeps all pixels
outside the selected subject and its one-pixel attachment boundary bit-exact.

The train-only reliability policy is deliberately conservative:

- Bypass SMIRK when `abs(pose_y) > 0.45` radians.
- Within that pose band, use blend `0.10` when jaw opening is below `0.075`.
- Otherwise use blend `0.25`.
- Use Gaussian scale `1 / 53` of the local face crop.

No exact-depth metric or known expression label is available to this policy at
inference time.

## Exact-depth result

The identity-disjoint challenge contains 32 deterministic CC0 MakeHuman rows:
24 train, 4 validation, and 4 sealed. It spans four expressions and four
camera/background/lighting contexts.

| Metric | Baseline | SMIRK policy |
| --- | ---: | ---: |
| Combined six-part failures | 286 | 277 |
| Improved rows | - | 5 |
| Tied rows | - | 27 |
| Regressed rows | - | 0 |
| Validation failures | 36 | 36 |
| Sealed failures | 38 | 37 |

All continuous median shape, raw-gradient, and normalized-RMSE gates were
non-regressing. Landmark/refinement-box IoU was `0.3483` to `0.4529`, mesh
coverage of the production face crop was `0.6118` to `0.7435`, and every
background and attachment-boundary value remained exact.

The selected policy used the high blend on 3 rows, the low blend on 5 rows,
and bypassed 24 turned faces. This matters: the ungated SMIRK candidate
improved six rows but regressed seven, so unrestricted application is closed.

## 30 mm STL result

Three rows representing train, validation, and sealed splits were replayed
through the unchanged production 30 mm emitter.

| Row | Split | Named-part failures | Background corr. | RMS retention |
| --- | --- | ---: | ---: | ---: |
| `mh_caucasian_male__mouth_open_06` | train | 8 to 7 | 0.999940 | 0.993427 |
| `mh_mixed_female__mouth_open_02` | validation | 12 to 12 | 1.000000 | 1.000000 |
| `mh_mixed_male__neutral_00` | sealed | 9 to 9 | 0.999912 | 0.993979 |

All three candidate STLs are watertight, winding-consistent, single-component
volumes with zero non-manifold edges and zero degenerate faces. Each has
232,320 faces, passes exact shell agreement, and remains within the 30 mm cap
tolerance.

## Comparison and decision

The earlier VGGHeads exact-depth policy reached `286 -> 276` failures with no
raw regressions, but its tested reference-normalized 30 mm follow-up regressed
both held-out rows. That normalization experiment was removed.

SMIRK is the first provider in this face iteration to pass both the broader
identity-disjoint raw gate and a bounded 30 mm STL gate. It still remains
research-only because:

- The official checkpoint needs a separate redistribution/usage audit.
- The FLAME model is a research asset and is not redistributed here.
- The pose policy bypasses most turned faces.
- The physical replay is three rows, not a production-scale print study.

The next useful experiment is a license-compatible camera-aligned provider or
a larger supervised reliability corpus for turned faces. The 30 mm background,
cap, attachment, topology, llama/object-depth, dark-skin, eyewear, and
cast-shadow controls must remain unchanged.

## Reproduce

```powershell
backend\.venv\Scripts\python.exe -m backend.benchmark.evaluate_smirk_face_training_smoke `
  --corpus-root backend\output\face_training_corpus_n320 `
  --cache-root backend\output\face_surface_fusion_cache_n320 `
  --provider-root backend\output\research\smirk `
  --checkpoint-path backend\output\research\smirk\pretrained_models\SMIRK_em1.pt `
  --flame-model-path backend\output\research\smirk\assets\FLAME2020\generic_model.pkl `
  --output-dir backend\output\smirk_face_training_challenge_n32_20260717 `
  --device cpu `
  --suite challenge
```

```powershell
backend\.venv\Scripts\python.exe -m backend.benchmark.run_smirk_face_training_30mm_smoke `
  --corpus-root backend\output\face_training_corpus_n320 `
  --cache-root backend\output\face_surface_fusion_cache_n320 `
  --smirk-smoke-root backend\output\smirk_face_training_challenge_n32_20260717 `
  --output-dir backend\output\smirk_face_training_challenge_30mm_n3_20260717
```

Compact hashes and metrics are in
`docs/benchmark-evidence/smirk_face_geometry_d338586/compact_evidence.json`.

