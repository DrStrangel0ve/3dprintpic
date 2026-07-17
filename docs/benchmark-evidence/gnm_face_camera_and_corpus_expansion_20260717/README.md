# Camera-conditioned and expanded-corpus GNM face geometry

This bounded slice tested three upstream changes to the small-face GNM relief
provider while preserving the production 30 mm face/background pipeline:

1. remove the generic DAv2 crop embedding and train from MediaPipe landmarks
   plus blendshapes only;
2. append the Face Landmarker canonical-face rotation from the same inference
   pass; and
3. approximately double the privacy-safe procedural training corpus.

Production remains the central-part mean-GNM incumbent. No candidate recovered
a named facial-part pass on the hard exact row, so no 30 mm or background
promotion replay was warranted.

## Research basis

The current MediaPipe Face Landmarker result exposes optional facial
transformation matrices alongside landmarks and blendshapes. MediaPipe's Face
Geometry documentation defines the transform as the map from the canonical
metric face to the runtime landmarks, describes canonical units as centimeters,
and explains that screen-space landmark Z uses weak-perspective scaling like X.
Those contracts support using the rigid rotation as pose provenance, but not
treating raw landmark Z as metric face depth.

- [FaceLandmarkerResult API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult)
- [MediaPipe Face Mesh and Face Geometry](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)
- [MediaPipe repository](https://github.com/google-ai-edge/mediapipe)

A direct dense interpolation of MediaPipe landmark Z was screened first and
closed: all tested blends increased the hard row from eight to nine named-part
failures. The transform was therefore evaluated only as a conditioning feature.

## Harness changes

- The image feature encoder is now an explicit checkpoint choice:
  `dav2-small` or `none`.
- New checkpoints use exact schema v4 and pin feature widths, MediaPipe version,
  Face Landmarker/name hashes, and encoder ID, revision, model hash, and license.
- Historical schema v1/v2 DAv2 checkpoints remain loadable. Experimental schema
  v3 checkpoints from this slice are intentionally rejected after review found
  that v3 did not pin encoder provenance.
- The optional 4x4 transform is validated as finite affine rigid geometry. Only
  its 3x3 rotation is appended; crop-dependent translation is excluded.
- Transform output stays disabled unless the active provider declares the
  requirement. The transform keyword is likewise passed only to providers that
  declare support, preserving the prior conditioned-provider interface.
- Delta corpus exclusions now require caller-supplied summary hashes, canonical
  row/spec equality, corpus seed and schema checks, a successful source
  preflight, unique/count-consistent rows, and a nonempty result.

The exact insertion harness reproduced the previous structured candidate
bit-for-bit (`maximum_absolute_delta=0`) before evaluating either challenger.
Across all three exact rows, enabling transform output changed neither bbox,
detector, 478 landmarks, nor 52 blendshape scores.

## Results

| Candidate | Validation-small RMSE | Sealed-small RMSE | Sealed win rate | Hard failures | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| DAv2 + structured incumbent | 0.008095 | 0.007009 | 0.80 | 8 | Prior hold |
| Structured only | 0.010082 | 0.008547 | 0.55 | Not run | Closed |
| DAv2 + structured + pose9 | 0.008090 | 0.007008 | 0.80 | 8 | Closed |
| DAv2 + structured, expanded | 0.007489 | 0.006695 | 0.75 | 8 | Hold |

Structured-only training failed sealed RMSE, paired-win, and four eye/brow part
gates. Pose9 made only a `0.0000011` sealed RMSE change. At hard-row strength
0.125 it improved gradient by `0.0000009` but slightly worsened shape and RMSE;
no strength Pareto-dominated the prior checkpoint.

The corpus expansion requested 240 train matrix rows, excluded 49 already
present rows, and rendered 191 unique train-only rows in 952.650 seconds. The
delta is explicitly not described as a balanced cumulative `n240` cohort: its
per-identity row counts range from 3 to 10. Identity/expression-balanced ridge
weights remain active. Strict replay found zero source overlap and zero asset
hash failures. Detector-clean training rows increased from 228 to 417.

The expanded schema-v4 checkpoint materially improved synthetic geometry. The
selector chose identity rank 16, expression rank 24, and ridge alpha 1000;
validation-small paired wins reached 100%. On the hard exact row, strength 0.25
improved shape correlation and normalized RMSE to 0.806652 and 0.175249, but raw
gradient correlation fell to 0.626091 from the production incumbent's 0.626152.
Every tested strength retained eight named-part failures. Full metrics and raw
artifact hashes are in `results.json`.

## Decision

Hold all three candidates. The larger procedural corpus improves synthetic GNM
coefficient prediction but does not close the real-photo domain gap. The next
provider/training lane should emit camera-aligned face geometry from real-image
supervision upstream; landmark-Z interpolation, structured-only regression,
pose9 conditioning, and further tuning of these same coefficients are closed.

The existing face/background relief path is unchanged, including full-scene
background prominence, 30 mm face height, cap and attachment constraints,
one-component watertight topology, and exact shell behavior.
