# SMIRK 105-landmark registration ablation

Status: **hold; production unchanged**.

This privacy-safe run tests whether turned-face failures are caused by simple
image-plane misregistration. It uses SMIRK's native 105 MediaPipe landmark
correspondences, robustly fits a 2D similarity transform, preserves every
provider z value, and rasterizes the registered mesh separately from the
control mesh.

## Result

All 32 registration fits passed their deterministic safety checks:

- median RMS: `0.966092 -> 0.671700` pixels;
- median RMS ratio: `0.737961`;
- minimum inlier ratio: `0.933333`;
- scale range: `0.893474` to `1.107219`;
- rotation range: `-3.853493` to `2.616845` degrees;
- minimum registered refinement-crop coverage: `0.591485`.

The geometric quality gate did not follow the landmark improvement:

| Variant | Applied | Failures | Improved | Tied | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| registered global alpha 0.10 | 32 | 287 | 5 | 22 | 5 |
| registered global alpha 0.25 | 32 | 288 | 6 | 16 | 10 |
| registered pose 0.60 jaw gate | 18 | 280 | 5 | 26 | 1 |
| registered pose 0.75 jaw gate | 32 | 288 | 5 | 19 | 8 |

The strongest registered variant regressed
`mh_caucasian_female__mouth_open_06` by one named-part failure. The unchanged
pose-0.45 control remained selected at `286 -> 277`, with five improved and
zero regressed rows. No registered output advanced to a 30 mm replay.

## Interpretation

The mesh was already accurately aligned in image space. The previously
regressing high-yaw smile row improved from `1.248946` to `0.803471` pixels
of landmark RMS, but its named-part failures still changed from `7` to `9`
under registered alpha 0.10. The remaining limitation is incorrect inferred
3D shape or visibility under pose, not global scale, rotation, or translation.

The next provider should therefore be trained for pose-invariant,
camera-aligned geometry. Post-hoc SMIRK similarity registration is closed.

## Validation

- 707 backend tests passed;
- 72 subtests passed;
- 2 existing warnings;
- focused SMIRK tests: 16 passed;
- Ruff and compileall passed.

Raw evidence is local-only at
`backend/output/smirk_face_training_registered_challenge_n32_20260717/evidence.json`.
Its SHA256 is
`0016f8477938ccf7a7d95d6a9e167b052e3b3f0d390d942095ecf68fca77a7e9`.
