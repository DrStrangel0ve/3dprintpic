# Face Relief Expression Gradient Replay

Date: 2026-07-13

This local replay checks the production face-aware STL path on two expressive
frontal portraits and one oblique portrait at 30 and 50 mm. Cached face-refined
depth is used so the comparison isolates geometry postprocessing. Source images
and generated celebrity-derived meshes are intentionally not committed here;
`summary.json` pins their provenance, cached-geometry hashes, and source hashes.

The two expression inputs are the official `samples/test_image1.png` and
`samples/test_image2.png` files from SMIRK commit
`c7de404c4389f073906a6db1adabf62efcea3f35`. The oblique input is a retained
local Lena fixture. This is a stress replay, not identity-ground-truth 3D
evaluation.

## Configuration

- cached Depth Anything V2 Large face-refined depth;
- MediaPipe 478-point Face Landmarker and gated relative-z prior;
- 192x192 / 76.8 mm for the expression fixtures;
- 300x300 / 120 mm for the oblique fixture;
- `max_relief_slope=2.0`, giving a 0.804 mm soft-gradient scale at the
  expression resolution;
- screened gradient reconstruction with screen weight `0.01`, gradient
  threshold ratio `1.2`, and boundary anchor weight `64`;
- acceptance gates of `0.80` face-detail correlation, `0.25-2.0` face-detail
  RMS retention, `12x` p99 / `24x` maximum cardinal and diagonal edge ratios,
  `0.5-1.15` height-span ratio, and `0.9` correction-span ratio.

## Measured Results

| Fixture | Height | Input / target / output edge p99 | Solver detail corr / RMS | Feature RMS retained | Runtime | Topology |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SMIRK image 1 | 30 mm | 6.390 / 3.608 / 4.161 | 0.8759 / 0.3927 | 0.9733 | 1.53 s | pass |
| SMIRK image 1 | 50 mm | 10.649 / 3.393 / 5.345 | 0.8148 / 0.2953 | 0.9722 | 1.57 s | pass |
| SMIRK image 2 | 30 mm | 6.236 / 3.558 / 4.624 | 0.8838 / 0.4144 | 0.9563 | 1.51 s | pass |
| SMIRK image 2 | 50 mm | 10.394 / 3.897 / 5.304 | 0.8289 / 0.3121 | 0.9370 | 1.58 s | pass |
| Lena oblique | 30 mm | 5.434 / 3.253 / 3.733 | 0.9889 / 0.8367 | 0.9038 | 4.56 s | pass |
| Lena oblique | 50 mm | 9.057 / 3.519 / 4.566 | 0.9704 / 0.7235 | 0.9677 | 3.97 s | pass |

All six sparse solves converged (`info=0`) in 375-404 iterations and passed
every reconstruction acceptance gate. Every STL is a watertight,
positive-volume, single-component mesh with consistent winding and zero
degenerate faces. "Topology pass" means all of those checks passed.

The solver-side detail metric compares Laplacian structure before and after
whole-surface gradient compression. The feature-retention metric measures the
subsequent eye/brow/nose/mouth update after its local print-safety projection;
they intentionally audit different stages.

## Visual Audit

The two-light height-field renders were reviewed at identical camera and light
settings. Compared with the previous production path:

- SMIRK image 1 retains the asymmetric mouth and widened eyes instead of
  embedding the face in an oval plate;
- SMIRK image 2 retains the open eye, wink, nose, lips, and continuous jaw at
  both heights instead of losing them to triangular falloff;
- the oblique fixture retains the hat brim, eye socket, nose, jaw, and shoulder
  transition instead of producing vertical wedges through the portrait.

This visual audit is intentionally qualitative. A single photograph does not
provide hidden-surface or metric facial ground truth. The deterministic
analytic curvature benchmark remains in the sibling
`face_relief_gradient_domain_local` evidence directory.
