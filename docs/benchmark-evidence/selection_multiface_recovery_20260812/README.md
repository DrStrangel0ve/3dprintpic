# Selection Multi-Face Recovery

## Regression

A private, evaluation-only three-person group photo reproduced a selected-relief
regression on the local RTX 3080 Ti. The strict 96-pixel detector accepted one
face and the pipeline stopped, so only one of three selected faces received the
face-depth refinement and downstream 30 mm protection masks.

No private input, selection mask, preview, depth map, or mesh is committed.

## Change

- Face detection for composed selections uses the untouched source photograph.
- The selection mask remains the acceptance boundary, so unselected faces are
  rejected even when they are detected in the source image.
- A bounded completion pass runs only when a selection has fewer than the
  configured maximum number of faces. It uses 5/6 of the strict minimum
  (96 to 80 pixels), keeps the existing 48-pixel hard floor, requires at least
  50% face-mask overlap with the selection, and deduplicates at 0.50 box IoU.
- Whole-image requests retain the original strict detector policy.
- The frontend again defaults to the measured 30 mm face-preservation path.

## Exact Replay

| Measurement | Before | Candidate |
| --- | ---: | ---: |
| Strict face candidates | 1 | 1 |
| Relaxed completion candidates | 0 | 3 |
| Selected faces detected | 1 | 3 |
| Selected faces refined | 1 | 3 |
| Unselected faces accepted | 0 | 0 |
| 30 mm face components measured | 1 | 3 |
| Minimum component detail correlation | n/a | 0.989736 |
| Component detail RMS retention | n/a | 0.859771-0.900595 |
| Face-height quality gates | incomplete coverage | passed |

The 30 mm confirmation used a 512 x 384 height field at approximately
0.501 mm/sample. Its emitted STL had 692,568 faces, one component, consistent
winding, zero degenerate faces, zero non-manifold edges, and was watertight,
manifold, and volumetric.

An exploratory 0.2 mm/sample mesh was rejected: it quadrupled mesh density and
made unstable high-frequency normals more visible. It is not part of the
production change.

## Validation

- Backend: 1,178 tests passed, 129 subtests passed.
- Focused face/API contracts: 86 tests passed, 23 subtests passed.
- Frontend: lint and TypeScript checks passed.
- Browser workflows: 11 passed, 1 integration test skipped by its existing
  environment gate.
- Live local health: PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Ti.
