# SAM 2.1 Large still-photo selection promotion

## Scope

This bounded local comparison addresses a production failure where a user needed
18 small masks to assemble three people and could still omit shirt regions. The
source was a private 1600 x 1200 group photo. Source pixels, rendered masks, and
prompt coordinates are not published; only aggregate measurements are retained.

## Revisions and runtime

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GiB |
| Tiny revision | `de431c4043854a71d8101e17995dfe596bf101a5` |
| Large revision | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| Runtime | Python 3.11, PyTorch 2.11.0+cu128, Transformers SAM 2 |
| Harness | `backend/benchmark_selection_models.py` |

The harness ran seven identical face, shirt, and difficult arm prompts through
Tiny and Large, retained all three raw SAM candidates, and applied the exact
production ranking policy.

## Results

| Metric | Tiny | Large |
| --- | ---: | ---: |
| Warm model inference | 26-32 ms | 56-75 ms |
| Peak allocated CUDA memory | 0.275 GiB | 0.700 GiB |
| Left face-vs-shirt selected-mask IoU | 0.9878 | 0.9947 |
| Middle face-vs-shirt selected-mask IoU | 0.9675 | 0.9969 |
| Right face-vs-shirt selected-mask IoU | 0.9628 | 0.9801 |

The right-shirt prompt originally chose a 3.03% shirt-only mask. The revised
credibility-plus-whole-object policy chooses the 12.80% full-person candidate.
The same rule rejects a larger 1.04% difficult-arm/background candidate because
its predicted score is only 0.018, below the 0.20 credibility floor.

A real `POST /selection/mask` smoke with Large returned the full middle person
from one shirt click at 16.368% coverage. Cold model load plus request was
11.847 s; the next warm HTTP request was 0.746 s. `POST /selection/compose`
completed in 0.537 s and released the selection cache before depth inference.

## Flat-base audit

The newest scene STL had an underside `Z` span of exactly `0.0 mm`; its emitted
top border row also had a `0.0 mm` span. The physical base was already planar.
The preview's nonzero camera longitude created the apparent slant, so the viewer
now opens with longitude zero and retains orbit controls.

## Decision

Promote pinned SAM 2.1 Large for still-photo point selection. Keep SAM 2.1 Tiny
for tracked video, where its temporal path and memory budget are evaluated
separately. Preserve no inpainting or generative completion in either route.
