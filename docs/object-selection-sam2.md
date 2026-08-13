# Still-image object selection

> Historical decision record. Still-photo production moved to the pinned SAM 3
> person-aware route on 2026-08-10. See `docs/object-selection-sam3.md`.

## Former production choice

Still-photo object selection uses `facebook/sam2.1-hiera-large` at immutable
revision `665f8e2ad61cf5f53d65644ff27c8ee525124610`. The checkpoint and official
[SAM 2 implementation](https://github.com/facebookresearch/sam2) are Apache-2.0
licensed. The runtime uses the Transformers SAM 2 point-prompt API already
present in the backend environment. Tracked video remains on SAM 2.1 Tiny; each
feature still has one production model.

The selection policy removes disconnected islands, rejects scene-sized
candidates above 65% coverage when a smaller mask exists, ignores candidates
below the measured 0.20 credibility floor, and then keeps the largest candidate
within 0.75 of the best predicted IoU. This favors the whole person rather than
a high-confidence shirt, face, or other subpart. Clicks always run at the exact
coordinate; a nearby hover preview is never committed as the selected mask.

## Measured failure and replacement

The production failure was reproduced from locally saved API evidence on a
private 2048 x 1536 night group photo. No private image or mask is committed.
DETR assigned a single segment to a person plus disconnected roads, buildings,
foliage, and railings. The two bad masks covered 10.508% and 20.676% of the full
image, and their union visibly leaked across the scene.

The first SAM 2.1 Tiny pass fixed DETR leakage, but a later 1600 x 1200 hard
photo exposed a second failure: selecting three people took 18 small masks and
could omit shirt regions. Tiny and Large were then run on seven identical face,
shirt, and arm prompts. Private source pixels and masks remain local.

| Measurement | Tiny | Large |
| --- | ---: | ---: |
| Warm inference range | 26-32 ms | 56-75 ms |
| Peak allocated CUDA memory | 0.275 GiB | 0.700 GiB |
| Left face-vs-shirt mask IoU | 0.9878 | 0.9947 |
| Middle face-vs-shirt mask IoU | 0.9675 | 0.9969 |
| Right face-vs-shirt mask IoU | 0.9628 | 0.9801 |

Large selected the same complete silhouette from either the face or shirt for
all three people. The broadened candidate rule also changes the right-shirt
prompt from a 3.03% shirt-only mask to the 12.80% full-person mask while rejecting
the 1.04% low-confidence arm/background region. Meta's published SAM 2.1 table
also ranks Large above Tiny, Base+, and Small on SA-V, MOSE, and LVOS v2. These
are bounded exact-photo measurements, not broad dataset claims.

`backend/benchmark_selection_models.py` is the reusable harness. Prompt
coordinates are supplied separately with `--points-json`, so no private source
or evaluation coordinates are committed. The composed-mask endpoint releases
the still-image model and empties the CUDA allocator before depth and face
inference.

## Regression contract

`backend/tests/test_main_stl_contract.py` verifies that candidate selection:

- keeps the complete object when a smaller subpart has a moderately higher IoU;
- expands a credible shirt prompt to the whole person even when the shirt has a
  much higher predicted score;
- rejects a larger mask when its score is below the credibility floor;
- rejects an over-large scene candidate;
- removes disconnected islands not attached to the click;
- releases the still-image selection cache before relief inference;
- retains the existing selection artifact and STL composition API contracts.

Frontend selection tests verify that the still-photo workflow sends the pinned
SAM model and that a click requests its own mask instead of committing stale
hover state.

## Flat-base verification

The newest failing-scene STL was audited directly: all underside vertices were
exactly `Z = 0.0` with `0.0 mm` span, and the emitted top border row also had
`0.0 mm` span. The mesh therefore has a physically planar print-contact base.
The preview had been using the viewer's default nonzero longitude, which made a
horizontal edge look slanted in perspective. The production preview now opens
at longitude zero while retaining orbit controls.
