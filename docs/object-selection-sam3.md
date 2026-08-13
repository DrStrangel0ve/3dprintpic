# Still-image object selection with SAM 3

## Production choice

Still-photo object selection uses the gated `facebook/sam3` checkpoint at
immutable revision `3c879f39826c281e95690f02c7821c4de09afae7`. The local
checkpoint is 3,439,938,512 bytes with SHA256
`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
Weights are not committed or redistributed.

The checkpoint is governed by Meta's
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE), not
Apache-2.0. Access must be requested on the official
[Hugging Face model page](https://huggingface.co/facebook/sam3). The runtime
uses the SAM 3 implementations shipped by the pinned local Transformers
environment rather than copying model source into this repository.

The default runtime is offline-only. For first-time provisioning, accept the
gated model terms, authenticate with Hugging Face, and set
`SELECTION_ALLOW_MODEL_DOWNLOAD=1` for one startup. The loader still pins the
revision above. After the snapshot is cached, remove that setting to return to
fail-closed, local-only loading.

## Why SAM 3 was chosen

The failing production photo was a private 2048 x 1536 night group image with
three overlapping, full-body people. A click on a face and a click on the same
person's shirt had to return the same complete silhouette. Point-only SAM 2.1
Large and SAM 3 Tracker both failed that requirement because their valid mask
candidates represented different semantic scales.

| Exact-photo face/shirt IoU | Left | Middle | Right |
| --- | ---: | ---: | ---: |
| SAM 2.1 Large point prompt | 0.1146 | 0.0875 | 0.0715 |
| SAM 3 Tracker point prompt | 0.0000 | 0.0848 | 0.0684 |
| SAM 3 `person` concept | **1.0000** | **1.0000** | **1.0000** |

At the selected mask threshold of `0.20`, the three SAM 3 person masks cover
17.079% of the frame. The old composed mask covered 21.558%; 21.175% of those
old selected pixels were outside the new three-person union, while the new
union missed only 0.501% relative to the old mask. This matches the visual
failure: the old selection contained substantial rail, road, foliage, and
building regions while still depending on click location for body coverage.

The production open-vocabulary smoke also selected:

| Click target | SAM 3 concept | Mask coverage |
| --- | --- | ---: |
| Bell tower | `building` | 3.946% |
| Right hotel | `building` | 4.946% |
| Road car | `vehicle` | 0.269% |

The generic `object` concept returned no useful candidate and was rejected.
The shipped broad concepts are `person`, `building`, `vehicle`, `animal`,
`plant`, and `furniture`. An unmatched click falls back to the official SAM 3
Tracker head from the same checkpoint. This keeps one checkpoint for the
still-selection feature while preserving a general point-prompt path.

## Runtime design

`POST /selection/precompute` runs the six concepts sequentially through one
resident `Sam3Model`, filters implausibly small and scene-sized masks, closes
only bounded enclosed mask holes, and caches packed masks with the source
image. On the exact photo, 63 retained masks occupy 23.625 MiB after bit
packing. Hover and click requests select the highest-scoring cached instance
under the exact pixel and then remove disconnected components not attached to
that seed.

Cold exact production-code precompute runs took 24.42-35.91 seconds on the
Windows host. The first concept inference took 1.67 seconds and subsequent
concepts took 0.22-0.24 seconds each once the model was warm. Packed person and
building click lookups then took 27-39 ms while unpacking only the selected
mask. The measured full-model peak allocation was 1.965 GiB on an RTX 3080 Ti.
The CUDA model cache is released before depth and face inference. Packed masks
remain available for other active sessions and are bounded by a 12-entry LRU
cache; applying one user's selection does not invalidate another user's work.
GPU inference runs in worker threads behind a process-local inference lock, so
health checks and unrelated API requests remain responsive during precompute.
The final live smoke completed 48 of 48 concurrent health probes successfully.

Mask-hole repair is deterministic binary topology cleanup. It is capped at
0.1% of image pixels per enclosed hole and never synthesizes RGB pixels. It
therefore repairs small shirt-texture holes without filling real openings
between arms and bodies. This is not image inpainting, and no inpainting model
is present in the production route.

## Training and data provenance

This repository did **not** train or fine-tune the SAM 3 checkpoint. Meta's
[SAM 3 paper](https://arxiv.org/abs/2511.16719) describes a scalable data engine
with four million unique concept labels, including hard negatives, across
images and videos. The official repository also publishes
[fine-tuning support](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md),
but local fine-tuning was deliberately not claimed because the measured
inference policy already fixed the failing case and no license-compatible
ground-truth corpus for this product-specific click behavior was assembled.

Local work optimized inference policy, not model weights:

- direct SAM 3 Tracker versus SAM 2.1 Large on identical private prompts;
- `person` mask thresholds `0.20`, `0.25`, `0.30`, and `0.35`;
- bounded versus unbounded binary hole closure;
- person-only versus six-concept precomputation;
- face/shirt consistency, selected coverage, leakage diagnostics, latency,
  peak VRAM, and packed-cache size;
- visual smoke checks for full people, buildings, and a vehicle.

Private source pixels, masks, and coordinates stay under ignored `output/`
paths. Only aggregate metrics, the source fingerprint, code, tests, and
checkpoint provenance are published. The compact record is under
`docs/benchmark-evidence/object_selection_sam3_local_20260810`.

## Regression contract

Backend tests verify bounded hole closure, face/shirt identity, mixed
concept-plus-tracker unions, cached SAM 3 instance reuse, tracker fallback,
artifact emission, first-time provisioning, and per-session cache isolation.
Frontend tests verify that still photos request `sam3-person-aware`, wait for
precomputation, use the packed-mask endpoint instead of live inference, and
never commit stale hover masks. Historical SAM 2.1 evidence remains in
`docs/object-selection-sam2.md` so the replacement decision is auditable.
