# Coarse-to-fine face geometry challenger

This evidence covers the benchmark-only
`dinov2_pyramid448_coarse_to_fine_normal_face_geometry` challenger at exact
commit `8bf315654e6bfa5970521b7e73245fc80302cce3`.

The architecture predicts broad camera-aligned facial geometry at 40x40,
then refines it at 160x160 with bounded detail. Inference uses only RGB,
local depth, the incumbent baseline, face support, fusion weight, pixel
coordinates, and the frozen DINOv2 pyramid. Exact depth and six named facial
part masks remain training/evaluation-only. Production was not changed.

## Validation

- Focused trainer, gate, and decoder contracts: 39 passed.
- Broader face-training regression suite: 70 passed.
- Validation loss is computed per row, so checkpoint selection is invariant
  to batch packing.
- All execution-critical trainer and metric modules are checksum-pinned and
  dirty-state checked.
- The FNR2R repository had no published license at review time. No FNR2R code
  or weights were imported; only the high-level coarse-to-fine idea informed
  this original implementation.

## Two-row capacity smoke

The exact 3080 Ti run used two distinct 70-pixel-face identities for 32
epochs. The loss fell from `0.3563183546` to `0.2848662287`, a ratio of
`0.7994711050` against the required maximum `0.97`. The capacity gate passed.

- Best epoch: 32
- Trainable parameters: 558,162
- Training runtime: 4.467 seconds
- Peak training VRAM: 0.273 GiB
- Process peak RSS: 2.444 GiB
- Checkpoint: 2,258,017 bytes,
  SHA256 `d24672a36a03c6444a164618952c22ea852f9a68cd8522844982165440bfa6af`
- Evidence: 62,402 bytes,
  SHA256 `11e4fbe46e5f946024c701184c75e24059afc7de0783680a67f2002a0c61ae46`

## Identity-disjoint result

The exact split run used 191 training, 32 validation, and 32 unopened sealed
rows. Training reached its best loss of `0.2353471133` at epoch 16 from
`0.3114751047`.

Every nonzero blend improved the aggregate median shape and RMSE. At full
strength, combined part failures fell from 353 to 316, median shape
correlation rose from 0.820402 to 0.824738, and normalized RMSE fell from
0.186267 to 0.185279. The strict selector nevertheless held the model because
individual named-part metrics regressed. Full-strength output had four
whole-face and 304 named-part regression events.

The lowest regression count was at alpha 0.10: zero whole-face regressions
and 237 named-part regressions. Those events were dominated by span retention
(81), bias (62), per-part physical RMSE (35), and shape correlation (26).
This points to local affine calibration as the next loss target rather than
more broad-form capacity.

- Training runtime: 69.777 seconds
- Peak training VRAM: 0.528 GiB
- Process peak RSS: 3.921 GiB
- Checkpoint: 2,257,953 bytes,
  SHA256 `49a9812dba1ed2aeee8404241ffe4641a60647fc52adfef15033087789ecff23`
- Evidence: 14,452,164 bytes,
  SHA256 `9f0a08e5a86e7d3161caf5f218cb5545b4b6af7602a118f5eb93a33095e3c7e9`

## Decision

`validation_strictly_improves=false`, so sealed evaluation, exact private
photos, background replay, and the 30 mm printability replay remained closed.
The production incumbent and its background-preservation path are unchanged.

The next bounded iteration should add differentiable per-part affine
calibration and worst-part non-regression terms that mirror the selector's
bias, span, physical RMSE, shape, and raw-gradient metrics. It should first
repeat the two-row capacity gate, then reopen the 32-row validation split only
if capacity still passes.
