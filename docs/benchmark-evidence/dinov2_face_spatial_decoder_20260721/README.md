# Aligned DINOv2 spatial face-relief decoder

This bounded study tested a frozen modern visual encoder as a direct,
camera-aligned face-depth training lane. The corrected decision is **hold**.
The decoder learned the two-identity capacity smoke and improved aggregate
identity-disjoint validation metrics, but every nonzero blend regressed at
least one paired named-part metric. The sealed split, exact private photo, and
30 mm physical replay were therefore not opened. Production remains unchanged.

## Research choice and provenance

The encoder is the official [DINOv2](https://github.com/facebookresearch/dinov2)
Small model, whose source and pretrained models are Apache-2.0. The local-only
checkpoint is pinned to Hugging Face revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 547 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `preprocessor_config.json` | 436 | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |
| `model.safetensors` | 88,249,960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |

Inference used `transformers.Dinov2Model` 5.13.0 with PyTorch 2.11.0 on an
RTX 3080 Ti. The full 160-pixel source crop is resized to 224 pixels without
the processor's resize or center crop, yielding a source-aligned
`384 x 16 x 16` token grid.

The privacy-safe corpus contains scan-derived parametric GNM faces and
deterministic procedural scenes only. Source geometry is training and
evaluation-only. The exact cached tensors are bound by:

- manifest SHA256 `37efec0884c9f6918a50b4b842cbe76911c06a677f503af14cbffd2a8cff1ac0`
- summary SHA256 `c15d826f7240904284ffbdeb78eeae1a2b193b1eb5c2b62852bc0b9bcf9e5049`
- ordered row-content SHA256 `134b6035fa3c50304a7c4d441ba21bfaa0baf4494785f2cdb8b60e473162c80b`
- 80 rows and 20,592,948 row bytes
- 720 source/depth/mask/part assets, 36,201,444 bytes, ordered content
  SHA256 `8cff6943e1197c85fa73ac5072b94f21ee3b349f42bb9aaaf1a64e4cb39e5f1c`

## Corrected method

The 241,473-parameter decoder combines frozen DINO tokens with nine aligned
channels: RGB, local depth, incumbent relief, face support, production fusion
weight, and source x/y coordinates. It emits a support-limited residual bounded
to `+/-0.35` before the existing production affine removal, correction cap,
and attachment feather.

The loss retains value, gradient, multiscale, normal, curvature, physical
amplitude, and correction terms. It also balances both eyes, both eyebrows,
nose, and mouth, and penalizes the worst named part. Promotion pairs every
validation row and all six parts across shape correlation, raw-gradient
correlation, normalized shape RMSE, affine RMSE, affine P95 error, absolute
bias, and span retention. Whole-face shape, gradient, and RMSE are also paired
per person so cheeks, jaw, or forehead cannot regress behind improved medians.
A gain elsewhere cannot compensate for a worse person or part.

An earlier exploratory result is invalid and is not used in this decision. It
used the processor's resize-and-center-crop transform, flipped already encoded
ViT tokens, and checked only failed-part name sets. The corrected path preserves
the full crop, disables token flipping, binds every cache row and evaluation
asset by content hash, and compares numeric whole-face and part telemetry.

## Capacity smoke

The corrected 32-epoch smoke used two distinct 70-pixel training identities:

- loss `0.2023812830 -> 0.1930403262`
- best/initial ratio `0.9538447594`, passing the fixed `0.97` limit
- selected epoch 32
- encoder: 0.3128 s and 0.0574 GiB peak VRAM
- training: 2.8216 s and 0.1795 GiB peak VRAM
- checkpoint: 981,729 bytes, SHA256
  `0bf2295812438e274cc2fbc489126f445b9c06ddd59dfa0a94e7aa2a3f3dbf0f`

## Identity-disjoint validation

The corrected split used 48 train, 16 validation, and 16 sealed rows. Training
selected epoch 16, reducing validation training loss from `0.2213355452` to
`0.2119721174` in 11.271 s with 0.4415 GiB peak VRAM.

| Method | Part failures | Shape corr. | Raw gradient corr. | RMSE | Whole/part regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent, alpha 0 | 189 | 0.839925 | 0.451201 | 0.179729 | 0 / 0 |
| DINOv2, alpha 0.05 | 189 | 0.840297 | **0.451480** | 0.179529 | 0 / 134 |
| DINOv2, alpha 0.20 | 187 | 0.841349 | **0.451949** | 0.178960 | 0 / 129 |
| DINOv2, alpha 0.50 | 180 | 0.842963 | 0.451563 | 0.178085 | 1 / 134 |
| DINOv2, alpha 0.75 | 177 | 0.843797 | 0.451229 | 0.177625 | 2 / 143 |
| DINOv2, alpha 1.00 | **176** | **0.844081** | 0.451024 | **0.177466** | 2 / 159 |

The aggregate direction is encouraging, especially for shape and RMSE, but no
blend is eligible. The selector therefore returns the incumbent at alpha zero.
At alpha `0.05`, the first recorded regression is the left-eye shape
correlation on `gnm_female_middle_eastern_v03__corners_down_04`,
`0.667715 -> 0.666793`. Every event is retained in the local exact evidence.
The diagnostic checkpoint is 981,545 bytes, SHA256
`96ce46af9708cbee09a6470f784273b3979cf4f7213f9bc5d2d23954ef07281c`.
The local evidence file is 7,283,645 bytes, SHA256
`62b3683d1e2c26667d64e11874908617026bc2aa9e47100232806a376ef92e17`.

## Decision and next experiment

- `validation_strictly_improves=false`
- `sealed_opened=false`
- `advance_to_exact_photo=false`
- production face relief and background handling unchanged

The current validation rows contain only 71-88 pixel faces, while the unopened
sealed rows contain 113-237 pixel faces. The next independent lane should use
the already generated identity-disjoint, size-balanced sets: 191 training rows,
32 validation rows, and 32 sealed rows. Both held-out sets contain 20 faces at
or below 90 pixels, and all three identity sets are disjoint.

Before training, their production tensors must be generated from one clean,
exact commit and content-hashed. The unrelated dirty production files in the
current worktree are deliberately not used to create mixed-version evidence.
No exact private-photo or 30 mm physical run is warranted until a candidate
passes the balanced validation gate.

Compact machine-readable values are in `results.json`. Raw synthetic artifacts
and checkpoints remain in ignored local research output and are not published.
