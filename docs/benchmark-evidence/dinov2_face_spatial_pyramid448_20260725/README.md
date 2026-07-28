# High-resolution DINOv2 face feature pyramid

This bounded study tests whether encoder spatial resolution was the main cause
of facial-part regressions in the 224-pixel DINOv2 relief decoder. The final
result is **hold**, with a meaningful but insufficient improvement.

The hardened 448-pixel feature pyramid reduces aggregate part failures and
paired regressions relative to the 224-pixel model. It still regresses at
least one whole-face or named-part measurement for every nonzero blend, so
the selector returns the incumbent at alpha zero. Sealed identities, the
exact private photo, and the 30 mm physical replay remain unopened.
Production is unchanged.

## Provider research

The preferred research lead was official
[DINOv3](https://github.com/facebookresearch/dinov3) ConvNeXt Tiny. Its
Transformers backbone API exposes multi-scale feature maps and the model is
designed for high-resolution dense tasks. The DINOv3 license grants use,
modification, and redistribution subject to its terms.

The official weights are gated. This machine is not authenticated to Hugging
Face and has no cached DINOv3 files, so the preflight failed closed without
trying to bypass the gate. The experiment therefore used the declared
Apache-2.0 fallback: the already pinned DINOv2 Small model.

MetaDepth HyDen was not used because its official repository is under the FAIR
Noncommercial Research License. Previously closed depth and post-hoc face
lanes remain closed.

## Exact method

Implementation commit:
`37647a672f8a2b5f8ac7b6bd3bbcc26b5ae07d0e`.

The historical `final-224` profile remains the default. The new
`pyramid-448` profile:

- resizes the complete 160-pixel source crop to 448 pixels without processor
  resize or center crop;
- obtains 32 x 32 DINOv2 patch grids from blocks 3, 6, 9, and 12;
- applies the pinned encoder's final LayerNorm to every selected stage;
- concatenates them into a `1536 x 32 x 32` tensor;
- caches the pyramid as FP16 and converts only streamed training batches to
  FP32, avoiding a second full host-memory feature stack;
- uses the existing bounded, support-limited residual decoder and exact
  production conditioning;
- retains the same affine removal, correction limit, attachment feather, loss,
  identity split, and numeric promotion gates.

Checkpoint schema v2 binds the encoder profile and projection shape and
rejects incompatible architecture metadata. Historical schema-v1 checkpoints
default to `final-224` and strict-load through the new loader. A focused
compatibility test confirms identical inference tensors from a historical
state dict. A clean-revision 224 capacity replay also passed its original gate
at `0.965524`; CUDA training itself is treated as numerically reproducible,
not bit-exact.

Evidence schema v3 records the exact invocation, clean Git revision, trainer
SHA256, Python/platform/package versions, model and corpus hashes, checkpoint
bindings, GPU memory, and host RSS. The executing trainer was 71,692 bytes
with SHA256
`491a68a2d90869f6f8449972dd35912e903ab41956e6e1f509f779b1ff3c8b8b`.

The exact DINOv2 Small revision is
`ed25f3a31f01632728cabb09d1542f84ab7b0056`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 547 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `preprocessor_config.json` | 436 | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |
| `model.safetensors` | 88,249,960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |

The corpus and cache bindings are unchanged:

- corpus summary SHA256
  `67abfa2b59e00f9cf7c0d3ae6a9b66960d7b97203b9d57b5b92ec791216d94bc`
- cache manifest SHA256
  `f408f166d7cd11fe6c6b9a84b9ac9df9d4d0a78da905f2295dd912b88dd4fe61`
- ordered cache-row SHA256
  `c23d0a83e03c60cebb66f3229ff36ae456a8578d8a28a426af5129efcaeba5ab`
- 191 train, 32 validation, and 32 sealed rows with disjoint identities

Source geometry remains training and evaluation-only.

## Capacity gate

The normalized two-identity, 32-epoch smoke passed the fixed `0.97` gate:

- loss `0.1939330250 -> 0.1867737621`
- best/initial ratio `0.9630838384`
- selected epoch 32
- feature extraction: 0.357 s, 0.0992 GiB peak VRAM
- training: 3.171 s, 0.1937 GiB peak VRAM
- process peak RSS: 2.3654 GiB
- checkpoint: 1,427,425 bytes, SHA256
  `2d0e8e59f1bd8731bc8bd57855cdbd3b30d1b0435f5fa646d21042ce0019ed3d`
- evidence: 57,193 bytes, SHA256
  `99250c681e1c78bd58d3f1363baf47760fb4e92db3213277442801f8c9d15084`

LayerNorm improves this ratio over both the pre-hardening 448 smoke
(`0.968216`) and historical 224 smoke (`0.965568`), so held-out validation was
warranted.

## Identity-disjoint validation

All models use the same 353-failure incumbent baseline:

| Model | Alpha | Part failures | Shape corr. | Raw gradient corr. | RMSE | Whole/part regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 final | 0.05 | **351** | **0.820948** | 0.399856 | **0.186136** | 1 / **238** |
| 448 normalized | 0.05 | **351** | 0.820926 | **0.399918** | 0.186141 | 1 / 239 |
| 224 final | 0.50 | 331 | **0.824861** | 0.401125 | **0.185295** | 3 / 264 |
| 448 normalized | 0.50 | **325** | 0.824815 | **0.402995** | 0.185295 | **1 / 252** |
| 224 final | 1.00 | 324 | 0.825028 | 0.401403 | **0.185191** | 5 / 336 |
| 448 normalized | 1.00 | **312** | **0.825039** | **0.406794** | 0.185197 | **1 / 309** |

The normalized 448 model selected epoch 16:

- validation loss `0.1833086610 -> 0.1709958613`
- 50.443 s training runtime
- 0.9046 GiB peak training VRAM
- 4.1936 GiB process peak RSS
- 352,065 trainable parameters
- 2.311 s frozen-feature extraction for 223 rows
- 0.1761 GiB peak encoder VRAM
- 701,497,344-byte FP16 feature cache

At full strength, normalization removes four more failures than the raw 448
run and twelve more than the 224 run. It also raises raw-gradient correlation
substantially and cuts whole-face regressions from five to one versus 224.
The result still violates the strict no-regression gate.

The remaining 309 named-part regressions at alpha 1.0 are:

| Part | Regression events |
| --- | ---: |
| Mouth | 72 |
| Left eyebrow | 58 |
| Right eye | 54 |
| Left eye | 46 |
| Nose | 44 |
| Right eyebrow | 35 |

The dominant metrics are part span retention, affine bias/RMSE, shape
correlation, and raw-gradient correlation. The first event remains left-eye
shape on the 71-pixel turned
`gnm_female_asian_v03__corners_down_04` row:
`0.752463 -> 0.746064`.

The split checkpoint is 1,426,985 bytes, SHA256
`65ca6d8982afcb82f986ad7edfa273ce87872c4e4b92da2388f718e290b6dd31`.
The exact local evidence is 14,439,999 bytes, SHA256
`2730c2f1faa95e93536c8f883d7e2a850e843935746f08466e96a9cb7ad8ae8a`.

## Decision

- `validation_strictly_improves=false`
- `sealed_opened=false`
- `advance_to_exact_photo=false`
- `advance_to_30mm_physical_replay=false`
- production face relief and background handling unchanged

Higher spatial resolution with normalized stages is retained as the stronger
research baseline, but another resolution increase is not justified. The next
bounded lane should predict correction safety from production-available
features and supervise that gate with training-only oracle improvement labels.
It must not require synthetic part masks at inference, and it must prove zero
numeric regressions on validation before any sealed, private, or 30 mm run.

Focused implementation and face-pipeline validation passed 67 tests. Compact
values are in `results.json`; synthetic tensors, raw regression events, and
checkpoints remain in ignored local research output.
