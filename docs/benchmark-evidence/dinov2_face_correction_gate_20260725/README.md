# Selective DINOv2 face-correction gate

Status: **hold**. Production remains unchanged.

This bounded study tests whether a learned confidence map can expose only the
safe parts of the normalized 448-pixel DINOv2 face-depth correction. The base
correction improves aggregate held-out geometry but regresses at least one
facial-part measurement on every validation identity. The selective gate also
improves aggregate metrics, but it does not rank the strict non-regression risk
well enough to authorize any production correction.

Sealed identities, the private exact-photo replay, and the 30 mm physical
replay remain unopened. The existing background-prominence, cap, attachment,
one-component watertight topology, exact-shell, object/llama, dark-skin,
eyewear, and cast-shadow controls are unchanged.

## Research basis

The design follows two primary-source ideas:

- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) learns a
  selective prediction function and treats coverage as an explicit quantity.
- [Adaptive Confidence Thresholding for Monocular Depth](https://openaccess.thecvf.com/content/ICCV2021/html/Choi_Adaptive_Confidence_Thresholding_for_Monocular_Depth_Estimation_ICCV_2021_paper.html)
  uses learned confidence to separate reliable and unreliable monocular-depth
  estimates.

The implementation is original and purpose-built for this benchmark. It does
not use target depth or synthetic facial-part masks at inference. Its declared
production inputs are:

1. RGB;
2. local and incumbent depth;
3. face support and fusion weight;
4. source-pixel coordinates;
5. the frozen normalized DINOv2 pyramid; and
6. the frozen candidate residual.

Training alone constructs an oracle gate from signed per-pixel depth and
gradient improvement. A small safety head predicts that gate, starts near
complete abstention, and is trained with geometry, oracle classification,
pixel-risk, and total-variation losses. Thresholding produces exact zeros
outside accepted support.

## Reproducibility

Implementation revision:
`0f410ea8739d98e528eb2fb724edb75e95880d1f`.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| Gate trainer | 37,552 | `434adc13da66e75b96d6b324ea208f872d3b00a968bca089b8b6263caa42193c` |
| Frozen decoder trainer | 71,692 | `491a68a2d90869f6f8449972dd35912e903ab41956e6e1f509f779b1ff3c8b8b` |
| Frozen decoder checkpoint | 1,426,985 | `65ca6d8982afcb82f986ad7edfa273ce87872c4e4b92da2388f718e290b6dd31` |

Both exact runs report an empty owned-file Git status. They used Python 3.11.8,
PyTorch 2.11.0+cu128, Transformers 5.13.0, and the local RTX 3080 Ti.

The frozen Apache-2.0 DINOv2 Small encoder remains pinned at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. The normalized pyramid uses
LayerNorm-applied blocks 3, 6, 9, and 12 at 448 pixels, producing
`1536 x 32 x 32` FP16 cached features.

The privacy-safe, identity-disjoint corpus remains bound to:

- corpus summary SHA256
  `67abfa2b59e00f9cf7c0d3ae6a9b66960d7b97203b9d57b5b92ec791216d94bc`;
- cache manifest SHA256
  `f408f166d7cd11fe6c6b9a84b9ac9df9d4d0a78da905f2295dd912b88dd4fe61`;
- ordered cache-row SHA256
  `c23d0a83e03c60cebb66f3229ff36ae456a8578d8a28a426af5129efcaeba5ab`;
- 191 train, 32 validation, and 32 sealed rows.

Source geometry is training and evaluation-only.

## Capacity gate

The exact two-hardest-identity, 32-epoch smoke passed:

| Measurement | Value |
| --- | ---: |
| Initial loss | 1.233459473 |
| Best loss | 1.041627884 |
| Best / initial | 0.844476780 |
| Maximum allowed ratio | 0.970000000 |
| Selected epoch | 32 |
| Trainable parameters | 108,897 |
| Training runtime | 4.649 s |
| Peak training VRAM | 0.1352 GiB |
| Process peak RSS | 2.3925 GiB |

The gate checkpoint is 449,369 bytes with SHA256
`d3638d00f320d2f58c7936ceb15a7ab441dbe206b1fba530cd7ff8720f3bf1a2`.
The exact evidence is 69,015 bytes with SHA256
`b3e9894dab206faee09f4204ace92736c9547915fa96e1873313e7ddb87d81ac`.

## Held-out validation

The 16-epoch identity-disjoint run selected epoch 16. Validation loss fell from
`1.112976462` to `0.309435766`. At that epoch, pixel-oracle precision was
`0.966424`, recall was `0.952285`, predicted soft coverage was `0.456626`, and
oracle coverage was `0.457402`. This apparently strong pixel classification
does not translate into facial-part safety.

| Threshold | Part failures | Shape corr. | Raw gradient corr. | RMSE | Median coverage | Whole / part regressions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 353 | 0.820402 | 0.399444 | 0.186267 | 0.000000 | 0 / 0 |
| 0.00 | **320** | **0.821308** | **0.400626** | **0.186050** | 0.446606 | 1 / 323 |
| 0.20 | 322 | 0.821195 | 0.400375 | 0.186107 | 0.403737 | 1 / 327 |
| 0.35 | 324 | 0.821180 | 0.400288 | 0.186115 | 0.358757 | 1 / 333 |
| 0.50 | 325 | 0.821172 | 0.400138 | 0.186111 | 0.328411 | 2 / 342 |
| 0.65 | 326 | 0.821153 | 0.399885 | 0.186107 | 0.293907 | 5 / 357 |
| 0.80 | 330 | 0.821107 | 0.399553 | 0.186115 | 0.246577 | 6 / 380 |
| 0.90 | 331 | 0.821027 | 0.399228 | 0.186132 | 0.191899 | 11 / 406 |

Every nonzero candidate regresses all 32 validation rows. The threshold-zero
candidate has 323 named-part regression events:

| Part | Events |
| --- | ---: |
| Right eye | 70 |
| Mouth | 56 |
| Left eye | 54 |
| Left eyebrow | 54 |
| Right eyebrow | 50 |
| Nose | 39 |

The dominant failures are span retention (90), affine bias (57), raw-gradient
correlation (49), shape correlation (35), normalized shape RMSE (31), affine
RMSE (30), and affine p95 error (28). Increasing the threshold reduces
coverage but increases strict regressions, so the gate's confidence ordering is
wrong for the actual six-part objective. More threshold tuning is not
justified.

The split checkpoint is 449,369 bytes with SHA256
`19b7966eb7781804de6a666fe5a82e3cb97bf85a9669cef034f1c96643106182`.
The exact local evidence is 14,782,601 bytes with SHA256
`7a98e17a4cc7ef0f6c5847ecde3e1512586b497ab4ace5bc6b70f0ba6e98a4a7`.
Peak training VRAM was 0.2591 GiB and process peak RSS was 5.1013 GiB.

## Decision and next lead

- `validation_strictly_improves=false`
- `sealed_opened=false`
- `advance_to_exact_photo=false`
- `advance_to_30mm_physical_replay=false`
- `production_changed=false`

This selective-gate configuration and training lane are closed. The result
does not rule out every possible selective-prediction method, but more
threshold tuning of this trained gate is not justified and it is not a
substitute for better upstream geometry.

[FNR2R](https://github.com/AutoHDR/FNR2R), introduced in
[Face Normal Estimation from Rags to Riches](https://arxiv.org/abs/2601.01950),
is a useful current research lead because it separates coarse facial shape from
fine normal refinement. Its official repository publishes no license, so its
code and checkpoints fail the product-use preflight and will not be imported.
The next bounded lane will instead implement an original coarse-to-fine
geometry trainer using only license-clean dependencies and verified
camera-aligned supervision. It will use the existing 417 detector-clean Google
GNM training rows and predict bounded dense depth with explicit normal and
gradient losses.

The CC BY 4.0
[C3I-SynFace](https://doi.org/10.1016/j.dib.2023.109087) part-2 archive was
already authenticated and tested in the earlier raw-EXR adapter study. That
120-row residual U-Net improved its synthetic validation and sealed splits but
regressed all three aggregate metrics on the exact production photos, so it
remains closed and will not be repeated. C3I can serve only as an
out-of-domain diagnostic for the new architecture. The exact prior hashes and
metrics are recorded in
`docs/benchmark-evidence/c3i_synface_raw_face_training_20260718`.

An adversarial audit after this run found that validation loss aggregation was
batch-composition dependent and that two execution-critical adapter modules
were absent from the run's code hash list. The raw evidence and compact metric
transcription were independently rechecked and still support the hold. Future
runs now evaluate validation losses per row and hash/status-check every
execution-critical trainer and evaluation module; this closed historical run
was not replayed merely to rewrite provenance.

Focused gate and decoder validation passed 28 tests. Compact values are in
`results.json`; full tensors, checkpoints, and raw regression events remain in
ignored local research output.
