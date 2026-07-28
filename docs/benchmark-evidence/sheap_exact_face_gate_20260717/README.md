# SHeaP exact small-face relief gate (2026-07-17)

Status: **hold, production unchanged**.

This bounded experiment tests whether camera-aligned FLAME geometry from
[SHeaP](https://github.com/nlml/SHeaP) can repair the measured 74-76 px face
failure without weakening the existing background or attachment boundary. It
uses the exact privacy-safe `small_side_lit_shelves_256` row that has ten
baseline named-part failures.

The experiment uses only CC0 MakeHuman geometry and deterministic procedural
scene data. Source geometry remains evaluation-only. No private image,
biometric embedding, FLAME asset, model weight, or full provider output is
published here.

## Source contract

The adapter pins official SHeaP source revision
`33f21125a1c353e1cb534cf9776fefd9e6cce719` and both official v1.0.0
checkpoints:

| Checkpoint | Bytes | SHA256 |
| --- | ---: | --- |
| Expressive | 348,292,433 | `4d769f493072aa2e98770ed1b71db784bc3ee0a2132a0fd36aab841ee591c5e2` |
| Paper | 348,352,867 | `e79addb1b56beb1cf5198020da27e790fa79681544ce1cece58460af9a923d25` |

Preflight rejects source drift, tracked source changes, missing dependencies,
or asset hash changes. The local FLAME source, its official converted tensor,
and eyelid tensor are separately hash-pinned and never redistributed. The
SHeaP code and weights are CC BY-NC 4.0 and the authenticated FLAME model is a
research-only local asset, so this provider is never production-eligible.

The exact gate additionally pins all 15 measurement-driving artifacts: source,
selection, exact depth, baseline, metadata, local face depth, face mask, six
named-part masks, landmark embedding, and detector model. Its transitive local
source manifest covers seven clean files plus source hashes for every imported
fusion and scoring function. This avoids depending on unrelated dirty module
content while still attesting the exact helper implementations that executed.

The camera path follows the official live-demo contract:

- landmark crop margin `0.9` and upward shift `0.5`;
- integer source slice followed by antialiased 224 px torchvision resize;
- 14.2539-degree perspective FOV and positive camera depth `1 - z`;
- torchvision half-pixel inverse resize back to source pixels;
- nearest visible surface from perspective-correct reciprocal-depth
  rasterization.

The official demo uses `face_alignment`; this harness uses MediaPipe only to
obtain the source crop and registration landmarks. The crop still applies the
official SHeaP landmark formula, and the SMIRK FLAME MediaPipe embedding is
pinned at 4,518 bytes with SHA256
`8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc`.
The MediaPipe face-landmarker model is pinned at 3,758,596 bytes with SHA256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The exact detected face is `[162, 102, 193, 139]`, the detector crop agrees with
it at IoU `0.9375`, and all 478 detector landmarks are output-hash pinned at
`2a7f0ca0376626b97b46e37827e92af6729bbe824f5e3ce776f3ef6cb3288a0c`.
The official-formula head crop is `[141, 76, 211, 146]`.

The run records Python and package versions, CUDA/cuDNN versions, GPU name and
capability, and deterministic Torch settings. Both runs used Python 3.11.8,
PyTorch 2.11.0+cu128, MediaPipe 0.10.35, and the local RTX 3080 Ti. Torch
deterministic algorithms were enabled with seed zero, TF32 disabled, cuDNN
benchmarking disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Exact gate

Each checkpoint runs one inference. The harness evaluates native camera
geometry and a separately labeled, guarded landmark-similarity projection at
provider alphas `0.125`, `0.25`, and `0.50`. Depth is never changed by the
image-space similarity fit. All candidates retain the exact background and
one-pixel attachment boundary.

Both checkpoints select `landmark_similarity_a0.125` and produce the same
strict decision:

| Exact metric | Baseline | Expressive | Paper |
| --- | ---: | ---: | ---: |
| Shape correlation | 0.804774 | 0.805138 | 0.805111 |
| Raw-gradient correlation | 0.625533 | 0.625799 | 0.625810 |
| Normalized shape RMSE | 0.176007 | 0.175860 | 0.175871 |
| Combined named-part failures | 10 | 10 | 10 |
| Strict named-part checks | - | 31 / 42 | 31 / 42 |
| Parts strictly improving shape, gradient, and affine RMSE | - | 3 / 6 | 3 / 6 |
| Provider face coverage | - | 0.928431 | 0.913725 |
| Inference time, seconds | - | 0.562268 | 0.634681 |
| PyTorch peak VRAM, GiB | - | 0.390424 | 0.390424 |

The expressive similarity fit uses scale `1.011152`, rotation `1.463728`
degrees, median residual `0.390839` px, and p95 residual `0.797898` px. The
paper fit uses scale `0.996627`, rotation `1.542264` degrees, median residual
`0.473719` px, and p95 residual `0.995190` px. Both pass positive-scale,
coverage, overall-quality, and exact-boundary guards.

The following named-part checks still fail for both checkpoints:

| Part | Failed checks |
| --- | --- |
| Left eye | affine span |
| Left eyebrow | shape RMSE |
| Mouth | affine bias |
| Nose | affine RMSE, affine bias |
| Right eye | shape correlation, affine RMSE, affine p95, affine bias |
| Right eyebrow | shape correlation, raw-gradient correlation |

## Decision

The tiny whole-face gains are real but insufficient. Neither official
checkpoint improves all six parts, neither reduces the ten named-part
failures, and both fail 11 of the 42 strict per-part checks. No 30 mm STL
replay was authorized and production remains bit-for-bit unchanged.

The expressive checkpoint is the stronger research sibling on shape
correlation, RMSE, coverage, and registration residual; the paper checkpoint
has only a very small raw-gradient advantage. This SHeaP configuration is
closed for the exact small-face correction lane. A future provider must improve
camera-aligned eye, brow, nose, and mouth depth before reaching the physical
relief gate.

## Validation

The focused SHeaP, SMIRK, and VGGHeads provider contracts pass 26/26 tests.
The complete dirty-worktree backend run reports 831 passed tests, 87 passed
subtests, 2 warnings, and 4 failures in the pre-existing production face-relief
path. Those failures are outside the three owned SHeaP files and were not
masked by changing unrelated work:

- canonical oblique 40 mm audited selection solve;
- canonical projected-nose 30 mm absolute gate;
- MakeHuman centered 30 mm background/shell gate;
- MakeHuman right-frame 40 mm relief gate.

## Raw evidence hashes

| Artifact | SHA256 |
| --- | --- |
| Exact source image | `2aca3775162f1b2320a47dbfead27e49e2d7b29654d80151d8ebc17315a8c074` |
| Selection mask | `853d871ac859dc0904cd866bc8e552a55efbc3af8ba59d2da3ee4aac3583d0ef` |
| Exact evaluation depth | `a21f6396c57abd850786e00d40cf2776e048f7b8d0664170e86cdca8fa27a21b` |
| Baseline depth | `417202239d046203027c01d8a9c2f8a6b8a5e057b95385174b144575f5df4fe6` |
| Adapter | `971dc1af4801fcaac1c3154acadee2cf1285cd34f62160e8aefe89135a2bb9c0` |
| Exact-gate evaluator | `c183c25ba9c55620bc38621892b953a8f2b4f1cf21be77dd84177088e1014f7b` |
| Expressive evidence | `cd53535f2f466804bc55b9c29024e9ba64d9e1e17ab9f96a52f6a7ec9318ae83` |
| Paper evidence | `e50aab143af6d3d41da9a254029afd9ef2563789ec17366b95648e7aa582a477` |

The full local arrays and licensed assets remain under ignored research output;
the committed `results.json` contains only compact metrics and provenance.
