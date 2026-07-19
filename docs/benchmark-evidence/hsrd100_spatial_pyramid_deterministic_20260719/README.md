# Deterministic HSRD-100 spatial-pyramid face-depth study

This bounded experiment tested whether frozen multiscale Depth Anything V2
features improve camera-aligned depth for the hardest 74-75 pixel turned faces.
It used a public, attribution-safe scan corpus and an equal-parameter control.
The final decision is **hold**: both trained heads beat the unrefined baseline,
but the deepest-only control beat the multiscale challenger on validation.
Production relief generation and the 30 mm background-prominence path remain
unchanged.

## Research and data choice

The source gate uses nine identities from the official HSRD-100 LOD1 release:

- [HSRD-100 dataset](https://huggingface.co/datasets/digitalrealitylab/HSRD-100)
- [HSRD-100 dataset card](https://huggingface.co/datasets/digitalrealitylab/HSRD-100/blob/main/README.md)
- pinned dataset revision
  `9cc5e9c138dd310dc88c95c80e5db0a3ecb7971a`
- data license: CC BY 4.0, Digital Reality Lab

HSRD-100 was selected because it publishes real high-resolution human scans
with a clear reusable license. C3I-SynFace was screened but rejected for
production training because its source paper describes a Reallusion/iClone
generation path, while the current
[Reallusion Content EULA](https://www.reallusion.com/Content/EULA/EULA.htm)
prohibits ordinary content-license use for machine learning, AI training, or
AI-generated output. Reallusion's
[license policy](https://www.reallusion.com/license/content.html) reserves AI
training for separate Enterprise arrangements. No such rights were established
for this project. Private photos and friend scans are excluded.

The frozen feature provider is the official
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small
checkpoint. The exact Hugging Face revision is
`5426e4f0f36572d16453bbda7a8389317b1bef99`, with Apache-2.0 weights.

## Hardened source gate

The final source-gate build contains 60 exact-height rows:

| Split | Rows | Identities | Yaw requirement |
| --- | ---: | ---: | --- |
| Train | 36 | 6 | Both signs |
| Validation | 16 | 2 | Both signs |
| Sealed | 8 | 1 | Both signs |

- exact face heights: 74 and 75 pixels
- accepted views: 30/36 (`0.833333`), above the hard `0.80` minimum
- required extracted source members rehashed against pinned ZIP member bytes
- all 660 generated corpus assets reverified against their recorded hashes and sizes
- emitted asset bytes: 80,831,441
- nearest-visible camera-Z resize prevents invented depth across occlusion edges
- intrinsics follow the actual rounded OpenCV output and half-pixel convention
- detector, model hash, package version, source license, and attribution are
  recorded per row
- HSR0161 is excluded because every published pose has a hat covering the
  cranial surface

The final source summary SHA256 is
`06e0d1fb40fdb013785dfac0912c5e7666751f5cb8de2a1d2b3041cae5956861`.
It differs from the initially measured gate only in the corrected privacy
wording; after normalizing that one field, the two summaries and all 60 row
artifacts are exact. The trainer requires this hash before loading any row.
Compact source-gate telemetry is in `source_gate.json`.

## Compared methods

Both heads have exactly 92,089 trainable parameters and the same initialization,
ordering, conditioning, loss, and optimizer. Depth Anything V2 remains frozen.

The challenger consumes all four 64-channel pre-fusion feature maps through a
24-channel top-down pyramid. The matched control resizes the deepest feature to
all four levels. Both predict a bounded camera-Z residual, limited to `+/-0.30`,
inside face support and its attachment boundary only.

Conditioning includes coarse camera-Z, exact crop rays, face support, attachment
boundary, and three luma-Laplacian scales. Direct losses cover camera-Z,
gradients, camera-ray normals, Laplacian response, attachment, named-part value
and gradient nonregression, residual magnitude, and the worse-axis gradient
correlation used by the promotion gate.

## Deterministic replay

Earlier diagnostic runs were invalidated after a promising apparent promotion
changed on CUDA replay. The authoritative A/B runs enable deterministic PyTorch
algorithms, deterministic cuDNN, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, highest
float32 matmul precision, and disable TF32.

The independent A and B runs produced:

- identical checkpoint SHA256:
  `374bc424885245a7bbf7095ec67aab753ff357f11971f9460b84c533c8047886`
- identical candidate and control histories
- identical validation and sealed selections
- identical decision flags
- runtime: 112.158 and 111.427 seconds
- peak VRAM: 0.3503 GiB on the local RTX 3080 Ti
- Python `3.11.8`, NVIDIA driver `591.86`, PyTorch `2.11.0+cu128`, CUDA
  runtime `12.8`, cuDNN `9.19.0`, and compute capability `8.6`
- Transformers `5.13.0`, Hugging Face Hub `1.22.0`, NumPy `2.4.4`, OpenCV
  `5.0.0`, and Pillow `12.2.0`

The final selector also requires identical row-ID sets and treats each
`(gate, named part)` failure as a tagged item. A new eye, eyebrow, nose, or
mouth failure can no longer be hidden by removing a different failure on the
same row. CUDA must be uninitialized when `run()` begins, and the deterministic
cuBLAS environment is established before imports that can load PyTorch.

## Results

| Validation method | Blend | Failures | Shape corr. | Raw gradient corr. | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.0 | 185 | 0.803557 | 0.606887 | 0.188266 |
| Multiscale candidate | 0.9 | 179 | 0.811476 | 0.611184 | 0.184956 |
| Deepest-only control | 0.7 | **172** | **0.818694** | **0.617693** | **0.181719** |

| Sealed method | Blend | Failures | Shape corr. | Raw gradient corr. | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.0 | 94 | 0.805111 | 0.602064 | 0.180751 |
| Multiscale candidate | 0.9 | **91** | **0.818347** | **0.607162** | **0.174980** |
| Deepest-only control | 0.7 | 93 | 0.809987 | 0.606292 | 0.178900 |

The multiscale candidate is baseline-safe on validation and wins the sealed
slice, but it lacks the required validation advantage over its matched control.
The strict selector therefore sets
`advance_to_exact_photo_and_30mm_replay=false`. Background pixels remain
bit-exact, and no private-photo or 30 mm physical replay was warranted.

## Decision

Hold and close coefficient tuning of this FPN family. The result supports frozen
foundation features plus a small camera-Z head, but it does not support the
multiscale hypothesis. A next training lane should combine materially broader,
identity-diverse camera-depth supervision with the deterministic control
architecture, or test a maintained camera-aligned provider with an equally
strict matched control. Production face height, full-scene background depth,
cap and attachment limits, one-component watertight topology, exact shell,
object/llama depth, dark-skin, eyewear, and cast-shadow gates are unchanged.
