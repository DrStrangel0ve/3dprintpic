# Upstream face-geometry provider research

Date: 2026-07-18. Scope: one upstream provider for the exact privacy-safe
74-75 pixel turned-face gate after SHeaP and LAM both measured hold. This review
uses only official repositories, project pages, papers, and asset terms.

## 1. Pixel3DMM full fitting: bounded next lead

[Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm) is the only reviewed
provider with available code, downloadable checkpoints, an explicit license,
posed FLAME geometry, and a fitted perspective camera. Pin source revision
`fcd1fa973c7715b02a8948dfc679dff53cf85924`; the repository has no release tag.

This is not a reopening of the closed Pixel3DMM normal-map/Poisson fusion lane.
The proposed challenger uses the official full optimization path: predict dense
UV and normals, fit posed FLAME vertices, retain world-to-camera `R`, `t`, focal
length, and principal point, transform vertices to camera space, rasterize
positive `-z_cam` under the repository's OpenGL convention, and inverse-warp
through the exact 512-pixel crop. Only that floating camera-space triangle depth
is eligible for the six-part gate.

- Official `uv.ckpt`: 2,246,794,577 bytes; available but no upstream digest.
- Official `normals.ckpt`: 1,469,022,184 bytes; available but no upstream digest.
- License: repository-wide CC BY-NC 4.0. Treat checkpoints as no more
  permissive; research-only.
- FLAME: registered 2020/2023 assets under
  [FLAME research terms](https://flame.is.tue.mpg.de/modellicense.html), never
  redistribute and never enable in production.
- Environment: official Python 3.9, CUDA 11.8, PyTorch 2.7/cu118, PyTorch3D,
  and nvdiffrast. Linux or WSL2 is the bounded setup path; native Windows is not
  documented.
- Compute: the paper reports roughly 30 seconds for 500 fitting steps; the
  official one-image recommendation is 800. A local RTX 3080 Ti smoke is
  practical but minute-scale.
- Main risk: the official README explicitly notes weaker UV prediction around
  the eyes, the exact region that failed the LAM gate.

The next action is an isolated preflight. Download both official checkpoints,
record local SHA256 values immediately, pin every FLAME/camera/crop asset, and
run one row only. Stop before fusion if perspective camera-space Z, crop replay,
or any asset term is ambiguous.

## 2. TEASER: fail closed

[TEASER](https://github.com/julia-cherry/Teaser_official) revision
`089d7ef62a3311d2d83c7066478cc78737dff9a4` exposes downloadable model and
landmark weights, but the repository and weights have no license. It also emits
only a weak-orthographic camera `(scale, tx, ty)`, so transformed FLAME Z is
crop-relative rather than reproducible perspective camera depth. Registered
FLAME 2020 is additionally required. Do not allocate GPU work to this lane.

## 3. Pix2NPHM: watchlist only

[Pix2NPHM](https://github.com/SimonGiebenhain/Pix2NPHM) revision
`ab56162cbd9cab2b3cd8033c3f94190cced820b0` currently provides no runnable
source, checkpoints, license, environment, or public camera-output contract.
The project paper is promising, but an exact smoke cannot be reproduced today.

## Decision

Prepare one research-only Pixel3DMM **full-fit** camera-depth smoke. TEASER and
Pix2NPHM are closed until their license/output or implementation blockers change.
No production path changes from this research decision.
