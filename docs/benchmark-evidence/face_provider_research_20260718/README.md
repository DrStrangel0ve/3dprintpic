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

- Official `uv.ckpt`: 2,246,794,577 bytes; local SHA256
  `dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8`.
- Official `normals.ckpt`: 1,469,022,184 bytes; local SHA256
  `e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da`.
  Upstream publishes neither digest, so these are transfer identities rather
  than upstream-authenticated checksums.
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

Both checkpoints were downloaded serially from the IDs in the official setup
script and match the documented byte sizes. The pinned source clone is clean,
but the local Linux preflight stops before dependency installation: the present
Ubuntu WSL2 distribution cannot start because Windows reports
`HCS_E_HYPERV_NOT_INSTALLED` and disabled virtualization/Virtual Machine
Platform support. This requires a host configuration change and possibly a
reboot, so it was not changed autonomously. No model inference has run.

The local adapter in
`backend/benchmark/pixel3dmm_full_fit_provider.py` now verifies that public
source/checkpoint transfer independently from runtime readiness. It also
implements the post-fit geometry contract without importing Pixel3DMM: apply
the fitted rigid head transform to the posed PLY, apply the OpenGL
world-to-camera transform, use positive `-z_cam`, project with the clean
`use_hack=False` intrinsics on the 256-pixel tracking grid, inverse-map the
stored crop with half-pixel centers, and rasterize reciprocal depth without
silhouette depth antialiasing. Nvdiffrast viewport coordinates are converted to
array indices with the required final `-0.5` offset. Crop transforms require
matching source hashes, frame IDs, and checkpoint/tracking image sizes.
Synthetic tests cover rotations, near-plane rejection, stale crop metadata,
projection, and perspective-correct depth.

This adapter deliberately reports runtime unready until registered FLAME/MICA
assets, the isolated Linux environment, and a source-landmark overlay replay
are all pinned. The final replay is required because preprocessing resizes via
both OpenCV and Pillow and the upstream crop code computes inclusive-looking
maxima that NumPy actually slices as exclusive bounds.

Validation passed 30 Pixel3DMM/SHeaP/VGGHeads camera-depth tests plus 6
subtests. The full backend suite completed with 908 passed plus 90 subtests and
only the same four pre-existing canonical/MakeHuman relief failures. A bounded
adversarial review caught and then verified fixes for the nvdiffrast half-pixel
array-index offset and stale in-bounds crop provenance.

The next action is an isolated Linux GPU preflight after that infrastructure
blocker changes, or on a genuine available Colab G4. Pin every
FLAME/camera/crop asset and run one row only. Stop before fusion if perspective
camera-space Z, crop replay, or any asset term is ambiguous.

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

Pixel3DMM **full-fit** checkpoint transfer and the fail-closed camera-depth
adapter are complete and tested, while runtime execution is blocked before
setup by the host WSL2 configuration.
TEASER and Pix2NPHM are closed until their license/output or implementation
blockers change. No production path changes from this research decision.
