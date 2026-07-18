# Pixel3DMM full-fit hardest-face smoke

This bounded research smoke evaluates a full Pixel3DMM fit on the hardest
privacy-safe face-fusion row, `fusion_caucasian_small_left_256`: a 74-pixel
face at -32 degrees yaw in a 256 px structured scene. The production relief
path is unchanged.

## Research basis

[Pixel3DMM](https://simongiebenhain.github.io/pixel3dmm/) predicts dense
screen-space surface normals and UV coordinates, then fits posed FLAME identity,
expression, head pose, and camera parameters to those cues. Its
[paper](https://arxiv.org/abs/2505.00615) reports training on more than 1,000
identities and 976K images and evaluates both posed and neutral geometry. That
upstream camera-aligned geometry is a better match for this project than another
RGB-to-depth post-process.

The official source and public checkpoints are CC BY-NC 4.0. Registered FLAME
assets are separately research-restricted. The maintained preprocessing wrapper
used here is also CC BY-NC 4.0, while its `face-parser` dependency has no
repository-level license. The complete lane is therefore research-only and is
not production-eligible even if its geometry gates later pass.

## Runtime and provenance

- Official Pixel3DMM source revision:
  `fcd1fa973c7715b02a8948dfc679dff53cf85924`.
- Maintained `easy-pixel3dmm` revision:
  `1d4e55dc3b9189627be0143a113696ec1769ef68`.
- Local runtime: Python 3.9.25, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 3080 Ti.
- The 74 px source did not pass the official two-stage face crop directly. A
  deterministic 4x PIL Lanczos supersample produced one accepted 218x218 crop.
- The full official single-image schedule ran 800 iterations in 77.286 seconds.
  Tracker-only peak VRAM was 0.215 GiB; the complete measured preprocessing
  peak was 3.084 GiB in the face parser.
- The posed mesh has 5,023 vertices and 9,976 faces. The adapter applies fitted
  head pose, OpenGL world-to-camera pose, official intrinsics, exact crop
  inversion, and perspective-correct reciprocal-depth rasterization.

The checked adapter now represents the 4x preprocessing transform directly.
It requires a bounded integer scale, distinct derived-image hash, and exact
parent-source hash. The numeric adapter labels the resize method and derived
dimensions as caller assertions because hashes alone cannot prove a resize;
this evidence replay independently verifies the staged source and derivative.
The replay produces the same 2,024-pixel raster mask as the captured run and
differs by at most `1.1920929e-7` in depth.

## Results

| Method | Coverage | Shape | Gradient | RMSE | Named failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production mean-GNM incumbent | 1.0000 | 0.806602 | 0.626152 | 0.175270 | 8 |
| Raw fitted camera mesh | 0.8297 | 0.963537 | 0.721012 | 0.094608 | 1 |
| Neutral support completion | 1.0000 | 0.657125 | 0.510755 | 0.228550 | 1 |
| Camera-rotated normal solve | 1.0000 | 0.668245 | 0.545082 | 0.225567 | 1 |
| Landmark-gated normal brow | 1.0000 | 0.657340 | 0.506106 | 0.228494 | 1 |
| Symmetric brow residual | 1.0000 | 0.668590 | 0.545573 | 0.225473 | 1 |

The raw fitted mesh improves the incumbent from eight named-part failures to
one. All six shared-face affine millimeter checks pass. Its one global failure
is incomplete head support, and its one local failure is the far eyebrow's
0.8 mm smoothed-gradient correlation.

Camera convention was checked without source geometry: after applying the
fitted head rotation, predicted normal slopes correlate 0.9144 horizontally
and 0.8558 vertically with rendered mesh-depth gradients. A screened solve
raises the far-eyebrow gradient from 0.5924 on the raw mesh, or 0.4440 after
neutral completion, to 0.6897 while retaining all other part gates. Stronger
normal or bilateral-brow corrections can pass the eyebrow at 0.8480 or 0.7573,
but then fail the adjacent far-eye shape gate. Those post-fit corrections are
closed rather than promoted.

## Decision

Hold Pixel3DMM full fit as a research challenger. It is the strongest measured
small turned-face provider so far, but no tested completion passes all six
part-shape and affine-millimeter checks together. Its non-commercial and
unresolved dependency licensing also prevents production use.

The existing 30 mm background-prominence path remains unchanged. No candidate
reached the face gate, so the cap, attachment, watertight topology, exact shell,
object/llama, dark-skin, eyewear, cast-shadow, or background-depth promotion
replays were warranted.

Compact measurements are in `metrics.json`; source, checkpoint, runtime patch,
and registered-asset provenance are in `runtime_manifest.json`. No registered
asset, model checkpoint, fitted mesh, private image, or source geometry is
included.

## Validation

```powershell
python -m pytest backend/tests/test_pixel3dmm_full_fit_provider.py -q
```

Focused result: 12 tests plus 17 subtests passed.
