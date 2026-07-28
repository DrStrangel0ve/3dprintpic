# Pixel3DMM plus MICA identity-prior hardest-face smoke

This bounded research smoke tests whether an official MICA identity prior fixes
the remaining far-eyebrow failure in the strongest Pixel3DMM fit. It uses the
same privacy-safe hardest row as the no-MICA control:
`fusion_caucasian_small_left_256`, a 74-pixel face at -32 degrees yaw. The
production relief path is unchanged.

## Research basis

[MICA](https://github.com/Zielon/MICA) maps an ArcFace identity embedding to
300 FLAME2020 identity coefficients. Pixel3DMM can initialize its identity from
those coefficients and regularize the fitted shape toward them while optimizing
pose, expression, camera, normal, and UV cues. This is an upstream geometry
prior, not a post-hoc depth filter.

The official MICA license permits single-user, non-commercial research use and
does not permit redistribution. FLAME assets have separate registered research
terms. Pixel3DMM and its public checkpoints are CC BY-NC 4.0. This combined lane
is research-only and cannot become a production dependency without separate
commercial rights.

## Reproducibility

- Official MICA revision: `af22e7a5810d474bc28a1433db533723d6bd2b07`.
- Official MICA checkpoint: 502,647,880 bytes, SHA256
  `4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b`.
- Runtime: Python 3.9.25, PyTorch 2.7.1+cu128, RTX 3080 Ti.
- Identity inference took 3.252 seconds and peaked at 0.778 GiB VRAM.
- The paired 800-iteration Pixel3DMM fit took 77.233 seconds and peaked at
  0.215 GiB tracker-only VRAM.
- The official MICA source remained clean. Its Python 2-era FLAME mask pickle
  required an LF-preserving checkout; a normal Windows CRLF checkout corrupts
  the binary payload.

InsightFace could not be installed in the isolated Python 3.9 environment
without a native MSVC build. The bounded smoke therefore used already pinned
PIPNet WFLW98 landmarks to compute the standard five-point ArcFace similarity
alignment. This compatibility alignment is deterministic and fully hashed, but
it is not the official antelopev2 detector path and is not described as such.

## Results

| Metric | No MICA | MICA prior | Delta |
| --- | ---: | ---: | ---: |
| Camera-mesh coverage | 0.829713 | 0.827976 | -0.001738 |
| Shape correlation | 0.963537 | 0.962043 | -0.001494 |
| Raw gradient correlation | 0.721012 | 0.726617 | +0.005605 |
| Normalized RMSE | 0.094608 | 0.096332 | +0.001724 |
| Named-part failures | 1 | 1 | 0 |

The far eyebrow remains the only named-part shape failure. MICA improves its
shape correlation from `0.918516` to `0.936059` and its raw gradient from
`0.720625` to `0.800954`. At the print-relevant 0.8 mm smoothing scale, however,
the same gradient falls from `0.592396` to `0.570144`, below the `0.75` gate.
All six measured part-level affine-millimeter checks pass in both fits, but the
overall affine check remains false because raw camera-mesh face coverage is
incomplete.

## Decision

Hold and close the MICA identity-prior lane. It adds identity-specific raw
detail but does not repair the mid-frequency turned-brow geometry, slightly
worsens global shape and RMSE, and remains research-only under its licenses.
No 30 mm physical replay was run because the exact six-part gate did not clear.

The production 30 mm face/background path, cap and attachment constraints,
one-component watertight topology, exact shell, object/llama depth, dark-skin,
eyewear, cast-shadow, and background-detail controls are unchanged. The next
experiment moves upstream to a larger camera-aligned RGB/depth training corpus
rather than another post-hoc face correction.

Compact measurements are in `metrics.json`; source, checkpoint, alignment,
runtime, and asset provenance are in `runtime_manifest.json`. No checkpoint,
registered asset, mesh, private image, or source geometry is included.

## Validation

```powershell
python -m pytest backend/tests/test_pixel3dmm_full_fit_provider.py -q
```

Focused result: 12 tests plus 17 subtests passed.
