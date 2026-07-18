# LAM camera-depth small-face gate

Status: **hold; LAM lane closed; production unchanged**.

This is a bounded upstream-geometry test on the privacy-safe
`small_side_lit_shelves_256` MakeHuman row. Its detected face is only 31 x 37
pixels inside a 74-75 pixel turned-head framing, the measured regime where the
current relief pipeline is weakest. Source geometry is used for evaluation
only. No private image or private geometry is published.

## Provider and runtime

- Provider: [official 3DAIGC/LAM](https://github.com/3DAIGC/LAM), current
  reference revision `339573649dd93df4cba8093a964e85a80d1b61f3`.
- Windows bundle source revision:
  `4368fb6a5d5a930af6b1af79ef76392f8a6c4278`, with the exact tracked diff and
  permitted untracked runtime files pinned by hash.
- Model: [3DAIGC/LAM-20K](https://huggingface.co/3DAIGC/LAM-20K), revision
  `a710cd3c40c86ffe3fc572e895ed12f3ead47289`, 2,356,556,212 bytes, SHA256
  `f527e6e78fd9743aad95cb15b221b864d8b6d356c1d174c0ffad5d74b9a95925`.
- License: Apache-2.0 source and CC BY-NC 4.0 primary weights. Auxiliary
  detector, matting, landmark, and FLAME asset terms are mixed or not
  separately established, so every asset is explicitly marked
  production-ineligible. This challenger cannot be enabled in production.
- Runtime: Python 3.10.11, PyTorch 2.7.0+cu128, CUDA 12.8, local RTX 3080 Ti.
  Native inference took 2.613 s and peaked at 3,600,306,176 allocated bytes;
  tracking, model loading, inference, and output took 47.437 s.

The official prebuilt Gaussian rasterizer attempted an impossible 156.02 GiB
allocation for a bounded 20,018-Gaussian render. The run therefore uses a clean
rebuild of the official dependency at revision
`8829d14f814fccdaf840b7b0f3021a616583c0a1`, GLM revision
`5c46b9c07008ae65cb81ab79cd677ecc1934b903`, with extension SHA256
`2b67a0b15d1956715c071416313a2eade8c98212371261dd1c257b65c8edffaa`.
The exact CUDA archives, MSVC compiler, Windows SDK headers/libraries, native
rasterizer smoke, model assets, and source state are checked before inference.

## Depth contract

LAM's `comp_depth` is alpha-weighted, not an expected camera-depth sample. The
adapter retains that native field. Render-space diagnostics divide by
`comp_mask` only where `comp_mask >= 0.01`; source-space output first bilinearly
warps the weighted numerator and alpha denominator through the captured crop
affine, then divides. The evaluator independently recomputes both contracts.
All four replay checks passed. A review caught and rejected the earlier
divide-before-warp attempt; no metrics from that invalid run are retained here.

The normalized source depth had 2,774 finite pixels, 0.742730 selection overlap,
and p05/p50/p95 camera depths of 0.762850/0.811826/0.863095. Coverage over the
incumbent's exact local face mask was 1.0.

## Exact result

| Variant | Shape corr. | Raw gradient corr. | Normalized RMSE | Part checks | Strictly improved parts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production incumbent | 0.804774 | 0.625533 | 0.176007 | 42 baseline references | n/a |
| Raw LAM diagnostic | 0.680725 | 0.551414 | 0.241800 | fails all six shape and affine parts | 0 |
| Selected LAM fusion, alpha 0.03125 | 0.804709 | 0.625500 | 0.176033 | 15/42 | 0/6 |

All seven nonzero strengths from 0.0078125 through 0.5 regressed at least one
strict part metric. The selected 0.03125 strength changed at most 0.003172 in
normalized production depth, preserved background and attachment-boundary
values exactly, and still regressed global shape, gradient, and RMSE. The
right eye was the clearest failure: shape correlation fell from 0.521052 to
0.441739 and raw-gradient correlation fell from 0.747069 to 0.718054.

The exact gate therefore emitted a deliberate hold and skipped the 30 mm STL
replay. LAM is closed for this face-relief path; further blending of the same
geometry would only make a measured regression larger.

## Reproduction

The provider preflight and run are implemented in
`backend/benchmark/lam_depth_provider.py`. `lam_exact_gate_payload_v2.zip`
contains the exact baseline, reference, and provider outputs required by the
evaluator: 1,711,853 bytes, SHA256
`41a57f2c0d25da6d63ffa443bd2972d3caabb48e237026d69cbfb841e8b2a6a4`.
After extracting it to `backend/output/research/lam/lam_payload`, replay with:

```powershell
python -m backend.benchmark.evaluate_lam_small_face_exact_gate `
  --baseline-row-dir backend/output/research/lam/lam_payload/baseline `
  --exact-row-dir backend/output/research/lam/lam_payload/exact `
  --provider-dir backend/output/research/lam/lam_payload/provider `
  --output-dir backend/output/research/lam/lam_payload_replay
```

Exit code 2 is expected for the measured hold. Compact machine-readable values
and all immutable provider hashes are in `results.json`.

Validation on this dirty shared worktree passed 47 focused provider, evaluator,
VGGHeads-gate, face-part, and C3I contract tests. The complete backend run
passed 901 tests plus 87 subtests and retained four pre-existing failures in
canonical/MakeHuman relief smokes whose production implementation files were
already modified outside this owned slice. None of those failures imports or
executes the new LAM adapter or evaluator.
