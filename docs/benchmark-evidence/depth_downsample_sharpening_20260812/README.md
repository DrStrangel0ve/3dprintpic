# Depth downsample sharpening

Date: 2026-08-12

## Problem

Depth Anything V2 Large preserves the uploaded source image but its official
processor reduces inference input toward a nominal 518 by 518 target while
preserving aspect ratio and constraining both axes to a multiple of 14. For
example, a 3840 by 2160 source becomes 924 by 518, not 518 by 291. The printable
relief path already applies halo-guarded depth-detail enhancement after
inference. Adding another unconstrained depth unsharp pass would duplicate that
filter and risk raised face rims.

## Selected method

`scale_aware_luminance_unsharp_before_model_resize_v1` operates only on the RGB
copy passed to the depth model:

- strength is request-bounded to 0.0 through 1.0; production uses 0.35;
- the source-space radius is `clip(0.45 * downsample_scale, 0.75, 6.0)` pixels;
- only Y luminance is filtered, preserving the original Cb/Cr chroma channels;
- the 8-bit threshold is 2, suppressing small noise changes;
- inputs at or below the processor target are returned byte-identically;
- the original source, SAM 3 masks, face-detection image, and RGB photo-detail
  source are never replaced;
- every depth artifact records method, strength, source size, nominal and actual
  processor sizes, reduction scale, radius, percent, and threshold.

The control is named `depth_downsample_sharpening`. It defaults to 0.0 at the
public backend boundary for historical replay, while both production frontends
send 0.35 explicitly. Depth Anything fallback calls preserve the requested
value. Non-DA2 providers fail closed on a nonzero request rather than silently
changing behavior only when a fallback happens.

## Deterministic resize gate

The focused fixture is a 1024 by 1024 luminance pattern reduced to 128 by 128
with the same bicubic stage used to represent model preprocessing.

| Metric | Baseline | Sharpened 0.35 |
| --- | ---: | ---: |
| Mean absolute horizontal gradient | 41.464565 | 49.440945 |
| Gradient-retention ratio | 1.000000 | 1.192366 |

The test requires at least 1.05x retention. The measured result is 1.192366x.
A separate fixture proves that an input smaller than 518 pixels is unchanged and
that a requested strength above 1.0 is capped.

## Full-model held-out gate

Depth Anything V2 Large was run on one non-published, evaluation-only held-out
scene at strengths 0.0, 0.15, 0.25, 0.35, 0.50, and 0.75. No source image or
derived artifact is included in this repository. The selected 0.35 setting was
the conservative tradeoff:

| Metric | Baseline | Sharpened 0.35 |
| --- | ---: | ---: |
| Correlation to baseline depth | 1.000000 | 0.999947 |
| Mean absolute normalized-depth change | 0.000000 | 0.002313 |
| P99 absolute normalized-depth change | 0.000000 | 0.012802 |
| RGB-edge depth-support ratio | 3.074685 | 3.080388 |
| Depth-gradient RMS | 0.112560 | 0.112355 |

The candidate slightly increases depth response on the strongest source edges
without increasing global depth-gradient energy. Strength 0.75 reached a support
ratio of 3.091604 but moved the normalized depth more than twice as far on
average, so it was rejected. This gate establishes bounded behavior rather than
claiming that sharpening creates geometric information absent from the model.

## Validation

- `backend.tests.test_relief_stl_controls`: 93 tests passed.
- `backend.tests.test_main_stl_contract`: forwarding, bounds, and unsupported
  provider checks pass.
- Full tracked backend discovery: 1,141 tests passed in 143.230 seconds after
  the corrected processor-geometry replay.
- `huggingface_space.tests.test_space_runtime`: 16 tests passed.
- `frontend` TypeScript typecheck passed.
- `frontend/tests/relief-selection-context.spec.ts`: 2 Chromium tests passed.

The local environment's installed PyTorch build is CPU-only even though the host
has an RTX 3080 Ti. The full-model sweep therefore ran on CPU; preprocessing and
model outputs are equivalent for this gate, while runtime and VRAM were excluded
from the decision.
