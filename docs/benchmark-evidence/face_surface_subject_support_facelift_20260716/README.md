# Subject-supported face residual and FaceLift preflight (2026-07-16)

Status: **production unchanged**. Subject-supported residual fusion is retained
as tested infrastructure, but every learned residual checkpoint remains on
hold. FaceLift was subsequently measured through its official Space and remains
research-only and on hold.

## Why the support changed

The prior learned residual was clipped to the detector face oval. That makes
the exact outer boundary safe, but it also prevents a camera-aligned provider
from correcting connected forehead, hairline, jaw, neck, and shoulder geometry
that controls how a face reads at 30 mm relief height.

`face_surface_support_mask` now finds the connected selected-subject component
that overlaps the detected face, unions it with the face mask, and falls back to
the detector face when that contract cannot be established. The 320-row CC0
corpus preflight selected the subject component on every row:

| Support statistic | Result |
| --- | ---: |
| Rows using selected-subject support | 320 / 320 |
| Minimum area expansion | 1.685774x |
| Median area expansion | 2.000377x |
| Maximum area expansion | 2.624638x |
| Zero-residual output change | 0 |
| Maximum correction on the exact outer two-pixel boundary | 0 |

Legacy checkpoints keep detector-face support. A checkpoint must explicitly
declare `surface_support_mode=selection-subject` to receive the wider support.

## Expressivity oracle

A diagnostic oracle passed the known target residual through the unchanged
production alignment, affine-component removal, correction cap, feathering,
and fusion. This asks whether the bounded fusion can express a useful fix
before asking a network to learn it.

| Alpha | Aggregate failures | Hard 76 px face failures | Hard-face mean correction | Hard-face max correction |
| ---: | ---: | ---: | ---: | ---: |
| 0.04 | 17 | 8 | 0.010553 | 0.022032 |
| 0.08 | 16 | 8 | 0.016135 | 0.044064 |
| 0.16 | 20 | 8 | 0.021788 | 0.088128 |
| 0.32 | 20 | 8 | 0.022843 | 0.110173 |
| 1.00 | 20 | 8 | 0.022843 | 0.110173 |

The current exact baseline has 19 aggregate failures and 10 failures on the
hard face. The oracle therefore proves that the support and bounded fusion are
not the limiting factor. Every oracle row retained exactly zero correction at
the outer boundary.

## Learned residual ablations

All runs use the deterministic 320-row privacy-safe MakeHuman CC0 corpus with
identity-disjoint train, validation, and sealed splits. Each model has 564,513
parameters and predicts a non-affine residual after production local-depth
normalization.

| Run | Added objective | Alpha | Validation failures | Sealed failures | Sealed non-regression | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| r5 | subject support only | 0.05 | 265 -> 262 | 298 -> 287 | 1.000 | exact replay |
| r6 | global physical value/gradient 0.50/0.25 | 0.10 | 265 -> 258 | 298 -> 283 | 0.925 | hold |
| r7 | global physical value/gradient 0.20/0.10 | 0.05 | 265 -> 262 | 298 -> 289 | 0.975 | hold |
| r8 | equal-part physical 0.50/0.25, non-regression 1.50 | 0.10 | 265 -> 260 | 298 -> 289 | 0.950 | hold |

r5 was the only sealed-safe checkpoint. Its exact production replay reproduced
the historical/current baseline at 19 failures and reduced the aggregate to
18, but the actual 76 px hard face stayed at 10 failures. The required
hard-face improvement gate therefore blocked the full STL replay.

Reducing r8 to alpha 0.05 still had 290 sealed failures and added a nose affine
failure on `mh_mixed_male__mouth_open_09`. Equal-part weighting did not turn the
compact residual family into a per-part-safe model. This post-hoc residual
family is closed; more alpha or loss-weight sweeps are not justified by the
measurements.

## Upstream provider research

[Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm) remains useful for
official camera-aligned normal inference, but direct FLAME fitting needs
separately registered FLAME assets and the UV checkpoint. The already measured
normal-to-Poisson fusion regressed the exact matrix and remains closed.

[FaceLift](https://github.com/weijielyu/FaceLift) is a more direct upstream
lead because it reconstructs a camera-conditioned 3D Gaussian face from one
image. The integration pins:

- official source revision
  `0a9420c1480db1e22daa6dee192b76a013295992`;
- model `wlyu/OpenFaceLift` revision
  `3ebac9f8e2d791f02507b8aa61520e1ba379a115`;
- the six large model files by SHA256;
- official front camera index 2 from the six-camera rig;
- Apache-2.0 source code and Adobe Research License v1.2 weights.

The six pinned weights total 9,316,311,496 bytes (8.676 GiB). The weights are
noncommercial research-only, so this provider is always marked
`production_eligible=false`.

The local preflight verified the exact clean source revision and all required
source files. It correctly reports `runnable=false` because no local model root
or weights were supplied. A separate official Space smoke completed on the
privacy-safe 76 px face row at Space revision
`268d06a78f9f23decf831e42ec33482c9ac3dd64`: 50 steps, seed 4, guidance 3.0,
54.229 seconds, and 140,410 output Gaussians.

The first 256 px diagnostic exposed an intrinsics bug in this harness: it reused
the native 512 px `fx`, `fy`, `cx`, and `cy`. The loader now records native
camera resolution and scales intrinsics independently for the requested render
width and height. Focused fixtures cover non-square scaling.

With corrected intrinsics, direct camera-depth rows had 11, 12, and 12 combined
named-part failures for camera indices 1, 2, and 3, versus 10 for the production
baseline. Bounded detector-face and selected-subject fusion stayed at 10
failures for every tested alpha from 0.02 through 0.20. FaceLift therefore
remains a negative provider result for this relief metric; no 30 mm STL
expansion was run.

## Gaussian depth normalization

The provider loader preserves FaceLift's native Gaussian PLY output. The
normalizer:

1. loads Gaussian centers, opacity logits, and log scales;
2. projects them with FaceLift's pinned front camera;
3. composites projected centers front-to-back using alpha;
4. derives a bounded screen-space radius from the Gaussian scale;
5. normalizes finite camera depth by p01/p99 with near depth remaining smaller.

Unit fixtures verify that a near Gaussian occludes a far Gaussian and that the
normalization preserves depth direction. This is a CPU-verifiable diagnostic
normalizer, not a replacement Gaussian renderer: it intentionally ignores
ellipsoid rotation and does not optimize splat appearance.

## Decision and next gate

The current production 30 mm face and background path is unchanged. Subject
support is retained because it is backward compatible, boundary exact, and
demonstrably more expressive, but no learned residual checkpoint is selected.
FaceLift is closed for this metric after its bounded one-row smoke.

The next measured provider lane is documented in
[`vggheads_small_face_exact_20260716`](../vggheads_small_face_exact_20260716/README.md).
Its small-face VGGHeads correction passes an exact depth gate and a 30 mm STL
replay only after subject and background normalization are decoupled. It remains
research-only pending provider asset/weight licensing and broader privacy-safe
identity coverage.

Source geometry remains training/evaluation-only. No private image, scan,
landmark, biometric embedding, model output, or checkpoint is published.

## Raw evidence hashes

| Artifact | SHA256 |
| --- | --- |
| Subject-support oracle | `420af8f04631141a5acb40de60cabcf5240764c7b7a53b8b04ede84d80a859b0` |
| r5 training evidence | `0d569f748ed73e2e0f89a9b1a2a8a715a7d60290e633457491521bc88a225294` |
| r5 exact replay | `05228d78d708824e06e1d4dcd016cbb9aac8a7a6dede7f1a4b09e88c002116ca` |
| r6 training evidence | `c998ed284e27b752333e7f29d5fdaeeafa4500698586e58e0c219c89f3f34f0a` |
| r7 training evidence | `d63fa0c41fa45528e08db965233807f6e2d1baefc53c7986af33dd5d0c43e6d2` |
| r8 training evidence | `1d7032085727fe01bf1b7f294985f9e9b8497a8ff74630b9a0490a36f6d661f6` |
| r8 alpha-0.05 post-hoc summary | `dc43809caaf66e9bf271a3a0ee7316a95da0ce7a639bc701db6e77a8ace2e46b` |
| FaceLift local preflight | `74e79274b83ec974cc67da9525b21c56bf1a63bf058b6cc8bcd441202f8ca5b1` |
| FaceLift Apache license | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| Adobe Research License v1.2 | `340b4a0d7a866b43e52128cfebb788925b20e451428eb118035d930e05eff806` |
| FaceLift official Space result | `a1c7ce6cf46e9f1f23a5062bf3bc0e089456fe91f7ab1d51e118c74fc127f452` |
| FaceLift direct-depth diagnostic | `51e60ba09f5519ff8251f003bab85ea46a121553610d667867c7d130afb957a7` |
| FaceLift bounded-fusion diagnostic | `92bb145b5ac44f1ddf3ca40fb229ca7de529fe161192567ecca10de5918f00db` |
