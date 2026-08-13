# SOTA depth challenger audit

Date: 2026-08-10

Decision: **hold every challenger; keep Depth Anything V2 Large as the one
production photo/scene depth model.**

This bounded, privacy-safe audit asked whether newer general-purpose monocular
depth models improve the final printable high-relief surface. It did not rank
models by a colorized depth preview. Every candidate used the same generated
CC0 MakeHuman source image, exact face/part masks for evaluation, 30 mm and
40 mm relief heights, a 96 mm XY target, a `0.65` background-depth ratio, and
the existing watertight STL emission path.

The exact source image SHA256 was
`368b04492a8b73d749e14646fae757331705ee3b095df19df6ae4851846ff973`.
Oracle geometry was evaluation-only and was never used to alter candidate
depth or emitted STLs.

## Candidates

| Provider | Official revision / artifact | License | Native contract used |
| --- | --- | --- | --- |
| Depth Anything V2 Large control | `depth-anything/Depth-Anything-V2-Large-hf@7581137eff8d4e94f6e796d3baea0e9fa79b22d2` | CC BY-NC 4.0 weights | Relative near-high depth, linear relief |
| DA3MONO-LARGE | `depth-anything/DA3MONO-LARGE@f465978e618db8cc79c83b8bbf24964857db1875`; source `3fe327a6abe2e5db95b54444ea95463dbfef5610` | Apache-2.0 | Relative far-high distance, inverse-depth relief |
| InfiniDepth | source `36c6e0c31887fafc210184ee43ca475230704095`; weights `b387ad877e922468fcd85190f24e5b78b28dcd66` | Apache-2.0 | Native relative disparity, near-high linear relief; no MoGe scaler |
| MetricAnything Student PointMap | source `616a5e6762f5fc40d1a4ef990fee04c800532f59`; weights `8f1f08c53c683d4e19864601dfaf78515f29a63b` | Apache-2.0 | Metric camera depth, inverse-depth relief; validity mask telemetry-only |
| MetricAnything Student DepthMap | same source; weights `cae9b4eb052e827048c9b385366c6d0dce83fb01` | Apache-2.0 | Focal-aware metric depth with official image-width fallback, inverse-depth relief |

InfiniDepth checkpoint SHA256:
`6f123506b266f61b9261878548946cfb8822e6233aaf53527dfc4b3dab1e7f9c`.

MetricAnything PointMap checkpoint SHA256:
`f3b49bf78bc2699e79a7c8a53498d4520b30d2c1620c5c4f6f02cc7e94229d58`.

MetricAnything DepthMap checkpoint SHA256:
`4553a58de46664c26616829183cfb38db09e986d706c4c43e06f6a37e2bebddc`.

## Measured result

The table reports 30 mm output, the user's failure height. `Parts failed` is
the exact six-part face gate. Background values are correlation / raw-gradient
correlation against the oracle emitted surface. None of the challengers passed
the complete face, affine-mm, and background promotion gates.

| Provider | Parts failed at 30 mm | Face normal mean / p05 | Background corr / grad | Peak VRAM | Result |
| --- | --- | ---: | ---: | ---: | --- |
| DA2 Large control | mouth | `0.9645 / 0.8784` | `-0.1226 / 0.0508` | `0.863 GiB` | Remains production incumbent |
| DA3MONO, 1008 px | left eye, mouth, nose, right eye | `0.9485 / 0.8284` | `-0.1062 / 0.1116` | `2.388 GiB` | Hold |
| InfiniDepth, 512 input/query | left eye, mouth, nose, right eye | `0.9268 / 0.7018` | `-0.0035 / -0.0269` | `2.388 GiB` | Hold |
| MetricAnything PointMap, level 9 | left eye, mouth, nose, right eye | `0.9357 / 0.7591` | `-0.0641 / 0.1671` | `2.845 GiB` | Hold |
| MetricAnything DepthMap | all six parts | `0.8222 / 0.4027` | `-0.2046 / 0.2381` | `8.883 GiB` | Hold |

At 40 mm, DA2 passed all six named-part shape gates. DA3MONO still failed the
mouth, nose, and right eye; InfiniDepth and PointMap still failed both eyes,
the nose, and mouth; DepthMap still failed all six parts. No challenger fixed
the background correlation gate.

InfiniDepth's official converted-depth tensor clipped `12.94%` of pixels to
its 200-unit ceiling on this image. The benchmark therefore used the model's
native relative disparity instead of feeding that plateau into relief
normalization. MetricAnything's predicted validity mask covered only `21.42%`
of this full scene, so it was recorded but deliberately not applied; applying
it would have removed source background and body pixels.

## Why DA2 remains selected

The production objective is a natural, printable face with useful scene depth,
not leaderboard recency. DA2 had the best 30 mm face-normal agreement, the fewest
failed facial parts, the lowest VRAM, and the strongest 40 mm named-part result.
The newer models add runtime and memory while losing eyes, nose, or mouth
geometry. Replacing DA2 would therefore be a measured regression.

This is a quality decision, not commercial-license clearance. The exact DA2
weight revision's Hugging Face model card is `CC-BY-NC-4.0`. Commercial use
requires separate permission or a future Apache-licensed model that passes the
same quality gates. The challengers in this audit are Apache-2.0, but none met
the product's face/background acceptance threshold.

The challenger adapters remain benchmark-only and fail closed on source commit,
weight revision/hash, CUDA availability, output finiteness, and depth semantics.
They are not offered in the product UI and are not automatic fallbacks.

## Raw summary integrity

The large local STL runs remain under ignored `backend/output/` directories.
Their complete `summary.json` files were reduced into `results.json`; hashes
below allow local evidence to be checked without committing generated STLs.

| Run | Summary bytes | Summary SHA256 |
| --- | ---: | --- |
| `makehuman_provider_relief_da3mono_sota_r1008_centered_n1` | `361687` | `c434f18bc41592d736ddcce104246ddc8a1b7640bc51c12134e181fc07182452` |
| `makehuman_provider_relief_infinidepth_sota_r512_q512_centered_n1` | `586744` | `776a8818d31aed1373ac49390adbb9d29fd24c1ed07b24eb1d5d635e4a3b7dec` |
| `makehuman_provider_relief_metricanything_sota_l9_centered_n1` | `587174` | `eb90e63c9c776292d56061f2db731a5620c4eaa01656bdcb05fc0e8bcc0bc288` |
| `makehuman_provider_relief_metricanything_depthmap_sota_centered_n1` | `362761` | `e4be3d41fba84322622d64b4f82abf4a83ce6e8a7f021467ab8c8b6ab866661f` |

Machine-readable compact telemetry is in `results.json`.
