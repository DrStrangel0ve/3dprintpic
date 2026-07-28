# C3I raw EXR small-face adapter gate

Status: **hold; production unchanged**.

This pass tests whether licensed, camera-aligned synthetic face depth can teach
a residual correction that survives the existing 75 px face and 30 mm relief
gates. It also hardens the exact metric path against Blender no-hit values and
missing candidate geometry. Raw images, EXRs, generated surfaces, checkpoints,
and private artifacts remain ignored; this directory publishes scalar telemetry
and hashes only.

## Research and corpus

The selected source is [C3I-SynFace female data part 2](https://data.mendeley.com/datasets/yzjdjj5w39/1),
DOI `10.17632/yzjdjj5w39.1`, licensed CC BY 4.0. The publisher archive is
`7,425,750,802` bytes with SHA256
`971c643a253e7939f357fb21831b73d44445ae935939cc6f50b662d8b89735a9`.
Its exact extracted inventory is 19,950 files: 6,650 RGB, 6,650 three-channel
float EXR depth, and 6,650 pose records.

The importer builds 120 balanced 75 px rows across four synthetic identities,
two backgrounds, five expressions, and three motion families. Identity `0020`
and `0024` supply 60 train rows, `0025` supplies 30 validation rows, and `0029`
supplies 30 sealed rows. One detector-incomplete pose fell back within its
original balanced cell. The source asset manifest SHA256 is
`cfc3e978bb6f1486e8f297dce5d2f61768d9cf7dd02d8fccdbfe4dd50e2e4255`;
the final transformed summary SHA256 is
`547511b2edfacd19fab5cf6018799652056827ea0852748f9e0f7dc72315dfa7`.

The EXRs use finite values near `1e10` for Blender no-hit pixels. The importer
now records and removes that support before linear resampling, and the exact
evaluator derives its 30 mm affine reference only from finite selected-face
geometry. Candidate holes retain their own validity mask so they reduce local
coverage instead of disappearing from the comparison. Unavailable or malformed
part metrics fail all six parts and emit structured hold evidence.

Microsoft FaceSynthetics was rejected during research because its public
release has RGB, segmentation, and 2D landmarks but no depth or 3D geometry;
its Research Use of Data Agreement also excludes commercial product use.
[RAP3DF V2](https://data.mendeley.com/datasets/kpdkpcs8zb/4) was integrated as
an evaluation-only Kinect diagnostic, not training data, because its published
depth scale and RGB registration contract are incomplete.

## Supervised result

The RGB/depth/mask residual U-Net has 564,513 parameters. Training used epoch
22, selected blend `0.2`, took 13.54 seconds, and peaked at 1.203 GiB allocated
VRAM on the local RTX 3080 Ti. Cache construction peaked at 1.973 GiB.

| Split | Metric | DAv2 baseline | Adapter |
| --- | --- | ---: | ---: |
| Validation, 30 rows | Named-part failures | 344 | 343 |
| Validation | Shape correlation | 0.962190 | 0.966550 |
| Validation | Gradient correlation | 0.838317 | 0.841799 |
| Validation | Normalized RMSE | 0.076726 | 0.072548 |
| Sealed, 30 rows | Named-part failures | 353 | 352 |
| Sealed | Shape correlation | 0.956007 | 0.959486 |
| Sealed | Gradient correlation | 0.789820 | 0.792438 |
| Sealed | Normalized RMSE | 0.086156 | 0.082953 |

Both splits had a 1.0 per-row non-regression ratio and no new global metric
failure. Checkpoint SHA256:
`fecc0dd9f468792fb69b5a54e578e965882fa201b517bb4bc856be07c32fd63f`.
Training evidence SHA256:
`b8ebffcff03a6ed6f433cf1e628a31bdbc973773be4334e155bdda7eb4b46faa`.

The geometry-only ablation removed RGB from both training and inference. It
improved validation failures from 344 to 343 but left sealed failures at
353, so the sealed strict-improvement gate correctly stopped it before a
production replay. Evidence SHA256:
`daeb885f8f7f3761da22cf709449c2acc268fb9627a03a5ab8aa14672fa65a37`.

## Exact production transfer

The exact source matrix was regenerated from clean commit
`c18a7e9a8907c4aa8bceb2015cf2da4ba052f6c9`; compact source summary SHA256 is
`e6b85dcc5870cf65d95788160b6ea45973a28972685e71e42ca6d0894ac5c5eb`.
The current baseline reproduced byte-for-byte and the residual remained finite,
bounded, and exactly zero at the attachment boundary.

| Blend | Baseline failures | Candidate failures | Shape | Gradient | RMSE | Hard face |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 25 | 25 | 0.895642 | 0.626082 | 0.134910 | 10 -> 10 |
| 0.05 | 25 | 25 | 0.895633 | 0.625869 | 0.134915 | 10 -> 10 |
| 0.10 | 25 | 25 | 0.895598 | 0.625627 | 0.134937 | 10 -> 10 |
| 0.20 | 25 | 26 | 0.895442 | 0.625057 | 0.135033 | 10 -> 10 |

The corresponding exact evidence SHA256 values are
`2c4c3baf48cb3669afed5fc73f33b3f059d7e816232df821728a898f47a23044`,
`bade1fac07a9b7302581e4e1da2e0a012ddc054c8a8bc098364cc6e36079082a`,
and `ebff1221e6df635c0dc7219bc76c021e8b4a80827ab3c3d1db702d625ba8542e`.
At `0.2`, the eyewear row gained a right-eyebrow affine failure and every
row's aggregate shape, gradient, and RMSE moved backward. Lower strengths
removed the added part failure but did not improve the hard face or any
aggregate metric.

## Decision

The raw C3I adapter and its geometry-only ablation are closed. Synthetic
identity-disjoint gains do not transfer to the exact production photos. No
checkpoint is shipped and no adapter 30 mm STL run is permitted. The existing
GNM small-face provider, background prominence, physical cap, attachment,
one-component watertight topology, exact shell, object/llama depth, dark-skin,
eyewear, and cast-shadow controls remain unchanged.

Machine-readable compact telemetry is in [results.json](results.json).
