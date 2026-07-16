# VGGHeads small-face 30 mm relief gate (2026-07-16)

Status: **research pass, production unchanged**.

This experiment targets the measured failure mode where faces around 75 px
high lose recognizable eye and nose depth when emitted as a 30 mm relief.
Larger faces and eyewear controls already behave better, so the provider is
fail-closed to non-occluded faces at most 96 px high.

All inputs are privacy-safe: CC0 MakeHuman geometry and deterministic
procedural scenes. Source geometry is evaluation-only. No private image,
biometric embedding, provider output, or model weight is published here.

## Provider selection

The bounded upstream review covered maintained camera-aligned face providers:

- [FaceLift](https://github.com/weijielyu/FaceLift) produced an official Space
  result, but direct camera-depth output had 11-12 named-part failures against
  the 10-failure baseline. Bounded detector-face and selected-subject fusion
  stayed at 10 failures for every tested alpha from 0.02 through 0.20. It is
  closed for this relief path.
- [SMIRK](https://github.com/georgeretsi/smirk) and
  [MICA](https://github.com/Zielon/MICA) require separately registered FLAME
  assets for the official inference path.
- [GNM](https://github.com/google/GNM) does not publish the complete inference
  stack needed by this harness.
- [VGGHeads](https://github.com/KupynOrest/head_detector) was the first
  official provider with public weights that ran locally and emitted useful,
  camera-aligned FLAME geometry on the exact small-face row.

VGGHeads source is pinned to
`fc41ffdec189a983d39eba5d208b586d2204d507`. The
`okupyn/vgg_heads` model is pinned to Hugging Face revision
`22672222d2631d8095d01afdf92cc8537e7433cd`; the local weight is
417,153,502 bytes with SHA256
`18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74`.

The source code is MIT, but the model card has no explicit weight license and
the bundled FLAME asset needs a separate redistribution audit. The provider is
therefore always recorded as `production_eligible=false`.

## Exact depth gate

The provider detected the 76 px hard face with confidence `0.942048`, estimated
yaw magnitude within `0.073212` degrees of the known 29-degree pose, and
projected 5,023 vertices / 9,976 faces. Inference took 2.127 seconds with
0.499 GiB PyTorch peak VRAM on the RTX 3080 Ti.

The selected correction uses:

- minimum-z projected front depth;
- nearest finite fill and p01-p99 normalization;
- affine fitting to production local face depth;
- a 3 px low-frequency residual at alpha 0.50;
- existing bounded face-surface fusion;
- a one-pixel exact attachment boundary and four-pixel subject-interior taper.

| Exact metric | Baseline | Candidate |
| --- | ---: | ---: |
| Three-row named-part failures | 19 | 18 |
| Hard 76 px face failures | 10 | 9 |
| Hard face shape correlation | 0.804774 | 0.805441 |
| Hard face gradient correlation | 0.625533 | 0.625513 |
| Hard face normalized RMSE | 0.176007 | 0.175738 |
| Background maximum raw-depth change | - | 0 |
| Attachment-boundary maximum raw-depth change | - | 0 |

The 191 px eyewear row and 249 px turned-face row bypassed the provider
bit-for-bit. The exact gate is eligible for a 30 mm STL replay.

## Background-safe normalization

The first 30 mm replay exposed a subtle coupling: the local face correction
raised the selected subject's p99, so global normalization compressed unchanged
background depth. Face failures improved from 9 to 8, but paired background RMS
retention fell to `0.979444`.

A hard pre-refinement normalization reference fixed the background but clipped
the corrected face peaks, returning to 9 failures. The accepted method uses
dual normalization:

1. normalize the subject interior with the corrected candidate distribution;
2. normalize background and the one-pixel attachment boundary with the
   pre-refinement depth distribution;
3. blend between them with the same four-pixel smooth subject-interior taper.

The optional reference is disabled by default, so legacy runs keep their
existing normalization exactly.

| 30 mm ablation | Face failures | Background RMS retention | Background correlation | Decision |
| --- | ---: | ---: | ---: | --- |
| Candidate normalization only | 8 | 0.979444 | 0.999973 | hold |
| Hard baseline reference | 9 | 0.999950 | 1.000000 | hold |
| Dual subject/background normalization | 8 | 0.999951 | 0.999999 | pass |

The final candidate recovered the right-eye affine gate: span retention moved
from `0.367641` to `0.654754`, with RMSE `0.673930` mm and p95 absolute error
`1.447640` mm.

## STL checks

The oracle, baseline, and candidate each emitted 232,320 faces and 116,162
vertices. Every mesh is one watertight, winding-consistent volume with:

- zero non-manifold edges;
- zero degenerate faces;
- positive volume;
- complete height-field shell agreement;
- maximum candidate surface height `30.006355` mm;
- far-background maximum `18.518951` mm below the `19.51` mm ceiling;
- maximum feasible attachment jump `0.8000002` mm at the `0.8` mm limit.

The final physical projection reports eight mutually incompatible one-pixel
attachment constraints, versus nine in the baseline. These are explicitly
reported and excluded from the feasible attachment gate; they are not silently
presented as satisfiable.

## Decision

The VGGHeads correction is the selected **research challenger** for small,
non-occluded faces, and the dual-normalization STL replay passes. Production is
still unchanged because provider asset and weight licensing is not yet
deployable and the evidence currently contains one eligible small-face
identity. The next promotion-sized experiment must expand the privacy-safe
corpus across identity, expression, pose, lighting, and background while
preserving these exact face, background, cap, attachment, topology, shell,
eyewear, object, and llama gates.

## Raw evidence hashes

| Artifact | SHA256 |
| --- | --- |
| FaceLift official Space result | `a1c7ce6cf46e9f1f23a5062bf3bc0e089456fe91f7ab1d51e118c74fc127f452` |
| FaceLift direct-depth diagnostic | `51e60ba09f5519ff8251f003bab85ea46a121553610d667867c7d130afb957a7` |
| FaceLift bounded-fusion diagnostic | `92bb145b5ac44f1ddf3ca40fb229ca7de529fe161192567ecca10de5918f00db` |
| VGGHeads exact gate | `9a9d21146f34b285bf2deaf950bbedebfecaff974727270e2e2cd625b8497a53` |
| Dual-normalization 30 mm replay | `e7f62413fe477f1dafd65712217937497333372d17b136b77cba81a864e580aa` |
