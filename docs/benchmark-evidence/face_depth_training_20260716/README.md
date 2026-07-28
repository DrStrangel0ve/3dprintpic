# Small-face depth training iteration (2026-07-16)

Status: **corpus and training method retained; all measured model candidates
hold**. No learned face model is enabled in production.

## Research screen

The exact hard row is the 256 px, 29-degree-yaw synthetic face whose visible
face height is 76 px. The existing production result has 10 combined named-part
failures.

Three current upstream directions were measured or preflighted:

- [OSDFace](https://github.com/jkwang28/OSDFace) at
  `ed7a700980fcc721267071955c3acfd723fc4901` restored the exact 53x63 production
  crop in 3.155 seconds at 2.698 GiB peak VRAM. Passing the restored crop through
  the pinned DAv2 model produced 9 best-case failures, but slightly regressed
  global shape, gradient, and RMSE versus the source-DAv2 9-failure candidate.
  This lane is closed.
- [MapAnything](https://github.com/facebookresearch/map-anything) at
  `c845b8f4f6cde0c20aecd87573656c3f69f5b2b0`, using the official
  [Apache checkpoint](https://huggingface.co/facebook/map-anything-apache),
  emitted finite camera depth and intrinsics on the RTX 3080 Ti. Raw ray depth
  improved global shape/RMSE, but every bounded production fusion retained 10
  named-part failures. This lane is closed for small-face detail.
- FaceVerse V4 and SMIRK were preflighted from their official repositories.
  FaceVerse's public weights were blocked behind an inaccessible OneDrive
  distribution path, while SMIRK requires separately registered FLAME assets.
  Neither was patched around or evaluated with unverified weights.

## Expanded CC0 corpus

The new fixture under
`backend/benchmark/assets/makehuman_cc0_face_training` is derived only from the
pinned MakeHuman CC0 source commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482`.

It contains:

- 8 identity groups and 4 expressions, producing 32 unique meshes;
- 320 deterministic scenes;
- identity-disjoint splits of 240 train, 40 validation, and 40 sealed rows;
- 192 faces at or below 90 px and 172 at or below 80 px;
- both yaw signs, 256/384 px images, three backgrounds, three lighting profiles,
  and 64 eye-band occlusions;
- complete exact masks for both eyes, both eyebrows, nose, and mouth on every
  row.

The compact fixture is 1,081,896 bytes with SHA256
`5da31444928a0d4d28522cd21dd4f5506946fdbb3c1436cabbbbea12ec054361`.
The ignored 320-row render summary has SHA256
`b4e8776e12e9f4b1842c1d69b5f3233a2e1e894d1d1f1ed4756a3186dc642243`.
No private image, face scan, landmark, or biometric artifact was used.

## DAv2 head-only experiment

The first training method freezes the DINOv2 backbone and DAv2 neck, then trains
only the 331,969-parameter depth head using:

- exact near-high face depth;
- two-times named-part weighting;
- value, gradient, and Laplacian losses;
- background distillation to the original DAv2 output;
- validation-only selection and a strict baseline/trained prediction blend.

The two-epoch smoke completed in 84.930 seconds at 0.865 GiB incremental peak
VRAM. Its raw epoch-2 head reduced validation failures from 200 to 183, but
regressed validation gradient and RMSE. A validation-only 10% blend was the
strongest strict candidate:

| Metric | Baseline | 10% blend |
| --- | ---: | ---: |
| Combined part failures | 200 | 194 |
| Median shape correlation | 0.908422 | 0.908472 |
| Median gradient correlation | 0.652653 | 0.652671 |
| Median normalized RMSE | 0.126702 | 0.126683 |

On the sealed identity, the same fixed blend changed failures from 242 to 241
while improving all three medians. A second run exposed CUDA nondeterminism in
the small optimization and correctly found no strictly eligible blend. The
harness now requests deterministic PyTorch/CuDNN algorithms and fails closed.

## Exact production replay

The favorable first-run 10% checkpoint was replayed through the unchanged
production face detector, 35% crop, local depth normalization, alignment,
landmark prior, eyewear handling, and bounded fusion.

The evaluator first reproduced all three historical local DAv2 crops:

- correlation: `0.9999995` to `0.9999997`;
- maximum absolute difference: `0.00139` to `0.00145`.

This proves the replay is measuring the learned head rather than a changed model
loader. The candidate then failed the production gate:

| Row | Face height | Baseline failures | Candidate failures |
| --- | ---: | ---: | ---: |
| Small side-lit shelves | 76 px | 10 | 10 |
| Dark-skin eyewear panel | 191 px | 3 | 3 |
| Strong turn layered scene | 249 px | 6 | 6 |

Aggregate failures stayed at 19. Median shape, gradient, and RMSE changed from
`0.895642 / 0.625533 / 0.134910` to
`0.895639 / 0.623579 / 0.134912`. A diagnostic 100% head replay was effectively
the same and also held, showing that simple blend attenuation is not the
problem.

## Decision

The head-only lane is closed. Its synthetic improvement does not survive
production crop normalization and face-depth fusion, and it does not improve
the actual 76 px failure.

The corpus, strict selector, and exact replay remain useful infrastructure. The
next training lane should predict a bounded residual **after** local DAv2
normalization and optimize that residual in production local-depth space.
That makes the learned correction non-affine and prevents normalization from
turning the model update into a near no-op.

Source geometry remains evaluation/training supervision only. No source mesh or
private artifact is emitted by the application.
