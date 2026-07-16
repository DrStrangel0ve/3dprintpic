# GNM Head face-training corpus

The current 30 mm relief pipeline is strongest on close frontal faces and
weakest when the visible face is only about 70 to 80 pixels tall. Recent
provider and fine-tuning experiments showed that another depth model alone is
not enough:

- Native FastAvatar `gsplat` expected-depth rendering matched the seven-failure
  baseline on the hard small-face row.
- ZipDepth decoder fine-tuning regressed the identity-disjoint validation split
  from 265 to 266 named-part failures and the sealed split from 298 to 299.
- Even exact target depth passed through the existing low-frequency fusion
  improved aggregate failures only to 262 and 296 while regressing 7.5% of
  rows. That establishes the current fusion as a bottleneck and closes this
  ZipDepth configuration.

## Selected data lead

[Google GNM Head](https://github.com/google/GNM) was released in July 2026 as
an Apache-2.0, scan-derived parametric head model. The pinned v3 asset has
17,821 vertices, 35,324 triangles, 253 identity dimensions, 383 expression
dimensions, separate skin/eye/teeth/tongue components, and a 68-landmark
definition.

GNM does not infer a head from an image. Its value here is privacy-safe,
camera-aligned supervision with substantially richer identity and expression
coverage than the existing eight-identity MakeHuman corpus. The official
semantic identity and expression samplers are small dense HDF5 decoders. The
benchmark reproduces those decoders with NumPy, avoiding a TensorFlow runtime
dependency while preserving the official weights exactly.

The tracked generator is
`backend/benchmark/gnm_face_training_corpus.py`. It pins the official source
revision and all three asset hashes, rejects tracked source modifications, and
emits source RGB, exact depth, selection masks, and six exact facial-part
masks.

## Matrix

The deterministic matrix contains 1,600 rows:

- 40 identities: five latent variants for every combination of two official
  gender labels and four official demographic labels.
- Identity-disjoint `960/320/320` train/validation/sealed splits.
- Eight expressions: surprise, happy, wide smile, left/right mouth motion,
  left/right wink, and corners down.
- Five scene conditions crossing 256/384 px inputs, both yaw signs, three
  backgrounds, three lighting profiles, horizontal offsets, and eye-band
  occlusion.
- Three of five conditions deliberately produce small turned heads, so 60% of
  rows target the measured 70 to 90 px failure regime.

GNM documents representation limitations in its binary gender labels and four
broad demographic groups. This corpus supplements rather than replaces the
CC0 MakeHuman controls, especially the dark-skin, eyewear, cast-shadow, and
varied-context held-out rows.

## Measured smoke

The exact tracked eight-row smoke rendered in 32.995 seconds. Face heights
ranged from 70 to 237 pixels with a 78.5 pixel median. Every named facial part
was visible, and every source mesh had zero degenerate faces.

The smoke then passed the real production-crop cache path after fixing a
variable-aspect batching bug in `train_face_surface_adapter`: processor outputs
are now grouped by tensor shape before model inference and restored to original
row order. All eight rows cached in 2.785 seconds at 1.136 GiB peak VRAM, with
finite depth targets and nonempty resized part masks.

Compact evidence is under
`docs/benchmark-evidence/gnm_head_face_corpus_smoke_n8`.

## Next gate

The next bounded action is an evenly sampled 80-row render and cache audit.
Only after that data slice passes should a new direct camera-aligned face-depth
training lane run. A candidate must improve the exact six-part shape and raw
gradient metrics without regressing any row, and then pass the existing 30 mm
face height, background depth/detail, cap/attachment, topology, exact-shell,
object/llama depth, dark-skin, eyewear, and cast-shadow controls.
