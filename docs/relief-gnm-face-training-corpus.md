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

## Expanded training result

The evenly sampled corpus was expanded to 80 identity-stratified rows and then
supplemented with 31 detector-clean novel v05 training identities. The
production-aware six-epoch residual trainer selected epoch 4 at alpha 0.25.
Validation failures moved only from 353 to 351, the sealed split remained at
156, and the exact 76 px production replay remained at 10 failures. Shape
correlation moved from 0.804774 to 0.805360, but gradient correlation regressed
from 0.625533 to 0.624516. This DAv2 residual-training configuration is closed.

Direct official Sapiens2 pointmap depth was also tested on the exact hard row.
Its best axis/sign normalization remained at 10 failures with shape/gradient
correlation 0.804890/0.623107. It is closed for this relief objective.

Apple SHARP is a technically relevant camera-frame metric-geometry provider,
but its checkpoint license explicitly excludes product development. FaceVerse
V4 source is MIT, but the published model links returned 403 during the
bounded preflight. Neither provider was patched around or used in production.

## Camera-aligned GNM foundation

GNM's released mean-head geometry and sparse 68-landmark definition are
sufficient for a bounded image-conditioned prior without its training stack:

1. Convert the existing 478 MediaPipe landmarks to dlib68 order using the
   pinned MIT correspondence.
2. Fit weak-perspective scale, translation, and axis-angle rotation to the GNM
   mean head with a robust weighted least-squares objective.
3. Z-buffer only the 24,820 exterior skin triangles into the detected face.
4. Affine-align that camera-facing surface to the current production depth.
5. Cap correction to 50% of the existing covered-face p05-p95 span and force
   it to zero at the one-pixel attachment boundary.
6. Use full strength only when aligned GNM/current-depth correlation is at
   least 0.80 and normalized RMSE is at most 0.26. Otherwise use 25% guarded
   strength; correlation below 0.65 or normalized RMSE above 0.35 skips the
   stage.

The stage is limited to genuine MediaPipe faces whose local support is at most
55 px. Larger faces, fallback regions, and every background pixel bypass it
exactly. Asset, fit, or coverage errors fail open to the existing depth.

The implementation pins GNM revision
`9c9419f191edd68644ef5cb6572e238248e9a81c`, model SHA256
`e50a702789af51347531e13721df1e5a26dcfc30ced9e243c7cede9fcac5db43`,
and landmark SHA256
`8b4b759042cae8b67062794306dae9d60fc7ba11ddad60461ba3e2bfaaeac222`.
The optional `GNM_HEAD_MODEL_PATH` and `GNM_HEAD_LANDMARKS_PATH` variables can
point to those exact assets; otherwise they are downloaded into the
per-user `~/.cache/3dprintpic` cache and verified before loading.

## Measured promotion gates

The privacy-safe exact replay uses the same three CC0 MakeHuman identities as
the hardened 30 mm gate. Aggregate named-part failures fell from 19 to 17.
The hard 76 px turned face improved from 10 to 8 failures, shape correlation
from 0.804892 to 0.806710, gradient correlation from 0.623512 to 0.626205, and
normalized RMSE from 0.175959 to 0.175226. The 191 px eyewear and 249 px
strong-turn rows bypassed the provider bit-for-bit. Correction outside the face
and at the attachment boundary was exactly zero.

The follow-up 15-row varied-scene replay covers three CC0 identities,
small/medium/close framing, both yaw signs, 256/384 px inputs, three lighting
profiles, and three backgrounds. The first uncalibrated full-strength pass
lowered failures but regressed one small-face gradient score by 0.0562. That
measured failure produced the reliability tiers above. The final paired replay
improved failures from 84 to 81 and median gradient correlation from 0.703735
to 0.707898. All three small faces improved shape, gradient, and RMSE; the
other 12 rows remained byte-identical to their controls.

The paired 30 mm STL replay improved the hard face from 9 to 7 emitted
named-part failures. The nose shape and right-eye affine gates recovered.
Background correlation was 0.999990 and centered-RMS retention was 0.999790;
the candidate retained a 15.962 mm p02-p98 background span. It remained one
watertight, winding-consistent manifold volume with 232,320 faces, zero
degenerates, complete shell agreement, and a 30.0077 mm maximum height. Far
background peaked at 18.4190 mm below the 19.5100 mm ceiling, and every
satisfiable attachment stayed within 0.800001 mm. Nine mutually incompatible
one-pixel constraints remain explicitly reported.

Compact evidence is under
`docs/benchmark-evidence/gnm_mean_face_foundation_20260717`. The stage advances
as the reliability-calibrated small-face production candidate. The algorithm
should change again only for a measured regression in face detail, background
depth, 30 mm cap/attachment, topology, exact shell, object/llama depth,
dark-skin, eyewear, or cast-shadow controls.
