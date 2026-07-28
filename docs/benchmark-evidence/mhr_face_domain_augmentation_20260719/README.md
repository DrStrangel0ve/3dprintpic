# MHR face domain-augmentation and parametric sample study

This bounded study tested two upstream paths for the remaining small, turned-face
failure without changing the production 30 mm relief pipeline:

1. directly decode pinned MHR identity/expression coefficients into
   crop-aligned floating camera-space depth; and
2. train the existing Depth Anything V2 face head with deterministic,
   training-only photo-domain augmentation.

Production remains unchanged. The best augmented checkpoint passes the exact
small-face and 30 mm physical replays, but it ties the incumbent at nine exact
hard-row failures and trades a tiny raw-gradient loss for tiny shape/RMSE gains.

## Direct parametric geometry

The new diagnostic decoder validates the pinned MHR assets and coefficients,
uses the official head identity slice, transforms posed vertices into positive
camera Z, projects them with a perspective camera, and rasterizes the nearest
floating surface into the detector crop. Unit tests cover coefficient bounds,
camera-Z preservation, and projection alignment. It reuses the existing
[official MHR](https://github.com/facebookresearch/MHR) Apache-2.0 source pin
`4998cec385b1aaa07abdefba71bfba2f83c7db32` and checksum-verified public model.

The bounded diagnostic sweep searched eight privacy-safe training expressions for one
promising identity, seven yaw offsets, and seven fusion strengths on the hardest
74-75-pixel turned face (392 candidates). It used known pose and source geometry
only for evaluation, so it is not promotion eligible. The best candidate reduced
combined named-part failures from 10 to 9, but did not improve shape,
raw-gradient correlation, and normalized RMSE together. Background pixels stayed
bit-exact. This closes further tuning of this finite sampled family for now; it
does not establish an upper bound for continuously optimized MHR coefficients or
camera parameters.

## Domain augmentation

Training augmentation is deterministic from the corpus seed, row ID, profile,
and augmentation index. It applies only to training rows; validation and sealed
rows remain the original rendered images. A preparation-boundary regression test
enforces this split invariant. Evidence and checkpoints record the profile,
prepared split counts, augmentation-function hash, and NumPy/OpenCV/Pillow
versions. Two profiles were measured:

- `deterministic-photo-domain-v1` adds aggressive resampling, blur, color,
  noise, and JPEG degradation. Three variants per row regressed the selector's
  small-face gradient. Two variants passed synthetic selection but were slightly
  worse than the incumbent on the exact hard row.
- `deterministic-photometric-v2` preserves image resolution and edge energy
  while applying bounded exposure, color, mild blur/noise, and JPEG changes.
  Two variants per training row selected epoch 1 and blend alpha 0.4.

The photometric profile reduced sealed combined named-part failures from 431 to
376 and sealed-small failures from 225 to 200. On the exact hard row it reduced
failures from 10 to 9, improved shape correlation from `0.804774` to `0.807042`,
improved raw-gradient correlation from `0.625533` to `0.625728`, and improved
normalized RMSE from `0.176007` to `0.175091`.

## 30 mm replay

The exact candidate was replayed through the physical STL path. Named-part
failures fell from 9 to 8 after print postprocessing. The output reached
`29.7260 mm`, remained one watertight manifold component with zero degenerate
faces, and matched the emitted shell exactly. Far-background emission passed with
zero cap violation, and every satisfiable attachment constraint stayed within
`0.800001 mm`. Nine mutually incompatible one-pixel boundary constraints remain
explicitly reported, so the raw attachment and combined physical-cap statuses
are false and are not described as passing. The full-scene background was
preserved with correlation `0.999983` and centered RMS retention `1.000183`.

## Decision

Hold the augmented checkpoint. It is a valid training lead and demonstrates
that edge-preserving photometric variation transfers better than aggressive
downsample/blur augmentation, but it is not a strict replacement for the
incumbent: both retain nine failures on the exact hard row, and the new
checkpoint's raw-gradient correlation is `0.000070` lower than the incumbent
while shape and RMSE improve only by about `0.00003` and `0.00001`.

The production face/background path is unchanged. Future work should add
license-compatible real-image supervision or a maintained camera-aligned face
geometry provider, and must retain these exact face, background, cap,
attachment, topology, and shell gates.

Machine-readable metrics and artifact hashes are in `results.json`.
