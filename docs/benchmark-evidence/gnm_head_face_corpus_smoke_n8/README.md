# GNM Head face corpus smoke

This compact evidence records the first tracked Google GNM Head corpus
generation and production-cache handoff. Generated face images, depth arrays,
source meshes, model assets, and checkpoints remain under ignored
`backend/output/` paths and are not published.

## Result

- Official source: `google/GNM` at
  `9c9419f191edd68644ef5cb6572e238248e9a81c`.
- License: Apache-2.0.
- Official model and both semantic-decoder hashes matched exactly.
- Tracked source was clean.
- Matrix: 1,600 rows with identity-disjoint `960/320/320`
  train/validation/sealed splits.
- Coverage: 40 identities, 8 expressions, 5 scene conditions, both yaw signs,
  256 and 384 px inputs, 3 backgrounds, 3 lighting profiles, and 640
  eye-band occlusion rows.
- The three small-face conditions are 60% of the matrix and use absolute yaw
  of at least 30 degrees.
- Eight-row smoke: 8/8 rows rendered in `32.9950s`.
- Face heights: minimum `70px`, median `78.5px`, maximum `237px`.
- All six named facial-part masks were nonempty; the minimum part had 25
  source pixels.
- All eight GNM source meshes had zero degenerate faces.
- Production crop cache: 8/8 rows in `2.7854s`, `1.1364 GiB` peak VRAM,
  all finite baseline/target arrays, minimum resized face support 5,669
  pixels, and minimum resized part support 63 pixels.

## Decision

Advance the GNM lane to a bounded, evenly sampled 80-row corpus/cache
calibration. This is a training-data decision, not a production face-relief
promotion. The current 30 mm face, background-depth, cap, attachment,
topology, and exact-shell gates remain unchanged.

## Reproduction

The generator intentionally keeps GNM optional:

```powershell
git clone https://github.com/google/GNM.git backend/output/research/gnm
git -C backend/output/research/gnm checkout 9c9419f191edd68644ef5cb6572e238248e9a81c
.\backend\.venv\Scripts\python.exe -m pip install h5py etils immutabledict
.\backend\.venv\Scripts\python.exe -m backend.benchmark.gnm_face_training_corpus `
  --gnm-root backend/output/research/gnm `
  --render-output backend/output/research/gnm_corpus/tracked_smoke_n8 `
  --limit 8
```

The raw generator summary SHA256 is
`ad150754a3311a32b2198e3402dd2bc0522714a6911a6c4b9f10e1f8d934df6b`.
The production-cache manifest SHA256 is
`96eb7f9706c54bfccf9bc852d1c7ccf4bbb9f3b608b403956972c36eb37c7a46`.
