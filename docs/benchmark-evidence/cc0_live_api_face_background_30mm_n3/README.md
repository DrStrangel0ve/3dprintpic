# CC0 varied-scene live API face and background replay at 30 mm

This three-scene replay exercises the real `/selection/compose` and
`/process_image` HTTP path at the standard 256-grid workflow size. It uses the
checksum-pinned MakeHuman CC0 heads and deterministic analytic backgrounds from
the existing photo-detail sweep. Only aggregate telemetry is tracked here.

## Finding and fix

The first centered request produced a printable STL and preserved background
depth, but face refinement reported zero detected faces. This was not a model
domain failure: MediaPipe found a 70 px-wide, 478-landmark face, then the fixed
96 px minimum discarded it. At a 256 px workflow size, that threshold required
a face to occupy 37.5% of the image width.

Revision `15a3a2580586e2d2fac4624a9d2b253e62980342` keeps the historical 96 px
limit for images with a short edge of at least 384 px and adapts it to 25% of
the short edge for smaller inputs, with a 48 px floor. All three 256 px scenes
therefore use a measured 64 px threshold and retain the landmark-based face
path.

The same revision replaces the brittle Haar-only fallback with OpenCV YuNet
before Haar. The model is the official OpenCV Zoo `2023mar` ONNX artifact,
pinned to repository revision
`47534e27c9851bb1128ccc0102f1145e27f23f98` and SHA256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
That artifact is the OpenCV 4.x-compatible model, matching the repository's
OpenCV 4.10 requirement. The fallback uses a 0.90 score threshold, validates
the five-point face geometry, bounds inference to 1024 px, and records model
provenance in the face audit. Both downloaded and environment-configured models
must match the pinned size and SHA256. Cache downloads are streamed,
size-bounded, and use unique atomic temporary files. If YuNet finds no face,
the documented final Haar fallback is still attempted; detector errors do not
include configured filesystem paths.

## Measured result

Each scene has a zero-photo-detail baseline and a candidate that omits both
background form fields, observing the endpoint defaults of 0.60 mm photo detail
and a 0.65 background-depth ratio. The clean server revision is exact for all
six requests. Source, selection-mask, and response content hashes are retained
in `summary.json`; job identifiers and local paths are not.

| Metric | Centered | Left frame | Right frame |
| --- | ---: | ---: | ---: |
| Face landmarks | 478 | 478 | 478 |
| Face normal mean cosine | 0.9985 | 0.9983 | 0.9973 |
| Face normal p95 error | 2.933 deg | 3.802 deg | 4.444 deg |
| Minimum face relighting correlation | 0.9956 | 0.9945 | 0.9904 |
| Intended-background source correlation | 0.6559 | 0.7307 | 0.5528 |
| Active source-detail correlation | 0.7121 | 0.6688 | 0.5615 |
| Added background RMS | 0.240 mm | 0.208 mm | 0.163 mm |
| Added background p95 | 0.440 mm | 0.380 mm | 0.236 mm |
| Background depth correlation | 0.999997 | 0.999988 | 0.999999 |
| Background gradient correlation | 0.999546 | 0.998704 | 0.999855 |

The background addition remains effectively zero inside the selected subject:
face-interior p99 change is at most `0.000110 mm`, and the maximum attachment
boundary change is `0.000780 mm`. Every candidate has zero far-background cap
violation and a maximum feasible attachment jump of `0.80000019 mm` against the
`0.8 mm` limit. The 20-27 mutually incompatible one-pixel attachment
constraints remain explicitly reported; emission status is not mislabeled as a
strict all-constraint pass.

Every baseline and candidate emits one watertight, manifold, consistently wound
component with zero degenerate faces. The exact shell check finds
`232,320/232,320` triangles, zero heightfield sample error, and at most
`0.000003052 mm` shell-coordinate error in every candidate.

All frozen face appearance, background appearance, source-detail, background
depth, physical cap, topology, request-binding, runtime-provenance, and exact
shell gates pass. The tracked summary includes every frozen threshold and every
numeric input used by the independent checks, including per-component relighting
records and explicit shell tolerances, rather than pass booleans alone.

The clean detector-control run at revision `162907c` detects the centered CC0
face with YuNet confidence `0.9266`. MediaPipe, YuNet, the official OpenCV 4.10
Haar cascade, and the complete production chain all emit zero detections and no
errors on eight 256 px negative controls: the three subject-removed analytic
backgrounds, three procedural 3D objects, a high-contrast checkerboard, and a
blank neutral image. Every control records its exact RGB SHA256. The full
tracked backend suite passes with 578 tests and 72 subtests; the two warnings
are pre-existing.

## Reproduction

Regenerate the CC0 fixture matrix first:

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_background_photo_detail_sweep `
  --output-dir backend/output/<cc0-fixture-run> `
  --detail-levels-mm 0,0.6 `
  --relief-height-mm 30
```

For each zero-detail scene, stage `source.png` and `face_parts/face.png` beneath
the clean server's ignored output directory. Post the source and staged mask to
`/selection/compose`, then post the returned selection job to `/process_image`
twice with these shared controls:

```text
target_dimension=256
z_scale=30
max_xy_size=96
```

The baseline explicitly posts `background_photo_detail_mm=0`. The candidate
omits `background_photo_detail_mm` and `selection_background_depth_ratio`.
Transform the staged selection mask with the recorded `surface_grid_transform`,
score the two retained `output_surface.npy` arrays with `_detail_metrics`, and
verify each `output_model.stl` against its retained heightfield with
`_stl_heightfield_agreement(..., expected_max_xy_size_mm=96)`.

Run the checked-in detector-control producer with the checksum-pinned official
YuNet model and OpenCV 4.10 cascade:

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_face_detector_controls `
  --fixture-root backend/output/<cc0-fixture-run> `
  --procedural-dataset backend/output/<procedural-dataset> `
  --yunet-model backend/output/<models>/face_detection_yunet_2023mar.onnx `
  --haar-cascade backend/output/<models>/haarcascade_frontalface_default.xml `
  --output backend/output/<detector-controls>/summary.json
```

The cascade is pinned to OpenCV tag `4.10.0`, SHA256
`0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0`.

The compact numeric result is in `summary.json`. It contains the frozen gate
definitions, complete appearance and physical telemetry, fixture and response
hashes, and the clean producer-bound detector controls. Source images, masks,
request records, job identifiers, heightfields, and meshes remain in gitignored
output.
