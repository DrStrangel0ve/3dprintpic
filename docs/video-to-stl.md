# Video to STL

The live video runner is `POST /run/video-to-mesh` on the companion FastAPI service. It accepts a turntable-style video, decodes and samples frames, creates temporally checked object masks, assigns orbit cameras, reconstructs a visual hull, repairs the mesh, and emits the same STL diagnostics used by the image runners.

This is a real runner, not a planner response. A successful response contains `output_model.stl`, `diagnostics.json`, `video_preparation.json`, the normalized multiview bundle, and URLs for every selected frame and mask.

## Capture Contract

The first live lane assumes a fixed camera and one object rotating through roughly 360 degrees:

- Keep the entire object visible with margin on every side.
- Use a static, uncluttered background that contrasts with the object.
- Keep lighting, zoom, and camera position fixed.
- Prefer a smooth 5-20 second rotation without pauses or hands crossing the object.
- Put the target near the center, or send normalized `object_point_x` and `object_point_y` values.
- Use MP4, MOV, AVI, MKV, M4V, or WebM. The default upload limit is 500 MB and the default duration limit is 180 seconds.

Handheld orbit videos need learned camera recovery and are not yet represented honestly by the turntable-camera prior.

## Tracked Subject Relief

`POST /run/video-to-relief` handles clips that do not contain an orbit, such as a person walking toward a fixed camera. It propagates a SAM 2.1 subject mask, measures perspective scale change, selects the largest sharp frame that still keeps the subject fully visible, estimates depth with the original scene context intact, applies the subject mask after depth inference, and emits a watertight 2.5D cutout relief.

This route deliberately does not assign invented side or rear cameras. It cannot recover an unseen backside; its output contract is a printable relief from the best observed view. Use `/run/video-to-mesh` only for a controlled turntable with genuine angular coverage.

The local `aarusnow.mp4` validation clip is 9.046 seconds, 478 x 850, and 537 frames. Pinned SAM 2.1 Tiny tracked 12 sharpness/motion samples in 12.986 seconds on the RTX 3080 Ti. Subject coverage grew by 17.3248x with only 0.0553 maximum normalized centroid jump, so the clip was classified as `approach` rather than rejected by the turntable-only area-stability gate. The selector chose source frame 450 at 7.581 seconds: subject height was 0.6282 of the frame with 0.1247 minimum border margin. Depth and STL generation took 4.718 and 1.100 seconds respectively.

The resulting 120.0 x 33.27 x 8.01 mm STL has 69,312 faces. It is watertight, manifold, consistently wound, positive-volume, single-component, and contains zero degenerate faces. The run does not claim face refinement: the face in the selected full-body frame was below the current detector's reliable operating size.

## Segmentation Safety

The built-in `turntable-grabcut` provider uses border-color evidence, GrabCut, and a temporal prior. It blocks reconstruction when masks are empty, nearly full-frame, fragmented, leaking heavily into the border, changing area implausibly, jumping in centroid, or losing temporal overlap. Failed strict checks return HTTP `422` and a fetchable `video_preparation.json` report; the mesh provider is not called.

The temporal prior is deliberately probable foreground. It becomes definite foreground only where the current frame agrees. A synthetic edge-on box test exposed the earlier failure mode: carrying the broad previous mask as definite foreground reduced IoU to about `0.51` at 90/270 degrees. The corrected prior raises the minimum on the same slice above `0.99`.

## Measured Smoke Slice

Evidence: [`docs/benchmark-evidence/video_pipeline_smoke_local_n3`](benchmark-evidence/video_pipeline_smoke_local_n3/README.md)

The deterministic local slice uses three procedural shapes, 24-frame MJPG turntable clips, eight selected views, ground-truth silhouettes, and visual-hull resolution 24.

| Frame selector | Runs | Median mask IoU | Minimum mask IoU | Median mesh Chamfer L1 | Median H95 | STL hard checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform | 3 | 0.9989 | 0.9925 | 0.0338 | 0.0833 | 3/3 pass |
| Sharpness/motion | 3 | 0.9991 | 0.9939 | 0.0605 | 0.1259 | 3/3 pass |

Uniform sampling remains the controlled-turntable default because final surface accuracy is better, even though the sharpness/motion selector has fractionally cleaner masks. The quality selector remains available for clips with blur or exposure variation.

Pinned SAM 2.1 Tiny evidence is stored in [`sam2_tiny_cpu_smoke.json`](benchmark-evidence/video_pipeline_smoke_local_n3/sam2_tiny_cpu_smoke.json). On a three-frame 256 px synthetic clip, revision `de431c4043854a71d8101e17995dfe596bf101a5` passed every mask gate, scored `0.99991` median and `0.99973` minimum IoU against ground truth, and completed in `23.295` seconds on CPU. This is a segmentation smoke, not yet a broad real-video quality claim.

## Modern Model Status

- **SAM 2.1 video:** a live Transformers `Sam2VideoModel` adapter is implemented for Tiny and Base+. It propagates a positive object point plus negative corner prompts and then runs the same mask gates. Tiny and Base+ are pinned to exact Hugging Face revisions. Tiny passed the isolated CPU smoke above; the long-lived global service environment still needs its Transformers/Hugging Face mismatch repaired before this provider is runnable there.
- **SAM 3.1 video:** this is the preferred modern segmentation comparison. Meta released SAM 3.1 in March 2026 with updated video checkpoints and Object Multiplex. It requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+, the external SAM 3 source, approved Hugging Face checkpoint access, and authentication. Preflight reports source and checkpoint readiness separately. Official source: <https://github.com/facebookresearch/sam3>.
- **VGGT-Omega:** this is the preferred modern camera/depth candidate for handheld clips. The official 1B model reports about 6.67 GB peak memory for 10 frames, so it should fit a free 12 GB 3080 Ti, but checkpoint access is gated. It currently emits cameras, depth, and points; it is not marked runnable here until a mesh extractor produces an STL that passes the same gates. Official source: <https://github.com/facebookresearch/vggt-omega>.
- **Multiview visual hull:** this is the only fully tested video-to-STL reconstruction provider in the current runner. It is deterministic and useful for controlled turntables, but concavities that never affect a silhouette cannot be recovered.

`GET /providers/video-to-mesh` returns machine-readable readiness for each of these lanes. Missing source, dependency, checkpoint, or STL extraction is a setup failure, not an implicit fallback.

## Reproduce

Run the focused tests:

```powershell
python -m pytest backend\tests\test_video_pipeline.py backend\tests\test_video_selection_service.py -q
```

Run the measured synthetic slice:

```powershell
python -m backend.benchmark.run_video_pipeline_smoke `
  --output-dir docs\benchmark-evidence\video_pipeline_smoke_local_n3 `
  --samples 3 --video-frames 24 --selected-frames 8 `
  --frame-size 96 --visual-hull-resolution 24
```

Run the pinned learned-segmentation smoke in an environment with compatible Transformers, Hugging Face Hub, PyTorch, and OpenCV packages:

```powershell
python -m backend.benchmark.run_video_segmentation_smoke `
  --output docs\benchmark-evidence\video_pipeline_smoke_local_n3\sam2_tiny_cpu_smoke.json `
  --provider sam2.1-hiera-tiny-video --device cpu `
  --selected-frames 3 --frame-size 256
```

Run the tracked-subject relief lane on a non-turntable clip:

```powershell
backend\.venv\Scripts\python.exe -m backend.benchmark.run_video_subject_relief `
  --input-video path\to\video.mp4 `
  --output-dir output\video-selection-runs\subject-relief `
  --segmentation-device cuda --depth-device cuda
```

For a personal clip, start the companion service, open `/docs`, and use `POST /run/video-to-mesh`. Begin with `uniform-frame-sampler`, `turntable-grabcut`, 12 selected frames, strict segmentation enabled, a 360-degree counter-clockwise rotation, and visual-hull resolution 32. Inspect the returned selected masks before judging the STL.
