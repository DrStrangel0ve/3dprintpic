# MapAnything Apache face-depth smoke

This is a fail-closed one-row screen of Meta's maintained
[MapAnything](https://github.com/facebookresearch/map-anything) camera-geometry
provider. The run uses the explicit
[`facebook/map-anything-apache`](https://huggingface.co/facebook/map-anything-apache)
checkpoint; both source and weights are Apache-2.0.

The reproducible adapter also pins official DINOv2 source revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. UniCeption normally requests
that code through Torch Hub's mutable branch cache. The adapter instead
requires a clean exact checkout, routes the request to `source="local"`, and
disables external DINOv2 weights before loading the hash-pinned MapAnything
checkpoint with `local_files_only=True`. It also requires the source-declared
`uniception==0.1.7` encoder package, pins the complete model-config SHA, forces
Hugging Face offline mode during construction, and rejects foreign preloaded
`dinov2.*` modules.

## Why this provider

Unlike another relative-depth model, MapAnything returns camera-frame
pointmaps, camera-Z depth, ray directions, recovered intrinsics, and camera
pose from one RGB image. That makes its geometry contract directly testable and
avoids pose-rotated Poisson reconstruction. The generic provider was screened
before any adapter training or STL emission.

## Exact run

The complete `small_side_lit_shelves_256` image was resized once to 518 square
and processed as one view. The run used official memory-efficient inference,
minibatch size one, fp16 autocast, and unmasked dense output. There was no face
crop, GNM stage, fusion, repair, affine normalization, or sign/scale search.

The RTX 3080 Ti run took 1.252 seconds after model load and reserved 5.635 GiB
at peak. All outputs were finite. Camera-Z depth matched the Z coordinate of
`pts3d_cam` exactly, ray norms stayed within 1.2e-7 of one, intrinsics were
valid, and the camera pose had the expected homogeneous row.

The final strict adapter replay took 0.937 seconds and allocated 5.192 GiB at
peak.
It reproduced the evaluated depth file exactly at SHA256
`a3b0e5193f0d6dc749a5d58a01b94bce3e6781d56d732cf18970dada8b1ebbc4`.

## Result

| Method | Shape corr. | Gradient corr. | Normalized RMSE | Part failures |
| --- | ---: | ---: | ---: | ---: |
| Current central-GNM incumbent | 0.806602 | 0.626152 | 0.175270 | 8 |
| MapAnything camera-Z | 0.809426 | 0.574901 | 0.174121 | 10 |

The generic pointmap improved low-frequency shape and affine RMSE but lost
facial gradients, added nose and right-eyebrow affine failures, and increased
the combined count from eight to ten. It therefore failed the raw gate. No STL
or 30 mm replay was emitted, and no tuning or fusion sweep was opened.

The lane is closed. The next training effort should add materially more varied,
license-clean paired face geometry rather than another generic monocular depth
provider. Machine-readable metrics and artifact hashes are in `results.json`.
