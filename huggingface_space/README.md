---
title: 3D Print a Picture
short_description: Photos to printable reliefs, scenes, and STL meshes.
license: other
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10.13
app_file: app.py
suggested_hardware: zero-a10g
models:
  - depth-anything/Depth-Anything-V2-Large-hf
  - facebook/sam3
  - VAST-AI/TripoSG
  - briaai/RMBG-1.4
tags:
  - image-to-3d
  - depth-estimation
  - 3d-printing
  - stl
  - zero-gpu
---

# 3D Print a Picture

This is the free, non-commercial Hugging Face edition of
[3dprintpic](https://github.com/DrStrangel0ve/3dprintpic). It turns one photo
into one of three downloadable artifacts:

- A watertight 2.5D relief STL with face-aware high-relief correction.
- A layered scene diorama with printable STL and colored GLB exports.
- A complete single-image TripoSG mesh normalized and repaired for STL output.

The application never inpaints or completes uploaded images. Object selection
uses the original selected pixels, relief depth is estimated from the original
scene, and generated files remain in ephemeral Space storage.

## Model choices

Only one production model is exposed for each learned feature.

| Feature | Model | Why it is used |
| --- | --- | --- |
| Object selection | `facebook/sam3` | Passed the measured multi-person silhouette and clothing-coverage selection tests. |
| Photo depth | `depth-anything/Depth-Anything-V2-Large-hf` | Beat newer challengers on the project's 30/40 mm face and background-retention gates. |
| Full image-to-mesh | `VAST-AI/TripoSG` | Won the held-out ten-object printable-STL comparison with every original hard gate passing. |
| STL repair | Trimesh pipeline | Shared watertightness, winding, component, volume, degeneracy, and complexity checks. |

Exact source and model revisions are pinned in `space_runtime.py`. Full
benchmark reports, negative results, training histories, and model-selection
rationale remain in the GitHub repository under `docs/benchmark-evidence`.

## ZeroGPU usage

GPU work is isolated to queued Gradio callbacks. Visitors use their own
Hugging Face ZeroGPU allowance. Relief and diorama runs request up to three GPU
minutes; a full TripoSG mesh can request up to five minutes. The queue is
serialized to keep model memory and output provenance deterministic.

## Privacy

Uploads are processed only to produce the requested preview and downloadable
files. They are not committed, persisted in a dataset, used for training, or
sent to an image-completion service. Temporary generation directories expire
automatically.

## Licenses

The application is intentionally free and non-commercial. Depth Anything V2
Large weights are `CC BY-NC 4.0`. TripoSG is MIT licensed. SAM 3 is provided
under Meta's SAM License. Each upstream model remains subject to its own
license and acceptable-use terms.
