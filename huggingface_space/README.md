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
preload_from_hub:
  - depth-anything/Depth-Anything-V2-Large-hf config.json,model.safetensors,preprocessor_config.json 7581137eff8d4e94f6e796d3baea0e9fa79b22d2
  - VAST-AI/TripoSG feature_extractor_dinov2/preprocessor_config.json,image_encoder_dinov2/config.json,image_encoder_dinov2/model.safetensors,model_index.json,scheduler/scheduler_config.json,transformer/config.json,transformer/diffusion_pytorch_model.safetensors,vae/config.json,vae/diffusion_pytorch_model.safetensors 2c1c516d22d58db486a058d98d31bb6177344e06
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
- A selected-object TripoSG mesh normalized and repaired for STL output.

The application never inpaints or completes uploaded images. Object selection
uses the original selected pixels, relief depth is estimated from the original
scene, and generated files remain in ephemeral Space storage.

## Model choices

Only one production model is exposed for each learned feature.

| Feature | Model | Why it is used |
| --- | --- | --- |
| Object selection | `facebook/sam3` | Passed the measured multi-person silhouette and clothing-coverage selection tests. |
| Photo depth | `depth-anything/Depth-Anything-V2-Large-hf` | Beat newer challengers on the project's 30/40 mm face and background-retention gates. |
| Full image-to-mesh | `VAST-AI/TripoSG` | Won the held-out ten-object printable-STL comparison with every original hard gate passing. It consumes the explicit SAM 3 cutout and does not invoke a second foreground model. |
| STL repair | Trimesh pipeline | Shared watertightness, winding, component, volume, degeneracy, and complexity checks. |

Exact source and model revisions are pinned in `space_runtime.py`. Full
benchmark reports, negative results, training histories, and model-selection
rationale remain in the GitHub repository under `docs/benchmark-evidence`.

## ZeroGPU usage

GPU work is isolated to one shared serialized Gradio queue. Visitors use their
own Hugging Face ZeroGPU allowance. Selection requests 45 seconds, relief and
diorama request 110 seconds, and a full TripoSG mesh requests 240 seconds.
Signed-in free users can combine selection with one full-mesh run inside the
daily five-minute allowance; anonymous users can run a full-scene relief within
their shorter allowance.

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
