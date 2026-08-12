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
own Hugging Face ZeroGPU allowance. When object selection is enabled, SAM 3
runs once after the upload in a 45-second reservation and returns a compact
object-region map. Hover previews and subsequent clicks use that cached map in
the browser and do not launch more GPU jobs. Each click toggles a persistent
kept region. Highlighting and draft assembly remain browser-local; click, Undo,
and Clear only invoke a lightweight non-GPU callback to invalidate any stale
applied mask. `Done selecting` sends the chosen region IDs and click coordinates
through one CPU callback, unions the exact connected SAM 3 components, and emits
one reusable selected image. The preview panel shows that isolated image on a
neutral background; the green overlay is retained only as a diagnostic artifact.
The neutral preview is not used for depth estimation.

For printable reliefs, `Select object` uses `source-depth-isolate`: Depth
Anything V2 sees the complete original photograph, preserving the same scene
context used by the local frontend. Face preservation also runs on that complete
source/depth pair. The refined depth and face feature masks are then aligned to
source coordinates, cropped to the selected bounds, and every unselected depth
sample is replaced by a non-finite value before mesh construction. The cropped
original RGB image supplies photo-detail features on that same grid. This keeps
local-quality depth cues while ensuring scenery cannot enter the selected-object
STL. The Background depth control remains exclusive to `Full scene`.
Before Depth Anything V2 performs its model-native reduction, relief requests
apply bounded scale-aware sharpening to source luminance at strength `0.35`.
The radius follows the source-to-model reduction ratio, chroma is unchanged, and
the step is skipped for images that are not reduced. This protects edges from
the nominal 518-pixel DPT resize (with aspect ratio and patch multiples
preserved) without adding a second unsharp pass to the relief or altering the
original RGB used for selection and photo-detail recovery.
Relief and diorama request 110
seconds, and a full TripoSG mesh requests 150 seconds. The
full-mesh reservation was reduced after a live ZeroGPU smoke showed that the
former 240-second decorator became a 360-second scheduler request, which could
not fit inside a free user's daily five-minute allowance. Signed-in free users
can combine selection with one full-mesh run. Anonymous visitors have a
two-minute daily quota and should sign in before using the mesh workflow.

The hover map is capped at a 1024-pixel long edge, uses score-resolved connected
regions, and is encoded as a lossless 24-bit PNG. It contains only region IDs,
labels, scores, and pixel counts. The original photo remains in Gradio's normal
temporary upload path. Only masks represented in the hover map are bit-packed,
individually compressed, and placed into versioned per-visitor `gr.State`; no
source RGB pixels are copied into that state. This explicit state handoff is
required because ZeroGPU GPU workers are forked processes and their
process-local globals disappear after inference. Source size, instance count,
and compressed bytes are hard-bounded. Gradio evicts packed state after 20
minutes; the browser tint is generated locally from the ID map.
Finishing a selection materializes the union of all kept cached SAM 3 masks
into the existing selection job without rerunning the model or invoking its
tracker.

### Owner setup

Object selection uses the gated `facebook/sam3` weights. The Space owner must
accept the SAM 3 access terms, create a read-only Hugging Face token, and store
it as the private Space secret `HF_TOKEN`. The application fails closed when
that secret is absent; it never places the token in source code or generated
artifacts. Startup also points `HF_XET_CACHE` at writable temporary Space
storage. This avoids the read-only `/home/user/.cache/huggingface/xet` failure
observed while downloading the gated SAM 3 checkpoint on ZeroGPU.

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
