# Single-photo scene diorama

The `Scene Diorama` photo route keeps multiple selected scene elements instead of
compositing them onto transparency. It is intended for photographs where a person,
building, furniture, or other background elements should remain part of one scene.

## Current reconstruction

1. The original image is passed to the selected monocular depth model.
2. Every clicked selection mask is retained separately with its semantic labels.
3. Person masks become rounded, depth-aware foreground volumes.
4. Building and facade masks become rear scene planes. Occluded facade columns are
   continued behind foreground subjects and extended to the print base.
5. Visible image detail adds shallow facade geometry, including window and masonry
   contrast when it survives the requested print resolution.
6. All layers and any required narrow supports are fused through one base.
7. Marching cubes produces a watertight mesh. The same mesh is exported as a
   vertex-colored GLB and as a printable STL.

The GLB can be viewed from any camera position. The frontend also exposes an orbitable
STL preview after a run.

## Observed and generated data

This is a single-view reconstruction, not a measurement of invisible geometry.

- Observed: source pixels, selection masks, and monocular depth estimates.
- Generated: geometry behind occluders, facade continuation, layer separation,
  side and rear extrusion, and print supports.
- Not currently invoked: a novel-view diffusion model or multiview fusion model.

The artifact metadata records those boundaries under `visible_evidence`,
`generated_geometry`, `novel_view_provider`, and `multiview_fusion_provider`.

## API

`POST /process_scene_diorama` accepts multipart form data:

- `file`: the original photo.
- `mask_paths_json`: output-root mask paths returned by the selection API.
- `selection_labels_json`: one string list per mask, such as `[["person"], ["building"]]`.
- `depth_model`, `depth_provider`, and `device`: depth inference settings.
- `max_size_mm`, `scene_depth_mm`, and `base_thickness_mm`: output envelope.
- `subject_depth_mm`, `facade_detail_mm`, and `depth_compression`: scene shaping.
- `minimum_feature_mm` and `max_samples`: print and raster limits.

The response includes `scene_url`, `stl_url`, `preview_url`, `diagnostics_url`, and
the full `scene_reconstruction` provenance block.

## Provider boundary

The local implementation deliberately keeps novel-view generation outside the mesh
builder. A future provider can generate nearby camera views, followed by camera/depth
fusion, while preserving this endpoint's GLB/STL and diagnostics contract. The current
metadata names Stable Virtual Camera and VGGT-Omega as candidate boundaries only; it
does not claim that either model ran.
