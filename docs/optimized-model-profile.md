# Optimized model profile

The default `verified-print-v1` profile is the highest-confidence stack in the
repository today. It deliberately favors final STL quality and printability
over textured-preview quality.

"Recommended" applies to the benchmarked model, inference, complexity, and
repair core. The winning benchmark also used a deployable depth-relief pass to
estimate object proportions before scaling. The live image runner does not yet
perform that extra prepass, so its minimum-dimension and aspect safeguards are
an operational adaptation rather than exact benchmark parity.

## Selected stack

| Route | Model | Settings that matter |
| --- | --- | --- |
| Object selection | DETR Panoptic | Locked production choice for cached, offline click-to-segment masks. |
| 2.5D relief | Depth Anything V2 Large | Two mesh samples per minimum printable feature and a `2.0 mm/mm` positive relief slope cap. |
| Single-photo full mesh | TripoSG | `50` steps, guidance `7.0`, printable repair, `40,000` fixed face ceiling, adaptive normalized-density ceiling `9.95`, `2.25` bbox aspect clamp. |
| Multiview full mesh | Visual hull | Resolution `32`, grid extent `1.9`, ortho scale `2.0`, one-pixel mask dilation, printable repair. |
| STL postprocess | Trimesh printable repair | Largest-body cleanup, bounded topology repair, watertight fallback, scaling, and diagnostics. |

The profile pins TripoSG to
`VAST-AI/TripoSG@2c1c516d22d58db486a058d98d31bb6177344e06` and its foreground
model to `briaai/RMBG-1.4@2ceba5a5efaec153162aedea169f76caf9b46cf8`.
This prevents a future upstream model update from silently changing the
meaning of the recommended profile.

## Why these settings won

The single-image comparison used the same ten held-out ModelNet rows for every
method. TripoSG with biharmonic prefill, inferred shape scaling, adaptive mesh
complexity, and printable repair won all ten paired comparisons against both
the masked baseline and the previous mirror-depth path. All ten outputs passed
the complete STL gate set.

| Single-image method | Rank score | Chamfer median | H95 median | Median faces |
| --- | ---: | ---: | ---: | ---: |
| TripoSG, biharmonic prefill | `1.1039` | `0.1593` | `0.3849` | `3,855` |
| TripoSG, masked input | `0.9675` | `0.1890` | `0.4861` | `584` |
| TripoSG, mirror prefill | `0.5623` | `0.1652` | `0.4234` | `10,677` |
| Mirror depth relief | `0.1992` | `0.1593` | `0.4768` | `36,860` |

For known missing-region masks, biharmonic prefill remains the preferred input
preparation. For normal user photos without such a mask, the live route keeps
the selected object intact and neutralizes only the background.

The multiview sweep compared resolutions `32`, `36`, and `40` on the same ten
objects. Resolution `32` achieved the best aggregate score and lowest mesh
complexity. Increasing resolution added geometry without a useful surface
accuracy gain.

| Visual-hull resolution | Rank score | Chamfer median | H95 median | Median faces |
| ---: | ---: | ---: | ---: | ---: |
| `32` | `1.1760` | `0.0770` | `0.2148` | `6,508` |
| `36` | `1.0687` | `0.0795` | `0.2163` | `8,340` |
| `40` | `1.0207` | `0.0830` | `0.2137` | `10,316` |

## Models not promoted

- Pixal3D is provisional. Its repaired STL passed hard printability checks on
  one object, but selector replay found `1.44x` Chamfer and `1.35x` H95 versus
  TripoSG. A paired 10-object run is required before promotion.
- TRELLIS.2 is unbenchmarked. Its pinned geometry adapter is available, but it
  has no paired STL result yet.
- Hunyuan3D Shape is held. The recovered 10-object run produced `0/10`
  successful candidate meshes.

These models remain visible as benchmark candidates through the profile API,
but the product UI exposes only the selected production model for each feature.

## API behavior

`GET /profiles` returns the profile, exact model revisions, evidence summary,
and candidate statuses. `GET /models` embeds the same catalog.

Both STL runners accept `profile_id=verified-print-v1`. The profile is already
the default, so a basic client only needs to upload its image or multiview
bundle. Any explicitly supplied runner field overrides the profile value.

```bash
curl -F "file=@object.png" \
  -F "profile_id=verified-print-v1" \
  http://localhost:8001/run/image-to-mesh
```

The response metadata and diagnostics record the selected profile. This makes
generated artifacts traceable to the exact model stack and tuning policy.

## Evidence locations

- Single-image decision: `docs/benchmark-evidence/g4_stl_first_triposg_inferred_adaptive_s40_n10`
- Multiview resolution sweeps: `backend/output/completion-benchmark/experiments/stl_first_visual_hull_local_s0_n10_res{32,36,40}`
- Pixal3D provisional result: `docs/benchmark-evidence/pixal3d-g4-s40-n1-r4.json`
- Hunyuan3D recovered run: `backend/output/completion-benchmark/colab_g4/g4_stl_first_hunyuan3d_shape_s40_n10_recovered`
