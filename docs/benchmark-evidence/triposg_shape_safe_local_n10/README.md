# TripoSG Shape-Safe Repair Replay

This local investigation explains why the previously promoted TripoSG output
could be printable according to the benchmark but visually useless. It replays
the ten cached raw meshes from
`g4_stl_first_triposg_inferred_adaptive_s40_n10`; it does not perform fresh
TripoSG inference and is not a new production promotion.

## Root Causes

1. The old repaired path silently replaced any still-unprintable mesh with its
   convex hull. On a representative chair, the native mesh had `5,150`
   vertices and `10,512` faces, while the repaired output became a convex
   `262`-vertex, `520`-face wedge.
2. Exact inferred-bbox normalization scaled all three axes independently. A
   cached bed with native extents `[0.875, 1.905, 0.040]` was forced to about
   `[95, 95, 48.4]`, turning a thin provider failure into a thick solid block.
   Across the ten mirror meshes, the required largest-to-smallest axis scale
   ratio ranged from `1.78x` to `22.4x`.
3. The biharmonic input image often contains one visible half of the object and
   a blurred synthetic half. TripoSG's official preprocessing removes the
   background and crops the detected foreground, so this blur can corrupt the
   conditioning image before geometry inference. The model itself is trained
   on coherent image/SDF pairs; it is not a dedicated partial-object completion
   model. See the [official repository](https://github.com/VAST-AI-Research/TripoSG)
   and [paper](https://arxiv.org/abs/2502.06608).
4. Simplifying a high-resolution voxel closure to the face budget could
   reintroduce nonmanifold edges. The subsequent printable repair then used the
   convex hull, hiding the simplification failure.

## Shape-Safe Path

The replacement path makes four bounded changes:

- use mirrored visible pixels as the default TripoSG input instead of blurred
  biharmonic completion;
- binary-search voxel resolution for the highest naturally printable closure
  that already fits the face budget;
- fail closed instead of using a convex-hull fallback;
- apply one uniform scale to the target maximum dimension, preserving the
  provider mesh's proportions.

The old exact-axis mode and hull fallback remain available for historical
replay, but the TripoSG STL smoke lane now selects the shape-safe behavior by
default.

## Cached Input Comparison

These metrics compare the cached native meshes before the new repair. Mirror
input is substantially more coherent in the observed view than masked or
biharmonic input, while biharmonic has the best median H95.

| input | front IoU | Chamfer L1 | H95 |
| --- | ---: | ---: | ---: |
| masked | `0.0826` | `0.2079` | `0.4777` |
| mirror | `0.3996` | `0.1732` | `0.5067` |
| biharmonic | `0.1739` | `0.1671` | `0.3849` |

## Exact Ten-Mesh Replay

The public repair and postprocess functions were run on all ten cached mirror
meshes with a `10,000`-face target, maximum voxel resolution `128`, uniform-max
normalization, and convex-hull fallback disabled.

| check | result |
| --- | ---: |
| meshes | `10/10` |
| watertight, volume, manifold | `10/10` |
| one connected component | `10/10` |
| zero degenerate faces | `10/10` |
| convex-hull fallback | `0/10` |
| maximum faces | `9,864` |
| median front IoU | `0.4692` |
| median Chamfer L1 | `0.1830` |
| median H95 | `0.5260` |
| median CPU repair time | `2.664 s` |

The geometry metrics are not directly comparable to the historical repaired
rows because the historical path altered aspect ratios. The new result is more
conservative: it produces valid STL geometry without inventing thickness or
silently substituting a convex hull. Native TripoSG outputs that are thin or
semantically wrong remain visibly wrong, which correctly prevents them from
being mistaken for successful reconstruction.

## Decision

Keep TripoSG behind the shape-safe repair and fail-closed checks. Do not treat
the cached model as production-ready based on the old promotion score. A fresh
provider run is still required to evaluate the new mirror-input default; if its
native mesh is wrong, the harness should reject it rather than fabricate a
plausible solid.
