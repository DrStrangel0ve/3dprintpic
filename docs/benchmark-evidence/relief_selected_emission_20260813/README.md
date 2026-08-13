# Exact selected-relief emission replay

This record covers the August 13, 2026 fix for broad flat or "sketched"
surfaces around selected objects. The motivating group photograph is private,
so the source image, selection masks, depth tensors, surface tensors, and STL
files remain local and ignored by Git. Only aggregate measurements and hashes
are published.

## Failure

The previous selected-relief path correctly estimated depth from the complete
photograph, but it also sent the complete finite context surface to the mesh
emitter. Unselected context was compressed by the configured background ratio
and could become a large, nearly flat plateau. In the replay, that plateau sat
at 8.9 mm: the 2.4 mm backing plus `0.65 * 10 mm` of relief.

The artifact was therefore not a depth-model hallucination or a SAM 3 mask
error. It was an output-scope error after otherwise useful full-scene depth and
face processing.

## Method

`full_scene_depth_selected_mask_emission_v1` keeps the local production path
unchanged through depth inference, face refinement, normalization, photo-detail
recovery, background constraints, and skyline handling. At final mesh emission
it then:

1. intersects the finite surface with the exact composed selection mask;
2. requires every selected sample to survive with exactly the same Z value;
3. adds only backing-height support needed to make selected islands printable;
4. gives thin selected edges enough 2 x 2 support to form actual triangles;
5. limits the input to 32 connected selection islands;
6. limits added backing support to 10% of selected area; and
7. fails closed unless the STL is one watertight, manifold, consistently wound
   positive-volume component with zero degenerate faces.

The connector width is computed in the selected crop's final physical scale,
not the pre-crop image scale. Added support is fixed at the 2.4 mm backing
height, so it cannot become another relief-height plateau.

## Exact private replay

The replay used the same 2,048 x 1,536 source, composed SAM 3 selection, cached
Depth Anything V2 Large result, three-face refinement, 384 x 512 relief grid,
10 mm relief height, 2.4 mm backing, 76 mm long edge, and 256 mm detail basis as
the reported local failure.

| Measurement | Result |
| --- | ---: |
| Selected surface samples | 68,751 |
| Selected samples retained | 68,751 |
| Selected samples missing | 0 |
| Selected samples with triangle support | 68,751 |
| Maximum selected Z change | 0.0 mm |
| Unselected context candidates suppressed | 57,269 |
| Added backing support | 585 pixels |
| Backing-support ratio | 0.8509% |
| Hard backing-support limit | 10% |
| Backing-support Z range | 2.4 to 2.4 mm |
| STL faces | 277,376 |
| STL bytes | 13,868,884 |

The final STL is one watertight, manifold, winding-consistent positive-volume
component with zero non-manifold edges and zero degenerate faces. Its bounding
box is 76.0 x 56.9628 x 12.4 mm and SHA-256 is
`468b26d1463743413c9396f093432d9c146958c4fec7282655c1596a9aae715e`.

## Adversarial coverage

Focused fixtures cover diagonal pixel contact that splits into two mesh-cell
components, one-pixel selected appendages, excessive selection fragmentation,
oversized backing bridges, exact selected-height parity, reference-surface
coverage, and watertight irregular-outline export. The default-off path remains
unchanged for full-scene reliefs and existing callers.

Final validation passed 1,196 backend tests plus 129 subtests, all 28 Hugging
Face runtime/UI tests, and 11 local Playwright tests with one intentional live
video skip. Frontend lint and TypeScript checks passed with no errors.
