# Closed-hole-free selected relief replay

This replay validates the final selected-object emission rule on the exact
private scene that exposed enclosed gaps between faces, people, and buildings.
The source photo, selection mask, depth arrays, generated preview, and STL stay
local and ignored; only aggregate measurements and the STL hash are tracked.

## Method

Implementation revision:
`4e04d502e1c222a121e3f54a52c2980ce2cf2257`.

`full_scene_depth_grounded_closed_hole_free_selection_emission_v3` first
preserves the selected surface, adds the bounded connector and grounded
foundation, and then labels empty pixels with eight-connectivity. Empty
components touching any image border are exterior and remain open. Every other
empty component is an enclosed hole and is emitted at the 2.4 mm backing
height. A second flood is a fail-closed postcondition: final emission is
rejected unless zero enclosed pixels remain.

Eight-connectivity is intentionally conservative. A diagonal route to the
outside counts as open skyline instead of being mistaken for a printable hole.
The operation changes topology only; it does not interpolate or modify selected
depth, face detail, RGB, or model weights.

## Exact replay

| Check | Result |
|---|---:|
| Selected pixels retained | 68,751 / 68,751 |
| Existing emitted samples changed | 0 |
| Maximum existing-surface delta | 0.0 mm |
| Enclosed components filled | 20 |
| Enclosed pixels filled | 5,570 |
| Largest enclosed component | 3,015 pixels |
| Fill height range | 2.4-2.4 mm |
| Enclosed pixels after fill | 0 |
| Mesh vertices / faces | 256,628 / 513,252 |
| Watertight | yes |
| Winding consistent | yes |
| Connected components | 1 |

The generated STL is 25,662,684 bytes with SHA256
`a2269290395e307023ef52965f75eab4fe3246f45eb4a90d613cc368d2bedd28`.

## Validation

- A synthetic topology fixture proves that a closed interior region is filled.
- The same fixture proves that an opening with only a diagonal path to the
  image border remains open.
- Backend, API-contract, Space-runtime, and hosted-parity coverage passed 191
  tests plus 29 subtests.
- The Space wrapper requires the v3 method, accepted fill provenance, and zero
  final enclosed pixels before returning a selected relief.
