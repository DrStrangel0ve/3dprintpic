# MakeHuman CC0 head fixture

This benchmark fixture is derived from the official MakeHuman source repository
at commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`:

- Repository: https://github.com/makehumancommunity/makehuman
- Base mesh: `makehuman/data/3dobjs/base.obj`
- Morph and expression targets: `makehuman/data/targets/`
- Asset license: `LICENSE.ASSETS.md` (CC0 1.0)

The MakeHuman project states that its core assets, including meshes and morph
targets, are CC0. The original base OBJ and target files also carry explicit
CC0 headers. The copied license is included beside the derived archive.

## Derivation

`backend.benchmark.makehuman_face_fixture` performs the complete deterministic
derivation:

1. Parse the original OBJ without reordering vertices.
2. Retain the exterior body and eye-helper triangles whose vertices are all at
   or above MakeHuman Y coordinate `5.35`.
3. Fan-triangulate source quads while preserving face order.
4. Apply three pinned MakeHuman macro/expression target combinations.
5. Derive eye, eyebrow, nose, and mouth vertex support from the official target
   directories; hidden eye-helper geometry is assigned to its matching eye.
6. Store arrays in a deterministic NPZ with fixed ZIP timestamps and ordering.

`asset.json` records the source hashes, aggregate part-target directory hashes,
profile target hashes and scales, derived topology/geometry hashes, and final
archive hash. The fixture is geometry-only. Skin, eye, brow, lip, pore, and
hairline colors are deterministic procedural benchmark materials and contain no
photographs or likeness data.

The original MakeHuman units and axes are preserved in the archive. Rendering
normalizes each head only after loading; exact depth and semantic masks are
generated from the same winning z-buffer.
