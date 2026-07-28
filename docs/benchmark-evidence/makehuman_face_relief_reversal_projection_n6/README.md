# MakeHuman varied-scene high-relief regression

This privacy-safe exact-depth pass targets the remaining failure where a face
cropped by the image frame could look acceptable at 30 mm and collapse at 40 mm.
It uses the repository's checksum-pinned MakeHuman CC0 fixture rather than a
private photograph or an ambiguously licensed scan dataset.

The six-row matrix contains three perspective identities at both 30 and 40 mm:

- a centered Caucasian female smile;
- an African male neutral face whose left head boundary is cropped;
- an Asian female asymmetric expression whose right head boundary is cropped.

Both cropped scenes retain at least 91.3% of the subject silhouette and 100% of
every eye, eyebrow, nose, and mouth support. The head itself, rather than only a
shoulder, must touch the frame. Each scene includes the same deterministic
panel, shelf, sloped depth, and high-frequency background structure.

## Research choice

MakeHuman's application code is AGPL, while its bundled core graphical assets
and exported models are CC0. The fixture pins source commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482`, source files, target hashes,
derived vertices, topology, and the asset license. This makes exact mesh,
z-buffer depth, masks, renders, and derived benchmark evidence redistributable.

- [MakeHuman license](https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.md)
- [MakeHuman export FAQ](https://static.makehumancommunity.org/makehuman/faq/can_i_sell_models_created_with_makehuman.html)
- [MPFB phenotype randomization](https://static.makehumancommunity.org/mpfb/docs/randomization/phenotype.html)

CharMorph's CC0 Vitruvian family and Blender's CC0 Human Base Meshes remain
useful independent future fixtures. MB-Lab was rejected for tracked exact
geometry because its database-derived models are AGPL rather than permissive.

## Measured failure and fix

At clean revision `8ec1add45b4d135a516976b73402bc1890c6db53`, the
right-cropped asymmetric 40 mm row fell through to the legacy fallback after a
good screened candidate missed only cardinal edge max (`25.4511` versus `24`).
The fallback reached `64.2410` degrees p95 normal error and only `0.557298`
minimum relighting correlation; all six named facial parts failed.

The accepted revision `6ddc5c45ff2753a1b90239f65248f3ca5e39d763`
extends the existing singleton-edge retry to cardinal max but does not waive the
edge gate. A provisional candidate is blended first, then only pixels incident
to significant cardinal or diagonal direction reversals are attenuated toward
the accepted source. Projection output is adopted only if every reversal is
removed. Missing coverage, invalid physical scale, an unadjustable reversal,
or an iteration limit fails closed. The unchanged baseline-aware post-blend
audit still checks exact finite coverage, detail, height span, correction span,
cardinal and diagonal edge ratios, edge excess, and direction reversals.

On the failing row, 248 pixels were adjusted over 10 iterations. Cardinal and
diagonal reversals fell from `77`/`73` to `0`/`0`. P95 normal error improved to
`5.4607` degrees and minimum relighting correlation to `0.989728`; every named
part passed. Background correlation/gradient correlation remained
`0.999993`/`0.999500`, with `3.9753` mm centered RMS and `15.8095` mm p02-p98
span at 40 mm.

## Final matrix

All six rows and all three 30-to-40 mm comparisons pass. Minimum relighting
correlation is `0.989728`; worst p95 normal error is `5.4607` degrees. Minimum
cross-height whole-face shape/gradient correlation is `0.999821`/`0.976690`,
and no named part fails. Forty-millimeter background spans are `15.2910`,
`18.0800`, and `15.8095` mm. Every feasible subject attachment step is at most
`0.8000002` mm. All six STLs are one watertight, winding-consistent, positive
volume with zero nonmanifold edges or degenerate faces, and all exact serialized
shell checks pass.

The tracked preview images are CC0-derived and contain no private input. Full
depth arrays and 11.6 MB STLs remain in ignored local output; `summary.json`
contains their sizes and SHA-256 hashes.

## Reproduction

```powershell
python -m backend.benchmark.run_makehuman_face_relief_smoke `
  --output-dir backend/output/makehuman_face_relief_reversal_projection_clean_n6

python -m pytest backend/tests -q
```

Clean `summary.json`: 398,107 bytes, SHA-256
`0c0e49eaa11ff6da9567d9c5663304700830c46f8c5dc1dc4204f529cdcf2c32`.

