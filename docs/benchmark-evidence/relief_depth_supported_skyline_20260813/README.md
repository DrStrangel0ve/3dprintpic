# Depth-supported skyline replay

This compact record covers the August 13, 2026 replacement of the structural
v2 skyline detector with an emission-only, depth-supported v3 detector. The
motivating group photo is private, so source pixels, masks, depth tensors,
surfaces, and STL files remain local and ignored by Git. Only aggregate
measurements and artifact hashes are published here.

## Failure and method

The structural v2 mask removed most smooth sky but still accepted a bright
cloud boundary beside a narrow clock tower. That produced a broad, almost flat
piece of mesh around the tower.

The v3 path:

1. computes the same Lab/Scharr structural boundaries as v2;
2. estimates the top-connected depth background from the top image band;
3. requires sustained depth departure beneath a candidate boundary;
4. permits mixed structural/depth evidence only at the actual leading edge;
5. treats every selected pixel as a hard silhouette constraint;
6. preserves alpha-defined cutouts without color or depth guessing;
7. falls back to v2 or the full rectangle when evidence is insufficient; and
8. applies the accepted mask only at final mesh emission.

The final rule is important: trimming cannot influence normalization,
smoothing, face reconstruction, background caps, or feature guards. Every
sample that remains in the STL must equal the same sample from an untrimmed run.

## Exact local replay

The matched jobs used the same 2,048 x 1,536 source, selection, Depth Anything
V2 Large inference, 384 x 512 relief grid, 10 mm relief height, 2.4 mm base,
76 mm long edge, and full-size 256 mm detail basis on the local RTX 3080 Ti.

| Measurement | Structural v2 | Depth-supported v3 |
| --- | ---: | ---: |
| Upper area removed | 0.276123 | 0.359029 |
| Finite surface ratio | 0.723877 | 0.640971 |
| STL faces | 569,276 | 504,076 |
| STL bytes | 28,463,884 | 25,203,884 |

Matched v3 trimmed/untrimmed controls additionally passed:

- depth tensors bit-identical;
- `126,020/126,020` retained surface samples bit-identical;
- zero selected pixels removed and all retained selected Z samples exact;
- zero face pixels removed and all retained face Z samples exact;
- reference-surface coverage exactly matched emitted-surface coverage;
- one watertight, manifold, winding-consistent positive-volume component;
- zero non-manifold edges and zero degenerate faces; and
- 76.0 x 56.9628 x 12.4 mm final bounding box.

The v3 STL SHA-256 is
`328fcc962a5988e4a58c15a1391faf6a71808b408425ae8fee0602bcdfc42cb8`.

## Regression coverage

Focused fixtures cover strong cloud edges above architecture, a selected
one-pixel spire, smooth cloud gradients, multiple roof heights, top-entering
objects, transparent cutouts, uniform fail-closed input, exact retained-height
parity against an untrimmed run, and watertight irregular-outline emission.
