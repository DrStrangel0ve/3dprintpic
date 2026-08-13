# Structural skyline replay

This compact record covers the August 12, 2026 replacement of the relief
writer's single-color top-background test. The motivating photo is private, so
the source image, masks, surfaces, and STL remain local and ignored by Git.
Only aggregate measurements are published here.

## Method

The production `Trim empty sky` control now:

1. resizes the original source to the exact relief grid;
2. uses alpha directly when the source has transparency;
3. otherwise smooths sensor-scale variation in Lab space;
4. detects supported Scharr structural boundaries rather than absolute sky
   color differences;
5. preserves objects that enter through the top edge;
6. interpolates only columns with no trustworthy boundary; and
7. fails closed to the full rectangle when no structural boundary exists.

The mask removes samples only above the first accepted boundary in each column.
All lower depth, face, background, scale-independent sampling, and backing-shell
stages remain unchanged.

## Exact local replay

| Measurement | Previous color mask | Structural boundary mask |
| --- | ---: | ---: |
| Upper area removed | 0.119349 | 0.276123 |
| Finite surface ratio | 0.880651 | 0.723877 |
| STL faces | 692,568 | 569,276 |
| STL bytes | 34,628,484 | 28,463,884 |
| Face pixels removed | - | 0 |
| Maximum face Z change | - | 0.0 mm |
| Maximum Z change beyond 3 px from the new boundary | - | 0.000000954 mm |
| Maximum Z change beyond 5 px from the new boundary | - | 0.0 mm |

The emitted STL passed watertightness, manifoldness, winding, positive-volume,
single-component, and zero-degenerate-face checks. The 17.8% face/file reduction
comes entirely from removing low-information top geometry.

## Regression coverage

Focused fixtures cover a smooth cloud gradient, multiple roof heights, a narrow
tower entering through the top edge, a transparent cutout, a uniform fail-closed
input, and watertight STL emission after an irregular skyline crop.
