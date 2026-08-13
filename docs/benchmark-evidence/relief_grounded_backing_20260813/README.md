# Grounded selected-relief backing replay

This record covers the August 13, 2026 follow-up to exact selected-only relief
emission. The motivating group photograph is private, so its RGB image, SAM 3
mask, depth arrays, surface arrays, and STL remain local and ignored by Git.
Only aggregate measurements and cryptographic hashes are published.

## Failure

Exact-mask emission correctly removed the broad raised context plateau, but a
selected building or skyline fragment could remain visibly suspended above the
bottom of the printable piece. The narrow connector made the STL one component
without filling the large lower voids. The desired result keeps sky and other
unselected content empty while placing a flat structural foundation below every
retained lower silhouette.

## Method

`full_scene_depth_grounded_selection_emission_v2` leaves the complete local
depth, face, detail, background, and skyline pipeline unchanged. At final mesh
emission it:

1. retains every source-supported selected sample at its exact prior Z value;
2. suppresses every unselected relief-height context sample;
3. joins detached selected groups with the existing minimum spanning connector,
   whose area remains capped at 10% of the selected mask;
4. fills downward from the lowest selected or connector sample in each occupied
   column to the lowest selected row;
5. adds a bottom rail at least one configured printable feature wide; and
6. assigns every connector, foundation, and rail sample exactly the configured
   backing height before the watertight shell is emitted.

The foundation is geometrically bounded to the horizontal and vertical extent
of the selected output. It does not fill holes above a column's lowest retained
sample, so upper sky remains open. Its area is not forced into the connector's
10% budget because the large lower foundation is the requested printable
geometry; its height and bounds are independently deterministic and audited.

## Exact private replay

The replay reused the exact cached 2,048 x 1,536 source-depth result, three-face
refinement, composed SAM 3 mask, 384 x 512 relief grid, 10 mm relief height,
2.4 mm backing, 76 mm long edge, and 256 mm recognition detail basis from the
reported local failure.

| Measurement | Result |
| --- | ---: |
| Selected samples | 68,751 |
| Selected samples retained | 68,751 |
| Selected samples missing or unsupported | 0 |
| Maximum selected Z change | 0.0 mm |
| Unselected context samples suppressed | 57,269 |
| Bounded connector pixels | 585 (0.8509%) |
| Grounded foundation pixels | 53,408 |
| Total backing-only pixels | 53,993 |
| Backing-only Z range | 2.4 to 2.4 mm |
| Bottom rail height | 6 px / 0.8924 mm |
| STL faces | 491,052 |
| STL vertices | 245,488 |
| STL bytes | 24,552,684 |

The emitted STL is one watertight, manifold, winding-consistent positive-volume
component with zero non-manifold edges and zero degenerate faces. Its bounding
box is 76.0 x 56.9628 x 12.4 mm. The STL SHA-256 is
`0055d5c7de4615ca71a2cd5698f78ac46ce7075ff6c402536803e17797de388c`.

## Regression coverage

Focused tests cover vertical fill only below each column's lowest selected
sample, a physically sized bottom rail, foundation fill below a bounded
component bridge, exact selected-height parity, backing/reference equality,
and final watertight topology. Full-scene reliefs keep the default-off path.

Validation passed 1,198 backend tests plus 129 subtests, 20 Hugging Face runtime
tests, and 8 Hugging Face UI tests. The two backend warnings and one UI warning
are existing dependency deprecation/runtime warnings, not test failures.

Machine-readable aggregate evidence is in `summary.json`.

## Local and Hugging Face parity

The measured backend implementation is immutable GitHub revision
`6b2c6729b8dd64be5265695c77c915e911d72176`; wrapper revision
`230012d76c6d57d51e42f144de35d1121e047b39` pins that exact source. A clean
checkout preflight resolved the pin before deployment. Hugging Face Space commit
`f1662d38ce77282cc77b2e9824f211f76e337e81` atomically replaced the same five
wrapper files against parent `3fede8b8ac26c32ec1d27b17739c1923f720fd49`.
All five downloaded files matched the tested payload byte for byte, and the
Space reached `RUNNING` on `zero-a10g`.

The restarted local route used the RTX 3080 Ti and wrapper revision `230012d`.
Its exact live replay completed in 31.762 seconds, retained all 68,751 selected
samples, emitted 53,408 foundation and 585 connector pixels at 2.4 mm, and
reproduced the 491,052-face watertight, manifold, single-component STL with zero
degenerate faces.
