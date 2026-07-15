# Background prominence v4

This privacy-safe confirmation raises the default selected-scene background
budget from `0.50` to `0.65`. Relief normalization is anchored to the selected
subject whenever scene context is enabled, so the stronger background cannot
rescale facial geometry. The historical flat path remains selected by
`selection_background_depth_ratio=0`.

The first 54-row attempt exposed one projected-nose failure at yaw `+45` and
40 mm: a whole-face detail correlation of `0.822` narrowly cleared the old
`0.80` internal acceptance floor and skipped the guarded high-detail retry.
The accepted implementation raises that internal floor to `0.85`; the retry
still has to pass the existing slope, correction, named-part, physical-cap,
complete-shell, and topology gates.

## Result

- All 54 canonical face rows pass at 20, 30, and 40 mm, yaw
  `-45`/`0`/`+45`, and background ratio `0.65`.
- Median background height above the plate rises from `15.0` to `19.5` mm.
- Median background p05-p95 span rises from `8.1601` to `11.4481` mm.
- Minimum face relighting correlation is `0.9961052`.
- Minimum named-part shape/gradient correlations are `0.9744342` and
  `0.9701458`.
- Worst named-part p95 absolute error is `1.2985969` mm.
- Ten of 54 rows use the guarded `8.0` high-detail retry; 44 stay on the
  `0.25` primary solve.
- Final background depth and gradient correlations are at least effectively
  `1.0`; the maximum far-background height is `26.0100002` mm in a 40 mm row.
- All 54 STLs are printable complete shells.

The companion six-scene analytic matrix also passes every face, background,
localized-structure, cap, attachment, negative-control, and printability gate.
Its minimum face correlation is `0.9705726`, minimum recoverable context
coverage is `0.8628985`, and maximum feasible attachment jump is
`0.8000002` mm.

The full generated evidence remains local. Its checksums are recorded in
`summary.json`; this directory contains only compact aggregate evidence.
