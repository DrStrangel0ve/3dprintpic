# GNM mean-face foundation gate

Status: **pass, reliability-calibrated small-face production candidate**.

This evidence covers the camera-aligned GNM mean-head stage for the measured
70-80 px face failure. Inputs are three deterministic CC0 MakeHuman face
scenes. Source images, exact geometry, depth arrays, model assets, and STLs
remain local and ignored; this directory publishes only scalar telemetry and
hashes.

## Provider decision

- Selected: [Google GNM](https://github.com/google/GNM) mean-head geometry,
  Apache-2.0, pinned at
  `9c9419f191edd68644ef5cb6572e238248e9a81c`.
- Landmark bridge:
  [Mediapipe_2_Dlib_Landmarks](https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks),
  MIT, pinned at `14e7480964c564b35debd878bde359c204388919`.
- Closed: the expanded production-aware DAv2 residual trainer remained at 10
  exact hard-row failures and regressed gradient correlation.
- Closed: official Sapiens2 pointmap inference remained at 10 failures.
- Blocked for product use: Apple SHARP's model license excludes product
  development.
- Blocked in preflight: FaceVerse V4's public model links returned 403.

GNM is not used as a learned image regressor. Existing MediaPipe landmarks fit
the released mean head under weak perspective, and the camera-facing skin
surface becomes a bounded depth prior. The stage runs only on MediaPipe support
at most 55 px, requires at least 85% z-buffer coverage and pose RMS at most
0.06, caps correction at half the current covered-face span, and has an exact
one-pixel zero-correction boundary. Alignment correlation/RMSE choose full
strength at `>=0.80`/`<=0.26`, guarded 25% strength inside the hard
`>=0.65`/`<=0.35` reliability limits, or a no-op outside them.

## Exact depth gate

| Metric | Paired control | GNM candidate |
| --- | ---: | ---: |
| Three-row named-part failures | 19 | 17 |
| Hard 76 px face failures | 10 | 8 |
| Hard shape correlation | 0.804892 | 0.806710 |
| Hard gradient correlation | 0.623512 | 0.626205 |
| Hard normalized RMSE | 0.175959 | 0.175226 |

The hard row fit 24,820 skin triangles with pose RMS `0.048389`, 93.89%
z-buffer coverage, and an inferred yaw-axis component of `0.491023` radians.
The depth correction was bounded to `0.103672`, was exactly zero outside the
face and at its attachment boundary, and remained finite. The 191 px eyewear
and 249 px strong-turn rows were byte-identical to their controls.

## Varied-scene gate

The 15-row CC0 matrix crosses three identities, small/medium/close framing,
both yaw signs, 256/384 px inputs, three backgrounds, and three lighting
profiles.

| Metric | Paired control | Calibrated candidate |
| --- | ---: | ---: |
| Named-part failures | 84 | 81 |
| Median gradient correlation | 0.703735 | 0.707898 |
| Median shape correlation | 0.944465 | 0.944465 |
| Median normalized RMSE | 0.107450 | 0.107450 |

All three small rows used 25% guarded strength. Their shape deltas were
`+0.001349`, `+0.000362`, and `+0.001202`; gradient deltas were `+0.002538`,
`+0.005408`, and `+0.009232`; RMSE improved on all three. Two rows reduced
named-part failures and none regressed. The 12 larger rows bypassed the stage
byte-for-byte. This calibration replaced an initial full-strength pass whose
worst small-face gradient delta was `-0.056155`.

## 30 mm STL gate

| Metric | Paired control | GNM candidate |
| --- | ---: | ---: |
| Emitted named-part failures | 9 | 7 |
| Background p02-p98 span | 15.9630 mm | 15.9623 mm |
| Maximum height | 30.0098 mm | 30.0077 mm |
| Watertight manifold components | 1 | 1 |
| Degenerate faces | 0 | 0 |

Paired background correlation is `0.999990` and centered-RMS retention is
`0.999790`. Candidate face relighting correlation is at least `0.972756`.
The STL has 232,320 faces, 116,162 vertices, complete height-field shell
agreement, consistent winding, and positive volume. Far background reaches
18.4190 mm under the 19.5100 mm ceiling. Every satisfiable attachment is at or
below 0.800001 mm; nine incompatible one-pixel constraints are reported
separately.

## Decision

Both gates pass. Keep the 55 px support gate, correction cap, exact boundary,
dual subject/background normalization, background-depth budget, physical cap,
attachment audit, topology, and shell checks unchanged while expanding to the
privacy-safe varied-scene corpus.

Raw exact evidence SHA256:
`82b36c9722d13a69a5299b8b54053f5428515eff764d795ea6a307b7b3b0693d`.

Raw varied-scene evidence SHA256:
`bc0c9ac37fc454ba5a0e11fc605da6ac8c7e736301d8cb4a7ced4f51a1d1d982`.

Raw 30 mm evidence SHA256:
`4334e8084cd87efa69d377530e5ad3d67808338d374e92ee0fef92a25b00b350`.
