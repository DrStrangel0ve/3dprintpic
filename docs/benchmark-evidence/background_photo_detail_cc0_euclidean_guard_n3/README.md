# CC0 face protection with Euclidean background recovery

This six-row regression combines three checksum-pinned CC0 generated heads
(centered, left-frame, and right-frame) with deterministic analytic
backgrounds. Each scene is emitted at `0.00` and `0.60 mm` photo detail using a
`30 mm` relief, `128 mm` footprint, and current production composition path.

At clean revision `2a8076539be9cb73777092788a3ba871df969e9e`, all six STLs
pass the physical cap, feasible attachment constraints, printability, and exact
complete-shell checks.

| Framing | Boundary max | Face p99 | Face max | Background correlation | Background p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centered | 0.000808 mm | 0.000318 mm | 0.000449 mm | 0.5011 | 0.4618 mm |
| Left-frame | 0.002238 mm | 0.000876 mm | 0.001457 mm | 0.5367 | 0.4040 mm |
| Right-frame | 0.013651 mm | 0.000202 mm | 0.000330 mm | 0.4103 | 0.2307 mm |

The right-frame case is the limiting attachment-boundary result and remains
well below the frozen `0.10 mm` gate. The tracked summary contains complete
per-row telemetry, artifact names, and hashes for the generated CC0 fixtures;
it contains no private input. Generated meshes and arrays remain under ignored
output.
