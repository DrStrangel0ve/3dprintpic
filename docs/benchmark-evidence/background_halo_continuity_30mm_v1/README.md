# Background halo continuity at 30 mm

This privacy-safe regression freezes the near-subject recovery contract for
background photo detail. It uses a deterministic analytic vase, shelf, and
window scene; no personal image or learned depth output is involved.

## Decision

The previous square maximum-filter plus Gaussian protection remained a hold:
only `0.5115` of calibrated detail returned in the `5-8 mm` band, and its
weakest local window retained `0.3157`. A pure Euclidean ramp recovered the
background but moved a CC0 face boundary by more than the `0.10 mm` gate.

The selected implementation keeps an exact `2 mm` inner guard and applies an
isotropic Euclidean smoothstep to full gain at `5 mm`. At clean revision
`2a8076539be9cb73777092788a3ba871df969e9e`, it measured:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Relative gain, 5-8 mm | 0.9215 | 0.70-1.10 |
| Relative gain, 8-12 mm | 0.9862 | 0.90-1.10 |
| Minimum local-window gain | 0.6484 | >= 0.50 |
| Source correlation, 5-8 mm | 0.9463 | >= 0.75 |
| Maximum low-pass sector/bin mean | 0.0329 mm | <= 0.04 mm |
| Low-pass absolute p99 | 0.0437 mm | <= 0.06 mm |
| Maximum adjacent radial-bin change | 0.0024 mm | <= 0.04 mm |
| New radial gradient reversals | 0 | <= 0.01 ratio |

Finite coverage is exactly `1.0`. The metric uses fixed millimeter bands,
independent far-field calibration, one-millimeter radial bins, twelve angular
sectors, and overlapping local windows. Delayed recovery, a localized moat,
missing coverage, selective high-energy omission, pitch mismatch, and grid
mismatch are covered by negative tests.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_background_halo_continuity_smoke `
  --output-dir backend/output/background_halo_continuity_30mm
```

The tracked `summary.json` contains the complete aggregate telemetry. Generated
arrays and STL artifacts remain under ignored output.
