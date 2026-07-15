# Retained-artifact Euclidean background recovery

This is the aggregate-only retained-artifact confirmation for the selected
`2 mm` inner guard plus Euclidean recovery to full gain at `5 mm`. It uses the
same pinned replay contract and explicit overrides as v1. It is not an
authenticated reproduction of the original requests and does not claim every
endpoint default.

The config, source photos, semantic masks, depth arrays, generated surfaces,
meshes, local paths, identifiers, and content hashes stay under ignored
`backend/output/`. The tracked summary passed the privacy allowlist and has
clean implementation provenance at
`2a8076539be9cb73777092788a3ba871df969e9e`.

| Scene | Intended-background correlation | RMS detail | p95 detail | Source-aligned capture | Boundary max | Face max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scene 01 | 0.6115 | 0.0581 mm | 0.1341 mm | 0.5684 | 0.000083 mm | 0.000011 mm |
| Scene 02 | 0.6354 | 0.0845 mm | 0.1996 mm | 0.6174 | 0.000055 mm | n/a |

All four baseline/candidate STLs are single-component watertight manifold
volumes with consistent winding, zero degenerates, and exact complete-shell
facet agreement. Broad-background correlation remains `0.999994/0.999921`,
RMS retention is `1.000051/1.000092`, and coverage is complete.

Only Scene 01 has a retained independent face-region mask, and it measures
`0.000011 mm` maximum face movement. Scene 02 is therefore not counted as an
independent private face-regression check; that contract is covered by Scene 01
and the separate three-face CC0 matrix.

Far-background cap violation is zero and every satisfiable attachment jump is
at most `0.800001 mm`. Strict status remains false because the two scenes have
`12` and `41` mutually incompatible one-pixel constraints; those conflicts are
reported rather than counted as feasible failures.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_private_background_photo_detail_replay `
  --scene-config backend/output/<private-run>/scene_config.json `
  --output-dir backend/output/<private-run>/artifacts `
  --aggregate-summary docs/benchmark-evidence/<run>/summary.json
```
