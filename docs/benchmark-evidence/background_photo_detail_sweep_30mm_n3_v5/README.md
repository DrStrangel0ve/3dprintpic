# Protected 30 mm background photo-detail sweep v5

This fail-closed 12-row sweep compares `0`, `0.12`, `0.30`, and `0.60 mm`
photometric background relief on three checksum-pinned CC0 MakeHuman scenes.
It runs the complete depth-to-STL path at exact implementation revision
`feca082ce3ac4af96f7c734dad14898e7ffbc0f0`.

## Result

`0.60 mm` passes every scene and remains the selected API default.

| Scene | Intended correlation | RMS (mm) | p95 (mm) | Aligned capture | Face boundary max (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| centered smile | 0.4881 | 0.1956 | 0.4483 | 0.3628 | 0.0525 |
| left frame | 0.5052 | 0.1818 | 0.4205 | 0.3703 | 0.0456 |
| right frame | 0.4101 | 0.0740 | 0.1727 | 0.1424 | 0.0325 |

The correlation and capture gates use source-defined background support outside
the 5 mm face-safe halo. All-background correlation remains in `summary.json`
as a diagnostic, so the protected halo is visible rather than silently omitted.
The weakest aligned capture is above the frozen `0.12` floor. A sparse candidate
affecting only about 10% of intended background is rejected.

Every row passes its physical cap, feasible attachment, complete shell, and
printable topology checks. At `0.60 mm`, worst face-interior movement is
`0.0043 mm`, and the largest attachment-boundary movement is `0.0525 mm`.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_background_photo_detail_sweep `
  --output-dir backend/output/background_photo_detail_sweep_30mm_n3_v5
```

The command intentionally has no `--allow-dirty` or `--allow-failures` escape.
`summary.json` is the complete structured run output; `sources/` contains the
three privacy-safe inputs used at every amplitude.
