# Production DA2 protected background-detail smoke v3

This smoke runs the pinned production Depth Anything V2 Large model once on the
hardest right-frame CC0 scene, then emits the exact same cached prediction at
`0` and `0.60 mm` photo-detail caps. Comparing those two surfaces cancels
upstream monocular face-depth error and isolates the background-detail change.

## Result

The pair passes on exact implementation revision
`562eb56bbe5cccc71a5fc144ffc636ab4c8f0769`.

- Model revision: `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`.
- Cached resized prediction SHA256:
  `3536a81703614b7a7676e7659861a50befd4d2a7cb1aa3ffd7f812b3eb672205`.
- Inference: `12.140 s`, `0.8632 GB` incremental/peak CUDA allocation.
- Intended-background correlation: `0.5871`.
- Background RMS/p95 detail: `0.1646/0.2532 mm`.
- Source-aligned capture: `0.7401`.
- Face-interior p99/max movement: `0.00039/0.00175 mm`.
- Attachment-boundary maximum: `0.01701 mm`.

Source hash, provider-depth hash, surface grid, sample pitch, and face mask match
between the two rows. Both baseline and candidate pass physical-cap, printable,
and complete-shell checks. Requested and effective detail telemetry is finite,
protected, and does not use the unprotected legacy fallback.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_background_photo_detail_provider_smoke `
  --output-dir backend/output/background_photo_detail_da2_s40_n1_v3 `
  --device cuda
```

The command is fail-closed and uses one inference for the two STL emissions.
`summary.json` is the complete structured run output; `sources/right_frame.png`
is the privacy-safe rendered input.
