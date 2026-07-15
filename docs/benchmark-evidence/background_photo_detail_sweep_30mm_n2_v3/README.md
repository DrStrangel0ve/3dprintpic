# Face-protected 30 mm background photo-detail sweep

This clean eight-row sweep compares `0`, `0.12`, `0.30`, and `0.60 mm` of
photometric background relief on the centered smile and hardest right-cropped
CC0 face. Each setting is evaluated after the complete depth-to-STL path.

## Result

`0.60 mm` is promoted as the new API default.

- Clean implementation revision:
  `91094d683772b926e13b4d134a2ceeb711a1e96e`.
- All eight meshes are one watertight manifold component with exact shell,
  physical cap, attachment, and downstream face/background gates passing.
- Centered active source-detail correlation/coverage at `0.60 mm`:
  `0.47744` / `0.39806`.
- Right-cropped active source-detail correlation/coverage:
  `0.70508` / `0.20345`.
- Realized background RMS/p95 detail is `0.12037/0.26198 mm` centered and
  `0.05846/0.08734 mm` right-cropped.
- Face-interior p99 change is `0.00050/0.00017 mm`; worst interior change is
  `0.00274/0.00088 mm`.
- The attachment-boundary maxima are `0.07193/0.04004 mm`, well below the
  `0.21 mm` bounded attachment allowance for a `0.60 mm` request.

The first sweep showed that a narrow protected halo let later physical capping
move the outermost selected pixels by up to `0.19 mm`. The final implementation
uses a wider quiet halo before photo detail is injected. It does not weaken the
existing cap or face-detail checks.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_background_photo_detail_sweep `
  --output-dir backend/output/background_photo_detail_sweep_30mm_n2_v3 `
  --allow-failures
```
