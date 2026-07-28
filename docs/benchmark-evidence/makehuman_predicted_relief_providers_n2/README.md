# Predicted-depth high-relief provider smoke

This privacy-safe smoke moves the exact MakeHuman face controls through actual
monocular depth inference and then through the same 30/40 mm STL emitter. It
uses one centered smiling face and the previously hardest right-cropped,
asymmetric face. The exact silhouette is shared as a declared perfect-selection
control. Exact depth and exact eye/eyebrow/nose/mouth masks are metric-only and
never affect challenger normalization or STL generation.

## Method

- Depth Anything V2 Large is the production control. Its native near-high
  relative output is pinned to model revision
  `7581137eff8d4e94f6e796d3baea0e9fa79b22d2` and normalized only on the
  selected face at STL emission.
- DA3Mono-Large is pinned to official source commit
  `3fe327a6abe2e5db95b54444ea95463dbfef5610` and model revision
  `f465978e618db8cc79c83b8bbf24964857db1875`. Its documented direct far-high
  depth is converted through inverse-depth relief semantics.
- Each model runs once per scene and the cached prediction is reused at 30 and
  40 mm. No oracle scale, shift, clipping range, or polarity is used.
- Every emitted STL independently passes the downstream context, physical-cap,
  attachment, topology, and exact-shell checks before oracle comparison.
- Both final runs have clean implementation provenance at revision
  `d0528a7df0036750b687d9b4cf388cbe9b3d5c2b`.

## Result

Depth Anything remains the face incumbent. On the centered face it passes the
normal/relighting and scale-free named-part gates at both heights, but misses the
strict millimeter gate on eye/eyebrow regions. On the cropped asymmetric face it
keeps scale-free named-part shape but misses physical face appearance and
millimeter gates.

The key background finding is that the compositor is not suppressing model
output. Depth Anything emits `18.5118` mm of p02-p98 background span at 30 mm
and `24.6825` mm at 40 mm on the centered scene, while every self-fidelity and
physical gate passes. The inferred ordering itself is wrong against the exact
scene: correlation is `-0.1226` and gradient correlation is `0.0506` at 30 mm.
Increasing amplitude would therefore amplify incorrect geometry.

DA3Mono-Large is closed after the centered smoke. It used `1.5414` GB peak VRAM
and a `0.5429` second warm inference pass, but lost three scale-free facial parts,
missed every part's millimeter gate, and retained negative broad background
correlation. It was not expanded to the cropped scene.

`comparison.json` contains the compact measured telemetry. Full generated
arrays and STLs remain local under `backend/output/` and are intentionally not
tracked.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_makehuman_face_provider_relief_smoke `
  --output-dir backend/output/makehuman_provider_relief_da2_pinned_clean_n2 `
  --providers depth-anything-v2-large --device cuda --allow-dirty --allow-failures

.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_makehuman_face_provider_relief_smoke `
  --output-dir backend/output/makehuman_provider_relief_da3mono_pinned_clean_centered_n1 `
  --providers da3mono-large --scene-profiles caucasian_female_smile `
  --device cuda --allow-dirty --allow-failures
```
