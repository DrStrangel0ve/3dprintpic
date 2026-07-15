# CC0 face-local gate and eyewear safety replay

This bounded replay measures the 30 mm live relief path at exact clean revision
`beaa8192a13af4e6b8940546e12a06764244eb79`. It uses three deterministic CC0
MakeHuman face scenes plus one procedural non-face control. Raw requests,
images, depth arrays, and meshes remain under ignored clean-worktree output;
`summary.json` contains only compact scalar telemetry.

## What changed

- The varied-context runner now stages and hashes the renderer's exact six face
  masks: both eyes, both eyebrows, nose, and mouth.
- Reconstruction is checked against exact source geometry per named part.
- `output_reference_surface.npy` versus `output_surface.npy` is checked
  separately, so upstream model errors cannot be blamed on relief shaping.
- Eyewear deocclusion now compares pre/post high-frequency gradient q95 in both
  eyes. A correction must retain `0.65-1.60x` in each eye or fail closed.
- Broad forehead, cheek, jaw, and silhouette measurements remain diagnostics;
  they do not override the six fixed facial-part gates.

## Measured result

The motivating eyewear correction retained only `0.3739x` left-eye and
`0.2831x` right-eye detail. It is now safely rejected. The accepted fallback
raises exact raw-gradient correlation from `0.6569` to `0.7545` for the left
eye and from `0.0715` to `0.2943` for the right eye compared with the rejected
correction. The fallback exactly matches the pre-deocclusion ablation.

All six-part emitted-surface retention checks pass on all three face rows.
Every candidate also passes background appearance/depth, physical cap,
attachment, topology, and complete-shell checks. Minimum background depth and
gradient correlation is `0.9999876` / `0.9985982`; p02-p98 span is
`7.9599-19.4983 mm`. All four STLs are one watertight manifold component with
zero degenerates.

The run remains a hold because exact named-part reconstruction still fails
upstream of relief shaping. The small face misses five parts, the eyewear row
misses mouth/nose/right eye shape, and the strong-turn row misses both eyes and
both eyebrows. No background gain or postprocessing change can repair those
errors.

## Modern provider ablation

Official [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3) source revision
`e15385837dc1e051a6cf376b3827f2e279537b29` was tested as a geometry-only
face-mesh prior. The checksum-pinned ResNet-50 model and 35,709-vertex face
model load under PyTorch 2.11 on the local RTX 3080 Ti at `0.180 GB` peak
VRAM. Both detector-bbox and official five-point aligned variants fail all six
parts on all three rows, so the provider is not fused into production.

## Reproduction

```powershell
python -m backend.benchmark.run_cc0_live_face_variation_matrix `
  --clean-server-repository <clean-revision-worktree> `
  --matrix varied-context `
  --output-dir backend/output/cc0_live_face_local_gate_beaa819_n4 `
  --base-url http://127.0.0.1:8015
```

The expected command exit is nonzero because the stricter exact reconstruction
selector intentionally holds the three face rows.
