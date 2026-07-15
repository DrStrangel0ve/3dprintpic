# Face-protected DA2 background residual learner

This privacy-safe run trained a `106,225`-parameter residual U-Net on 16
deterministic CC0/procedural scenes, selected its checkpoint on four separate
validation scenes, and opened the two existing exact centered/right-cropped
scenes only after selection. The model receives RGB, pinned DA2 depth, and a
selection mask. Selected DA2 face values are copied back byte-for-byte.

## Result

The lane is a hold. It proves that a small learner can recover local background
shape, but not broad depth ordering from this small synthetic corpus.

- Clean implementation revision:
  `9d46feadcfe143a06ed58fc934f2616fb420b982`.
- DA2 model revision:
  `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`.
- Best epoch: `19` of `80`; training runtime: `13.0911 s`.
- Incremental training VRAM: `0.4615 GB` on the local RTX 3080 Ti.
- Checkpoint SHA256:
  `05fdef1b8c2a0b898f799303b0a3dace5e84f65e0d334c419637cf3cf44ceda9`.
- Every supervised target pixel was inside the bounded residual range.
- Validation minimum gradient correlation reached `0.8948`, but minimum broad
  correlation was only `0.0845`; the weakest training correlation was
  `-0.4833`.
- Sealed centered/right-cropped gradient correlation improved from
  `0.5346/0.5427` to `0.8318/0.8767`, while broad correlation remained
  incorrect at `-0.4449/-0.4719`.
- All selected face hashes remained exact. No STL was emitted.

The failure is therefore objective/information limited rather than a target
range, memory, topology, or face-protection failure.

## Modern normal/depth follow-up

Meta's May 2026 HyDen-MoGeV2 surface-normal checkpoint was selected as the
newest suitable normal prior. Official source HEAD was
`810b77c3e56712813a0de42a130cfbe6f5b19b90`; model revision was
`d2f22df8e53cb67c27e046601181d6b214fa942e`. The checkpoint requires manual
FAIR noncommercial license acceptance and the local Hugging Face token is
invalid, so it was not downloaded or patched around.

The official Apache-2.0 Lotus-2 public Normal and Depth demos were then smoked
on one CC0 centered scene. Both retained the person but flattened the analytic
background. Background luminance p02-p98 span was only `0.02966` for Normal and
`0.05295` for Depth. Output checksums were
`8ae79aab8fa1539c7a62e48a091874a7293932728b787231e6e0ac22f0413f34` and
`796a518c2af4bf54e1e2dbf91a5e14e3a425eff62cb3e82b9da13a8e4c75b840`.
Neither output was integrated or expanded.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_background_residual_training_smoke `
  --output-dir backend/output/background_residual_da2_n20_s1 `
  --device cuda --epochs 80 --allow-failures
```

