# G4 Hunyuan3D-2mv Repair Ablation

This directory preserves the compact evidence for Colab G4 run
`g4_stl_first_hunyuan3d_2mv_repair_ablation_s40_s6_n4`, completed on
July 11, 2026. The run evaluated nine methods on four deliberately difficult
held-out ModelNet rows and completed all `36/36` method/sample rows.

## Decision

Do not promote a Hunyuan3D-2mv repair variant from this slice. The configured
candidate, `hunyuan3d_2mv_voxel_r256_a001_repaired_stl_inferred_bbox_direct_mesh`,
remained `hold` against the promoted TripoSG control.

The original `stl-quality` score made the legacy convex hull look best because
`1.2653` of its `1.5729` score, or `80.4%`, came from reducing complexity far
below the printer budget. That method used a convex hull on all four samples
and expanded median volume fill by `12.2245x`. This run therefore exposed a
ranking-policy bug rather than a better STL architecture.

| method | original score | Chamfer | H95 | held-out IoU | hull rate | fill drift | faces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan legacy repair | `1.5729` | `0.1844` | `0.3499` | `0.5034` | `1.00` | `12.2245` | `233` |
| Hunyuan voxel r192, area 1% | `1.3815` | `0.2072` | `0.3577` | `0.4868` | `0.75` | `3.8409` | `265` |
| Hunyuan voxel r192, all | `1.1034` | `0.1864` | `0.3539` | `0.4840` | `0.50` | `0.5621` | `5,430` |
| Hunyuan voxel r256, area 1% | `1.0980` | `0.1779` | `0.3682` | `0.4703` | `0.50` | `0.4064` | `5,447` |
| Hunyuan voxel r256, all | `1.0979` | `0.1779` | `0.3682` | `0.4697` | `0.50` | `0.3617` | `5,447` |
| TripoSG control | `0.5841` | `0.1847` | `0.3617` | not measured | `0.25` | `0.000003` | `10,712` |
| Hunyuan raw mesh | `-6.9875` | `0.0470` | `0.1479` | `0.8007` | n/a | n/a | `754,531` |

Chamfer and H95 use hidden source geometry and are retained as diagnostics.
They are no longer inputs to the deployable `stl-quality` score. Complexity is
also a hard eligibility limit rather than a reward for excessive
simplification. The corrected policy ranks deployable meshes with held-out
camera agreement and STL validity, then applies repair-integrity gates:

- convex hull fallback rate `<= 0.25`;
- median absolute fill-ratio drift `<= 0.5`;
- at least `75%` of samples with absolute drift `<= 0.5`;
- no sample with absolute drift above `4.0`.

Under that policy, every Hunyuan repaired variant is ineligible. The r256
candidate has a `0.50` hull rate and one `7.0766x` fill-drift outlier. Its
selection remains `hold`; the corrected selector also fails closed because
the older single-image TripoSG control lacks comparable held-out-view rows.

## Repair Finding

Voxel resolution was not the main failure. The same bed and sofa samples fell
back at both r192 and r256. The provider produced `10.7k`-face decimated voxel
meshes, after which the generic cleanup path reduced failing outputs to
`134-210` hull faces. Successful samples retained approximately `10.7k`
faces. This isolated the failure to the post-decimation cleanup/printability
transition.

The follow-up implementation now:

- preserves an already printable voxel-decimated mesh instead of cleaning it
  a second time;
- records pre-clean and post-clean watertightness, volume, winding,
  components, manifold edges, degenerate faces, positive volume, and bbox
  health;
- computes absolute repair fill drift;
- evaluates single-image meshes on views held out from their primary input.

## Provenance

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `94.971` GiB.
- PyTorch: `2.11.0+cu128`; compute capability `12.0`.
- Runtime code: `68f97d24c3ee87eb68c910efaa21d32fad4f4de3`.
- Hunyuan3D-2mv model revision: `3a761b539b29fe4ff64714813aa9560fd66f5de0`.
- Hunyuan3D-2mv source revision: `f8db63096c8282cb27354314d896feba5ba6ff8a`.
- Input payload: `806,390` bytes, SHA256
  `f37656983f3e5688ee1c8d5fe513e6bf4218d829cd120f5309f12c1c1aa2d32e`.
- Compact result archive: `1,258,420` bytes, SHA256
  `eb368249c84692a2c3f01938615915b543261d2e1ab9109f0d99a34e35523568`.
- Provider median inference: `34.7425` seconds; peak CUDA VRAM: `5.8008` GiB.

The full mesh archive is intentionally not committed. The files here retain
the original and corrected rankings, both selection decisions, sanitized
per-sample repair metrics, provider/GPU provenance, the corrected local ingest
report, and the visual contact sheet needed to audit this conclusion.
