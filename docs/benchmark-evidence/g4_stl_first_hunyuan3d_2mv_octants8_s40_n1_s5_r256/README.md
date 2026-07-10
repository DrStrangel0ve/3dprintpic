# G4 Hunyuan3D-2mv Octants Smoke Evidence

This directory preserves compact evidence for Colab run
`g4_stl_first_hunyuan3d_2mv_octants8_s40_n1_s5_r256`, completed on July 11,
2026. The smoke evaluated six methods on held-out ModelNet asset
`table/train/table_0247.off`. Hunyuan consumed the four cardinal views; the
four intermediate octant views at 45, 135, 225, and 315 degrees were reserved
for held-out silhouette agreement.

## Result

The repaired Hunyuan mesh passed every STL promotion gate and led the
deployable smoke score. This is one-row execution evidence, not a promotion
claim; the same ten-row slice used to promote TripoSG is the deciding run.

| method | score | Chamfer | H95 | held-out IoU | components | faces | complexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan3D-2mv, repaired and inferred bbox | `0.6308643` | `0.1641288` | `0.3861005` | `0.7914006` | `1` | `4,372` | `9.0454495` |
| Mirror depth relief | `0.1183079` | `0.1946848` | `0.4253555` | n/a | `1` | `36,860` | `11.1772520` |
| Masked depth relief | `0.0000000` | `0.1838510` | `0.3718838` | n/a | `1` | `36,860` | `11.1729047` |
| Hunyuan3D-2mv raw mesh | `-5.4496193` | `0.3767970` | `1.1326047` | `0.0101568` | `10` | `15,468` | `10.2349665` |

The raw output was watertight but split into ten components and exceeded the
scale-free complexity gate. Printable repair retained the dominant geometry,
closed it into one manifold body, calibrated it to the deployable inferred
bbox, and reduced complexity below `10`. The repaired STL is watertight,
manifold, winding-consistent, positive-volume, and single-component. Its
volume-fill ratio changed from `0.0006557` in model space to `0.7172162` after
calibration, so raw and repaired measurements are intentionally reported as
separate artifacts.

The provider process reported `5.5495` GiB peak allocated and `5.7266` GiB
peak reserved CUDA memory. Its native timing separated `10.1321` seconds of
model loading from `4.0860` seconds of diffusion inference; the first complete
provider invocation, including startup and export, took `20.1984` seconds.
The repaired row was a content-addressed cache hit and added `0.0593` seconds
of repair work rather than rerunning inference.

## Provenance

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, about `95` GiB.
- Runtime code: `8de0bc749da24ec3b2c018ff6b37d391bbc763b1`.
- Official source: `Tencent-Hunyuan/Hunyuan3D-2` commit `f8db63096c8282cb27354314d896feba5ba6ff8a`.
- Model: `tencent/Hunyuan3D-2mv` revision `3a761b539b29fe4ff64714813aa9560fd66f5de0`, subfolder `hunyuan3d-dit-v2-mv`.
- Input payload: `38,240` bytes, SHA256 `48b675a222d04c1e7b3f76090b0e206e91daecd47bc7a71ab97d4a1b75c6f807`.
- Result archive: `2,497,447` bytes, SHA256 `0385ffe62c009eb2170032f048afd74b2e8769301398921c4026aeb82683a423`.
- Colab and local re-ingest both returned `promote-challenger`; the local archive matched the Colab byte count and SHA256 exactly.
- Source geometry was diagnostic-only and was never read by the provider or its cache key.

The full result archive remains local. The committed files retain raw and
repaired per-sample metrics, selection and ingest decisions, provider/runtime
preflight, ranked metrics, result summary, and the contact sheet needed to
audit this smoke result.
