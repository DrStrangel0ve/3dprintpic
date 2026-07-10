# G4 Hunyuan3D-2mv vs TripoSG Evidence

This directory preserves compact evidence for Colab run
`g4_stl_first_hunyuan3d_2mv_vs_triposg_octants8_s40_n10`, completed on July
11, 2026. The run evaluated seven methods on the same ten held-out ModelNet
assets used for the promoted TripoSG slice and completed all `70/70`
method/sample rows. Hunyuan consumed four cardinal renders per asset; four
intermediate octant renders were held out for view agreement.

## Decision

Keep promoted TripoSG as the production incumbent. Repaired Hunyuan is the
STL architecture leader and wins the paired composite objective, but the
incumbent-aware selector correctly returns `hold` because two worst-sample
surface-regression ceilings fail.

| method | score | Chamfer median | H95 median | held-out IoU median | faces median | complexity median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan3D-2mv, repaired and inferred bbox | `1.5784637` | `0.1741899` | `0.3504748` | `0.6551098` | `352` | `6.5212088` |
| TripoSG biharmonic prefill, inferred bbox | `0.2849006` | `0.1956434` | `0.4608482` | n/a | `10,710` | `9.9497980` |
| Mirror depth relief | `0.0563840` | `0.1750209` | `0.5071317` | n/a | `36,860` | `11.1802663` |
| Hunyuan3D-2mv raw mesh | `-3.7386852` | `0.0488328` | `0.1558482` | `0.8998789` | `876,135` | `14.6145664` |
| Source mesh oracle, diagnostic only | `1.6292790` | `0.0732593` | `0.2881702` | n/a | `92` | `6.4681769` |

Hunyuan repaired wins `10/10` paired objective comparisons against TripoSG,
with paired CI95 lower bound `0.6987292` and score margin `1.2935632`. It also
passes all twelve STL gate families on every sample: watertightness, volume,
manifoldness, winding, positive volume, one body, bbox volume/aspect,
nonmanifold edges, degenerate faces, component excess, and scale-free
complexity. Candidate complexity ranges from `5.6232598` to `9.9494379`.

The selector remains on `hold` for two failed surface checks spanning three
outlier assets:

- Worst paired Chamfer ratio `1.6689293x` on `mesh_dir_8a68107048_v00`
  (`0.1766` Hunyuan versus `0.1058` TripoSG), above the `1.10x` ceiling.
- Paired Chamfer ratio `1.1347448x` on `mesh_dir_3ea1828365_v00`
  (`0.2134` Hunyuan versus `0.1881` TripoSG), also above the ceiling.
- Worst paired H95 ratio `1.1249693x` on `mesh_dir_c3a9748da7_v00`
  (`0.3638` Hunyuan versus `0.3234` TripoSG), above the `1.10x` ceiling.

Median surface ratios favor Hunyuan (`0.9042x` Chamfer and `0.8174x` H95),
so this is a bounded repair-quality problem rather than a failed multiview
model. The raw Hunyuan meshes demonstrate the opportunity: median Chamfer is
`0.0488328`, median H95 is `0.1558482`, and median held-out IoU is `0.8998789`.
They are not printable, however. Raw meshes range from `534,048` to
`1,628,024` faces; only `6/10` are watertight/volumetric, `0/10` are manifold,
and only `3/10` are single-body. Component count ranges from `1` to `41`.

Printable repair is reliable but sometimes too aggressive. It converts all
ten outputs into watertight, manifold, single-body STLs and reuses the raw mesh
cache `10/10`, but repaired face counts range from `144` to `10,742`. The next
optimization should filter and close the dominant raw surface before adaptive
decimation, then optimize Chamfer/H95 under the existing complexity cap. The
surface ceilings should not be weakened.

Median raw-provider inference was `34.7639` seconds at 50 steps and octree
resolution 380. Peak allocated/reserved CUDA memory was `5.6822/5.8008` GiB.
Cached repaired invocations took `0.0115` seconds before a median `1.1009`
seconds of mesh repair.

## Provenance

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, about `95` GiB.
- Runtime code: `8de0bc749da24ec3b2c018ff6b37d391bbc763b1`.
- Official Hunyuan source/model pins: `f8db63096c8282cb27354314d896feba5ba6ff8a` / `3a761b539b29fe4ff64714813aa9560fd66f5de0`.
- Input payload: `1,069,043` bytes, SHA256 `5a86953669f72f1f2e544706603bed3e487761da01c8e957d8b0f13b774cf9b6`.
- Full result archive: `444,218,826` bytes, SHA256 `65ac1aba0e30c287c48de66cc5514a0796d7f67e14ed47dcf8c20afff2dd8e06`.
- Locally verified metrics archive: `1,110,155` bytes, SHA256 `09156bbe5d877ba392d5690df9f63d0e972dc041b5f8e676a9b2ecc7990c60c9`.
- Local ingest of the compact archive reproduced the seven ranked rows, `70` per-sample rows, and `promote-challenger` STL architecture result.
- The incumbent-aware `selection_decision.json` remains `hold`; architecture replacement and provider promotion are intentionally separate decisions.
- Source geometry and source bbox were diagnostic-only and were never provider inputs.

The 444 MB full mesh archive is intentionally not committed. The locally
re-ingested metrics archive remains outside git. This directory retains
provider/runtime preflight, exact archive provenance, the resolved experiment
configuration, all `70` sanitized per-sample metric rows, a focused 30-row
raw/repaired/incumbent extract, aggregate ranking and selection, the local
ingest report, and the four-sample contact sheet needed to audit the result.
