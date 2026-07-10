# G4 Adaptive Inferred-Bbox Evidence

This directory preserves the compact evidence for Colab run
`g4_stl_first_triposg_inferred_adaptive_s40_n10`, completed on July 10,
2026. The run evaluated seven methods on the same ten held-out ModelNet rows
`40-49` and completed all `70/70` method/sample rows.

## Decision

`triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh` was
promoted over `mirror` with no failed selection checks. It won `10/10` paired
comparisons against both `masked` and `mirror`; the paired CI95 lower bounds
were `1.0655100` and `0.9277428`, respectively. Every candidate sample passed
all twelve per-sample STL gate families, including watertightness,
manifoldness, positive volume, one connected body, bbox health, and scale-free
complexity.

| method | score | Chamfer median | H95 median | faces median | complexity median |
| --- | ---: | ---: | ---: | ---: | ---: |
| TripoSG biharmonic prefill, inferred bbox, adaptive cap | `1.1038838` | `0.1593129` | `0.3849049` | `3,855` | `8.2388673` |
| TripoSG masked input, inferred bbox, adaptive cap | `0.9674651` | `0.1890342` | `0.4861368` | `584` | `7.0139660` |
| TripoSG mirror prefill, inferred bbox, adaptive cap | `0.5622659` | `0.1651543` | `0.4233816` | `10,677` | `9.9482615` |
| Mirror depth relief | `0.1991798` | `0.1592503` | `0.4768460` | `36,860` | `11.1885910` |
| Source mesh oracle, diagnostic only | `2.9898396` | `0.0746460` | `0.2376423` | `103` | `5.3730671` |

The promoted candidate's ten complexity values range from `6.4386047` to
`9.9499179`, below the selector maximum of `10`. Its median inferred-bbox
diagnostics against hidden ground truth were shape log MAE `0.2112335`, shape
relative MAE `0.1939597`, and centered IoU `0.5528558`; those hidden metrics
were diagnostic only and were not inputs to mesh generation.

## Provenance

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `95.5928` GB.
- Runtime code: `008f45b676960c38390080bee644f18883602433`.
- Input payload: `1,626,001` bytes, SHA256 `4556853d62b9150dda9ea74b684c410f2ddba2cdc99ead8946b1187d5299d3d3`.
- Result archive: `29,533,390` bytes, SHA256 `8cb4a836c3e5a1da205edb8c9dd80cf4bde26921c18110b13d5b1199ff358a78`.
- Local re-ingest completed successfully and reproduced `promote-challenger`.

The full result archive is intentionally not committed. The files here retain
the selection decision, ranked aggregate metrics, sanitized candidate
per-sample metrics, runtime/GPU provenance, in-run ingest report, result
summary, and visual contact sheet needed to audit the documented conclusion.
