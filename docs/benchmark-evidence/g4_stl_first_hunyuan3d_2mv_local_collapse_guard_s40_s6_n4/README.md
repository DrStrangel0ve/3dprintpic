# G4 Hunyuan Local-Collapse Audit

This directory preserves compact evidence for Colab G4 run
`g4_stl_first_hunyuan3d_2mv_local_collapse_guard_s40_s6_n4`, completed on
July 12, 2026. It replayed the same three Hunyuan repair outliers and one
control using exact runtime commit
`3da31816575fa26b370c7601c02c2d8ebb2fe8eb`.

All `36/36` method/sample rows completed. The one-edge repair is safe but does
not fix these provider meshes. All three attempts stayed inside every geometry
limit, but the original degenerate face remained after the collapse. No attempt
was accepted, hull use remained unchanged, and the held-out ten remains gated.

## Decision

The configured r192 unfiltered candidate remains `hold` against TripoSG. It
still uses convex-hull fallback on `2/4` samples, has median fill drift
`0.5621`, and retains a `7.0545x` bathtub outlier. TripoSG remains the only
promotion-eligible mesh provider in this slice.

| method | score | held-out IoU | local attempts | accepted | geometry preserved | max retained-vertex displacement | hull rate | median fill drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan legacy | `0.3681` | `0.5034` | n/a | n/a | n/a | n/a | `1.00` | `12.2245` |
| Hunyuan r192, area 1% | `0.2777` | `0.4868` | `0.50` | `0.00` | `1.00` | `0.0` | `0.75` | `3.8409` |
| Hunyuan r192, all | `0.2672` | `0.4840` | `0.25` | `0.00` | `1.00` | `0.0` | `0.50` | `0.5621` |
| Hunyuan r256, area 1% | `0.1997` | `0.4703` | `0.00` | `0.00` | n/a | n/a | `0.50` | `0.4064` |
| Hunyuan r256, all | `0.1993` | `0.4697` | `0.00` | `0.00` | n/a | n/a | `0.50` | `0.3617` |
| TripoSG control | `-0.0756` | `0.4378` | n/a | n/a | n/a | n/a | `0.25` | `0.000003` |
| Hunyuan raw | `-5.9691` | `0.8007` | n/a | n/a | n/a | n/a | n/a | n/a |

Scores use held-out mean/min silhouette agreement and STL validity. Hidden
source-mesh Chamfer/H95 remain diagnostic-only, and complexity remains a hard
eligibility cap.

## Local Edit Audit

The repair uses topology-preserving quadric edge collapse with endpoint
placement. It targets exactly two fewer faces and rejects additions, more than
two removed faces, or more than one removed vertex. The retained-vertex
displacement limit is `0.001` of the reference bbox diagonal, in addition to
the volume, bbox, center, surface Chamfer, and H95 limits.

Every real attempt removed exactly two faces and one vertex:

| method/sample | face change | vertex change | vertex displacement | volume change | degenerate faces after | accepted | final fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r192 area, bathtub `001bfe840e` | `0.0001854` | `0.0001863` | `0.0` | `0.0` | `1` | no | `176`-face hull |
| r192 all, bathtub `001bfe840e` | `0.0001854` | `0.0001863` | `0.0` | `0.0` | `1` | no | `176`-face hull |
| r192 area, chair `3ea1828365` | `0.0001872` | `0.0001998` | `0.0` | `0.0` | `1` | no | `354`-face hull |

This distinguishes the failure from the broad-filter result. The global
filter changed too much geometry; the local filter preserves geometry but
does not select the edge responsible for the degenerate triangle. Tightening
or loosening the post-hoc acceptance limits will not solve this case.

## Next Experiment

Move the intervention upstream into topology-preserving voxel-mesh
simplification. Compare endpoint versus optimal placement and nearby target
face budgets on the exact cached raw meshes, then select only configurations
that emit zero-degenerate pre-clean meshes while preserving held-out views,
volume, complexity, and all printability gates. Do not weaken the selector or
expand to ten rows until a four-row variant reduces hull use to at most one.

## Provenance

- Runtime code: `3da31816575fa26b370c7601c02c2d8ebb2fe8eb`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `94.971` GiB.
- Runtime: `597.744s`.
- Rewritten run manifest SHA256:
  `9f0aaec46541735ef6ed95110e151ee205565b51c5be7af6d4b015125d99c89a`.
- Compact result archive: `1,297,705` bytes, `207` entries, SHA256
  `ed34c41d81d392b0673acc7e8e529ef94b6eb1b584e1809c8b50204dfec4ab6e`.

The full mesh archive is intentionally not committed. This directory keeps
the rankings, selector decision, sixteen voxel-stage rows, local-edit and
displacement metrics, provider/GPU provenance, local ingest report, and
visually checked contact sheet needed to audit the decision.

Validation: full backend suite `310` passing; complete benchmark regression
module `193` passing; fault-injected tests cover spatial displacement,
edit-budget overshoot, and the printable/geometry acceptance conjunction.
