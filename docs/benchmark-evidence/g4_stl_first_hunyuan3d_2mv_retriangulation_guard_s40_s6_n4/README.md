# G4 Hunyuan Retriangulation Guard Audit

This directory preserves compact evidence for Colab G4 run
`g4_stl_first_hunyuan3d_2mv_retriangulation_guard_s40_s6_n4`, completed on
July 12, 2026. It replayed the same three Hunyuan repair outliers and one
control as the cleanup-bypass audit, using the same cached provider outputs.

All `36/36` method/sample rows completed. The result is intentionally
negative: the broad PyMeshLab marching-cubes edge-collapse filter must not be
used as a one-face repair. It either failed to remove the degenerate face or
changed far more geometry than the name suggested. The new fail-closed guard
rejected every attempted retriangulation, restored the prior fallback
behavior, and kept TripoSG as the only promotion-eligible mesh provider.

## Decision

Do not expand this Hunyuan repair to the held-out ten. The configured r192
unfiltered candidate remains `hold` against TripoSG. It still fails:

- paired win rate against the current method;
- convex-hull fallback rate (`0.50` versus the `0.25` limit);
- median repair fill drift (`0.5621` versus the `0.50` limit);
- per-sample fill drift (`7.0545x` versus the `4.0` limit);
- fill-drift coverage.

| method | score | held-out IoU | retriangulation attempts | accepted | geometry preserved | hull rate | median fill drift | faces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan legacy | `0.3681` | `0.5034` | n/a | n/a | n/a | `1.00` | `12.2245` | `233` |
| Hunyuan r192, area 1% | `0.2777` | `0.4868` | `0.50` | `0.00` | `0.00` | `0.75` | `3.8409` | `265` |
| Hunyuan r192, all | `0.2672` | `0.4840` | `0.25` | `0.00` | `0.00` | `0.50` | `0.5621` | `5,430` |
| Hunyuan r256, area 1% | `0.1997` | `0.4703` | `0.00` | `0.00` | n/a | `0.50` | `0.4064` | `5,447` |
| Hunyuan r256, all | `0.1993` | `0.4697` | `0.00` | `0.00` | n/a | `0.50` | `0.3617` | `5,447` |
| TripoSG control | `-0.0756` | `0.4378` | n/a | n/a | n/a | `0.25` | `0.000003` | `10,712` |
| Hunyuan raw | `-5.9691` | `0.8007` | n/a | n/a | n/a | n/a | n/a | `754,531` |

Scores use held-out mean/min silhouette agreement and STL validity. Hidden
source-mesh Chamfer/H95 remain diagnostic-only, and complexity remains a hard
eligibility cap.

## Guard Finding

The parent diagnostic run accepted one apparently printable r192
area-filtered chair and reduced the aggregate Hunyuan hull count by one. The
stage audit showed that this was not a local repair:

- the chair lost `82.815%` of faces and `88.374%` of vertices;
- absolute volume changed by `40.739%`;
- maximum bbox-extent change was `6.052%`;
- normalized surface Chamfer/H95 were `0.009835/0.022576`;
- the final mesh had only `1,836` faces.

The r192 bathtub remained non-printable after the same filter and still had
one degenerate face. It nevertheless lost `43.826%` of faces and `44.039%` of
vertices and changed volume by `5.301%`.

The accepted-repair gate now requires all strict printability predicates and
all of these geometry limits:

- face-count and vertex-count change at most `2%` each;
- absolute volume change at most `1%`;
- maximum bbox-extent change at most `1%`;
- normalized bounds-center shift at most `0.5%`;
- normalized symmetric surface Chamfer at most `1%`;
- normalized surface H95 at most `3%`.

Every value must be finite. An audit exception also rejects the candidate.
The full per-row measurements are in `repair_stage_per_sample.csv`; the three
run comparison is in `retriangulation_comparison.csv`.

## Next Repair

The measured [local-collapse follow-up](../g4_stl_first_hunyuan3d_2mv_local_collapse_guard_s40_s6_n4/README.md)
replaced the broad simplifier with one topology-preserving edge collapse and
an absolute two-face/one-vertex budget. It preserved geometry on every real
attempt but left the degenerate face in place, so all three attempts were
rejected and the decision remained `hold`. The next intervention belongs in
the upstream voxel decimator rather than another post-hoc repair.

## Provenance

- Runtime code: `d8e082e992d3f9d334db5469780c946247dd81aa`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `94.971` GiB.
- Runtime: `594.801s`.
- Rewritten run manifest SHA256:
  `9f0aaec46541735ef6ed95110e151ee205565b51c5be7af6d4b015125d99c89a`.
- Compact result archive: `1,298,196` bytes, `207` entries, SHA256
  `bda740e5915ce86a0b0e888a352f3f78fb6e818e4e88a9fb45aa1dbfd5c56949`.
- Parent diagnostic archive: `1,296,817` bytes, `207` entries, SHA256
  `5ea5459ee94b13b3fa17d4f745feb9bdd58a38ac601c6397360781b04588ca3b`.

The full mesh archives are intentionally not committed. This directory keeps
the rankings, selector decision, sixteen voxel-stage rows, guard metrics,
provider/GPU provenance, local ingest report, and visually checked contact
sheet needed to reproduce the decision.

Validation at this checkpoint: full backend suite `308` passing; complete
benchmark regression module `191` passing; no actionable findings in the
bounded follow-up code review.
