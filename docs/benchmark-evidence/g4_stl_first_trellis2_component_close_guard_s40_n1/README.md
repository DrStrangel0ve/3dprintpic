# TRELLIS.2 Guarded Component-Close Smoke

This bundle records `g4_stl_first_trellis2_component_close_guard_s40_n1`,
completed on exact runtime commit
`a628e172a1ab61d3bc57eb991dc9854065c617a5`. The six one-row arms ran on
an NVIDIA RTX PRO 6000 Blackwell Server Edition. Five rows completed and the
new no-hull TRELLIS repair failed closed before it could emit an STL.

## Decision

The strict decision is `hold`; TripoSG remains the production incumbent and
the prepared five-row TRELLIS expansion was not run. The candidate reached
its adaptive face target only by relaxing topology and boundary preservation,
then failed the mandatory self-intersection audit with `2,943` intersecting
faces. Expanding a candidate that emitted `0/1` printable outputs would have
violated the predeclared gate.

The guarded path behaved as intended: it preserved the diagnostics from the
failed provider invocation instead of silently accepting a geometrically
invalid mesh or falling back to a convex hull.

## Repair Trace

For `table_0247`, the exact cached raw TRELLIS mesh had `657,642` faces and
`632` connected components. The one-percent surface-area filter retained four
components and `641,832` faces. Bounded hole closure found `54` eligible loops
and added `98` faces.

The optimal decimator then produced:

| stage | faces |
| --- | ---: |
| topology and boundary preserved | `20,834` |
| topology relaxed, boundary preserved | `18,862` |
| topology and boundary relaxed | `10,704` |
| requested target after hole reserve | `10,704` |

The final boundary-relaxed mesh had normalized surface Chamfer/H95
`0.01994`/`0.07827` against raw TRELLIS, maximum bbox-extent drift `0.07826`,
and absolute volume drift `12.8568`. It was rejected because its self-
intersection count was `2,943`; no hull fallback was permitted.

## One-Row Controls

| method | success | Chamfer | H95 | printable | hull | fill drift | complexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TripoSG incumbent | `1/1` | `0.12637` | `0.31447` | yes | no | `0.000000008` | `9.94987` |
| raw TRELLIS diagnostic | `1/1` | `0.20944` | `0.43053` | no | n/a | n/a | `15.89697` |
| existing TRELLIS repair | `1/1` | `0.16787` | `0.39442` | yes | yes | `271.4760` | `5.55301` |
| guarded component-close | `0/1` | n/a | n/a | no output | no | n/a | n/a |

Held-out rendering was unavailable for this failed smoke output, so no view
agreement claim is made. Peak CUDA VRAM was also unsupported by the TRELLIS
provider wrapper in this run. End-to-end orchestration took `219.018s`; the
failed repair used `15.9343s`, including `0.4860s` of geometry diagnostics.

## Cached Ablations

The exact raw mesh remained cached while the G4 was connected, so three
bounded diagnostic variants were tested before runtime teardown:

- Closing bounded holes before strict simplification was worse: `14`
  components, `12,847` nonmanifold edges, and `5,468` self-intersections.
- A filled voxel surface at resolution `128`, followed by simplification to
  `10,704` faces, retained `20` nonmanifold edges and `817` self-intersections.
  Its surface Chamfer/H95 were `0.01170`/`0.03008`, but absolute volume drift
  was `33.677`.
- Native filled voxel meshes were printable without decimation. Resolutions
  `40` through `96` failed the `9.95` complexity cap; resolutions `16`
  through `24` passed complexity but had absolute volume drift from `60.581`
  to `91.628`.
- Native shell-only voxel meshes were printable and self-intersection-free at
  every tested resolution. The under-cap resolutions `16`, `20`, and `24`
  still had relative fill drift of `130.424`, `139.016`, and `104.091`.
  `voxel_shell_probe.csv` preserves all ten measurements.

These results close the current post-hoc TRELLIS repair lane. No tested shape-
preserving path simultaneously produced a printable mesh, respected the
scale-free complexity cap, and satisfied repair-drift guards.

## Metric Limitation

The raw mesh is open and inconsistently wound, yet the current fill-drift
metric uses its signed volume divided by bbox volume as the baseline. Here
that raw fill ratio is `0.00290124`; the filtered raw value is `0.00687578`.
Signed volume is not a stable enclosed-volume estimate for an open mesh, so
the very large fill-drift values are useful warnings but not reliable shape
measurements on their own.

This run does not waive or retune the existing gate after seeing the result.
The next harness change should introduce a topology-aware, surface-derived
volume proxy for open raw meshes and add explicit paired Chamfer/H95 promotion
guards. Until both are validated on fixed fixtures and historical evidence,
the original fill gate remains authoritative.

## Provenance

- Input payload: `139,838` bytes, SHA256
  `3d240cee35f89dd342e7db780ce4785b261fb1ec0d7322f308eb16df75c5b02d`.
- Compact result archive: `314,308` bytes, `73` entries, SHA256
  `271aac157ac2978dbef3fadb653bb06eee75158f32e0cadee141f781d4d523b6`.
- GPU guard: minimum `90 GB`, name matching RTX PRO 6000, Blackwell, or G4;
  observed VRAM was `95.5928 GB`.
- TRELLIS source revision: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`.
- TRELLIS.2 model revision:
  `af44b45f2e35a493886929c6d786e563ec68364d`.

The compact aggregate, failed-provider metrics, selector output, preflights,
contact sheet, and locally regenerated ingest report are included. The G4
runtime was disconnected and deleted after evidence transfer.
