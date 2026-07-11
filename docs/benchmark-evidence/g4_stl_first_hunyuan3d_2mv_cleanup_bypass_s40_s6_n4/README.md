# G4 Hunyuan Cleanup-Bypass Audit

This directory preserves compact evidence for Colab G4 run
`g4_stl_first_hunyuan3d_2mv_cleanup_bypass_s40_s6_n4`, completed on
July 12, 2026. The run replayed the same three Hunyuan repair outliers and
one control used by the preceding ablation. It changed only the runtime code:
printable voxel-decimated meshes bypassed generic cleanup, and every voxel
variant recorded pre-clean and post-clean printability predicates.

All `36/36` method/sample rows completed. Do not expand this repair to the
held-out ten. Cleanup bypass preserved five already-printable voxel meshes,
but total convex-hull use remained `14` rows across all repaired methods,
exactly matching the original ablation.

## Decision

The configured r256 area-filtered candidate remains `hold` against TripoSG.
Its only failed selector checks are:

- convex-hull fallback rate `0.50`, above the `0.25` limit;
- one `7.0766x` absolute fill-ratio-drift outlier, above the `4.0` limit.

TripoSG is the only promotion-eligible mesh provider in this slice. Hunyuan
repairs score higher on held-out views, but every variant still fails repair
integrity gates.

| method | score | held-out IoU | preclean printable | cleanup bypass | hull rate | median fill drift | faces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hunyuan legacy | `0.3681` | `0.5034` | n/a | `0.00` | `1.00` | `12.2245` | `233` |
| Hunyuan r192, area 1% | `0.2777` | `0.4868` | `0.00` | `0.00` | `0.75` | `3.8409` | `265` |
| Hunyuan r192, all | `0.2672` | `0.4840` | `0.50` | `0.50` | `0.50` | `0.5621` | `5,430` |
| Hunyuan r256, area 1% | `0.1997` | `0.4703` | `0.25` | `0.25` | `0.50` | `0.4064` | `5,447` |
| Hunyuan r256, all | `0.1993` | `0.4697` | `0.50` | `0.50` | `0.50` | `0.3617` | `5,447` |
| TripoSG control | `-0.0756` | `0.4378` | n/a | `0.00` | `0.25` | `0.000003` | `10,712` |
| Hunyuan raw | `-5.9691` | `0.8007` | n/a | n/a | n/a | n/a | `754,531` |

Scores use held-out mean/min silhouette agreement and STL validity. Hidden
source-mesh Chamfer/H95 remain diagnostic-only, and complexity remains a hard
eligibility cap.

## Stage Audit

The audit separates three repair cases that previously looked identical:

- `mesh_dir_001bfe840e_v00` at r192 is already watertight, volumetric,
  winding-consistent, manifold, and single-body before cleanup. Its only
  failed predicate is one degenerate face. Generic cleanup then fails and the
  method falls back to a `176`-face hull with `7.0545x` fill drift.
- `mesh_dir_8a68107048_v00` is genuinely invalid before cleanup. At r192 it
  has three components and six nonmanifold edges; at r256 it has 65 components
  and 97 nonmanifold edges. Cleanup bypass cannot safely rescue this mesh.
- Component-area filtering is not reliably beneficial. It turns the otherwise
  printable r192 chair into a one-degenerate-face hull and turns the otherwise
  printable r256 bed into a seven-component, six-nonmanifold-edge mesh that
  generic cleanup must recover.

The bounded follow-up therefore runs marching-cubes edge-collapse
retriangulation only when every predicate passes except exactly one degenerate
face. It re-audits the result and skips generic cleanup only if the full strict
audit then passes. Multi-component or nonmanifold outputs still follow the
existing cleanup/fallback path.

## Runner Finding

Evaluation completed, but the generated shell runner initially exited before
writing its result archives. The benchmark runs inside a subshell that changes
to `$REPO_DIR`; the post-run Python executes outside that subshell and could not
import `backend` from `/content`. The completed outputs were recovered by
rerunning only the packaging block with the repository on `PYTHONPATH`.

The generator now explicitly changes to `$REPO_DIR` before post-run ingest and
archive creation. No GPU inference was repeated during recovery.

## Provenance

- Runtime code: `d122a749b95eba893c6c881419a920e66a60860d`.
- Payload branch commit: `f3e1f26`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `95.593` GiB.
- Input payload: `806,446` bytes, SHA256
  `9b5e9fab0be0c3524a5fa32453cf3089f9c788e825512a904794444f19d90393`.
- Manifest SHA256:
  `fd796eb9b7fc423c8d5750fbb18e96591f72df5e38d83f68b352173ace96410f`.
- Compact result archive: `1,325,106` bytes, SHA256
  `fea5c1acf27ee1bffeebaeb8abe95a2a2fb21a5c04caa54f0cba0c63fde018ac`.
- Full mesh archive: `297,359,404` bytes, SHA256
  `7fea49d929a9cba1bd4b1e782c9fa645380afe31f700ae1700e8c39fdf62ef51`.

The full mesh archive is intentionally not committed. This directory retains
the rankings, selection decision, sixteen voxel-stage rows, provider/GPU
preflights, local ingest report, contact sheet, and payload provenance needed
to audit the decision.
