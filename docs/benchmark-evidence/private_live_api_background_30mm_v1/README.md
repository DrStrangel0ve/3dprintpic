# Exact-input live API background replay at 30 mm

This one-row retained-artifact replay closes the gap between the direct relief
harness and the actual `/process_image` HTTP path. The source, selection mask,
face-refined depth, composed context depth, face region, and feature-weight mask
are checksum-pinned locally. Only aggregate telemetry is tracked.

## Finding and fix

The first live request returned HTTP 200 and a printable STL, but reported
`background_photo_detail.enabled=false` with `reason=no_photo_detail`. Selection
mode inferred full-scene depth correctly, then passed the neutral subject-only
image into photographic relief generation. The selected background therefore
contained no structure to restore.

Commit `a80df41` makes the photo-detail stage use the same original full-scene
source as context-depth inference while leaving the neutral selected image in
place for face refinement. Commit `2453f28` additionally retains the emitted
and reference heightfields under ignored output so the serialized STL shell can
be verified exactly. Commit `7c9c7fe` binds each response to the source revision
loaded by the server and adds the committed omission-aware launcher and
fail-closed numeric summarizer.

## Clean result

The final request ran from a detached clean worktree at
`7c9c7fe962bea2c99ba1c9f75d6fe810c04435fe`. The launcher held the `30 mm`,
`128 mm`, 512-grid, and sigma `0.35` physical controls constant while omitting
the two background fields. The response observed the endpoint defaults of
`0.60 mm` photo detail and `0.65` background depth ratio.

| Gate | Result |
| --- | ---: |
| Photo-detail method | Euclidean 2 mm guard, full at 5 mm |
| Face normal mean cosine | 0.9911 |
| Face normal p95 error | 7.9260 degrees |
| Minimum face relighting correlation | 0.9167 |
| Face detail correlation / RMS retention | 0.9656 / 0.9476 |
| Background depth correlation | 0.999994 |
| Background gradient correlation | 0.999260 |
| Background RMS retention | 1.000051 |
| Feasible attachment jump max | 0.800001 mm |
| Far-background cap violation | 0 mm |
| STL topology | one watertight manifold volume, zero degenerates |
| Serialized shell | 309,440/309,440 facets, zero error |
| Runtime | 59.859 s |

All exact-input checks pass: the live face-refined depth and composed context
depth match the pinned direct-core arrays, as do the source, selection, face,
feature-weight, and pre-high-relief reference-surface artifacts. The private
request record binds the response hash and proves neither background field was
posted. The response's import-time revision stamp matches the clean worktree.
Face appearance uses the existing frozen gates, not thresholds selected for
this row; face, background, and physical-cap decisions are independently
recomputed from their numeric telemetry instead of trusting producer booleans.

Strict attachment status remains false because 12 one-pixel constraints are
mutually incompatible. Far-background and every feasible attachment constraint
pass, so this remains an emission pass rather than being mislabeled strict.

## Reproduction

Run a clean server for the implementation revision, copy the checksum-pinned
selection bundle into that worktree's ignored output, and post the source plus
the following form controls with
`backend.benchmark.run_private_live_api_background_replay`. The launcher
deliberately omits `background_photo_detail_mm` and
`selection_background_depth_ratio` and writes an ignored request record.

```text
target_dimension=512
z_scale=30
max_xy_size=128
sigma=0.35
```

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_private_live_api_background_replay `
  --scene-config backend/output/<private-run>/scene_config.json `
  --scene-label scene-01 `
  --clean-repository <clean-worktree> `
  --output-dir <clean-worktree>/backend/output/<private-run> `
  --base-url http://127.0.0.1:8005
```

Then generate the tracked aggregate:

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.summarize_private_live_api_background_replay `
  --scene-config backend/output/<private-run>/scene_config.json `
  --scene-label scene-01 `
  --response-json <clean-worktree>/backend/output/<private-run>/response.json `
  --request-record <clean-worktree>/backend/output/<private-run>/request_record.json `
  --clean-repository <clean-worktree> `
  --direct-context-depth backend/output/<private-run>/composed_depth.npy `
  --direct-reference-surface backend/output/<private-run>/reference_surface.npy `
  --expected-revision 7c9c7fe962bea2c99ba1c9f75d6fe810c04435fe `
  --aggregate-summary docs/benchmark-evidence/private_live_api_background_30mm_v1/summary.json
```

The summarizer fails closed for an unbound response or posted background field,
a dirty/mismatched runtime revision, non-exact private input or reference
surface, disabled/wrong photo-detail gate, numerically failed face/background/
cap telemetry, failed topology, missing heightfields, shell facet disagreement,
or any private path, hash, filename, or job identifier in the aggregate output.
