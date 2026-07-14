# Natural 30 mm relief and context-preserving selections

## Problem

Two measured failures shared the same visible symptom but had different causes:

- A 30 mm portrait relief passed the old quality gate after retaining only 41.9% of aggregate facial curvature. The two faces retained 39.0% and 47.3% separately, and the successful global solve bypassed the computed protected face core.
- An object-selection run inferred depth after replacing 81.7% of the original scene with a neutral background. On the four meaningful selected components, full-scene inference contained 3.0x to 4.5x more local gradient energy than cut-out inference. Skyline trimming then split the selected output into four printable-but-disconnected slabs.

The fix therefore does not treat this as one global smoothing parameter.

## Research direction

The implementation follows the useful common ground in current monocular geometry and relief literature:

- Infer geometry with the original scene context intact, then apply the semantic mask. Microsoft MoGe-2 exposes metric point/depth and optional normal maps and is the next provider candidate, but the context-ordering fix applies to the current provider as well. See the [official MoGe repository](https://github.com/microsoft/MoGe) and [MoGe-2 paper](https://arxiv.org/abs/2507.02546).
- Combine globally coherent low-frequency depth with bounded high-frequency gradients instead of trusting a single smoothed depth field. This is consistent with [multi-resolution monocular depth fusion](https://arxiv.org/abs/2212.01538) and the nonlinear gradient reconstruction used in [Digital Bas-Relief from 3D Scenes](https://gfx.cs.princeton.edu/pubs/Weyrich_2007_DBF/relief.pdf).
- Do not force strong discontinuities through an ordinary dense Poisson penalty. The separate semantic domains and boundary guards follow the motivation of [discontinuity-preserving normal integration](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Discontinuity-preserving_Normal_Integration_with_Auxiliary_Edges_CVPR_2024_paper.pdf).
- Apple Depth Pro remains a useful metric-depth comparison because it targets sharp boundaries, but its custom research license is less straightforward than MoGe-2's MIT model/code path. See the [official Depth Pro repository](https://github.com/apple/ml-depth-pro).

## Implemented pipeline

### Selected objects

1. `/selection/compose` stores the EXIF-normalized source, selected appearance, mask, and source fingerprint as one opaque compose job. The browser sends only its job ID; `/process_image` loads that exact trio atomically and rejects missing, tampered, or mismatched provenance.
2. Pending click/compose requests are generation-bound in the browser. Replacing the photo invalidates them synchronously, so a late response cannot pair an old mask with a new source.
3. Depth inference runs on the original full-scene artifact. Face refinement and RGB detail use the same normalized orientation, while the selected artifact remains the appearance source.
4. `compose_selection_depth_with_context` preserves every selected depth sample exactly. The immediate mask boundary still extends toward a robust base plane at a physical slope derived from relief height, XY pitch, and the configured maximum slope.
5. Farther from the subject, a low-noise copy of the original scene depth receives a bounded 45% depth budget. A 1.5 mm smoothstep transition and `max(support ramp, context)` composition keep the subject attached while allowing buildings, terrain, and other background layers to remain legible.
6. The old flat-background result is replayable with `selection_background_depth_ratio=0`; historical evidence is therefore not silently reinterpreted.
7. Context-selected reliefs use a rectangular base instead of skyline trimming, preventing disconnected vertical slabs.
8. A selection-specific screened gradient solve compresses large internal terraces while retaining moderate selected-object gradients. Tiny nonmetric fragments are excluded from the selection metric, while every detected face still fails closed if it cannot be measured.
9. When faces and other selected subjects coexist, both solvers run. The object solve is feathered around the protected head region and is accepted only if every face still clears correlation and RMS-retention gates.
10. Curvature telemetry uses finite values or `null` plus an explicit flat-reference violation, so a rejected candidate cannot turn a valid fallback STL into a JSON serialization failure.
11. Final-surface background telemetry measures centred depth correlation, RMS and percentile-span retention, gradient correlation/RMS, absolute mean shift, and subject-boundary jumps outside the selected subject. Lower and upper bounds reject both flattening and artificial amplification.
12. A second cap runs in physical millimetres after gamma and every surface solver. Far context cannot exceed 45% of the requested relief height. High and low subject boundaries both constrain the immediate attachment; incompatible constraints are counted explicitly rather than silently certified.
13. The comparison remains the original bounded pre-solver surface; it is never rewritten from the candidate being scored. A failed final gate may restore that reference through a physical feather, but the result is accepted only when the slope guard, far cap, satisfiable attachment constraints, coverage, and preservation metrics all pass.
14. Nonpositive inverse-depth samples are excluded before selected-depth percentiles are computed, and large invalid patches fall back to the finite support surface. Candidate coverage is measured against the reference context so missing geometry cannot pass by omission.

### Faces at high relief

1. Moderate gradients inside an eroded face core retain 80% of their source amplitude. Extreme gradients and the face boundary still use nonlinear compression.
2. Quality is measured for every disconnected face, not only over their union. The minimum per-face gates are correlation 0.8 and curvature RMS retention 0.6.
3. A post-solve high-pass restoration is capped at 0.6 mm and reprojected against the accepted surface. Its morphological boundary is explicitly reset and audited.
4. Feature embossing is audited again after application. The update is attenuated until every face still passes, preventing glasses or landmark weights from degrading an otherwise accepted solve.
5. Opaque eyewear de-occlusion and its feature-exclusion mask remain active before these stages.

## Exact private replay results

The source photos and generated meshes remain local and ignored. Aggregate evidence is tracked in `docs/benchmark-evidence/relief_30mm_face_context_v2/summary.json`.

| Check | Previous | New live path |
|---|---:|---:|
| Portrait face curvature RMS retention | 0.390 / 0.473 | 0.921 / 0.967 |
| Portrait face curvature correlation | 0.913 / 0.934 | 0.954 / 0.966 |
| Eyewear residual reduction | unavailable in stale process | 63.3% / 58.5% |
| Face restoration boundary correction | not audited after solve | 0.000 mm |
| Llama/group depth input | neutral-background cut-out | original full scene |
| Full-scene local gradient gain | baseline | 2.96x / 3.81x / 4.11x / 4.55x |
| Llama/group STL components | 4 | 1 |
| Final STL topology | watertight/manifold | watertight/manifold |
| Degenerate faces | 0 | 0 |

The exact current-code RTX 3080 Ti replay completed in 21.6 seconds for the cold portrait run and 8.9 seconds for the warm llama/group run. The portrait also exercised the combined face-plus-selection path with face protection passing. Both emitted a single watertight, manifold, consistently wound positive volume with zero degenerates.

## Background context follow-up

The first context-preserving implementation still replaced all pixels beyond the subject's support halo with one base value. This protected connectivity but made the background look intentionally suppressed. The v2 compositor restores broad scene geometry without giving it equal visual priority to the selected subject.

The exact hardened 30 mm portrait replay used a 45% background budget, 1.5 mm boundary feather, and 0.6 mm depth smoothing. It restored context to 325,370 pixels (16.95% of the source grid) with 0.971 source-depth correlation. On the final STL, the old background's 5th-to-95th percentile height span was 4.357 mm; the new span is 12.690 mm, a 2.913x gain. Centred background depth energy increased 2.569x. Against the original bounded reference, the final gate passed with 0.960 depth correlation, 0.878 RMS retention, 0.988 span retention, 0.751 gradient correlation, and 0.667 gradient RMS retention. Mean shift was -0.344 mm, coverage was 100%, and the boundary-jump p99/max were 1.244/2.396 mm.

After the final feature guard, the two portrait face components remained above the face gates at 0.943/0.948 curvature correlation and 0.931/0.968 RMS retention. The llama/group replay passed at 0.979 correlation, 0.910 RMS retention, and 0.937 span retention while keeping a 17.169 mm robust background span and 6.330 mm centred depth RMS. Its four selected-object component correlations remain 0.753, 0.929, 0.818, and 0.938. In both runs, far context was held to 13.510 mm with zero cap violation, and every satisfiable attachment was held to the 0.800 mm neighbor-step limit. The strict audit separately reported 26 portrait and 100 llama pixels where neighboring selected heights make those one-step constraints mutually incompatible; those cases are not mislabeled as strict cap successes. Both outputs remain single-component watertight manifold positive volumes with consistent winding and zero degenerates.

Aggregate follow-up evidence is in `docs/benchmark-evidence/relief_30mm_background_context_v2/`. Private source images, masks, renders, and meshes remain local and ignored.

## Validation

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend
npm run typecheck
npm run lint
npm run test:ui
```

Measured state on 2026-07-14:

- Backend: 423 passed, 8 subtests passed.
- Frontend typecheck: passed.
- Frontend lint: passed with zero warnings.
- Playwright: 9 passed, 1 intentionally skipped, including the delayed-compose replacement-photo race.
- Exact live portrait and llama/group requests: HTTP 200 with all STL hard checks passing.
