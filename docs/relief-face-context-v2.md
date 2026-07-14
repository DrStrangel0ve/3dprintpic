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
4. `compose_selection_depth_with_context` preserves every selected depth sample exactly. Outside the mask it extends the nearest selected boundary toward a robust base plane at a physical slope derived from relief height, XY pitch, and the configured maximum slope.
5. Context-selected reliefs use a rectangular base instead of skyline trimming, preventing disconnected vertical slabs.
6. A selection-specific screened gradient solve compresses large internal terraces while retaining moderate selected-object gradients. Tiny nonmetric fragments are excluded from the selection metric, while every detected face still fails closed if it cannot be measured.
7. When faces and other selected subjects coexist, both solvers run. The object solve is feathered around the protected head region and is accepted only if every face still clears correlation and RMS-retention gates.
8. Curvature telemetry uses finite values or `null` plus an explicit flat-reference violation, so a rejected candidate cannot turn a valid fallback STL into a JSON serialization failure.

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

## Validation

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend
npm run typecheck
npm run lint
npm run test:ui
```

Measured state on 2026-07-14:

- Backend: 415 passed, 8 subtests passed.
- Frontend typecheck: passed.
- Frontend lint: passed with zero warnings.
- Playwright: 9 passed, 1 intentionally skipped, including the delayed-compose replacement-photo race.
- Exact live portrait and llama/group requests: HTTP 200 with all STL hard checks passing.
