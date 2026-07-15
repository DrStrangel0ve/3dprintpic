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
8. A selection-specific screened gradient solve compresses large internal terraces while retaining moderate selected-object gradients. Its calibrated screened data weight is `2.0`, and recoverable local gradients retain 90% of their source amplitude. Tiny nonmetric fragments are excluded from the selection metric, while every detected face still fails closed if it cannot be measured.
9. When faces and other selected subjects coexist, both solvers run. The object solve is feathered around the protected head region and is accepted only if every face still clears correlation and RMS-retention gates.
10. Curvature telemetry uses finite values or `null` plus an explicit flat-reference violation, so a rejected candidate cannot turn a valid fallback STL into a JSON serialization failure.
11. Final-surface background telemetry measures centred depth correlation, RMS and percentile-span retention, gradient correlation/RMS, absolute mean shift, and subject-boundary jumps outside the selected subject. Lower and upper bounds reject both flattening and artificial amplification.
12. A second cap runs in physical millimetres after gamma and every surface solver. Far context cannot exceed 45% of the requested relief height. High and low subject boundaries both constrain the immediate attachment; incompatible constraints are counted explicitly rather than silently certified.
13. The background comparison combines the final processed foreground with the original background before applying the same physical cap to reference and candidate. This holds required support-ramp changes constant without letting the candidate rewrite the scene context being scored.
14. Nonpositive inverse-depth samples are excluded before selected-depth percentiles are computed, and large invalid patches fall back to the finite support surface. Candidate coverage is measured against the reference context so missing geometry cannot pass by omission.
15. Face/head gradient updates are clipped to the selected subject before the physical attachment stage. Overlapping 6 mm windows reject localized background loss that can hide inside strong global scores, and unavailable preservation telemetry blocks STL emission.
16. Final physical-normal and relighting telemetry scores faces, selected non-face surfaces, and background components independently. A disconnected island or an enclosed background hole cannot disappear into an aggregate score.
17. If a final background candidate fails preservation after an already capped reference surface passed, fallback restoration uses that accepted reference as the slope-audit baseline. The foreground remains unchanged, and the reference is still re-capped and re-audited before emission; a flattened rejected candidate can no longer prevent valid context from being restored.

### Faces at high relief

1. Moderate gradients inside an eroded face core retain 80% of their source amplitude. Extreme gradients and the face boundary still use nonlinear compression.
2. Quality is measured for every disconnected face, not only over their union. The minimum per-face gates are correlation 0.8 and curvature RMS retention 0.6.
3. A post-solve high-pass restoration is capped at 0.6 mm and reprojected against the accepted surface. Its morphological boundary is explicitly reset and audited.
4. Feature embossing is audited again after application. The update is attenuated until every face still passes, preventing glasses or landmark weights from degrading an otherwise accepted solve.
5. Opaque eyewear de-occlusion and its feature-exclusion mask remain active before these stages.
6. The high-relief face solve uses a uniform `0.25` screened data term at 20, 30, and 40 mm. The earlier `0.05` setting passed 30 mm narrowly but rejected the larger face at 40 mm, sending it through a fallback whose worst-light correlation fell to `0.233819`. The uniform setting raises the exact weaker-face scores to `0.862279`, `0.869303`, and `0.896382` while all face edge ratios remain below their physical limits.
7. Final face telemetry compares physical surface normals and four deterministic Lambertian lights. Coverage and every disconnected face component must pass independently, so missing geometry or one damaged face cannot hide inside a union score.

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

The original exact RTX 3080 Ti replay completed in 21.6 seconds for the cold
portrait run and 8.9 seconds for the warm llama/group run. The final cached-depth
`2.0` confirmation took 5.50 seconds for the portrait and 2.37 seconds for the
llama/group scene. The portrait exercised the combined face-plus-selection path
with face protection passing. Both emitted a single watertight, manifold,
consistently wound positive volume with zero degenerates.

## Background context follow-up

The first context-preserving implementation still replaced all pixels beyond the subject's support halo with one base value. This protected connectivity but made the background look intentionally suppressed. The v2 compositor restores broad scene geometry without giving it equal visual priority to the selected subject.

The exact hardened 30 mm portrait replay used a 45% background budget, 1.5 mm boundary feather, and 0.6 mm depth smoothing. It restored context to 325,370 pixels (16.95% of the source grid) with 0.971 source-depth correlation. On the final STL, the old background's 5th-to-95th percentile height span was 4.357 mm; the new span is 12.690 mm, a 2.913x gain. Centred background depth energy increased 2.569x. Against the original bounded reference, the final gate passed with 0.960 depth correlation, 0.878 RMS retention, 0.988 span retention, 0.751 gradient correlation, and 0.667 gradient RMS retention. Mean shift was -0.344 mm, coverage was 100%, and the boundary-jump p99/max were 1.244/2.396 mm.

After the final feature guard, the two portrait face components remained above the face gates at 0.943/0.948 curvature correlation and 0.931/0.968 RMS retention. The llama/group replay passed at 0.979 correlation, 0.910 RMS retention, and 0.937 span retention while keeping a 17.169 mm robust background span and 6.330 mm centred depth RMS. With the final selected-surface solve, its four measurable physical-relighting correlations are 0.967, 0.951, 0.988, and 0.844. In both runs, far context was held to 13.510 mm with zero cap violation, and every satisfiable attachment was held to the 0.800 mm neighbor-step limit. The final strict audit separately reported 63 portrait and 138 llama pixels where neighboring selected heights make those one-step constraints mutually incompatible; those cases are not mislabeled as strict cap successes. Both outputs remain single-component watertight manifold positive volumes with consistent winding and zero degenerates.

Aggregate follow-up evidence is in `docs/benchmark-evidence/relief_30mm_background_context_v2/`. Private source images, masks, renders, and meshes remain local and ignored.

## Privacy-safe varied-scene regression

A deterministic six-scene follow-up varied portrait placement, expression, yaw,
foreground-to-background contrast, and broad background geometry without using
any private image. It exposed a real v2 failure: the depth-connected head region
could let face-solver updates reach well outside the user's selection and flatten
localized background structure. The v3 path clips those updates to the subject
and scores the original background with identical final foreground geometry and
physical support constraints.

All six 30 mm v3 runs pass the face, source-context, local/global background,
physical-cap, attachment, provenance, and printability gates without fallback on
implementation commit `7ff4024`. The minimum face correlation is 0.954; minimum
final background and gradient correlations are 0.999981 and 0.999497. Every STL
is a single watertight, winding-consistent volume with no nonmanifold edges or
degenerate faces. The localized-deletion control still fails in five windows
despite global depth/gradient correlations of 0.983/0.825.

The runnable harness is `backend/benchmark/run_relief_scene_regression.py`, and
compact per-scene evidence is in
`docs/benchmark-evidence/relief_scene_regression_local_n6/`.

## Relief-height appearance sweep

The follow-up harness `backend/benchmark/run_relief_visual_sweep.py` expands the
privacy-safe regression to 20, 30, and 40 mm. Four mask topologies cover a
centered subject, an edge-clipped subject, a near-full-frame subject, and two
disconnected components with an internal hole. It verifies physical normals,
four deterministic light responses, candidate coverage, every face, selected
non-face, and background component, background depth, exact complete-shell STL
agreement, and pairwise cross-height face shape consistency.

The first exact two-face replay found a real aggregate-metric blind spot: the
larger face failed at `0.773947` worst-light correlation and `1.405562` maximum
lighting RMS retention even though the union passed. A nine-variant cached-depth
ablation isolated the face solver's low screening weight. Raising only that
face-aware term from `0.01` to `0.05` makes both exact face components pass at
`0.819375` and `0.915401` worst-light correlation. The exact portrait keeps
`0.999996` background depth correlation, the physical caps pass, and the mesh
remains one watertight manifold component with zero degenerates.

Independent selected-surface scoring then exposed six high-relief failures that
the face/background checks did not see. The worst near-full-frame 40 mm row had
only `0.493615` relighting correlation and `55.8160` degrees p95 normal error.
A bounded synthetic ablation first cleared screening `1.0` with 90%
recoverable-gradient retention, but an exact per-component llama/group replay
then exposed a remaining component at only `0.745409` relighting correlation
and `38.7870` degrees p95 normal error. Screening `1.5` still missed the `0.8`
correlation gate. The selected `2.0 / 0.9` setting raises that component to
`0.843994`, `0.957626` normal p05 cosine, and `16.7388` degrees p95 normal
error. Screening `3.0` is rejected before emission because its cardinal edge
ratio reaches `24.0606`, above the hard `24` limit.

The clean schema-v2 12-row matrix passes every gate on commit `9467322`. Minimum face
normal cosine and lighting correlation are `0.991248` and `0.971095`; minimum
aggregate selected-surface normal cosine and lighting correlation are `0.980935`
and `0.895048`; the worst individual selected-component correlation is
`0.847740`. Minimum background depth correlation is `0.999774`. Maximum
selected-solver cardinal p99/cardinal max/diagonal max ratios are `10.3559`,
`17.1018`, and `12.0929` against limits of `12`, `24`, and `24`. Across
20/30/40 mm pairs, minimum normalized face shape and gradient correlations are
`0.997841` and `0.993255`, with maximum normalized shape RMSE `0.023164`. Every
STL exactly reproduces its measured heightfield plus all top, bottom, and wall
facets, and passes topology checks. Full evidence is in
`docs/benchmark-evidence/relief_visual_sweep_local_n12/`.

The exact llama/group confirmation records full appearance, depth, cap, and
topology telemetry. All four measurable selected components pass, while one
isolated one-pixel selection fragment is explicitly unmeasurable. The principal
background component and every global/local background-depth gate pass. A
separate enclosed 144-pixel background region has no 64-sample interior after
the strict 1.5 mm boundary exclusion and is reported as unmeasured, not silently
certified. This exact-input caveat does not occur in the privacy-safe matrix,
whose disconnected subject and enclosed background hole are both large enough
to be measured component by component.

## Uniform face-height follow-up

The first exact 40 mm replay exposed two coupled failures. The `0.05` face
candidate missed its per-component detail gate, and the rejected fallback
surface reduced the weaker face's worst-light correlation to `0.233819`. The
same run's accepted background reference could not be restored because the
slope guard compared it against the already flattened rejected candidate.

A bounded `0.06` through `0.30` exact-input ablation found `0.25` to be the
first face term that passed. Applying that setting uniformly also improves the
weaker exact face from `0.845156` to `0.862279` at 20 mm and from `0.819365` to
`0.869303` at 30 mm. At 40 mm it reaches `0.896382` worst-light correlation,
`0.985779` mean-normal cosine, and `16.4955` degrees p95 normal error. Face
solver cardinal p99/max and diagonal max ratios are `6.4311` / `18.1081` /
`12.8252`, below `12` / `24` / `24`.

The accepted-reference fallback keeps exact background depth correlation above
`0.999997` with 100% coverage at all three heights. Every exact output is a
single watertight manifold volume with zero degenerates and exactly matches all
309,440 expected shell facets. The exact 20-to-40 gradient-consistency
diagnostic remains a disclosed near miss at `0.967754` versus `0.97`; 20-to-30
and 30-to-40 pass. At 40 mm the selection-wide candidate independently exceeds
its edge gate and fails closed, so the accepted face/background surface is not
replaced.

The clean privacy-safe 12-row matrix passes on implementation revision
`53d37b08672519dfb1cd9c1b6b27c673a416e0c7`. Its minimum face relighting
correlation improves from `0.971095` to `0.990004`, minimum background depth
correlation is `0.999973`, and every complete STL shell and printability check
passes. Evidence is in
`docs/benchmark-evidence/relief_visual_sweep_face025_n12/`.

## Named facial-part follow-up

Whole-face correlations can conceal a collapsed nose, flattened mouth, or one
damaged eye. This follow-up adopts the facial-part principle used by
[3DDFA-V3](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_3D_Face_Reconstruction_with_the_Geometric_Guidance_of_Facial_Part_CVPR_2024_paper.pdf),
but evaluates printable surface geometry rather than reproducing its training
loss. The relief metrics follow the multiscale gradient and curvature motivation
of [Digital Bas-Relief from 3D Scenes](https://gfx.cs.princeton.edu/pubs/Weyrich_2007_DBF/index.php)
and [Estimating Curvatures and Their Derivatives on Triangle
Meshes](https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/index.php).

Face refinement now persists fixed full-depth-grid masks for left/right eyes,
left/right eyebrows, mouth, nose, and the containing face. The mask loader
requires all six parts, verifies the recorded input-depth dimensions, and
replays the exact resize, horizontal flip, mesh resize, and crop transform used
by the STL exporter. A single detection is reused across relief heights, so the
metric never benefits from detector jitter. MediaPipe landmarks define only the
2D regions; its relative landmark depth is not interpreted as metric millimeters.
See the official [Face Landmarker Python
guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python).

Each part is compared after one shared robust face normalization. Independent
gates cover coverage, shape correlation, normalized RMSE, x/y gradient
correlation, slope and curvature q95 retention, and Wasserstein distances at 0
and 0.8 mm physical smoothing radii. Flat signals, missing pixels, incomplete
parts, stale transforms, invalid scales, and nonfinite derivatives all fail
closed. Synthetic mouth-flattening and nose-oversharpening controls must fail.

The initial fixed two-face replay exposed failures hidden by the whole-face
score. With the requested 0.8 mm RGB-derived emboss, the 20-to-40 mm comparison
failed mouth slope/curvature, nose gradient/slope, and right-eye gradient gates;
the nose gradient correlation was only `0.47284`. A bounded ablation showed the
late emboss duplicated and distorted geometry already reconstructed by the
screened face solver, while the attachment bridge remained beneficial. The
accepted high-relief path now suppresses only that emboss and retains the 0.8 mm
bridge. Lower-relief and rejected-screening paths are unchanged.

After the change, all six exact face/height comparisons pass. Worst named-part
shape and gradient correlations are `0.992224` and `0.958704`; worst normalized
RMSE is `0.016113`. Minimum per-face relighting correlation is `0.929934`,
`0.910359`, and `0.902234` at 20, 30, and 40 mm. Exact background depth
correlation remains above `0.999997`, and all three 309,440-facet STLs pass the
complete-shell and single-component printability checks.

The clean analytic 12-row matrix passes every gate on revision `f2c6cb8`. Its
worst named-part shape/gradient correlations are `0.999951`/`0.992325`, and no
part fails. Evidence is in
`docs/benchmark-evidence/relief_named_face_parts_n12/`; private artifacts remain
local and ignored.

## Canonical 3D face and selection-bound follow-up

Cross-height consistency still cannot prove that every relief is faithful to a
real 3D face: the same distortion could recur at all heights. Revision
`f6a473c3f9f25b0f6e58e36c2574663aefd687f6` therefore adds a checksum-pinned,
Apache-2.0 MediaPipe canonical face oracle and renders it at yaw `0`, `-30`, and
`+30` degrees at 20, 30, and 40 mm. Visible facial-part labels share the primary
z-buffer, and one face-wide affine calibration is followed by independent
millimeter errors for the eyes, eyebrows, nose, and mouth.

All 9 rows pass. Minimum face relighting correlation is `0.999385`, minimum
mean-normal cosine is `0.999954`, and the worst named-part p95 error is only
`0.110641` mm. Every STL is a single watertight manifold volume with zero
degenerates and an exact 232,320-facet shell. The companion 12-scene analytic
matrix also passes every face, selection, background, cap, shell, topology, and
negative-control gate; minimum background depth correlation is `1.0`.

The matrix exposed one asymmetric failure at yaw `-30`, 40 mm. The primary face
candidate failed only cardinal p99 (`13.2044` versus `12`), while the old
fallback badly damaged the nose and mouth. A `0.10` screening retry is now
eligible only for that exact singleton failure and must pass all unchanged
gates. The accepted row reaches `10.9944`. Face updates are also hard-bounded to
the selection: a measured `1.3163` mm pre-bound background correction becomes
exactly zero. The exact bounded surface is then re-audited for detail, span,
correction, and cardinal/diagonal edges; failed or unavailable telemetry routes
to the existing fallback.

Evidence and reproduction commands are in
`docs/benchmark-evidence/canonical_face_relief_yaw_height_n9/`. Historical
private portrait and llama evidence remains separate because the original
private selection masks are deliberately not tracked; no new exact-input claim
is inferred from cached depth arrays alone.

## Varied canonical identities and deeper background context

The next privacy-safe pass deforms the pinned canonical face into six
deterministic identity/expression profiles, including broad and projected
noses, high cheeks, asymmetry, and deep-set eyes. Exact topology, signed
triangle Jacobians, area and edge ratios, normal bending, connectedness, and
self-intersections are audited before rendering. The matrix covers yaw `-45`,
`0`, and `+45` degrees at 20, 30, and 40 mm.

The first expansion found a real 40 mm profile failure. The primary screened
solve lost detail and the legacy fallback produced 13-14 mm nose RMSE. A
bounded screening-`8.0` retry now becomes eligible only when its remaining
failures are raw edge maxima at source discontinuities. It remains provisional
until the exact blended surface passes the baseline-aware edge, detail,
correction, span, attachment, face-appearance, and STL gates. Projected-nose
profiles can cascade from the existing `0.10` cardinal-edge retry into this
guarded high-detail candidate only when the low-screening attempt itself loses
detail.

Metric schema v3 separates the unsmoothed raster derivative from the 0.8 mm
physical gradient. The strict `0.75` gate applies to the physical signal, while
the raw signal retains a `0.25` anti-collapse floor and all raw slope,
curvature, and Wasserstein gates remain active. Thin profile masks fall back to
their complete visible support when erosion would retain less than 35%; both
shape and affine millimeter metrics use the same rule and record the attempted
support ratio.

The named-part metric is schema v3 and its affine millimeter companion is
independently versioned as schema v2, so the support-semantics change cannot be
mistaken for historical v1 telemetry.

The previous varied-identity pass raised the selection-background default from
45% to 50%. On its two marginal 40 mm frontal scenes, recoverable context
coverage increased from 0.596/0.588 to 0.640/0.630. That historical setting is
superseded by the 65% production default measured in the follow-up below.
Background depth, RMS, span, and gradient retention remained effectively 1.0,
the far-background cap was 20.01 mm at 40 mm relief, and feasible attachment
steps remained at 0.8000002 mm or less.

All 54 clean rows pass on revision
`c0bc7ccb81ac724e6cdb3a99b7d406f9dbb2d36a`. Minimum named-part shape and
physical-gradient correlations are 0.9746 and 0.9697; worst p95 part error is
1.2892 mm; minimum face relighting correlation is 0.9961. Every output is one
watertight, winding-consistent positive volume with no nonmanifold edges or
degenerate faces, and every exact shell check passes. Evidence is in
`docs/benchmark-evidence/canonical_face_variants_height_v3_n54/`.

## Perspective provider and background-prominence follow-up

A new deterministic MakeHuman-derived CC0 fixture adds perspective projection,
smooth vertex normals, procedural skin/eye/brow color, exact z-buffer depth,
and exact source-derived facial-part supports. The orthographic renderer remains
byte-identical when perspective mode is not requested. On the first 30 mm
provider row, Depth Anything V2 remains the best face-depth model and narrowly
misses only one eye-gradient gate. MoGe-2 and Apache-2.0 DA3 Base retain more
background gradient structure but regress facial shape and reverse the broad
synthetic background ordering, so neither is promoted. Compact evidence is in
`docs/benchmark-evidence/makehuman_perspective_depth_provider_n1/`.

The background compositor now defaults to a `0.65` scene budget instead of
`0.50`. Positive-context normalization is anchored to the selected subject,
so raising the background cannot rescale the face. The first 54-row attempt
exposed a separate projected-nose blind spot: a whole-face correlation of
`0.822` cleared the old `0.80` internal solver floor while the nose exceeded
its millimeter gate. Raising only that internal floor to `0.85` selects the
already-audited high-detail retry for 10 of 54 rows.

The final 54-row matrix passes every face, named-part, background, physical-cap,
shell, and printability gate. Median background height rises from `15.0` to
`19.5` mm and median p05-p95 background span rises from `8.1601` to `11.4481`
mm. Minimum face relighting correlation is `0.9961052`, worst named-part p95
error is `1.2985969` mm, and all 54 STLs remain printable. The companion
six-scene matrix also passes. Compact evidence is in
`docs/benchmark-evidence/relief_background_prominence_v4/`.
Both matrices were rerun with clean implementation provenance at exact revision
`2dea653881460a6b4b39d67fe1c2d2ce8ae62f7d`.

The two exact 30 mm inputs that originally exposed the portrait and llama-group
failures were then replayed from clean head
`535daf3cacbf3d57c2474240322290d06f8296c4`. Relative to the tracked `0.45`
baseline, portrait background centered RMS and p02-p98 span rise from
`4.6482`/`13.3445` to `6.8165`/`19.5000` mm; llama-group rises from
`6.5078`/`21.6562` to `7.8375`/`22.0282` mm. Portrait face-component metrics
improve, all four llama selected-component correlations and RMS retentions
improve, and both STLs remain printable. Compact aggregate-only evidence is in
`docs/benchmark-evidence/relief_background_prominence_exact_v5/`; no private
image, mask, depth, response, preview, or mesh is tracked.

## Framing, attachment, and selected-surface follow-up

The next ten-scene framing expansion added small faces, near-full-frame faces,
left/right clipping, and four new background structures. It exposed an
independent border failure: `base_border_px` flattened selected face pixels that
touched the image frame, creating 7-8.5 mm one-pixel cliffs. Positive-context
selections now preserve only the detected face at the frame border; non-face and
ratio-zero borders retain their historical flattening behavior. An exact
ratio-zero replay keeps identical emitted-surface and loaded-triangle-vector
hashes.

Concave selection corners could also leave a background pixel adjacent to two
incompatible selected heights. A one-pixel localized projection now adjusts only
the conflicting selected boundary and one inward feather pixel, then reapplies
the unchanged physical cap. The clean ten-scene matrix passes on revision
`1a3cfc9efd8649e460c4aeba1a93f46fa2214297`: minimum face correlation is
`0.8691`, the largest actual boundary max is `1.5167` mm, every background
structure is retained, and every STL is one printable watertight volume.

The exact llama replay then found a different failure hidden by background-only
metrics. Its selected-object gradient candidate preserved component shape but
was rejected solely because an unchanged source discontinuity exceeded the raw
cardinal maximum. The selected-object path now uses the same fail-closed
baseline-aware post-blend audit as the guarded face path. Only edges touched by
the candidate are exempted from unchanged source jumps; exact finite coverage,
changed-edge excess, detail, height span, correction span, cardinal/diagonal
edges, background, cap, and topology gates remain active. Signed reversals also
fail when they form four combined edges or one sparse reversal exceeds four
physical steps.

On the paired cached-depth llama input, the old fallback's selected-surface
lighting correlation is `0.1201` with `80.79` degrees p95 normal error. The
audited candidate reaches `0.9507` and `10.73` degrees, with a weakest meaningful
component of `0.8848`. Changed-edge p99/max ratios are `1.553`/`4.906` against
`12`/`24`; background RMS/span are `7.8597`/`22.0333` mm and depth/gradient
correlations are `0.999921`/`0.988976`. Its two cardinal and one diagonal
reversals remain bounded at `1.951`/`1.044` physical steps. The exact portrait
face metrics are unchanged. Both outputs remain printable, while incompatible attachment
constraints remain explicitly counted rather than called strict successes.

The clean 12-row 20/30/40 mm sweep also passes with minimum face and selected
lighting correlations `0.9896` and `0.9228`. Evidence, checksums, exact aggregate
telemetry, and reproduction commands are in
`docs/benchmark-evidence/relief_framing_boundary_selection_v6/`.

## Cropped perspective identities at 30 and 40 mm

The next privacy-safe pass uses the pinned MakeHuman CC0 head fixture instead
of private photos. Three perspective identities cover different skin tones, a
smile, a neutral face, and an asymmetric expression. The latter two are cropped
far enough that the head itself touches the left or right frame while all exact
eye, eyebrow, nose, and mouth supports remain visible. Structured background
depth is present in every scene.

This matrix exposed a new asymmetric 40 mm failure. The screened candidate
retained facial detail but exceeded only cardinal edge max, then the legacy
fallback damaged every named part. A bounded retry now projects only significant
edge-direction reversals back toward the accepted source. The projected surface
is used only when all reversals reach zero and the existing post-blend audit
passes; failed or unavailable projection telemetry remains fail-closed.

All six final rows pass at revision
`6ddc5c45ff2753a1b90239f65248f3ca5e39d763`. The formerly failing row improves
from `64.2410` to `5.4607` degrees p95 normal error and from `0.557298` to
`0.989728` minimum relighting correlation. Its 150 measured reversals fall to
zero after changing 248 pixels. Background depth/gradient correlation remains
`0.999993`/`0.999500`, with `15.8095` mm p02-p98 span at 40 mm. Every row keeps
one printable watertight component and an exact serialized STL shell.

Research notes, CC0 previews, before/after aggregate telemetry, checksums, and
reproduction commands are in
`docs/benchmark-evidence/makehuman_face_relief_reversal_projection_n6/`.

## Predicted depth through the physical STL path

The next bounded lane replaces exact depth with real monocular predictions on
the centered smile and the hardest right-cropped asymmetric scene. Each model
runs once per scene, then its unchanged cached depth is emitted at 30 and 40 mm.
The exact silhouette is a declared selection control; exact depth and named
facial-part masks are metric-only and cannot affect challenger normalization,
feature weighting, or STL generation.

Depth Anything V2 Large keeps every generated mesh printable and passes its own
context, cap, attachment, shell, and 30-to-40 consistency checks. On the
centered face it also passes physical normal/relighting and scale-free
named-part shape. The remaining eye/eyebrow millimeter misses are upstream model
error. The cropped face retains scale-free part shape but misses the stricter
physical appearance and millimeter gates.

This run also separates background suppression from background estimation
error. At 30/40 mm, the centered Depth Anything meshes retain
`18.5118`/`24.6825` mm of background span, so the compositor is not flattening
the scene. Against the exact scene, however, broad background correlation is
`-0.1226` and gradient correlation is `0.0506` at 30 mm. Increasing background
gain would amplify incorrect ordering rather than recover depth.

The current Apache-2.0 DA3Mono-Large checkpoint was tested from exact official
source/model revisions. It fit easily on the local 3080 Ti (`1.5414` GB peak)
but regressed facial shape and did not repair broad background ordering, so it
was not expanded past the centered control. Implementation, measured telemetry,
and reproduction commands are in
`docs/benchmark-evidence/makehuman_predicted_relief_providers_n2/`.
Both provider runs have clean implementation provenance at exact revision
`d0528a7df0036750b687d9b4cf388cbe9b3d5c2b`.

## Metric-depth follow-up

The official Apache-2.0 DA3Metric-Large checkpoint was then added with exact
source/model pins and run only on the centered 30/40 mm control. Absolute and
incremental VRAM are tracked separately on the selected CUDA device, and raw
face/background ordering is recorded as diagnostic-only telemetry. The oracle
still cannot affect normalization, polarity, clipping, feature weighting, or
STL generation.

DA3Metric does not qualify for expansion. At 30 mm it reaches only `0.900212`
normal mean cosine, `48.4292` degrees p95 normal error, and `0.725143` minimum
relighting correlation; every named part fails shape and millimeter gates. Its
background span is already `18.6208` mm, yet broad correlation is `-0.210890`.
The 40 mm row worsens to `53.6326` degrees while background correlation remains
negative at `-0.216549`. All emitted meshes themselves remain printable and
pass context, cap, attachment, shell, and cross-height checks.

The new ordering audit explains why another gain adjustment is inappropriate.
DA2 follows its near-high contract on the face (`-0.982817` against far-high
depth) but reverses that sign within the background (`+0.504785`). A polarity
flip was rejected because the pinned DA3 source defines positive direct depth
and choosing the flip from oracle scores would leak evaluation into emission.
Clean evidence at implementation revision `5738972` is in
`docs/benchmark-evidence/makehuman_da3metric_relief_centered_n1/`.

## Learned background correction and modern geometry cues

A bounded `106,225`-parameter residual U-Net was trained on 16 deterministic
CC0/procedural scenes with four validation scenes and two sealed final scenes.
The selected face is masked from the network input and restored byte-for-byte;
exact depth is training/evaluation-only. The run is a hold. It improves sealed
background gradient correlation from `0.5346/0.5427` to `0.8318/0.8767`, but
broad correlation remains wrong at `-0.4449/-0.4719`. Every target was reachable
and incremental training VRAM was only `0.4615 GB`, so more gain or capacity is
not justified by this failure.

The May 2026 HyDen-MoGeV2 normal model was researched next. Its official model
is manually gated under the FAIR noncommercial research license and the local
token cannot access it, so no weights were downloaded. The official Apache-2.0
Lotus-2 Normal and Depth public demos were each smoked on one CC0 row. Both
preserved the person while flattening the analytic background, so neither was
integrated. Pins, output checksums, measured holds, and reproduction are in
`docs/benchmark-evidence/background_residual_da2_n20_s1/`.

## Stronger face-protected photo relief

The final background pass no longer decides which scenery deserves detail from
depth percentiles when a valid semantic subject mask exists. It activates the
source-defined background while preserving a physical 5 mm quiet halo around
the subject. A bounded p96 normalization and `0.75` detail-contrast curve make
weak photographic structure printable without exceeding the requested cap.
Missing or empty protection masks retain the legacy `0.12 mm` limit.

The clean 12-row oracle sweep covers centered, left-frame, and right-frame CC0
faces at `0`, `0.12`, `0.30`, and `0.60 mm`. At `0.60 mm`, intended-background
correlation is `0.4101-0.5052`, RMS detail is `0.0740-0.1956 mm`, and p95 detail
is `0.1727-0.4483 mm`. Source-aligned capture is `0.1424-0.3703`, above the
frozen `0.12` coverage floor. Worst face-interior and attachment-boundary
movement are `0.0043` and `0.0525 mm`. Every STL passes cap, attachment,
topology, and complete-shell checks.

A production Depth Anything V2 pair then reuses one exact cached prediction for
the `0` and `0.60 mm` emissions. It reaches `0.5871` intended-background
correlation and `0.1646/0.2532 mm` RMS/p95 detail, while face-interior and
attachment-boundary maxima remain `0.00175/0.01701 mm`. Both STLs are printable
complete shells. Non-finite, negative, or greater-than-`0.60 mm` API values are
rejected before inference; internal calls also fail closed or clamp to the hard
cap. The tracked backend default is `0.60 mm`; frontend controls remain outside
this evidence claim because the current UI edits are part of unrelated local
work.

Exact summaries, privacy-safe sources, hashes, and fail-closed reproduction are
in `docs/benchmark-evidence/background_photo_detail_sweep_30mm_n3_v5/` and
`docs/benchmark-evidence/background_photo_detail_da2_s40_n1_v3/`.

## Pinned retained-artifact background-detail confirmation

The current `0.60 mm` photo-detail algorithm was replayed at 30 mm on locally
retained motivating-scene artifacts. It keeps the historical 512-grid, sigma,
footprint, gamma, and detail-boost controls while explicitly overriding the
selection-background ratio (`0.45 -> 0.65`) and photo detail
(`0.12 -> {0.00, 0.60} mm`). This is a pinned retained-artifact comparison, not
an authenticated reproduction of the original requests, and it does not cover
every current endpoint default. The new runner keeps the config, every private input, and all
generated geometry under ignored output and publishes a scalar allowlist only.
It locally checksum-pins every retained input, cross-checks selection
fingerprints, selection-job linkage, recorded mask identity, stable historical
controls, and the expected Depth Anything V2 identity, and never publishes
those checksums or identifiers. The original metadata has no request-time depth
or mask digest, so the documentation does not claim that unavailable provenance.
It also replays the exporter's recorded resize, horizontal flip,
mesh resample, and crop when measuring source alignment; direct photo-to-final-
grid resizing is no longer accepted for this evidence lane.

Intended-background correlation is `0.6141` on Scene 01 and `0.6296` on Scene
02. RMS/p95 added relief is `0.0512/0.1184 mm` and
`0.0784/0.1843 mm`, with source-aligned capture of `0.5430` and `0.6076`.
Scene 01 face movement is effectively zero (`0.000002 mm` maximum), and the
selected-subject attachment boundary changes by at most `0.0146/0.0129 mm`.
The original broad background remains intact: correlation is
`0.999994/0.999921`, RMS retention is `1.000051/1.000088`, and coverage is
complete.

All four baseline/candidate STLs are single-component watertight manifold
volumes with consistent winding, zero degenerates, and exact complete-shell
facet agreement. Far-background cap violation is zero and every satisfiable
attachment jump is at most `0.800001 mm`. Strict cap status remains explicitly
false for `13/41` mutually incompatible one-pixel constraints; this is not
misreported as strict success. Aggregate evidence and the generic reproduction
contract are in
`docs/benchmark-evidence/private_background_photo_detail_30mm_v1/`.

## Isotropic near-subject background recovery

The initial face-protection halo kept the subject stable, but its square
maximum-filter support plus Gaussian recovery suppressed valid scenery too far
from the silhouette. A stricter privacy-safe radial test measured only `0.5115`
relative gain at `5-8 mm` and `0.3157` in the weakest local window. A first
Euclidean replacement restored that detail but was rejected after an
independent CC0 review found `0.1306-0.1875 mm` boundary movement.

The selected gate therefore preserves an exact `2 mm` inner guard and then
uses an isotropic Euclidean smoothstep to full gain at `5 mm`. On the clean
analytic 30 mm regression, calibrated detail retention is `0.9215` at
`5-8 mm` and `0.9862` at `8-12 mm`; the weakest local window is `0.6484`.
There are zero new radial gradient reversals, full finite coverage, and all
low-pass moat/overshoot limits pass. The metric follows halo research that
treats edge artifacts as overshoot or gradient reversal and treats missing
source-aligned structure outside the declared guard as overprotection:
[RWDR (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html)
and [Mixed-Domain Edge-Aware Image Manipulation](https://cg.cs.tsinghua.edu.cn/papers/TIP_2013_Edge-Aware.pdf).

The three-face CC0 matrix keeps worst subject-boundary movement to
`0.013651 mm` and worst face-interior movement to `0.001457 mm`; every emitted
STL passes cap, feasible attachment, topology, and exact-shell checks. The
aggregate-only retained-photo replay independently passes all four emissions.
Only its Scene 01 has an independent retained face-region mask; its maximum
face movement is `0.000011 mm`, while Scene 02 is not counted as a private
face-regression measurement.
It improves Scene 01 source-aligned capture from `0.5430` to `0.5684` and
Scene 02 from `0.6076` to `0.6174`, while reducing attachment-boundary movement
from `0.0146/0.0129 mm` to `0.000083/0.000055 mm`. Broad background correlation
remains `0.999994/0.999921` with complete coverage.

Evidence and reproduction contracts are in
`docs/benchmark-evidence/background_halo_continuity_30mm_v1/`,
`docs/benchmark-evidence/background_photo_detail_cc0_euclidean_guard_n3/`, and
`docs/benchmark-evidence/private_background_photo_detail_30mm_v2_euclidean_guard/`.

## Exact live API confirmation

The first exact-input HTTP replay found that the validated direct-core
background algorithm was not active in selection mode. Full-scene depth was
inferred correctly, but the mesh call received the neutral subject-only image;
photo relief consequently failed closed with `reason=no_photo_detail`. The API
now uses the original full-scene source for photographic relief while retaining
the neutral image for face refinement. Endpoint jobs also retain emitted and
reference heightfields under ignored output for exact shell verification.

The clean 30 mm replay at revision `7c9c7fe` omits both background form fields
and observes the current `0.60 mm` / `0.65` defaults. Its refined depth,
composed context, source, selection, face region, and feature-weight inputs all
match the pinned direct-core artifacts, as does the retained pre-high-relief
reference surface. The private request record binds the response hash and exact
posted-field set, while the response binds the loaded clean revision. Photo
relief is enabled with the 2 mm Euclidean guard and full recovery at 5 mm.

Face normal mean cosine is `0.9911`, p95 normal error is `7.9260` degrees, and
minimum relighting correlation is `0.9167`; all frozen appearance gates pass.
Background depth/gradient correlation is `0.999994/0.999260` with `1.000051`
RMS retention and complete coverage. The STL is one watertight manifold volume
with zero degenerates. Its complete serialized shell matches all `309,440`
expected facets with zero coordinate or RMS error. Aggregate telemetry and the
privacy-safe reproduction contract are in
`docs/benchmark-evidence/private_live_api_background_30mm_v1/`.

## Standard-grid face detection and varied CC0 live replay

The next privacy-safe live replay exposed a scale bug before relief shaping:
the centered 256 px fixture produced a valid 70 px-wide MediaPipe face with 478
landmarks, but the fixed 96 px minimum discarded it. The detector threshold now
adapts down to 25% of the short edge for small images, with a 48 px floor, while
preserving the historical 96 px limit from a 384 px short edge upward. The
measured effective threshold is 64 px for the standard 256-grid workflow.

The fallback path is also modernized from missing-package-dependent Haar to
official OpenCV YuNet before Haar. The OpenCV 4.x-compatible `2023mar` model is
pinned by repository revision and SHA256, downloaded through a size-bounded
atomic cache, and accepted only at confidence 0.90 with valid five-point face
geometry. A real-model smoke detects the centered CC0 face at 0.9266 while the
normal production path retains MediaPipe's 478-landmark masks.

The clean live HTTP matrix at revision `15a3a25` covers centered, left-framed,
and right-framed CC0 heads at 30 mm. All three faces are detected and refined.
Worst face normal mean cosine is `0.9973`, worst p95 normal error is `4.4443`
degrees, and worst minimum relighting correlation is `0.9904`. The default
0.60 mm background detail adds `0.163-0.240 mm` RMS and `0.236-0.440 mm` p95
source-aligned relief; intended-background source correlation is
`0.5528-0.7307`. Face-interior p99 movement remains at most `0.000110 mm`.

Background depth correlation remains at least `0.999988`, gradient correlation
at least `0.998704`, and coverage is complete. Every emission has zero
far-background cap violation, all feasible attachment jumps are at most
`0.80000019 mm`, and the 20-27 incompatible one-pixel constraints remain
explicit. All six baseline/candidate STLs are single-component watertight
manifold volumes with zero degenerates. Exact shell verification matches
`232,320/232,320` facets with zero heightfield sample error.

The pinned YuNet model detects the centered positive control at confidence
`0.9266`. On clean control revision `1a2bfe7`, MediaPipe, YuNet, the official
OpenCV 4.10 Haar cascade, and the complete production chain all produce zero
detections and no errors on eight 256 px negative controls covering
subject-removed analytic backgrounds, procedural 3D objects, a high-contrast
checkerboard, and a blank image. Every input is RGB-checksum-bound by the
checked-in producer. Environment-supplied models are checksum-verified, an
empty YuNet result continues to Haar, and configured filesystem paths are
excluded from detector errors.

Aggregate telemetry and reproduction details are in
`docs/benchmark-evidence/cc0_live_api_face_background_30mm_n3/`.

## Validation

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend
npm run typecheck
npm run lint
npm run test:ui
```

Measured state on 2026-07-15:

- Backend: 578 passed, 72 subtests passed (2 existing warnings).
- Frontend typecheck: passed.
- Frontend lint: passed with zero warnings.
- Playwright: 9 passed, 1 intentionally skipped, including the delayed-compose replacement-photo race.
- Exact live portrait and llama/group requests: HTTP 200 with all STL hard checks passing.
