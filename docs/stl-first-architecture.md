# STL-First Architecture

The project goal is printable geometry. Image completion, depth estimation, novel-view synthesis, Gaussian splatting, and NeRF-style reconstruction are useful only when they improve the final STL. The benchmark should therefore rank methods by mesh and print-readiness outcomes instead of visual preview quality alone.

## Baseline Modes

The current completion benchmark remains the baseline for regression tracking:

- **Fast 2.5D depth relief:** estimate or repair a depth map from one image, convert it into a relief-style mesh, then export STL.
- **Masked input baseline:** preserve the visible object region and measure how much geometry the downstream depth-to-STL path can recover without hallucinating hidden structure.
- **Mirror completion baseline:** mirror the visible side across a symmetry plane, then feed the completed image/depth into the existing STL path.
- **Biharmonic/inpainting baseline:** fill masked regions with classical or lightweight completion before depth estimation and STL export.
- **Source mesh oracle:** when rendered 3D assets are available, use the original mesh as an upper-bound reference for STL metrics.

These modes are intentionally simple. They provide stable comparisons while newer image-to-3D methods are integrated.

## STL-First Target Modes

### 1. Fast Depth-Relief STL

This mode should stay available because it is predictable, quick, and useful for plaques, bas-relief prints, lithophanes, and shallow printable objects. Improvements should focus on clean silhouettes, thickness control, watertight backing geometry, and reliable scaling.

### 2. Single-Image Full Mesh Generation

Modern image-to-3D models should be evaluated as direct STL candidates, not just as prettier previews. Candidate backends can include Hunyuan3D, TripoSR/TripoSG-style pipelines, SV3D-style multiview generation plus reconstruction, or newer open image-to-mesh systems when they can run reproducibly.

Each backend should receive the same image crop, object mask, and scale target where possible. The output should be normalized, repaired if needed, exported to STL, and scored with the same mesh-quality checks as the depth-to-STL baseline.

### 3. Video or Multiview Reconstruction

When the user provides video or multiple photos, the pipeline should select useful frames, estimate camera motion or viewpoints, segment the object, reconstruct a mesh, and export STL. This path should compare selected-frame reconstruction against using every frame so the system can prefer quality over brute force.

Gaussian splatting and NeRF are optional reconstruction backends in this lane. They should be used only if they produce better final meshes or better STL repair inputs. Splat visual fidelity by itself is not a success metric.

## Planner-to-Runner Contract

The app-facing planner should stay lightweight and deterministic. It should expose model ids and route plans, while heavy runners attach behind those ids and write benchmark-compatible artifacts. A planner response is not a successful reconstruction by itself; it is a promise that the backend knows which stages, inputs, and promotion gates the runner must satisfy.

Required planner stages:

- **Photo full mesh:** object selection, image-to-mesh, STL postprocess.
- **Selected-frame video:** frame selection, object selection, camera pose, video reconstruction, STL postprocess.
- **Whole-video reconstruction:** frame sampling, camera pose, video reconstruction, STL postprocess.

Runner inputs should preserve both geometry context and object focus. Selected-frame video runs must keep full uncropped frames for keypoint/camera matching, plus object masks or crops for the reconstruction target. Single-image mesh runs should receive the same visible image, object mask, and target STL scale used by the benchmark.

Runner outputs must be machine-checkable before the UI marks a job as ready:

- `output_model.stl` or an equivalent STL path.
- `diagnostics.json` with watertightness, manifoldness, positive volume, component count, bbox extents/aspect, face count, and repair actions.
- Optional preview images or meshes for inspection, treated as secondary evidence.
- A promotion decision that distinguishes planner-only readiness from STL-ready reconstruction.

Local planner contract smoke on July 10, 2026: the draft planner service compiled, `/health` and `/models` returned `200`, photo `/plan` returned `object-selection -> image-to-mesh -> stl-postprocess`, video `/plan` returned `frame-selection -> object-selection -> camera-pose -> video-reconstruction -> stl-postprocess`, and an invalid camera-pose id returned `400` with the valid model ids. This validates the planning contract only; the next implementation step is to attach a runner that emits the STL and diagnostics artifacts above.

Live depth-relief runner contract on July 10, 2026: `/process_image` now writes `output_model.stl`, `diagnostics.json`, and `metadata.json` in each job directory. The response returns `stl_url`, `diagnostics_url`, and inline `stl_diagnostics` using the same watertightness, manifoldness, winding, volume, connected-component, bbox, and face-density fields used by the benchmark promotion gates. This turns the fast 2.5D path into a machine-checkable STL runner while full-mesh and video runners are attached behind the planner ids.

Live single-image mesh runner contract on July 10, 2026: the companion service now exposes `GET /providers/image-to-mesh` and `POST /run/image-to-mesh`. The preflight endpoint checks server-side provider repo configuration, Python availability, and lightweight entrypoint/import readiness without loading models or downloading checkpoints. The runner accepts a photo and a provider id from the model catalog, invokes the existing `backend.benchmark.run_image_to_mesh_provider` adapter, normalizes/repairs/scales the mesh, and emits `output_model.stl`, `diagnostics.json`, and `metadata.json` under the same artifact contract as the relief runner. Provider repo paths, provider Python, and runner timeout are server-side environment configuration only. The response distinguishes `printable` from `stl-emitted` by running the hard STL diagnostic gates, and the web app's `Full Mesh STL` route treats failed gates as a blocked emitted artifact rather than a printable model. Provider dependencies remain external, so missing TripoSR/TripoSG/Hunyuan/SPAR3D/SF3D installs fail as runner setup errors instead of pretending a planner response is a finished STL.

Live multiview mesh runner contract on July 10, 2026: the companion service now exposes `GET /providers/multiview-to-mesh` and `POST /run/multiview-to-mesh`. The live runner accepts a primary image, uploaded view/mask files, and a `multiview_input.json`-compatible bundle, rewrites bundle paths to server-owned job artifacts, invokes `run_image_to_mesh_provider --provider multiview-visual-hull`, and emits `output_model.stl`, `diagnostics.json`, `metadata.json`, and the normalized `multiview_input.json`. It rejects bundle paths that were not uploaded or already inside the service output tree, and it uses the same hard STL diagnostic gates as the single-image runner before reporting `printable`.

Single-image mesh calibration update on July 10, 2026: `backend.benchmark.run_stl_first_smoke` exposes `--mesh-target-bbox-source none|source|mirror|inferred|reference` and `--direct-mesh-reference-method`. `inferred` is the deployable alias: it derives target extents from the sibling depth-relief/mirror STL and produces names such as `_stl_inferred_bbox_direct_mesh`. `source` reads hidden ground-truth mesh extents, so it is an oracle-only diagnostic and is never promotion eligible. `mirror` remains available for historical configs, while new generated experiments should use `inferred` to state the provenance clearly.

**Measured G4 bbox tuning.** Run `g4_stl_first_triposg_bbox_tuning_s40_n10` completed `70/70` rows (`7` methods x `10` samples) on a Colab G4 RTX PRO 6000 Blackwell with 95 GB VRAM. The TripoSG biharmonic-prefill mesh calibrated to depth-relief bbox extents scored `0.9280740`, with median Chamfer `0.1417979`, H95 `0.3871739`, bbox aspect `1.9614717`, and scale-free complexity `9.3621272`; `mirror` scored `0.1991798`. All candidate printability medians passed, but the decision remained `hold` because `5/10` per-sample complexity rows exceeded the maximum of `10`. The source oracle scored `2.9898396` and remains explicitly diagnostic/nondeployable.

**Measured Pixal3D G4 comparison.** Run `g4_stl_first_pixal3d_vs_triposg_s40_n5_r5` completed the same five held-out samples for pinned Pixal3D, pinned TripoSG, the depth-relief baselines, and the source oracle. Pixal3D repair reached `5/5` watertight, manifold, positive-volume, single-component STLs and led TripoSG on the composite score (`1.2783` versus `0.9521`), but the decision remained `hold`. Pixal3D's median Chamfer/H95 were `0.1822/0.4822` versus TripoSG's `0.1264/0.3492`; paired win rate versus TripoSG was `0.60`, with worst-sample surface ratios `1.5485x/2.4093x`. The contact sheet confirms the cause: raw Pixal meshes contain roughly one million faces and many fragments, while printable fallback collapses them to only `40-128` faces. The next Pixal optimization is shape-preserving component filtering and closure, not weaker selection gates.

**Measured TRELLIS.2 challenger.** The benchmark has a separate `/content/trellis2-venv` setup path for pinned TRELLIS.2 source and model revisions. It substitutes public exact-revision DINOv3 and BiRefNet snapshots for gated references, rewrites the pipeline manifest to local offline paths, uses the official `(x,y,z) -> (x,z,-y)` export convention, and validates pinned `xformers` before inference. Run `g4_stl_first_trellis2_vs_triposg_s40_n5_r3` completed all five TRELLIS raw/repaired rows and all five TripoSG control rows on G4. Repaired TRELLIS produced `5/5` watertight, manifold, positive-volume, single-component STLs and led the composite score (`1.3779` versus `0.9521`), but remained `hold`: worst paired Chamfer and H95 ratios were `1.3447x` and `1.2580x`, above the `1.10x` ceiling. One deterministic seed retry recovered an empty sparse-structure sample; bounded volume-first component repair recovered a `2,768,786`-face, `133,533`-component desk mesh. The final machine-readable record is `docs/benchmark-evidence/trellis2-g4-s40-n1-r3-n5-r1-r3.json`.

**Guarded TRELLIS.2 repair follow-up.** Run
`g4_stl_first_trellis2_component_close_guard_s40_n1` tested one exact cached
raw mesh before authorizing the prepared five-row expansion. Area filtering
and bounded hole closure preserved most of the surface, but optimal
simplification reached its `10,704`-face target only after topology and
boundary relaxation. The final no-hull mesh had `2,943` self-intersections
and was rejected, so the candidate emitted `0/1` STLs and the five-row run was
not launched. Cached voxel prefill and shell variants also failed either
topology, complexity, or repair-drift guards. This closes the current
post-hoc TRELLIS repair lane; compact evidence is under
`docs/benchmark-evidence/g4_stl_first_trellis2_component_close_guard_s40_n1`.

**Promoted inferred-bbox TripoSG run.** `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_inferred_bbox_generated.json` applies `--mesh-max-normalized-face-density-log1p 9.95`, deriving each mesh face cap from normalized bbox volume and the same scale-free metric used by the promotion gate. Run `g4_stl_first_triposg_inferred_adaptive_s40_n10` completed `70/70` rows and promoted the biharmonic-prefill candidate over `mirror`: score `1.1038838` versus `0.1991798`, no failed checks, and all ten candidate complexities at or below `9.9499179`. Compact evidence is committed under `docs/benchmark-evidence/g4_stl_first_triposg_inferred_adaptive_s40_n10`.

**Pinned Hunyuan3D-2mv lane.** The `hunyuan3d-2mv` provider uses official `Tencent-Hunyuan/Hunyuan3D-2` source commit `f8db63096c8282cb27354314d896feba5ba6ff8a` and `tencent/Hunyuan3D-2mv` model revision `3a761b539b29fe4ff64714813aa9560fd66f5de0`, filtered to the standard `hunyuan3d-dit-v2-mv` FP16 safetensors checkpoint. It passes a keyed PIL dictionary in the model's required `front, left, back, right` slot order, maps benchmark camera azimuths to those slots with a bounded angular tolerance, and converts each rendered silhouette into an RGBA alpha cutout. The adapter never reads `source_mesh` or `asset_path`; those fields remain diagnostic-only. Every selected image, mask, camera, model/source revision, generation setting, and wrapper source hash participates in the raw-mesh cache key, so raw and identically repaired STL rows share one inference without cross-sample cache leakage. CUDA-synchronized generation runtime plus PyTorch peak allocated and reserved memory are carried into `provider_metrics.json` and the cache metadata; the generic peak field is explicitly labeled `torch_peak_reserved`, not presented as an NVML device-wide measurement.

Fresh Colab packages can include `--include-hunyuan3d-2mv-setup`. The generated launcher creates `/content/hunyuan3d-2mv-venv`, preserves Colab's matched Torch/Torchvision CUDA pair, installs only the shape dependencies, checks that the Torch wheel contains the live GPU architecture, executes a CUDA matrix-multiply probe, preflights the official pipeline import, archives the structured provider preflight, and downloads the exact filtered model snapshot before evaluation. The renderer supports four-view `cardinal` and eight-view `octants` layouts with stable appearance across views. In the eight-view layout Hunyuan consumes exact cardinal inputs while the harness renders each raw/repaired result at the four intermediate angles and records held-out silhouette IoU. Anchor-only manifests emit one canonical row per asset, keeping extra renders out of the sample count and ground-truth camera transform; `--asset-manifest --asset-manifest-start-index 40` selects the exact ten assets used by the promoted TripoSG slice. The provider is governed by the Tencent Hunyuan 3D 2.0 Community License, which is source-available rather than OSI-open-source and includes territory, scale, distribution, and acceptable-use restrictions; deployment must review that license independently of benchmark quality.

**Measured Hunyuan3D-2mv G4 smoke.** Run `g4_stl_first_hunyuan3d_2mv_octants8_s40_n1_s5_r256` completed all six method rows on the G4 Blackwell runtime with `run_status=0`, a runnable official-provider preflight, and no setup errors. The repaired multiview mesh scored `0.6308643`, with Chamfer `0.1641288`, H95 `0.3861005`, four-view held-out silhouette IoU `0.7914006`, one connected body, `4,372` faces, and scale-free complexity `9.0454495`. Peak reserved CUDA memory was `5.7266` GiB. The raw mesh had ten components, complexity `10.2349665`, Chamfer `0.3767970`, and held-out IoU `0.0101568`; printable repair therefore changes the deployment outcome materially and raw/repaired rows must remain separate. The result archive was recovered locally at `2,497,447` bytes with SHA256 `0385ffe62c009eb2170032f048afd74b2e8769301398921c4026aeb82683a423`, and local re-ingest reproduced `promote-challenger`. This one-row decision is smoke-only; compact evidence is committed under `docs/benchmark-evidence/g4_stl_first_hunyuan3d_2mv_octants8_s40_n1_s5_r256`, and the completed paired n10 result is documented below.

**Measured Hunyuan3D-2mv versus TripoSG n10.** Run `g4_stl_first_hunyuan3d_2mv_vs_triposg_octants8_s40_n10` completed `70/70` rows. Repaired Hunyuan scored `1.5784637` versus TripoSG `0.2849006`, won `10/10` paired objective comparisons, and passed every per-sample STL gate. Its median Chamfer/H95 were `0.1741899/0.3504748`, held-out IoU was `0.6551098`, and scale-free complexity was `6.5212088`. The architecture ingest recommends the Hunyuan multiview mesh, but provider promotion remains `hold`: worst paired Chamfer and H95 ratios were `1.6689293x` and `1.1249693x`, above the `1.10x` ceilings. Raw Hunyuan is far more faithful (`0.0488328/0.1558482` median Chamfer/H95 and `0.8998789` held-out IoU) but has a median `876,135` faces, `0/10` manifold outputs, and only `3/10` single-body outputs. Repair reaches `10/10` printable STLs but can collapse to `144` faces. This makes shape-preserving repair the next target; do not add more diffusion steps or weaken selector gates. Compact evidence is committed under `docs/benchmark-evidence/g4_stl_first_hunyuan3d_2mv_vs_triposg_octants8_s40_n10`.

Deployable multiview baseline on July 10, 2026: the benchmark provider wrapper now includes `multiview-visual-hull`, a deterministic silhouette-carving backend that consumes the existing `multiview_input.json` bundle, projects a voxel grid through each view's orthographic camera, exports a mesh, and then uses the same repair/scale/STL path as direct image-to-mesh providers. It is not a learned reconstruction model, but it gives the `multiview-mesh` lane a real STL-producing baseline that can be compared against depth-relief, source oracle, and TripoSR/TripoSG/Hunyuan-style single-image meshes without requiring external COLMAP/VGGT/SV3D setup. The calibrated 10-sample procedural slice at visual-hull resolution 32 promoted visual hull over the depth-relief `mirror` baseline under STL-quality scoring after source-mesh bundle oracles were correctly kept in the diagnostic-only lane; denser resolution-48/40/36 probes ranked visual hull highly but failed at least one strict per-sample mesh-complexity gate.

## Evaluation Metrics

Metric schema v2 separates historical signed-volume telemetry from the
canonical repair-fill gate. Closed, consistently wound, single-component raw
meshes use signed-volume magnitude. Open, fragmented, or topology-unknown raw
meshes use a bounded component-centered unsigned-tetrahedron surface proxy;
the matching repaired proxy is computed only for that row. Reliability
status, reason, topology coverage, self-intersection coverage, proxy support,
and proxy-use rate are retained in evidence. Legacy `repair_volume_*` fields
remain unchanged and historical archives derive canonical values from them.

All standard selection paths now apply explicit worst-sample Chamfer and H95
ratios of `1.10x` against the current method. The migration replay is under
`docs/benchmark-evidence/stl_metric_schema_v2_replay`; it keeps TripoSG as a
grandfathered incumbent but requires a schema-v2 confirmation before that old
archive can support a new promotion.

**Prepared Step1X-3D geometry lane.** The next bounded challenger uses the
geometry stage from [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D),
pinned to source `cb5ac944709c6c913109070c7b90c3447f57f3d4` and public model
snapshot `bf7084495b3a72222f36549b7942948aa4d9daa7`. The official model emits a
TSDF-derived `trimesh` mesh, making it a stronger upstream printability
hypothesis than another repair pass over fragmented TRELLIS-family output.
The benchmark adapter runs geometry only, disables optional SageAttention,
preserves the provider-native raw mesh without floater removal or upstream
decimation, and records PyTorch peak allocated/reserved VRAM. Raw and repaired
rows share one content-addressed inference; only the repaired row receives the
same inferred bbox, printable repair, 40,000-face ceiling, and scale-free
complexity cap used by the promoted TripoSG lane.

Fresh packages can include `--include-step1x3d-setup`. The generated G4 setup
retains Colab's PyTorch `2.11` CUDA `12.8` build, checks compute capability
`(12, 0)` and `sm_120`, installs only geometry dependencies, and excludes
texture/training-only NVDiffRast, PyTorch3D, Kaolin, DeepSpeed, CuPy, and
SageAttention. Source, CUDA, provider import, exact snapshot download, and
required-file checks all fail before benchmark work. The exact one-row config is
`backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_step1x3d_triposg_s40_n1.json`:
it remeasures the biharmonic-prefill TripoSG incumbent under metric schema v2
and compares Step1X raw/repaired output on held-out row 40. Step1X remains
unmeasured until that G4 smoke completes; no expansion is authorized by this
integration alone.

After the compact archive is checksum-verified locally, ingest it with both
repaired methods named explicitly:

```powershell
.\backend\.venv\Scripts\python -m backend.benchmark.ingest_stl_results step1x=C:\path\to\g4_stl_first_step1x3d_schema_v2_results_compact.tar.gz --output-dir backend\output\completion-benchmark\ingested\g4_stl_first_step1x3d_schema_v2 --require-schema-v2-repair-method triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh --require-schema-v2-repair-method step1x3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh
```

The optional strict ingest contract reads per-sample CSV telemetry rather than
aggregate means. It exits with status `2` if either named method is absent, if
canonical values are non-finite, if `repair_fill_ratio_supported` is false, if
surface-proxy provenance disagrees with the metric, or if a legacy fallback
label appears. This local acceptance step does not change the immutable Colab
payload.

The benchmark should rank candidates with STL-facing metrics:

- **Watertightness:** whether the exported STL encloses a valid solid.
- **Manifoldness:** non-manifold edges, self-intersections, inverted faces, and disconnected fragments.
- **Printable thickness:** minimum wall thickness, thin spikes, holes, and fragile bridges.
- **Mesh complexity:** triangle count, file size, repair time, and whether decimation preserves shape.
- **Surface accuracy:** Chamfer distance, normal consistency, and volume agreement when a ground-truth mesh exists.
- **Visual/depth agreement:** rendered views and depth maps should still match the input image, but only as supporting evidence.
- **Pipeline reliability:** setup success, runtime, GPU memory use, error rate, and deterministic artifact production.

Architecture reports now separate the **deployable score leader** from the **promotion-eligible winner**. The score leader is the best ranked deployable lane by the `stl-quality` objective, while the promotion winner must also pass hard STL gates: successful artifact emission, watertightness, volume/manifold validity, consistent winding, positive volume, one connected body, non-flat bounding box, no non-manifold/degenerate geometry, sane aspect ratio, and printable face density. When raw per-sample STL diagnostics are present, promotion requires every evaluated sample for that method to pass those hard gates; aggregate medians alone are not enough. This keeps a direct image-to-3D model visible when it is promising, but prevents it from replacing a cleaner deployable baseline until the STL itself is printable across the slice.

Ingest reports now include an **architecture replacement decision** for each run. It compares the best promotion-eligible full-mesh or multiview STL challenger against the best promotion-eligible depth-relief baseline, reports the score delta, and emits either `promote-challenger` or `keep-depth-relief`. A high-scoring direct mesh that fails hard STL gates remains visible as the score-leading challenger, but it cannot replace the depth-relief path until it is printable across the evaluated samples.

The STL-quality objective now scores and gates mesh complexity with `stl_faces_per_normalized_bbox_volume_log1p_median`, which divides bbox volume by the largest bbox side cubed before computing face density. This keeps complexity comparable across model-space direct meshes and print-scale depth-relief STLs. The older absolute `stl_faces_per_bbox_volume_log1p` remains available for backwards compatibility, but it should not decide rank or promotion by coordinate scale alone. Provider option `--mesh-max-normalized-face-density-log1p` closes the loop by decimating each mesh to a cap computed from this exact metric; the next TripoSG inferred-bbox run uses `9.95`, just below the promotion maximum of `10`.

Ground-truth mesh datasets should be used whenever possible. A small initial batch of 10-20 rendered 3D models is enough to compare candidates before spending time on larger training or tuning runs.

## Next Experiments

1. Keep the current depth/inpaint benchmark as the regression baseline.
2. Keep pinned TripoSG as the promoted full-mesh incumbent and preserve its adaptive scale-free face-density limit `9.95`.
3. Validate a topology-aware, surface-derived volume proxy before using fill drift to compare an open generative raw mesh with a watertight repair; preserve the current metric alongside it during calibration.
4. Add explicit paired Chamfer and H95 promotion guards to the standard STL selector instead of leaving those surface metrics diagnostic-only.
5. Keep the current post-hoc Hunyuan and TRELLIS repair searches closed. Move the next provider experiment upstream to a model or reconstruction path that emits cleaner connected geometry.
6. Score every depth-to-STL and direct-mesh output with the same watertightness, manifoldness, thickness, complexity, runtime, and Chamfer metrics; keep every source-mesh/source-bbox lane diagnostic-only.
7. Promote only a deployable backend that passes every per-sample STL gate, then expand beyond the current calibration slice.

The winning path should be chosen by final STL quality. A model that creates a convincing preview but produces fragile, non-manifold, or unprintable geometry should lose to a less flashy path that generates a clean printable solid.
