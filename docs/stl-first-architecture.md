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

**Ready inferred-bbox run.** Use `backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_inferred_bbox_generated.json`, which includes provider flag `--mesh-max-normalized-face-density-log1p 9.95`. The implemented flag adaptively derives each mesh's face cap from its normalized bbox volume and the same scale-free density metric used by the promotion gate. This targets the only failed check from the measured run while retaining the existing fixed face target as an upper bound. The path is locally covered by the full benchmark regression suite; its G4 quality result is still pending.

Deployable multiview baseline on July 10, 2026: the benchmark provider wrapper now includes `multiview-visual-hull`, a deterministic silhouette-carving backend that consumes the existing `multiview_input.json` bundle, projects a voxel grid through each view's orthographic camera, exports a mesh, and then uses the same repair/scale/STL path as direct image-to-mesh providers. It is not a learned reconstruction model, but it gives the `multiview-mesh` lane a real STL-producing baseline that can be compared against depth-relief, source oracle, and TripoSR/TripoSG/Hunyuan-style single-image meshes without requiring external COLMAP/VGGT/SV3D setup. The calibrated 10-sample procedural slice at visual-hull resolution 32 promoted visual hull over the depth-relief `mirror` baseline under STL-quality scoring after source-mesh bundle oracles were correctly kept in the diagnostic-only lane; denser resolution-48/40/36 probes ranked visual hull highly but failed at least one strict per-sample mesh-complexity gate.

## Evaluation Metrics

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
2. Run the generated TripoSG inferred-bbox config on the same held-out samples with adaptive scale-free face-density limit `9.95`.
3. Score depth-to-STL and direct mesh outputs with the same watertightness, manifoldness, thickness, complexity, and Chamfer metrics; keep every source-mesh/source-bbox lane diagnostic-only.
4. Add contact sheets that show input image, mask, completed image when relevant, depth preview, mesh render, and STL repair status.
5. Promote only a deployable backend that passes every per-sample STL gate, then expand beyond the current 10-sample calibration slice and iterate on preprocessing, masking, scaling, repair settings, and model-specific parameters.
6. Add a video/multiview experiment once the single-image STL scoring harness is stable.

The winning path should be chosen by final STL quality. A model that creates a convincing preview but produces fragile, non-manifold, or unprintable geometry should lose to a less flashy path that generates a clean printable solid.
