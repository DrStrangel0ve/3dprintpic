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

## Evaluation Metrics

The benchmark should rank candidates with STL-facing metrics:

- **Watertightness:** whether the exported STL encloses a valid solid.
- **Manifoldness:** non-manifold edges, self-intersections, inverted faces, and disconnected fragments.
- **Printable thickness:** minimum wall thickness, thin spikes, holes, and fragile bridges.
- **Mesh complexity:** triangle count, file size, repair time, and whether decimation preserves shape.
- **Surface accuracy:** Chamfer distance, normal consistency, and volume agreement when a ground-truth mesh exists.
- **Visual/depth agreement:** rendered views and depth maps should still match the input image, but only as supporting evidence.
- **Pipeline reliability:** setup success, runtime, GPU memory use, error rate, and deterministic artifact production.

Ground-truth mesh datasets should be used whenever possible. A small initial batch of 10-20 rendered 3D models is enough to compare candidates before spending time on larger training or tuning runs.

## Next Experiments

1. Keep the current depth/inpaint benchmark as the regression baseline.
2. Add a direct image-to-mesh smoke lane that runs one or more modern models on the same rendered samples and exports repaired STL files.
3. Score depth-to-STL and direct mesh outputs with the same watertightness, manifoldness, thickness, complexity, and Chamfer metrics.
4. Add contact sheets that show input image, mask, completed image when relevant, depth preview, mesh render, and STL repair status.
5. Promote the best-performing backend to a larger 10-20 sample run, then iterate on preprocessing, masking, scaling, repair settings, and model-specific parameters.
6. Add a video/multiview experiment once the single-image STL scoring harness is stable.

The winning path should be chosen by final STL quality. A model that creates a convincing preview but produces fragile, non-manifold, or unprintable geometry should lose to a less flashy path that generates a clean printable solid.
