# Face-Aware Relief Depth

The depth-relief API can refine detected faces before STL generation. The
refiner reruns the selected depth model on a padded face crop, robustly aligns
the crop prediction to the global depth map, and transfers only additional
high-frequency facial detail. It does not replace the global head geometry.

Corrections are weighted toward the eyes, brows, nose, and mouth. The weight
falls smoothly to zero at the face oval, preventing the vertical ridge that a
hard face-depth splice creates. A robust face-depth range also caps the maximum
correction before the existing physical feature-size and slope filters run.

## API Controls

`POST /process_image` accepts:

| Field | Default | Purpose |
| --- | ---: | --- |
| `face_refinement_mode` | `auto` | `off`, `auto`, or required `on` |
| `face_detail_strength` | `1.0` | Strength of incremental face-crop detail |
| `face_feather_ratio` | `0.20` | Width of the zero-seam transition relative to face size |
| `face_max_correction_ratio` | `0.08` | Maximum correction relative to robust face-depth range |

`auto` is fail-open: images without a face, unreadable inputs, and unavailable
face-crop inference keep the original global depth map. `on` fails the request
when a detected face cannot be refined.

## Detectors

The default backend uses MediaPipe's 478-point Face Landmarker when the optional face
requirements are installed:

```powershell
pip install -r backend/requirements-face.txt
```

Without MediaPipe, the already-required OpenCV package supplies a frontal-face
box. The fallback still uses separate anatomical masks for the eyes, nose, and
mouth, but MediaPipe provides more accurate face contours and feature regions.
Current MediaPipe builds use the Face Landmarker Tasks model. On first use the
backend downloads the official `face_landmarker.task` file into
`~/.cache/3dprintpic`, verifies its pinned SHA256, and reuses it afterward. Set
`FACE_LANDMARKER_MODEL_PATH` to a pre-provisioned model for offline deployments.

## Artifacts

An applied refinement writes these files beside the normal depth output:

- `output_depth_data_face_refined.npy`
- `output_depth_face_refined_preview.png`
- `output_face_refinement_weight.png`
- `output_face_refinement_region.png`
- `output_face_refinement_occlusion.png`
- `output_face_refinement_metadata.json`

The metadata records the detector, crop and depth boxes, landmark count,
alignment scale and offset, correction limit, applied correction, and maximum
correction at the face boundary. The boundary value should remain effectively
zero.

## Coarse Landmark Shape Prior

MediaPipe Face Landmarker returns relative `z` as well as image-space `x/y` for
478 landmarks. The face pipeline now retains that signal instead of discarding
it. The first 468 landmarks are rasterized into a smooth crop-space surface,
robustly aligned to the coarse generic depth map, and blended inside the face
oval. It is a weak-perspective shape prior, not metric depth.

The prior fails closed unless all of these checks pass:

- at least 100 finite, visible landmarks;
- a non-clipped face contour;
- a nose-center yaw proxy no greater than `0.32`;
- absolute correlation of at least `0.15` with smoothed generic depth;
- a finite relative-depth span.

Correction strength scales with correlation and is capped at half of the
configured face correction ratio. The outer two pixels of the face mask are
held at zero correction. Hair, the head silhouette, and background remain from
the generic depth model.

## Opaque Eyewear Occlusion

Dark or reflective sunglasses can be a semantic outlier for monocular depth.
The generic model may turn a wraparound lens into one rigid depth sheet across
the eyes and brow. Treating that sheet as protected face detail preserves and
then amplifies the error in a tall relief.

The face stage now detects this narrow case before printable feature
enhancement. It builds an eyewear envelope from the MediaPipe eye and brow
landmarks, estimates skin luminance below the envelope, and accepts only one
broad connected dark component. Detection fails closed unless dark coverage is
at least `0.50`, component coverage is at least `0.45`, and component width is
at least `0.60` of the visible face size. Separate eyes, brows, and ordinary
shadows do not meet those gates in the regression fixtures.

For an accepted opaque occlusion, the hidden low-frequency surface comes from
the already aligned MediaPipe relative-z prior. The original depth may retain
only a signed accessory residual of `0.02` times the robust face-depth span.
The total correction is capped at `0.15` times that span and is rejected if
more than `5%` of the occlusion core saturates the cap, the residual does not
fall by at least `35%`, or the final residual remains above `0.05` times the
span. A feathered audit mask removes the accepted region from both printable
feature enhancement and the later maximum-filter bridge. The hard exclusion
is applied after those operations expand their support, so they cannot stamp
the lens sheet back into the STL.

Accepted opaque eyewear also changes how the coarse shape prior is calibrated.
Eyewear can lower agreement with the generic depth precisely because the
generic estimate has collapsed the lens region into a sheet. After the same
contour, yaw, finite-span, and `0.15` correlation gates pass, the shape-prior
confidence therefore has a `0.70` floor and its correction-cap scale increases
from `0.50` to `0.75`. The configured `0.08` face-range cap, feathered
zero-boundary blend, and all deocclusion quality gates still apply. Ordinary
faces and rejected eyewear detections retain the previous calibration
bit-for-bit.

The privacy-held two-face replay at 30 mm accepted both opaque eyewear regions.
Prior residual fell by `64.86%` and `60.51%`; cap saturation was `0%` and
`0.41%`. The regenerated STL retained the same `258,864` triangles,
`128 x 96 x 26.119 mm` bounds, one watertight volume, consistent winding, and
zero degenerate faces. Only aggregate evidence is committed under
`docs/benchmark-evidence/face_eyewear_occlusion_local`; the personal source,
crops, masks, renders, depth arrays, and STLs remain local and ignored.

The final source-preservation replay fixed a later regression where context
selection fed an interpolated canvas into global depth. It now uses the
checksum-verified original scene for depth, face refinement, and photo detail,
while a source-derived neutral cutout with a `1.5`-pixel feathered mask edge is
detector-only. Both faces were
recovered. The weak face's coarse-shape maximum correction increased from
`0.004470` to `0.018363` normalized depth (`4.109x`); the second increased from
`0.010719` to `0.018270` (`1.704x`). Final minimum component face-detail
correlation/RMS retention were `0.954011` / `0.815322`. Background depth
correlation, RMS retention, and gradient correlation were `0.999967`,
`0.999854`, and `0.995648`. The 30 mm result emitted one watertight, manifold,
consistently wound volume with zero degenerates. Aggregate evidence is under
`docs/benchmark-evidence/face_source_preservation_eyewear_20260728`; no private
photo or derived artifact is committed.

## Reliefs Above 12 mm

The old high-relief path used a one-sided minimum envelope to enforce a hard
neighbor-slope limit. It could only lower a high sample toward a low neighbor.
At 30-50 mm that operation propagated background lows through a portrait as
large triangular wedges. Reinserting a rigid 12 mm face afterward preserved
some features, but produced an oval plate around the face.

Face-aware reliefs above 12 mm now use screened gradient-domain dynamic-range
compression instead:

1. Horizontal and vertical gradients are measured on the complete requested
   relief before the old face-height transform.
2. Modest gradients are retained while large gradients are compressed with a
   smooth `asinh` response. The transition is `1.2 * sample_pitch *
   max_relief_slope`.
3. Audited silhouette edges remain eligible for preservation; large internal
   face edges are not silently exempted.
4. A sparse screened-Poisson system reconstructs one integrable height field.
   The screen weight is `0.01`, outer base anchors use weight `64`, and
   conjugate-gradient nonconvergence fails closed to the prior attachment path.
5. Eye, brow, nose, and mouth enhancement runs after reconstruction and is
   locally projected so it cannot introduce a new unchecked cliff.

A converged solve is not automatically accepted. The production path audits
face-region Laplacian correlation and RMS retention, cardinal and diagonal
edge distributions, output height span, and correction span. Missing metrics
or breached bounds reject the candidate to the prior attachment path. The
attempt and its rejection reason remain under
`face_height_stabilization.gradient_compression_attempt`.

`max_relief_slope` is therefore a gradient-compression scale in this path, not
a claimed hard upper bound. A closed relief generated from a height field has
no geometric undercuts; treating every steep front-surface gradient as an
overhang was both unnecessarily destructive and the direct cause of the face
falloff. Telemetry explicitly reports `hard_slope_limit_enforced: false`, input,
target, and output edge statistics, height spans, correction magnitudes, solver
status, and iteration count.

The deterministic analytic sweep in
`docs/benchmark-evidence/face_relief_gradient_domain_local` now compares the
legacy quadratic path, the former rigid screened-Poisson attachment, and the
new whole-surface reconstruction. At 30 and 50 mm, reconstructed facial
curvature correlates with the requested face at `0.9966` and `0.9949`; RMS
curvature retention is `0.9727` and `0.9440`. The held local real-image replay
contains two expressive frontal faces and one oblique face at both heights.
All six emitted STLs are watertight single-component volumes with consistent
winding and zero degenerate faces. The fresh quality-gated replay has minimum
face-detail correlation `0.8148`, minimum solver-side curvature RMS retention
`0.2953`, and final feature-update retention of `0.9370-0.9733` on the two
expression fixtures. Aggregate, source-provenance, runtime, solver, acceptance,
and topology telemetry is in
`docs/benchmark-evidence/face_relief_expression_gradient_local`.

## Research Basis

The geometry formulation follows the gradient-domain bas-relief family rather
than linear height scaling:

- [Digital Bas-Relief from 3D Scenes](https://gfx.cs.princeton.edu/pubs/Weyrich_2007_DBF/index.php)
  compresses gradient magnitudes and reconstructs an integrable height field.
- [Gradient Domain High Dynamic Range Compression](https://cris.huji.ac.il/en/publications/gradient-domain-high-dynamic-range-compression-13/)
  provides the underlying large-gradient attenuation and Poisson reconstruction
  pattern.
- [Preserving detailed features in digital bas-relief making](https://doi.org/10.1016/j.cagd.2011.03.003)
  combines nonlinear gradient compression, Laplacian detail restoration, and
  silhouette attachment.
- [MediaPipe Face Mesh V2 model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
  documents the 478 three-dimensional landmarks and their visibility limits.

Modern learned face reconstruction was also reviewed before choosing the local
path:

- [SMIRK](https://github.com/georgeretsi/smirk) is the strongest
  expression-focused parametric lead, but its practical setup requires the
  separately licensed FLAME assets and an older PyTorch3D/CUDA environment.
- [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm) is newer, but is
  non-commercial, also requires FLAME, and uses per-image optimization.
- [FaceLift](https://github.com/weijielyu/FaceLift) targets novel-view 3D
  Gaussian reconstruction rather than a directly printable watertight mesh.
- [Fast 3D Reconstruction of Faces With Glasses](https://openaccess.thecvf.com/content_cvpr_2017/html/Maninchedda_Fast_3D_Reconstruction_CVPR_2017_paper.html)
  segments glasses, excludes their measured depth from face fitting, and
  reconstructs the underlying face before optionally modeling the accessory.
- [Extreme 3D Face Reconstruction](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_Extreme_3D_Face_CVPR_2018_paper.html)
  separates a coarse facial foundation from bounded mid-level detail under
  occlusion.
- [FOCUS](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Robust_Model-Based_Face_Reconstruction_Through_Weakly-Supervised_Outlier_Segmentation_CVPR_2023_paper.html)
  demonstrates that outlier segmentation should be part of robust face-model
  fitting rather than treating every observed pixel as facial geometry.

Those remain isolated research providers. MediaPipe is the deployable default
because its maintained Apache-licensed landmark runtime supplies expression and
coarse shape without restricted identity-model assets. Hard occlusion and large
yaw still reject to generic depth; unseen identity-specific geometry cannot be
recovered as ground truth from one photograph.
