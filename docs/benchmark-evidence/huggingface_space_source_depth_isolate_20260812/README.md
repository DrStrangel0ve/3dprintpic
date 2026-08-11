# Hugging Face Space source-depth isolate relief

## Failure

The selected-object Space route estimated depth from a tight RGB crop whose
unselected pixels had already been replaced by a neutral background. That input
was useful for guaranteeing an isolated STL, but it removed the surrounding
scene cues available to the local frontend. It also inherited backend defaults
for several relief controls instead of sending the local production values.

The result was a real pipeline difference despite both routes naming the same
Depth Anything V2 Large checkpoint. Faces and shallow object structure could be
materially worse in the Space.

## Architecture

The new `source-depth-isolate` mode separates inference context from mesh scope:

1. Load the immutable original image from the composed selection job.
2. Run the pinned Depth Anything V2 Large model on the complete photograph.
3. Run the existing face-preservation stage on the complete original RGB image,
   full-scene depth, full selection ROI, and full detection image. This matches
   the local route's face scale and context.
4. If the provider returned a lower-resolution grid, resample the entire refined
   depth and full face weight/region/occlusion masks into source-pixel coordinates.
   Only then apply the exact integer source crop. This avoids spatial shifts from
   rounding and stretching a crop taken in the lower-resolution grid.
5. Replace every depth sample outside the exact cropped union mask with `NaN`.
6. Mesh that selected-only depth with the aligned original-RGB crop as the photo
   detail source. No selection background blending or inpainting is performed.

The historical `isolate` mode is unchanged and remains regression-tested. The
Space alone moves to the hybrid mode.

## Local parity

The Space now submits the local production relief defaults explicitly instead of
depending on API defaults: smoothing sigma `0.35`, detail boost `0.8`, printable
feature depth `0.4 mm`, feature bridge depth `0.8 mm`, background detail boost
`2.4`, photo detail `0.60 mm`, relief gamma `0.75`, a two-pixel base border,
`2.0 mm/mm` maximum slope, `0.4 mm` nozzle, `0.8 mm` minimum feature, and the
same face strength/feather/correction controls. Its default surface-detail
request increases from 420 to 520 samples.

## Contract evidence

The focused endpoint fixture uses an `80x60` source, an asymmetric selected union
with a `56x46` padded crop, and a deliberately lower-resolution `40x30` provider
depth:

| Stage | Observed contract |
| --- | --- |
| Depth model input | Full original `80x60` RGB |
| Provider depth | Full-scene `40x30` grid |
| Face pass input | Full `80x60` RGB/detection image and full `40x30` depth |
| Aligned crop after face pass | `56x46` depth and original RGB |
| Face feature maps at meshing | Weight, region, and occlusion all exactly `56x46` |
| Meshing source | Original-RGB `56x46` crop |
| Meshing mask handoff | `None`; isolation is encoded as non-finite depth |
| Background depth ratio | `0.0` |
| Final depth file | `output_depth_data_selected_source_isolate.npy` |
| Provenance method | `full_source_depth_then_isolated_crop_v1` |

The fixture uses an asymmetric selection and requires
`isfinite(final_depth) == cropped_selection_mask` exactly. It also compares the
selected coordinate ramp against a full-grid resize followed by the exact crop,
and compares all three applied face feature masks pixel-for-pixel at the meshing
boundary. This catches both background leakage and reduced-grid misregistration.

## Validation

- Focused old-isolate, new-hybrid, and Space request-contract tests: 3/3 passed.
- Full backend regression suite: 1,136 tests passed in 147.141 seconds.
- Final selection/STL contract replay after provenance hardening: 44 tests passed.
- Full Hugging Face Space suite: 23 tests passed.
- Python compilation and `git diff --check`: passed.

This is structural and deterministic evidence. A live visual comparison remains
the final deployment smoke because ZeroGPU inference output cannot be reproduced
faithfully by mocked unit fixtures.
