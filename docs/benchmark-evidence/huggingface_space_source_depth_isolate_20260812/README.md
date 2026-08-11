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
3. Map the selected source-pixel bounds into the returned depth-grid coordinate
   system using floor/ceil bounds, then resample to the exact RGB crop size when
   the provider returns a lower-resolution grid.
4. Run the existing face-preservation stage on the aligned original-RGB crop and
   cropped full-scene depth. Its ROI, region mask, detail weights, and occlusion
   masks therefore share one coordinate system.
5. Replace every depth sample outside the union selection mask with `NaN`.
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

The focused endpoint fixture uses an `80x60` source, a selected rectangle with a
`56x46` padded crop, and a deliberately lower-resolution `40x30` provider depth:

| Stage | Observed contract |
| --- | --- |
| Depth model input | Full original `80x60` RGB |
| Provider depth | Full-scene `40x30` grid |
| Aligned crop before face pass | `56x46` depth and original RGB |
| Meshing source | Original-RGB `56x46` crop |
| Meshing mask handoff | `None`; isolation is encoded as non-finite depth |
| Background depth ratio | `0.0` |
| Final depth file | `output_depth_data_selected_source_isolate.npy` |
| Provenance method | `full_source_depth_then_isolated_crop_v1` |

The fixture also requires less than 60% finite depth after masking, proving that
the crop is not merely visual metadata and that unselected samples are absent at
the mesh boundary.

## Validation

- Focused old-isolate, new-hybrid, and Space request-contract tests: 3/3 passed.
- Full backend regression suite: 1,136 tests passed in 183.128 seconds.
- Full Hugging Face Space suite: 23 tests passed.
- Python compilation and `git diff --check`: passed.

This is structural and deterministic evidence. A live visual comparison remains
the final deployment smoke because ZeroGPU inference output cannot be reproduced
faithfully by mocked unit fixtures.
