# Source-Preserving Eyewear Face Replay

This compact bundle records the final privacy-held two-face regression used to
close the production relief path. The source photo, selection masks, face
crops, depth arrays, rendered previews, and STL remain local.

The final replay ran from tracked implementation revision
`fc8308f75c43872ab22fff528438ac62c4dbabec`. Only ignored local service logs
were present in the worktree.

The old context route estimated depth from a source-free interpolated selection
canvas. The corrected route estimates all geometry and photo detail from the
original source image. A source-derived neutral cutout with a 1.5-pixel
feathered mask edge is used only to help the face detector find both people.
Product selection requests
`selection_infill_mode=none`, the frontend exposes no inpainting feature, and
the production relief endpoint rejects legacy completion modes.

Both accepted opaque-eyewear faces keep the existing contour, yaw, correlation,
finite-span, correction, deocclusion, and zero-boundary gates. The new
eyewear-only calibration applies a `0.70` confidence floor and a `0.75`
correction-limit scale after those gates pass.

`summary.json` contains only aggregate measurements. It carries no source
filename, image hash, biometric coordinates, crop, mask, depth field, or mesh.
