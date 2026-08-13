# Hugging Face Space multi-object selection

## Goal

Match the production frontend's selection workflow while retaining the Space's
single SAM 3 precompute:

1. Hover previews a cached region.
2. Click toggles that region into or out of a persistent kept set.
3. Undo removes the most recently kept region; Clear removes all kept regions.
4. Done selecting composes the kept components and emits one selected image.
5. The preview shows only that isolated selected image, not the diagnostic tint.

## Implementation

- Hover, click toggling, Undo, Clear highlighting, and request assembly execute
  entirely in the browser with no `fetch` and no GPU call.
- The hidden request contains unique region IDs plus the normalized click point
  for each region. Points retain connected-component precision when one SAM 3
  instance contains disconnected regions.
- The CPU apply callback validates the image fingerprint and session schema,
  decompresses each referenced instance at most once, extracts the clicked
  component, unions all components, and creates one normal selection job.
- The user-facing preview returns `selected_image.png` on a neutral background.
  `selection_overlay.png` remains available in the job directory for diagnostics.
  Since the 2026-08-12 source-depth migration, this preview is not a depth-model
  input.
- Printable selected-object reliefs now submit
  `selection_mode=source-depth-isolate`. Depth is inferred from the original
  full photograph, then cropped to the union-mask bounds and stripped of every
  unselected sample before meshing. The prior `isolate` route remains covered as
  a legacy contract but is no longer used by the Space.
- Re-clicking a kept region removes it. This gives direct correction in
  addition to Undo and Clear.
- Every draft mutation sends one lightweight, non-GPU invalidation event. It
  clears any previously applied selection and cancels an older in-flight Done
  request, so generation cannot consume a stale mask set.

## Validation

- Browser-JavaScript contracts require a persistent `Map`, click toggling,
  Undo/Clear hooks, and no network `fetch`.
- A focused runtime fixture composes two disconnected cached masks, checks the
  exact union pixel count, labels, mask count, and multi-point provenance.
- The existing ZeroGPU pickle-boundary, state TTL, payload-bound, stale-upload,
  hover alignment, and no-second-model-call contracts remain active.
- Adversarial review specifically covered edit-after-Done and edit-during-Done
  races; both now fail closed through draft invalidation and cancellation.
- Local browser smoke used a public 512-pixel astronaut image and a deterministic
  three-region SAM 3 fixture. Two clicks produced one combined selection with
  42.6% image coverage. Re-clicking one object after Done cleared the applied
  preview and returned the UI to a one-object draft before generation could use
  the previous two-object job.
- The original multi-select validation passed 21 Space tests and 43 focused
  backend selection/STL contract tests. The source-depth migration is validated
  separately under `huggingface_space_source_depth_isolate_20260812`.
