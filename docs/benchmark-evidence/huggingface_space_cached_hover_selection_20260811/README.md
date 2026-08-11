# Cached SAM 3 hover selection

Date: 2026-08-11

## Objective

Replace per-click SAM 3 inference with one bounded object-map preparation so a
user can hover over detected objects, see the green selection tint immediately,
and click without consuming another ZeroGPU allocation.

## Design

- The existing pinned `facebook/sam3` concept model remains the only selection
  model. No inpainting or second foreground model was added.
- SAM 3 instance masks and scores are computed once when `Select object` is
  active and an image is uploaded.
- The server stores packed full-resolution masks in the existing bounded
  selection cache, then releases the selection model before returning.
- For browser hit testing, masks are reduced to a maximum 1024-pixel long edge.
  The highest-scoring mask wins overlaps, disconnected winning components get
  separate IDs, and tiny fragments below 0.001% of map area are omitted.
- Region IDs are encoded losslessly in RGB channels of a PNG. The manifest
  contains this PNG plus labels, confidence scores, and pixel counts; it does
  not contain generated image pixels.
- Browser `pointermove` events read the cached ID map and draw a green canvas
  tint. Switching objects and leaving the image produce no network request.
- A click sends only normalized coordinates through a hidden Gradio event. The
  server resolves the exact packed full-resolution mask and writes the normal
  selection artifacts without calling SAM 3 again.
- Source-image fingerprints bind a precompute session to its upload. Stale or
  missing sessions, non-finite coordinates, and uncovered points fail closed.
- Each public hover region is bound server-side to the exact full-resolution
  instance that produced it. A bounded nearest-pixel correction covers only
  downsampling-edge disagreement and never calls the model.
- Rapid upload preparation uses `always_last`; image replacement and scope
  changes cancel delivery of in-flight stale results. Stale image decodes are
  ignored, and Gradio's `object-fit: scale-down` layout is handled explicitly.
- Original RGB pixels are removed from the global precompute cache as soon as
  the packed masks are stored. Packed entries idle for 20 minutes are pruned on
  the next Space action and remain protected by the backend's count bound.

This follows Gradio's official custom JavaScript event model while keeping all
GPU inference in the existing serialized queue:
<https://www.gradio.app/guides/custom-CSS-and-JS>.

## Validation

The local browser smoke used the Space-pinned Gradio 5.49.1 runtime and a
generated two-shape image; no personal image or private artifact was used.

- Upload in `Select object` mode automatically prepared two cached regions.
- Hovering the first region exposed `person selection preview` on the overlay.
- Moving directly to the second exposed `object selection preview`.
- Moving outside the source image cleared the hover preview.
- Clicking the first region triggered the CPU-only cached-selection event and
  returned `Selected person with SAM 3 (19.0% of image).`
- The canvas matched the displayed image's contained content rectangle at
  640 x 400 source resolution and retained the selected tint on pointer leave.
- A separate 96 x 60 source stayed at exactly 96 x 60 inside Gradio's larger
  `object-fit: scale-down` frame; its hover target remained pixel-aligned.
- The overlaid source still allowed Gradio's `Remove Image` control to clear the
  image and reset selection state.

Focused automated validation:

```text
python -m unittest huggingface_space.tests.test_space_runtime huggingface_space.tests.test_space_ui
Ran 17 tests in 0.078s
OK
```

The new tests cover overlap score precedence, disconnected region IDs, the
single precompute call, model release, packed-mask click resolution without a
second model call, cache source-pixel removal and expiry, latest-upload queueing,
three shared GPU preparation events, and absence of browser `fetch()` calls in
hover handling.

## Deployment gate

The browser interaction is locally verified. The Hugging Face Space should be
rebuilt with the updated `app.py`, `space_runtime.py`, tests, and README, then
smoked once with a non-private image. A successful deployment must show one
45-second preparation allocation followed by repeated hover/click interactions
with no additional ZeroGPU jobs.
