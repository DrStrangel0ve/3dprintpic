# SAM 3 still-selection promotion evidence

This compact bundle records the 2026-08-10 local promotion of the pinned
`facebook/sam3` checkpoint for still-image object selection.

The exact 2048 x 1536 source is private. No image, prompt coordinate, mask, or
friend-identifying artifact is published. Reproducibility is provided through
the source fingerprint, immutable model revision and weight hash, reusable
benchmark harness, production tests, and aggregate metrics in `summary.json`.

The deciding regression required a face click and a torso click to select the
same full person. SAM 3's `person` concept reached 1.0 IoU for all three people;
both point-only challengers failed. A six-concept production smoke then
selected the same people plus the full tower, a separate hotel, and a car.

No model fine-tuning was performed. The selected change is an inference-policy
promotion with deterministic bounded mask cleanup and bit-packed precompute
caching. RGB inpainting remains disabled.

The final production-code smoke retained 63 candidates, completed cold
precompute in 24.417 seconds, served the cached person click in 27 ms, and
passed 48 of 48 health probes issued while inference was active. The smoke
records no source pixels or prompt coordinates.
