# Perspective face depth provider smoke

This bounded research smoke uses a deterministic CC0 MakeHuman-derived head,
procedural materials, a perspective camera, and exact z-buffer face/background
depth. It is a provider preflight, not an STL promotion result.

Official current sources were reviewed before inference:

- [MakeHuman licensing](https://static.makehumancommunity.org/about/license.html)
  permits CC0 exports and derived assets.
- [MoGe](https://github.com/microsoft/MoGe) provides the 2025 MoGe-2 depth and
  normal models under MIT terms.
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) lists
  DA3 Base as a 120M Apache-2.0 model and documents single-image depth.

## One-row finding

The tracked oracle/DA2/MoGe harness was replayed from clean exact revision
`2dea653881460a6b4b39d67fe1c2d2ce8ae62f7d`. The separate bounded DA3 probe
uses the source and model revisions recorded in `summary.json`. Its raw depth
and provider metadata are checksummed, but its derived evaluator telemetry was
not emitted as a replayable clean-harness summary and remains informational.

- The oracle passes every face, affine-mm, and background gate.
- Depth Anything V2 Large is the closest face model: minimum named-part shape
  correlation `0.9201`, minimum physical gradient correlation `0.7456`, and
  maximum p95 error `0.8674` mm. It narrowly misses the `0.75` eye-gradient
  gate and reverses the synthetic background's broad ordering.
- MoGe-2 Vit-B preserves more background gradient structure (`0.8208`) but has
  weaker face gradients (`0.4534`) and `1.5024` mm maximum p95 error.
- The informational DA3 Base probe runs on the local RTX 3080 Ti in `0.773`
  seconds with `0.802` GB
  peak allocated VRAM. It preserves background gradients best (`0.9473`) and
  keeps background span near `1.034x`, but its minimum facial-part shape and
  gradient correlations fall to `0.7069` and `0.4074`; it is held.
- A bounded DA2 face-interior unsharp/detail sweep did not improve the limiting
  eye gradient and is closed.

No provider is promoted from this one stylized scene. DA2 remains the production
face-depth incumbent while future provider work needs a larger textured,
perspective scene set and final STL gates.
