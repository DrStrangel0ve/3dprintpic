# Face-depth rectified-flow pilot (2026-07-19)

This bounded research pilot tested whether a small conditional,
endpoint-supervised rectified-flow parameterization could recover
camera-aligned facial depth more naturally than an equal-parameter
deterministic predictor. It did not clear the synthetic gate, so the exact-photo
and 30 mm physical replays were intentionally not run and the production relief
path remains unchanged.

## Research decision

The upstream review did not find a current provider that simultaneously met
the 74-75 pixel turned-face requirement, reproducible camera-depth contract,
and production-compatible asset terms. Research-only near leads included
[HQ-Head](https://github.com/Biggaoga/HQ-Head-Generation) and
[HRN](https://github.com/youngLBW/HRN). Arc2Avatar depends on restricted
InsightFace recognition weights, and SOAP did not publish a usable license at
review time. Those providers were not silently treated as production options.

The selected pilot follows the pixel-space conditional flow-matching direction
demonstrated by [DepthFM](https://github.com/CompVis/depth-fm), but starts from
random weights and trains only on the checksum-pinned MHR camera-depth corpus.
The implementation uses the public
[Diffusers](https://github.com/huggingface/diffusers) `UNet2DModel`; it uses no
latent VAE and no generative foundation-model weights.

This corpus is suitable for bounded research comparison but is explicitly
`production_training_eligible=false`. The model is therefore incapable of
promotion regardless of metric outcome. Source geometry remains evaluation and
training supervision only; it is not shipped as user-facing output.

## Matched experiment

Both candidates have 1,808,001 randomly initialized parameters and optimize the
same emitted camera-Z endpoint objective. The only intended method difference
is stochastic flow-path training/inference versus direct deterministic endpoint
prediction. The flow loss converts velocity to its endpoint before applying the
geometry objective; it is not textbook target-velocity MSE. Its `flow` and
`camera_z` terms intentionally give endpoint value error a combined weight of
1.5 in both candidates.

- Input: RGB, frozen DAv2-Large coarse depth, validity, face support, exact
  crop-intrinsic rays, focal confidence, and the current dynamic state.
- Output: a floating relative camera-Z residual at 96 x 96 pixels.
- Geometry losses: equal-weight six-part camera-Z, gradient, camera-space
  normal, ordering, boundary, and Laplacian terms.
- Inference: eight Euler steps and four fixed posterior seeds.
- Uncertainty: posterior MAD calibrated once on validation and reused unchanged
  on the sealed split.
- Leakage controls: identity-disjoint 240/40/40 train/validation/sealed splits;
  sealed labels load only after training and checkpoint selection complete.
- Emission controls: unsupported flow state is projected away at initialization
  and every Euler step; residual resize is support-normalized; the emitted
  full-image correction is exactly zero outside face support. Crop-local
  evaluation arrays are diagnostics, not full production relief outputs.

The corpus summary SHA256 is
`d8717cced1d8f6753c15a7367dc1c84c5e62e0439f87b4380cca65f55fd665a5`.
Frozen DAv2 model and processor SHA256 values are respectively
`4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9`
and
`d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84`.

## Results

The 1,200-step run completed on the local RTX 3080 Ti in 366.456 seconds with
2.110 GiB peak VRAM. Both methods selected step 1,200. Validation part failures
favored the deterministic control at every checkpoint: 371 vs 429 at step 300,
269 vs 378 at 600, 244 vs 370 at 900, and 227 vs 296 at 1,200.

| Sealed metric | Rectified flow | Matched control | Gate result |
| --- | ---: | ---: | --- |
| All-row named-part failures | 313 | 230 | Fail |
| Median shape correlation | 0.950464 | 0.962022 | Fail |
| Median raw-gradient correlation | 0.814594 | 0.798637 | Pass |
| Median normalized RMSE | 0.102231 | 0.090322 | Fail |
| Small/turned named-part failures | 243 | 190 | Fail |
| Small/turned paired wins | 2/30 | - | Fail, required >= 24/30 |
| Small/turned failure reduction | -27.895% | - | Fail, required >= 25% |

Validation-calibrated 90% MAD coverage was 0.909545 and passed its 0.85-0.95
range. MAD/error Spearman correlation was only 0.123415 versus the required
0.35, so the uncertainty ranking gate failed. No claim is made from a raw
four-sample empirical 90th percentile.

The complete local `results.json` is 659,456 bytes and has SHA256
`230cb3f2d96c1838653cc0949ad72d5e505b34774c44ca895d6e7b13c3f50c3a`.
The full local run folder contains 403 files and 185,721,376 bytes, including
checkpoints and per-row artifacts. Only the compact result below is committed.
Best checkpoint hashes are:

- Rectified flow:
  `0cf43b5e02f853cb6d1d06a821a9dd4b1dfaf1bdd03530434d916f53ed741000`
- Deterministic control:
  `bfecd157dae71e31d5924f67861f631095372813ee23b5cfaf47bde0a8686eab`

## Decision

Hold and close this compact rectified-flow lane. Its small gradient gain does
not compensate for worse shape, RMSE, facial-part reliability, paired outcomes,
or uncertainty ranking. The advance gate correctly prevented private/exact
photo evaluation and the 30 mm face/background physical replay.

The existing production path is untouched: background prominence, 30 mm face
height, cap/attachment constraints, one-component watertight topology, exact
shell behavior, object/llama depth, dark-skin, eyewear, and cast-shadow controls
remain governed by their prior measured gates.

The run names source revision `1d5d8d8`, at which these four new source/test
files were still untracked. Their content SHA256 values in the run provenance
match the committed bytes exactly; the compact evidence preserves that detail
rather than pretending the pre-run revision already contained them.

Focused validation passed 25 tests and Ruff. The full shared dirty-tree backend
run completed with 987 tests plus 104 subtests passing and the same four known
canonical/MakeHuman production-relief fixture failures outside these owned
files. This evidence does not describe that dirty worktree as fully green.
