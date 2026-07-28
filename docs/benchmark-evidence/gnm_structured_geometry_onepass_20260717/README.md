# One-pass expression-conditioned GNM geometry

This slice tests whether richer expression evidence can improve the guarded
GNM face foundation without weakening background depth or 30 mm printability.
It is privacy-safe: source geometry is used only for train/evaluation targets,
and the real-photo controls are public CC0 assets.

## Method

The Face Landmarker Tasks API is invoked once per face. The same result and
face index provide all 478 landmarks and the optional ordered 52-element
blendshape vector. The production default keeps blendshape output disabled;
only a checkpoint that declares the richer schema opts in. Checkpoints pin the
MediaPipe version, task hash, ordered-name hash, schema, and feature dimension.

The trainer combines these 256 structured values with a pinned 768-value
Depth Anything V2 Small embedding. Separate low-rank ridge heads predict GNM
identity and expression coefficients. At inference the predicted surface must
Pareto-dominate the mean GNM surface on correlation and normalized RMSE against
live production depth, or the row uses the established mean-face fallback.

The train-only geometry target contract rejects validation/sealed rows before
loading a decoder or writing targets. Its private supervision manifest contains
only row identifiers, target paths, dimensions, and hashes; public summaries do
not expose source meshes or target paths.

## Synthetic result

The corpus added 160 identity-stratified train rows and combined them with the
existing training and novel-identity slices. Detector-clean prepared counts
were 228 train, 46 validation, and 48 sealed rows with 100% detector coverage.

The selector chose rank 16 for both heads at ridge alpha 1000. On 34 small
validation faces, median normalized vertex RMSE moved from 0.010953 to
0.008095, with a 97.1% paired win rate. On 20 small sealed faces it moved from
0.008484 to 0.007009, with an 80% paired win rate. All six named facial parts
improved on both splits. This is a real synthetic-geometry gain.

## Real-photo gates

Identity coefficients remained disabled because prior exact sweeps regressed
the CC0 controls. Expression strengths 0.125, 0.25, 0.5, and 1.0 were tested on
the same three exact rows. Strength 0.125 was the only setting that
Pareto-improved the current central-part mean-GNM incumbent on the hard face:

| Method | Shape corr. | Gradient corr. | Normalized RMSE | Part failures |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 0.806602 | 0.626152 | 0.175270 | 8 |
| One-pass, 0.125 | 0.806633 | 0.626163 | 0.175257 | 8 |

Full strength regressed to ten failures and failed closed. The selected 0.125
candidate passed the 15-row varied gate and the safety portions of the 20-row
all-small gate. However, it changed only one varied row and seven all-small
rows relative to the incumbent, and recovered no additional named-part pass.
The all-small absolute part-failure rate remained 0.9542, so that gate remains
hold.

The 30 mm candidate retained a 29.9805 mm span, background correlation
0.999996 and centered-RMS retention 1.000006 versus its paired baseline. It is
one watertight, winding-consistent component with zero degenerate faces and
exact shell agreement. It still has eight named-part failures, exactly the
same as the production incumbent, and its absolute shape and affine-mm part
gates remain false.

## Decision

Hold. The model learns expression geometry on identity-disjoint synthetic
faces, but the live-depth selector correctly rejects most domain-gap outputs.
The surviving real-photo improvement is too small to justify a second neural
backbone in production. The mean GNM central-part foundation remains the
default. The one-pass detector/checkpoint contract stays in the benchmark
harness for future providers or materially more varied supervision.

Machine-readable metrics and raw evidence hashes are in `results.json`.
