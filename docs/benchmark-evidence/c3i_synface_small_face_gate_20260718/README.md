# C3I-SynFace small-face gate

This evidence records a privacy-safe, source-geometry-only screen of the
current face-depth path and the published C3I FaceDepth architecture. No source
RGB, depth image, checkpoint, or private-person artifact is tracked here.

## Corpus

- Demo assets: official C3I-SynFace code revision
  `dc8adfffbfd38818b72d0ad3776eb66724fafe95`.
- Demo-asset license: not stated in the source repository. The demo assets are
  evaluated by exact hash but are not redistributed.
- Raw-data lead: C3I-SynFace female data part 1, DOI
  `10.17632/z4454fyd8b.1`, CC BY 4.0. That license is not imputed to the
  separate GitHub demo assets.
- Rows: ten official RGB/grayscale-depth demo pairs, each transformed with the
  same full-frame geometric transform to a 78 px detected face in a 256 px
  canvas.
- Use limit: the official demo depth is an 8-bit preview. It supports this
  relative-shape smoke only; it is not raw EXR, metric-scale, or
  identity-disjoint training evidence.
- Every row retained a complete MediaPipe face and all six exact part masks.

The publisher advertises `female_data_part1.7z` as 7,738,200,071 bytes with
SHA-256
`5448d6d6577172a3f93c237c8cfe3e1896b238d256810e775855c890bf94b478`.
One redirected resume produced 7,772,454,919 bytes with SHA-256
`82ffb9c586de847a1c6cc0c9befc75ccfc907cb3562294e5d21ab7eb0979e2fe`;
it was rejected and removed. Fresh publisher and direct-object retries were
bounded out after sustained transfer rates near 30 KB/s. No unverified archive
was extracted or retained. The importer now exposes an exact size/hash
preflight for the next acquisition attempt.

## Current path

The current DAv2-plus-production-refinement replay completed all ten rows:

| Metric | Global DAv2 | Current refinement |
| --- | ---: | ---: |
| Combined named-part failures | 109/120 | 104/120 |
| Median shape correlation | 0.950707 | 0.951227 |
| Median raw-gradient correlation | 0.548052 | 0.563008 |
| Median normalized RMSE | 0.080707 | 0.079684 |

The current path improves every aggregate, but remains a hold: one row regresses
by two part checks and the absolute failure rate is 86.7%. This corpus is a
useful hard screen, not promotion evidence. Every evaluated asset is
authenticated against the portable corpus manifest before inference.

## FaceDepth checkpoint

The official README links a MobileNetV2 FaceDepth checkpoint, but the published
SharePoint URL returned HTTP 404 on 2026-07-18. A public research mirror at
`Yimin-zhou/2DImage-Relighting` revision
`bdf553705d4c996fb25d1747a0a005d80fa8b06d` contains a 57,906,985-byte blob
with SHA-256
`95ac733d761459e7f2072ba2862467c4d605b4cf4ae90491a757d93553123b17`.
The evaluator pins those exact bytes and loads them with
`torch.load(..., weights_only=True)`. Because no publisher checksum or explicit
code/checkpoint license is available, the mirror is research-only and is not
redistributed.

The bounded matrix compared RGB/BGR ordering, the documented 480x640 stretch,
an aspect-preserving fully convolutional input, raw crop geometry, and the
existing production fusion. The best variant was aspect-preserving BGR fusion:

| Metric | Current refinement | Best FaceDepth fusion |
| --- | ---: | ---: |
| Combined named-part failures | 104/120 | 106/120 |
| Median shape correlation | 0.951227 | 0.950727 |
| Median raw-gradient correlation | 0.563008 | 0.555085 |
| Median normalized RMSE | 0.079684 | 0.079993 |

Raw aspect-preserving BGR geometry failed 118/120 part checks. The best fused
variant improved one row by one part check, regressed three rows by one check,
and tied the remaining six. It failed aggregate, per-row, gradient, absolute
quality, and strict-improvement gates. Raw crop splices are diagnostic-only and
cannot trigger promotion. FaceDepth is closed; no 30 mm STL replay or
production change was made. The run took 25.873 seconds at 0.918 GiB peak
PyTorch-allocated inference memory on the local RTX 3080 Ti.

## Files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `corpus_preflight.json` | 6,953 | `dc8b495d5fd5a14a529e298406d30ccbf90b5cabd1adafbfec252711f1b1ec88` |
| `corpus_summary.json` | 35,646 | `1b7a35d45f0a1e45f05694ecc711869b223a5f9f319ebf90968ee8b12d054bb9` |
| `current_path_results.json` | 30,954 | `3278a688c1dd2b9e73bc0e9cab4d03e492c56df1f40e1b4ea2c93c173530f504` |
| `facedepth_results.json` | 166,467 | `e1769868770d519007cb8fb9056864cf3eac82ebd35c63864e6f945b83744562` |
