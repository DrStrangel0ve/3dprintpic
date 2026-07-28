# RAP3DF V2 authenticated real-sensor gate

Status: **hold; production unchanged**.

This pass adds a fail-closed acquisition path for the official
[RAP3DF V2 release](https://data.mendeley.com/datasets/kpdkpcs8zb/4) and tests
whether it can serve as real-sensor evidence for 74-75 pixel faces. RAP3DF V4
is CC BY 4.0 and contains matched visible-light and Kinect One depth captures
from real volunteers under the ethics approval stated by the publisher. It is
evaluation-only in this project and is not copied into Git.

## Acquisition and decoder correction

`backend.benchmark.acquire_rap3df_v2` reads the publisher ZIP without extracting
it wholesale, queries Mendeley's anonymous public file API, and stages only the
selected database, RGB, and raw-depth members. Every member must match the
publisher file ID, content ID, status, byte count, and SHA256 before it is
written. Authenticated size and compression-ratio limits are checked before
decompression, temporary files are created exclusively, and the complete slice
is published atomically. ZIP traversal, duplicate/case-ambiguous members,
conflicting outputs, missing folders, and malformed metadata fail closed
without leaving unattributed volunteer assets behind.

The official V4 ZIP is 66,792,678 bytes with SHA256
`92a967bdacba4a7e5d387232f2d3308ad656022953c0139f615def2f607ccc5e`.
The deterministic three-identity, six-pose acquisition contains 37 authenticated
files for 18 candidate rows. Its emitted manifest SHA256 is
`e9cd56b008386403b30dba17d6a32d5e3b00ffceec92ea2bbb6c0841794b5f1b`;
the importer's normalized manifest identity is
`fa0c001ddb310a160949d7f0000b543e26e2ec524ab2ab326ce3a45a2a10b61c`.

The real archive exposed a fixture-blind decoder defect: publisher frames are
119 pixels wide by 149 high, while the old constant treated those axes as
height by width. Reshaping the raw little-endian uint16 stream to `(149, 119)`
correlates `0.990055` with the publisher depth preview and has valid-mask IoU
`1.0`. The old transposed interpretation correlates only `0.111878` with mask
IoU `0.189655`. The importer now pins the verified row-major orientation.

RAP3DF's low-resolution faces also need a lower bounded detector floor. Native
detection now uses 24 pixels; only a failed native attempt may use a deterministic
3x Lanczos detection pass. Masks and bounding boxes are inverse-mapped with
nearest-neighbor masks and conservative floor/ceil bounds. The scale is recorded
per row.

## Full-slice source audit

The 18-row expansion is not a valid six-part benchmark:

| Source check | Passing rows |
| --- | ---: |
| One complete 478-landmark/six-mask face | 17 / 18 |
| Nonzero raw depth under all six named parts | 13 / 18 |
| Nose median is nearer than face median | 5 / 18 |

All three left-turn rows pass the part and orientation checks. No right-turn
row has valid depth under all six parts; one near-profile image cannot produce
all six landmark masks even at 3x. Front, up, down, and random-pose failures
also show that the release does not provide a reliable pixel-registration
contract for the selected RGB and raw depth streams. The importer therefore
correctly rejects the full expansion instead of filling source holes or
inventing millimeter calibration.

Because the known all-pose default cannot pass those source gates, corpus import
now requires an explicit pose selection. Corpus rows are built in a private
sibling directory and become visible only after the complete summary and every
asset hash have been written. A post-hardening smoke against the real publisher
archive authenticated and imported the first identity's left row successfully.
Its acquisition manifest SHA256 is
`d48ebe5cc404cd3ee0cb21c343de89d42fcec9e4fd7f170910a75e684f8a4e43` and
its corpus summary SHA256 is
`49498f759027ea32dfeb56c13463dd2ebb72eb33d46a821093935eca8b35fc1c`.

## Bounded model diagnostic

Only the first identity's front and left rows passed source preflight. They were
imported at 75-pixel face height and evaluated with the pinned Depth Anything V2
Large model plus the current production face refinement on the local RTX 3080
Ti. MediaPipe was present for the final replay.

| Method | Part failures | Shape | Gradient | Normalized RMSE |
| --- | ---: | ---: | ---: | ---: |
| Global DAv2 depth | 24 / 24 | 0.138583 | 0.036092 | 0.334709 |
| Current face refinement | 24 / 24 | 0.126159 | 0.022200 | 0.335503 |

Inference took 4.575 seconds and peaked at 0.862 GiB allocated VRAM. The
refinement regressed every aggregate metric and one row had reversed candidate
depth semantics. The result is a hold, but not a trustworthy absolute ranking:
the publisher does not establish metric depth scale or pixel-perfect RGB/depth
registration. It is sufficient to show that the current synthetic gains do not
transfer cleanly to this real-sensor source.

## Decision

Keep RAP3DF evaluation-only and stop at the two-row diagnostic. Do not train on
these 18 volunteer rows, do not expand the benchmark, and do not tune the
production refiner against ambiguous registration. The useful deliverables are
the authenticated acquisition utility, corrected raw orientation, bounded
low-resolution detector path, and a measured real-domain warning.

The production GNM face path, 30 mm background prominence, cap and attachment
constraints, one-component watertight topology, exact shell, object/llama
depth, dark-skin, eyewear, and cast-shadow controls remain unchanged. No real
volunteer image, raw depth, transformed crop, model output, or private artifact
is published. Compact scalar evidence is in `results.json`.

## Validation

```powershell
python -m pytest backend/tests/test_acquire_rap3df_v2.py backend/tests/test_rap3df_corpus.py -q
```

Initial pre-review result: 17 tests passed.

Post-review focused result: 21 tests passed. The acquisition/corpus plus both
face-depth evaluator suites pass 37 tests. Ruff passes all four owned Python
and test modules. The complete backend replay reports 922 passed plus 104
subtests, with only the same four pre-existing canonical/profile and MakeHuman
relief failures recorded before this lane; RAP3DF introduced no new failure.
