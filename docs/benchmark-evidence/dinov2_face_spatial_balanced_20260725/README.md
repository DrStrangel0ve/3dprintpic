# Size-balanced DINOv2 face-relief study

This bounded study closes the current DINOv2 residual-decoder lane with a
**hold**. It made two useful production-facing improvements:

1. The privacy-safe corpus and production cache now cover 255 identity-disjoint
   rows across small, medium, and close faces.
2. A selected-subject YuNet path now detects the 70-pixel turned face that
   previously stopped cache generation, without lowering the full-image
   detector threshold.

The learned correction improves aggregate validation geometry, but every
nonzero blend regresses at least one paired whole-face or named-part metric.
Size/yaw-weighted training reduces the full-strength failure count by two, but
slightly worsens the median metrics and increases whole-face regressions.
Neither decoder is promoted. The sealed identities, exact private photo, and
30 mm physical replay remain unopened, and production relief behavior is
unchanged.

## Corpus and cache

The deterministic composer combines three checksum-pinned, mutually disjoint
GNM sets:

| Split | Rows | Identities | Source summary SHA256 |
| --- | ---: | ---: | --- |
| Train | 191 | 24 | `683087359d25f12db6702ad6d774b7c3cce49e70c307db9cfcc33f37d7b259de` |
| Validation | 32 | 8 | `0c85aa696aac7c45145ea016fd8fc3ffa0ce9c54ff1d8f18bdba58bd28053e80` |
| Sealed | 32 | 8 | `17c42393a8e08c856a848d609e827a82fde2e708956a2f53e6e9bcb7444d9b55` |

The combined corpus contains 255 rows, 40 identities, and 2,295 assets:

- summary SHA256
  `67abfa2b59e00f9cf7c0d3ae6a9b66960d7b97203b9d57b5b92ec791216d94bc`
- ordered artifact SHA256
  `8940492360c1def86bb9da66910758f25148ed3b80e31297f4a1aa6f10080d09`
- source-summary manifest SHA256
  `c7c5962f3af4a044edce2de8b79132fa864e74fd3980aa45250359aae1bddc54`
- 115,767,617 asset bytes

Every asset hash is revalidated while composing. The composer rejects row,
identity, relative-path, source-image, exact-depth, and selection-mask leakage
across splits. Source geometry is training and evaluation-only.

The production cache uses the exact pinned Depth Anything V2 Large revision
`7581137eff8d4e94f6e796d3baea0e9fa79b22d2`. It completed 255/255 rows in
329.611 seconds with 0.864 GiB peak VRAM:

- cache manifest SHA256
  `f408f166d7cd11fe6c6b9a84b9ac9df9d4d0a78da905f2295dd912b88dd4fe61`
- ordered row-content SHA256
  `c23d0a83e03c60cebb66f3229ff36ae456a8578d8a28a426af5129efcaeba5ab`
- 65,582,615 cached-row bytes
- detector scopes: 200 selection ROI, 50 full image, 5 tight selection ROI
- minimum support precision `0.821310`
- minimum exact-support coverage `0.275877`

## Small turned-face detection

The first clean cache attempt failed on
`gnm_female_black_v02__corners_down_00`, a 70-pixel turned face. YuNet had
found a geometrically valid face at about 0.72 confidence, but the global 0.90
threshold and frontal landmark-ratio checks rejected it.

The corrected path relaxes confidence and eye/mouth separation only inside a
user-selected connected component. It still requires valid eye/nose/mouth
ordering and at least 50% selection overlap. Full-image detection remains at
the original 0.90 threshold. The failed row is now recovered at:

- detector: `selection-roi:opencv-yunet-2023mar`
- source bounding box: `[82, 102, 107, 142]`
- confidence: `0.733252`
- selection overlap: `0.821310`

## Decoder results

Both studies use frozen official DINOv2 Small features at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056` and the same 241,473-parameter
camera-aligned residual decoder. The two-identity capacity smoke passed for
both runs.

The validation incumbent has 353 combined part failures, median shape
correlation `0.820402`, median raw-gradient correlation `0.399444`, and median
normalized RMSE `0.186267`.

| Training | Alpha | Part failures | Shape corr. | Raw gradient corr. | RMSE | Whole/part regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unweighted | 0.05 | 351 | **0.820948** | **0.399856** | **0.186136** | 1 / 238 |
| Weighted | 0.05 | 352 | 0.820916 | 0.399834 | 0.186138 | 1 / 248 |
| Unweighted | 1.00 | 324 | **0.825028** | **0.401403** | **0.185191** | **5** / 336 |
| Weighted | 1.00 | **322** | 0.825000 | 0.401293 | 0.185220 | 6 / **333** |

The weighted curriculum used a 128-pixel small-face reference, up to 2.0x
size weight, 0.75 yaw strength at 38 degrees, and a 2.5x combined cap. The
validation weight median was 2.3424. It selected epoch 13 with weighted loss
`0.185835 -> 0.174713`.

The first regression in both runs remains the left-eye shape correlation on
the 71-pixel, 38-degree
`gnm_female_asian_v03__corners_down_04` row:

- unweighted alpha 0.05: `0.752463 -> 0.751785`
- weighted alpha 0.05: `0.752463 -> 0.752107`

Weighting makes that one change smaller but creates more regressions elsewhere.
The validation selector therefore correctly returns alpha zero for both runs.

## Decision

- `validation_strictly_improves=false`
- `sealed_opened=false`
- `advance_to_exact_photo=false`
- `advance_to_30mm_physical_replay=false`
- production face relief and background handling unchanged

The existing 30 mm face/background path remains the production result: it
retains the measured face height and background span while preserving cap,
attachment, one-component watertight topology, and exact shell gates. This
study does not weaken those controls.

The next useful face-quality lane needs stronger camera-aligned real-image
supervision or an upstream model that emits reliable fine facial geometry.
More weighting of this same small synthetic decoder is closed.

Focused validation passed 61 tests. A clean-worktree all-backend invocation
reached an unrelated Git-repository provenance test that cannot resolve the
linked worktree metadata under the current filesystem sandbox; it is not a
face-pipeline failure.

Compact machine-readable values are in `results.json`. Raw cache tensors,
synthetic render assets, exact regression events, and diagnostic checkpoints
remain in ignored local research output.
