# Selector-aligned face geometry loss

This evidence covers the benchmark-only
`dinov2_pyramid448_coarse_to_fine_normal_selector_aligned_face_geometry`
challenger at exact commit
`6970cf2cd3cba367f4b4c693bda800d4294153d9`.

The challenger keeps the existing coarse-to-fine architecture and adds
training-only, differentiable approximations of the production selector:
shared-face affine fitting plus per-part physical RMSE, bias, p95 error, span,
shape RMSE, shape correlation, and raw-gradient correlation. Production
inference still uses only RGB, local depth, the incumbent face surface,
support/fusion maps, coordinates, and frozen DINOv2 features. Exact depth and
the six named part masks remain training/evaluation-only. Production was not
changed.

## Capacity gates

The strong selector setting, with mean/worst non-regression weights `4.0/8.0`,
mostly learned to abstain. Its two-row loss moved from `0.3563183546` to
`0.3560992777`, a ratio of `0.9993851653`; this failed the required maximum
ratio of `0.97`, so no split run was allowed.

The weaker `0.25/0.5` setting retained capacity. Its loss fell to
`0.3272459358`, a ratio of `0.9184088655`, and opened the identity-disjoint
split.

## Identity-disjoint result

The exact split contained 191 training rows, 32 validation rows, and 32
unopened sealed rows. Training reached its best validation loss of
`0.2624169644` at epoch 13 from `0.3114751047`.

The loss reduced strict regression events relative to the unaligned
coarse-to-fine run. The best regression count occurred at alpha `0.75`: 202
named-part regressions and one whole-face regression, compared with 237
named-part regressions and no whole-face regressions for the previous run's
best non-regression point at alpha `0.10`.

That gain came with excessive abstention. At full strength, total named-part
failures moved only from 353 to 352, median shape correlation from `0.820402`
to `0.820918`, and normalized RMSE from `0.186267` to `0.186122`. The
unaligned loss had reached 316 failures and `0.824738` shape correlation.
Strict non-regression therefore still failed.

- Training runtime: 575.809 seconds
- Peak training VRAM: 0.529 GiB
- Process peak RSS: 4.097 GiB
- Checkpoint: 2,258,401 bytes,
  SHA256 `574281adbff494e5e737886a081b472e85d22b99fd0a8f4a15eb424ec656ed69`
- Evidence: 14,310,921 bytes,
  SHA256 `a4a0f21f1dc7cc95a8f006d97f98d59777541591b616fa86620bee1df9eb1f09`

## Decision

`validation_strictly_improves=false`. Sealed evaluation, exact private
photos, background replay, and the 30 mm physical replay remained closed.
The production incumbent and its background-preservation path are unchanged.

Both measured selector-weight settings are closed. The next bounded
experiment should increase identity, expression, pose, illumination, and
occlusion diversity in the privacy-safe training split before changing this
architecture or loss again.
