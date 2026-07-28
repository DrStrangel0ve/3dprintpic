# DA3Metric-Large centered high-relief smoke

This bounded privacy-safe run compares pinned Depth Anything V2 Large and the
official DA3Metric-Large checkpoint against the exact emitted MakeHuman oracle
at 30 and 40 mm. It uses the same centered smiling CC0 face, structured scene
background, perfect-selection control, physical STL gates, and pairwise metrics
as the preceding provider run.

## Pins and semantics

- Implementation: clean revision
  `5738972a4f5d14ce66c4241855e99b2b62a4f36e`.
- DA3 source: official clean commit
  `3fe327a6abe2e5db95b54444ea95463dbfef5610`.
- DA3Metric model: `depth-anything/DA3METRIC-LARGE` revision
  `4010e39f3634a45bc60553321fb49fb760bd594e`, Apache-2.0.
- DA2 control: revision
  `7581137eff8d4e94f6e796d3baea0e9fa79b22d2`.

The pinned DA3 head uses an exponential positive output and the official source
describes metric conversion as a global focal-length multiplier. That scalar
cancels under selection-only relief normalization, so the unchanged canonical
output is evaluated with documented far-high/inverse-depth semantics. The
standalone checkpoint returns `is_metric=false` and no intrinsics; both facts
are recorded rather than overstated.

## Result

DA3Metric is closed without cropped-scene expansion. At 30 mm its face normal
mean is `0.900212`, p95 normal error is `48.4292` degrees, and minimum relighting
correlation is `0.725143`. Every eye, eyebrow, nose, and mouth shape gate fails;
the worst part is the mouth at `13.3627` mm p95 affine error. The 40 mm result is
worse at `53.6326` degrees and `17.8143` mm.

Background amplitude is again not the limiting factor. DA3Metric emits
`18.6208` mm span at 30 mm and `27.0948` mm at 40 mm, but oracle correlation is
`-0.210890`/`-0.216549`. Gradient correlation reaches only
`0.292176`/`0.335536`. A polarity flip was not adopted: it would contradict the
pinned model contract and would be selected using oracle evidence.

All six oracle/control/challenger STLs pass their own context, cap, attachment,
topology, exact-shell, and cross-height checks. The DA3Metric process absolute
peak is `2.1674` GB with DA2 resident; incremental DA3Metric peak is `1.5322`
GB. The run stops solely on oracle face/background quality gates.

Raw diagnostic telemetry reveals a region-specific DA2 issue useful for the
next fusion lane. Against far-high exact depth, DA2 is `-0.982817` on the face,
consistent with its near-high contract, but `+0.504785` on the background,
which reverses that contract locally. These correlations are diagnostic-only;
the oracle never enters normalization or STL generation.

## Reproduction

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.run_makehuman_face_provider_relief_smoke `
  --output-dir backend/output/makehuman_provider_relief_da2_da3metric_clean_centered_n1 `
  --providers depth-anything-v2-large,da3metric-large `
  --scene-profiles caucasian_female_smile --device cuda --allow-failures
```
