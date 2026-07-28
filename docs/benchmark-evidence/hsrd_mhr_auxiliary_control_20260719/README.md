# HSRD/MHR auxiliary face-depth control

Status: **hold**. Production is unchanged.

This experiment tested whether exact camera-Z supervision from Meta MHR can
improve the small, turned faces that remain the dominant 30 mm relief failure.
It also preflighted the stronger HSRD native-photo supervision path before any
photo archive was downloaded or any camera pose was assumed.

## Research decision

HSRD publishes the original RealityCapture project, but its XML project file
does not contain solved camera matrices. They are held in the binary
`sfm0.dat`. The downloaded `HSR0015-Body-032` RealityCapture archive exactly
matches SHA256
`2f437b6cbc805440333665f769dd27ad6c3781808a79c840e4eab1fa8bb901bf`.
The contained project and registration files are independently hashed in
`results.json`.

RealityScan is not installed on this machine. The official CLI documents
`exportRegistration` and XMP export as commands executed by the RealityScan
application, and the official camera report variables expose the required
intrinsics, distortion, rotation, and translation only through that running
application. No official standalone `sfm0.dat` schema or parser is published.
The calibration lane therefore failed closed: no binary reverse engineering,
guessed field of view, or guessed camera pose was used.

Primary references:

- <https://rshelp.capturingreality.com/en-US/appbasics/allcommands.htm>
- <https://rshelp.capturingreality.com/en-US/appbasics/reports_fav_cameras.htm>
- <https://rshelp.capturingreality.com/en-US/tools/xmpalign.htm>
- <https://dev.epicgames.com/documentation/realityscan/getting-realityscan>

The fallback used the exact 240 train-only rows from the deterministic MHR
camera-depth corpus. Synthetic depth is useful, but published depth-domain
work consistently identifies synthetic-to-real shift as a first-order risk.

Relevant primary research:

- Zhao et al., CVPR 2020, *Domain Decluttering*: <https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Domain_Decluttering_Simplifying_Images_to_Mitigate_Synthetic-Real_Domain_Shift_and_CVPR_2020_paper.html>
- He et al., CVPRW 2021, *Semi-Synthesis*: <https://openaccess.thecvf.com/content/CVPR2021W/WAD/html/He_Semi-Synthesis_A_Fast_Way_To_Produce_Effective_Datasets_for_Stereo_CVPRW_2021_paper.html>
- Li et al., 2024, *PatchRefiner*: <https://arxiv.org/abs/2406.06679>

## Matched protocol

Both lanes use the same 92,089-parameter deepest-only DAv2-Small decoder, exact
initial state, HSRD batch order, 216 HSRD optimizer steps, optimizer settings,
gradient clipping, and fixed final epoch. The challenger adds a predetermined
`0.25 * MHR loss` to the same HSRD step; it receives no additional optimizer
step. MHR validation and sealed rows are never prepared.

The strict gate now completes HSRD validation before opening any sealed asset.
The matched contract also fails closed unless both lanes provide valid initial
and schedule hashes, positive and equal exposure/step counts, identical
optimizer records, fixed final epochs, and valid auxiliary provenance.

## Authoritative result

The exact validation-only replay is bound to trainer SHA256
`b9890a4f9b4bdc3dd2af2d4b2878d72b0dd5856b777b794778dd76b8044b476e`
and summary SHA256
`de51ed8ebbe4069b56badaa7ad86d4b99cecfcf0894192e240a1978e4d1a754c`.
It reproduced the earlier training state hashes while enforcing the corrected
sealed boundary.

Final review then hardened only the evidence contract to require the exact
auxiliary weight and schedule seed. Published trainer SHA256 is
`2a24e07ee22ac494966620b59a9582798730901041f412b1162b9944ebc0c48c`.
Its post-hardening smoke passed every matched-training check and again stopped
before sealed access; its summary SHA256 is
`0789fcd849615b7498929b3e2e253de79498ce1bbaea6c2c9ba35dfc10e070fc`.
The hardening does not change numerical training or selection behavior.

Persistent auxiliary loss reduced validation named-part failures from 177 to
172 and improved median shape correlation from `0.822647` to `0.827267`.
However, raw-gradient correlation fell from `0.617559` to `0.613607`, so the
candidate did not strictly beat the HSRD-only control. The run stopped with
`sealed_evaluated=false`, no sealed feature preparation, and no selected sealed
metrics. Background pixels remained bit-exact and peak VRAM was `0.4635 GiB`
on the local RTX 3080 Ti.

An earlier harness version opened the eight sealed rows after this validation
failure. Those values are now explicitly quarantined as contaminated
diagnostics. The six-epoch handoff was chosen after those values were visible,
so that entire staged run is also non-decision evidence. Their hashes remain in
`results.json` for audit; they must not guide another MHR/HSRD variant.

No private exact photo or 30 mm physical replay was run, and production was not
changed.

## Reproduction

Persistent auxiliary schedule, which reproduces the authoritative hold:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.benchmark.train_hsrd_mhr_auxiliary_control `
  --hsrd-corpus-root D:\path\to\hsrd100_lod1_source_gate_v4_n60 `
  --mhr-corpus-root D:\path\to\mhr_semantic_v2_training_n320 `
  --output-dir D:\path\to\output `
  --expected-hsrd-summary-sha256 06e0d1fb40fdb013785dfac0912c5e7666751f5cb8de2a1d2b3041cae5956861 `
  --expected-mhr-summary-sha256 d8717cced1d8f6753c15a7367dc1c84c5e62e0439f87b4380cca65f55fd665a5 `
  --epochs 12 --batch-size 2 --auxiliary-weight 0.25 `
  --loss-profile correlation-strict
```

The historical staged diagnostic adds `--auxiliary-epochs 6`, but it is closed
and not valid promotion evidence.

Raw RGB/depth targets, private images, feature tensors, and checkpoints are not
included in this compact evidence directory. HSRD is attributed to HSRD-100,
Digital Reality Lab (2025), under CC BY 4.0. MHR source/assets are pinned to
release v1.0.1 under Apache-2.0. Source geometry remains
training/evaluation-only.
