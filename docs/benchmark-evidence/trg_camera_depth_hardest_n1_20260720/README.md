# TRG camera-depth hardest-row smoke

Status: **hold**. Production is unchanged and the 30 mm STL replay was not run.

This bounded research smoke evaluated the official ECCV 2024 TRG model on the
privacy-safe `small_side_lit_shelves_256` MakeHuman row. TRG was selected after
screening DAD-3DHeads, DICE, and PerspNet/6DoF Face because it emits a posed
1,220-vertex face mesh in camera coordinates and has a current public source
release. The adapter preserves the official 192-pixel crop and 5,000-pixel
focal setting, strict-loads the pinned checkpoint, and does not modify upstream
source files.

## Provenance

- Source: `asw91666/TRG-Release`
- Source revision: `3916650722576599126b6242730a2210f597ac71`
- Checkpoint: `trg_240717/checkpoint-30/state_dict.bin`
- Checkpoint bytes: `150684140`
- Checkpoint SHA256: `75d745e5a0fe96f170cdce61ded64f080dfcbf45d3514a60db8f097fc758c77b`
- Source repository license: MIT
- Production eligibility: false. The checkpoint has no separate upstream
  terms, the training set is research-only, and one bundled config module
  retains an MPG proprietary header.

TRG's camera Z is **focal-conditioned, not calibrated metric depth** for an
arbitrary input camera. The official demo fixes focal length to 5,000 pixels;
the adapter records that fact and uses the output only as a structural prior.

## Measured result

The mesh registered correctly: projected face-bbox IoU was `0.793445`, actual
face-support coverage was `0.910185`, and all 2,304 triangles were free of
degeneracy before image-space rasterization. Inference took `0.327085 s` on the
local RTX 3080 Ti and peaked at `171689472` allocated CUDA bytes.

Fresh current-code baseline metrics were `0.806585` shape correlation,
`0.626915` gradient correlation, `0.175277` normalized RMSE, and 10 combined
named-part failures. These are a schema-current replay of the preserved GNM
artifact; they must not be substituted silently for its historical evidence.

| Blend | Shape | Gradient | RMSE | Named-part failures |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.806950 | 0.627019 | 0.175129 | 9 |
| 0.250 | 0.807276 | 0.626982 | 0.174997 | 9 |
| 0.500 | 0.807812 | 0.626491 | 0.174779 | 8 |

The 0.5 blend won the declared lexicographic sweep, but it failed promotion:

- aggregate gradient correlation regressed;
- not all six shape parts passed;
- not all six affine-mm parts passed;
- five of six parts regressed on at least one guarded shape, raw-gradient,
  RMSE, p95 error, bias, or span-retention metric;
- background pixels and one-pixel subject attachment remained bit-exact.

Only the nose passed every paired no-regression check. The lane is therefore
closed at one row, with no 30 mm STL replay and no production change.

## Harness changes

The new isolated adapter and evaluator add:

- revision-, size-, and checksum-pinned preflight;
- a no-source-edit compatibility shim for the bad ResNet LFS pointer and old
  `torchgeometry` rotation conversion;
- official crop/intrinsic reconstruction and full-image camera projection;
- runtime, peak VRAM, camera transform, fixed-focal provenance, projected bbox,
  raster coverage, and camera-Z telemetry;
- a three-strength structural-prior sweep against the current GNM artifact;
- six-part shape/raw-gradient and affine-mm no-regression checks;
- bit-exact background and attachment checks;
- face-support coverage rather than misleading padded-crop coverage.

The full local evidence file is 215,451 bytes with SHA256
`892219ea446a3f47f735eed8eb2514fc32757d2eca7fa63b8f0c36f5c8df0dc4`.
It remains under ignored `backend/output/research`; `results.json` contains the
compact publishable record.
