# Face fusion training slice (`aa77abc`, n=15)

Status: **all tested fusion candidates hold**. The deterministic privacy-safe
matrix and its exact supervision are retained for subsequent training work, but
none of the Pixel3DMM or learned residual experiments is enabled in production.

## Experimental design

The run used exact clean revision
`aa77abcc4d73b2a617875b3510d6385cac765aaf` and 15 generated MakeHuman CC0
scenes. Each of the three identities spans:

- 256 and 384 pixel inputs;
- small, medium, and close framing;
- negative, frontal, and positive yaw;
- all three backgrounds and all three lighting profiles across the matrix.

The matrix is disjoint from the existing varied-context row identifiers and
rejects close semantic duplicates of those held-out scenes. The three existing
small-face, eyewear, and strong-turn rows were not used to choose parameters.

The live run completed 15/15 rows. Producer and server revisions matched, the
clean worktree stayed unchanged, and every paired background-detail check
passed. The overall run intentionally reports hold because exact facial-part
gates are stricter than the global face correlation gates.

## Baseline finding

The current candidate recorded 77 combined named-part failures over 15 rows.
Median global metrics were:

| Metric | Median |
| --- | ---: |
| Shape correlation | 0.944540 |
| Gradient correlation | 0.704562 |
| Normalized RMSE | 0.107163 |

Apparent face size is a strong failure axis. The closest frontal row (264 px
face height) passed every named-part check at 0.984984 shape correlation. The
small turned rows at 74-75 px had 9-11 combined part failures and shape
correlation between 0.734025 and 0.884611. The crossed 256/384 design shows this
is apparent face size and pose, not input resolution alone.

## Pixel3DMM probe

The probe used the [official Pixel3DMM repository](https://github.com/SimonGiebenhain/pixel3dmm)
at `fcd1fa973c7715b02a8948dfc679dff53cf85924` and its public normal checkpoint:

- checkpoint SHA256: `e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da`;
- device: NVIDIA GeForce RTX 3080 Ti;
- 15-row inference: 4.803 seconds after model load;
- peak allocated VRAM: 3.585 GiB.

The official model predicts normals in FLAME coordinates. Rotating them by the
known rendered yaw improved median horizontal normal/exact-gradient correlation
from 0.305 to 0.392, but vertical agreement remained weak.

## Candidate decisions

### Pose-rotated normal integration

A 320-candidate bounded search varied screened-Poisson weight, normal scale,
blend, and correction cap. The best detailed candidate increased combined
training-slice failures from 77 to 78. It is a hold.

### Compact residual U-Net

A compact 128 px residual U-Net was trained by whole scene, comparing RGB plus
current depth against RGB, depth, and Pixel3DMM normals. On the three internal
validation scenes, the normal input won narrowly (35 versus 36 failures), but
both were much worse than the unchanged baseline (11 failures). After retraining
on all 15 rows, the untouched varied-context faces had 34 failures versus the
current baseline's 19.

A separate 72-candidate low-amplitude/low-frequency blend sweep selected a zero
blend. This lane is closed rather than shipping a nominal no-op model.

### Production-native face fusion

The existing MediaPipe/local-depth fusion was replayed from cached local depth
with 36 combinations of detail strength and correction ratio. Internal
validation selected detail strength 2.0 and correction ratio 0.04, tying the
baseline at 11 failures. On the untouched varied-context rows it regressed from
19 to 21 failures, primarily around eyewear. The production defaults remain
unchanged.

## Conclusion

Post-hoc normal integration, a small synthetic residual model, and wider tuning
of the current fusion do not solve the small turned-face failure without
damaging already-correct facial parts. The next face provider should emit
camera-aligned geometry or depth directly, or the depth model should be trained
on a substantially broader identity/expression corpus. Future work should keep
this exact 15-row slice for development and the existing three face rows for
one-time held-out confirmation.

## Reproduction

```powershell
python -m backend.benchmark.run_cc0_live_face_variation_matrix `
  --clean-server-repository <clean-revision-worktree> `
  --matrix face-fusion-train `
  --output-dir backend/output/cc0_face_fusion_train_aa77abc_n15 `
  --base-url http://127.0.0.1:8015
```

The expected command exit is nonzero because named facial-part gates
intentionally hold the baseline. `results.json` contains the compact row metrics
and hashes for the ignored raw evidence.
