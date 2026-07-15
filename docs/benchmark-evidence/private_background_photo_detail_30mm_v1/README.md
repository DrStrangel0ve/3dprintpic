# Exact private background-detail replay

This evidence validates the current `0.60 mm` face-protected photo-detail
algorithm under the exact 30 mm, 512-grid request configuration used for the
two motivating scenes. It does not claim coverage of every endpoint default.
Only aggregate measurements are tracked. The source photos, semantic masks, cached
depth, emitted surfaces, STL files, local paths, job identifiers, and content
hashes remain under gitignored `backend/output/`.

## Contract

- Relief height: `30 mm`
- Footprint: `128 mm`
- Selection background depth ratio: `0.65`
- Compared detail levels: `0.00` and `0.60 mm`
- One cached composed depth is reused within each pair.
- The exact production selection-plus-face union is excluded from background scoring.
- The available Scene 01 face region is independently checked.
- Photo evidence follows the recorded resize, horizontal flip, mesh resample,
  and crop transform before correlation is measured.

The ignored config and every private input are locally checksum-pinned. Selection
fingerprints, selection metadata, the cached request metadata, and the expected
Depth Anything V2 model identity must agree before emission. No private checksum
or identifier is copied into tracked evidence.

The run used clean implementation revision
`0d654c4cba69fb911ecf6c38fb3362e42855f278`.

## Results

| Scene | Intended-background correlation | RMS detail | p95 detail | Source-aligned capture | Subject boundary max | Face max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scene 01 | 0.6141 | 0.0512 mm | 0.1184 mm | 0.5430 | 0.0146 mm | 0.000002 mm |
| Scene 02 | 0.6296 | 0.0784 mm | 0.1843 mm | 0.6076 | 0.0129 mm | n/a |

Both baseline/candidate pairs pass requested/effective detail telemetry,
background preservation, physical cap, every feasible attachment constraint,
single-component watertight/manifold/winding/volume topology, zero degenerates,
and complete-shell facet agreement. Candidate broad-background correlation is
`0.999994` for Scene 01 and `0.999921` for Scene 02; RMS retention is
`1.000051` and `1.000088`, respectively.

Strict attachment status remains false because Scenes 01 and 02 contain `13`
and `41` mutually incompatible one-pixel constraints. Those
conflicts are reported rather than hidden. Far-background cap violation is zero,
and every satisfiable attachment jump is at most `0.800001 mm`.

## Reproduction

Create a private JSON config under an ignored output directory. Each scene needs
a non-semantic `scene-NN` label plus checksum-pinned source, depth, selection,
selection metadata, cached request metadata, and expected model identity.
Face-aware scenes can also provide `face_region_mask`, `feature_weight_mask`,
`face_metadata`, and `expected_face_count`.

```powershell
.\backend\.venv\Scripts\python.exe `
  -m backend.benchmark.run_private_background_photo_detail_replay `
  --scene-config backend/output/<private-run>/scene_config.json `
  --output-dir backend/output/<private-run>/artifacts `
  --aggregate-summary docs/benchmark-evidence/<run>/summary.json
```

The runner refuses a config, private input, or output directory not covered by
`.gitignore` and audits the aggregate summary for private paths, image/mesh
filenames, content hashes,
and private job identifiers before writing tracked evidence.
