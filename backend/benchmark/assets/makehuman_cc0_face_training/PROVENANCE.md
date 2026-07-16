# MakeHuman CC0 face training fixture

This directory contains a deterministic derivative of the MakeHuman base mesh
and target files. The source checkout is pinned to:

- repository: <https://github.com/makehumancommunity/makehuman>
- commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`
- asset license: CC0 1.0 Universal, reproduced in `LICENSE.ASSETS.md`

The fixture contains 32 profiles organized as eight identity groups and four
expressions. All geometry is produced by applying the target paths and scales
recorded in `asset.json` to the pinned public base mesh. No private image, scan,
landmark, or biometric artifact is included.

Files:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `asset.json` | 81,399 | `607489fd20730452b10a5bd8a1c1e96999082d9c2b789d25c51a89261853c634` |
| `LICENSE.ASSETS.md` | 7,003 | `5f3ab0cf6f7ebe92efe4b83213131c617d308d164eeefd5da230373640b0c226` |
| `makehuman_cc0_face_training.npz` | 1,081,896 | `5da31444928a0d4d28522cd21dd4f5506946fdbb3c1436cabbbbea12ec054361` |

Regenerate from the exact source checkout:

```powershell
python -m backend.benchmark.makehuman_face_training_corpus `
  --source-root <makehuman-checkout> `
  --asset-dir backend/benchmark/assets/makehuman_cc0_face_training
```

The NPZ writer fixes member order, timestamps, permissions, compression, array
dtype, and array shape before hashing.
