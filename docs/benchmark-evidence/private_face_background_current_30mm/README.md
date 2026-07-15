# Exact-current private face and background replay at 30 mm

This bounded replay checks the current face-refinement and background-relief
path on the two retained inputs that originally exposed the portrait collapse
and shallow llama-background failures. It ran from a detached clean worktree at
implementation revision `4549b28eae7864799ab4adb0fc4d4a7499fae99b`.

Only aggregate telemetry is tracked. Source images, masks, identifiers, input
hashes, depth arrays, API responses, previews, scene configs, and meshes remain
under ignored output.

## Result

- The direct retained-artifact replay passes both the portrait and llama rows.
  Its background photo-detail, physical-cap, topology, and complete-shell
  measurements are numerically identical to the earlier Euclidean-guard
  baseline.
- The live portrait request detects and refines both faces through the current
  YuNet-guided eyewear path. Face normal mean cosine is `0.991031`, p95 normal
  error is `7.9510` degrees, minimum relighting correlation is `0.916314`, and
  face detail correlation/RMS retention are `0.965541`/`0.947619`.
- The weakest measured face component retains `0.959306` detail correlation
  and `0.930945` RMS. Coverage is complete and the face height-span ratio is
  `0.997264`.
- Live portrait background depth correlation, gradient correlation, and RMS
  retention are `0.99999448`, `0.99925874`, and `1.00005102`.
- Intended-background photo-detail correlation is `0.611473` for the portrait
  and `0.635361` for the llama row. Source-aligned capture is
  `0.568383`/`0.617403`.
- Both direct outputs and the live portrait are one-component watertight,
  manifold, winding-consistent positive volumes with zero degenerate faces.
  Their serialized STL shells agree with the emitted heightfields exactly.
- Far-background and every satisfiable attachment constraint pass. Maximum
  feasible attachment jump is `0.8000002 mm`; incompatible one-pixel boundary
  constraints remain reported separately instead of being hidden.

The first comparison against the historical byte-exact face artifacts failed
only the exact-array checks because the current detector now refines both
eyewear regions. All independently computed appearance, background, cap,
topology, and shell gates still passed. A new ignored direct reference was then
built from the current live face-refined depth, without changing thresholds or
reusing the emitted surface. The final same-revision comparison passes exact
refined depth, face masks, composed context depth, reference surface, and every
downstream gate. No algorithm change is justified by this replay.

## Evidence

- `direct_summary.json` contains the two-scene retained-artifact replay. SHA256:
  `e97a9162b8f3e5b0518fb39b928560fe3025f97d8141130d2549bbe9a3f8f1be`.
- `live_portrait_summary.json` contains the exact-current live API replay.
  SHA256: `ff4d5430c1f1048838fa40fc68f94d18e92d86a6b26ac0f6701f1c6b038b3e7f`.

The summaries are produced by the existing omission-aware private replay and
summarizer. Reproduction requires the locally retained checksum-pinned bundle,
a clean server for the implementation revision, and the fixed `512` grid,
`30 mm` relief height, `128 mm` physical size, and sigma `0.35` controls. The
live launcher deliberately omits the background fields so endpoint defaults of
`0.60 mm` photo detail and `0.65` background depth ratio are exercised.
