# High-relief face and background confirmation

This evidence confirms the uniform `0.25` high-relief face screening term and
the accepted-reference background fallback on implementation commit
`53d37b08672519dfb1cd9c1b6b27c673a416e0c7`.

## Privacy-safe matrix

The deterministic matrix covers four scene and mask topologies at 20, 30, and
40 mm. All 12 rows pass face, selected-surface, background, physical-cap,
complete-shell, and printability gates.

- Minimum face normal mean cosine: `0.997599`.
- Minimum face relighting correlation: `0.990004`.
- Maximum face p95 normal angle: `7.7202` degrees.
- Minimum background depth correlation: `0.999973`.
- Minimum selected-surface relighting correlation: `0.924550`.
- Maximum selected-solver cardinal p99/max/diagonal max ratios:
  `10.3589` / `17.1045` / `12.0948`, below `12` / `24` / `24`.
- Cross-height minimum shape/gradient correlations: `0.999857` / `0.998895`;
  maximum normalized shape RMSE: `0.005851`.
- Every reopened STL exactly matches its full top, bottom, and wall shell and is
  one watertight, winding-consistent volume with zero nonmanifold edges or
  degenerate faces.

The clean run took `15.246` seconds. `summary.json` is 236,437 bytes with
SHA256 `7395e00dd5392f2bdf679f3173f307aaaf80073069baabdaca2df1bd2352b7f2`.
Generated NPY and STL artifacts remain under ignored `backend/output` paths;
their sizes and hashes are recorded in the summary.

## Exact private replay

Only aggregate metrics are tracked in `exact-private-summary.json`. The source
photo, masks, depth arrays, renders, and meshes remain local and ignored.

The weaker face's worst-light correlation improves from `0.8452` to `0.8623`
at 20 mm and from `0.8194` to `0.8693` at 30 mm. At 40 mm, the old `0.05`
candidate failed its per-face detail gate and the fallback surface fell to
`0.2338`; the uniform `0.25` solve passes at `0.8964`. All three heights retain
background depth correlation above `0.999997`, 100% coverage, and exact
309,440-facet printable shells.

The exact 20-to-40 gradient-consistency diagnostic is `0.967754`, narrowly
below its `0.97` threshold, while 20-to-30 and 30-to-40 pass. This is reported,
not promoted away: at 40 mm the selection-wide candidate reaches an edge ratio
of `30.5811` and correctly fails closed, leaving the independently accepted
face and background surface. Per-height face appearance, physical caps, and
printability all pass.

Run the privacy-safe matrix from the repository root:

```powershell
python -m backend.benchmark.run_relief_visual_sweep `
  --output-dir backend/output/relief_visual_sweep_uniform025_clean `
  --summary-path docs/benchmark-evidence/relief_visual_sweep_face025_n12/summary.json
```
