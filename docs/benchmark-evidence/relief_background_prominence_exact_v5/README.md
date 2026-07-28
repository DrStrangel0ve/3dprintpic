# Exact-input background prominence v5

This bounded replay checks the `0.65` selected-scene background budget on the
two exact 30 mm inputs that originally exposed the face-collapse and shallow
llama-background failures. It ran from a clean worktree at exact revision
`535daf3cacbf3d57c2474240322290d06f8296c4` with the same Depth Anything V2
Large, CUDA, relief, selection, face, slope, and printability settings as the
tracked `0.45` baseline. Each comparison reuses the same private source,
selection mask, selected region, and camera framing; only the implementation
and requested background ratio differ.

Only aggregate telemetry is retained here. The private source images, masks,
depth arrays, API responses, previews, and meshes are excluded from git.

## Result

- Portrait background centered RMS increases from `4.6482` to `6.8165` mm
  (`1.466x`), and p02-p98 span increases from `13.3445` to `19.5000` mm
  (`1.461x`).
- Llama-group background centered RMS increases from `6.5078` to `7.8375` mm
  (`1.204x`), and p02-p98 span increases from `21.6562` to `22.0282` mm.
  This is a cap-limited redistribution, not a proportional `0.65 / 0.45`
  expansion claim.
- Portrait face-component correlations improve from
  `0.9434`/`0.9483` to `0.9523`/`0.9712`; RMS retention improves from
  `0.9315`/`0.9684` to `0.9498`/`0.9736`.
- All four llama selected-component correlations and RMS-retention values
  improve. The new minima are `0.9149` and `0.6169`, up from `0.7527` and
  `0.2378`.
- Both new outputs retain complete background coverage and pass the enforced
  depth, gradient, far-cap, feasible-attachment, and emission checks.
- Both STLs are one watertight, manifold, winding-consistent positive volume
  with zero degenerate faces.

The strict attachment flag remains false because mutually incompatible
one-pixel boundary constraints are reported rather than hidden (48 portrait,
140 llama-group). Every satisfiable attachment step is at most `0.8000002` mm,
and the far-background cap has zero violation. This is accepted under the
existing permissive-emission policy as a measured confirmation on these exact
fixtures, not as proof of strict feasibility or universal proportional depth
gain. No additional algorithm change is needed for the measured failure.
