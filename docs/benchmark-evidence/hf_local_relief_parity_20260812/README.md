# Hugging Face/local selected-relief parity (2026-08-12)

## Failure

The Hugging Face Space and local frontend both called `POST /process_image`,
but their selected-object request contracts differed. The local frontend used
full-scene context with a locked subject. The Space requested
`source-depth-isolate`, disabled subject locking, forced background depth to
zero, cropped to the selection bounds, and replaced every unselected depth
sample with `NaN`.

That isolated route could still emit a formally watertight STL because the
mesher closed every new mask boundary down to the base. Visually, however, those
closures were mask-shaped through-holes between arms, torsos, and nearby scene
regions. Watertightness alone therefore did not catch the user-visible failure.

## Selected correction

The Space now sends the local selected-relief contract verbatim for the fields
that control geometry:

| Field | Hosted value |
|---|---:|
| `selection_mode` | `context` |
| `selection_subject_lock` | `true` |
| `selection_background_depth_ratio` | not submitted; shared backend default `0.65` |
| `device` | `auto` |
| `invert` | `false` |
| `relief_polarity` | `raised-print` |
| `completion_mode` | not submitted; shared backend default `none` |
| `mesh_resolution_multiplier` | local preset (`2.0` default) |
| `printer_profile` | `Bambu Lab P1S` |
| `printer_max_x/y/z_mm` | `256/256/256` |
| `printer_clearance_mm` | `0` |
| `nozzle_diameter_mm` | `0.4` |
| `minimum_feature_mm` | `0.8` |
| `base_thickness_mm` | local control (`2.4` default) |

Depth and face refinement operate on the complete original photograph. The
selection mask remains a subject-protection signal inside the common relief
postprocessor; it is no longer used as a validity mask. The mesh receives a
full finite rectangular surface, exactly as the local frontend does.

The hosted mesh-detail control is restricted to the local production choices:
384 samples (`1.5x`), 512 (`2x`, default), 768 (`3x`), and the backend-capped
900 (`4x`). Unknown budgets fail closed instead of creating a hosted-only
profile.

The Space's physical controls also use the local defaults and domains: 100% of
the 256 mm Bambu P1S edge, 10 mm relief height, 2.4 mm backing thickness, and
print-scale steps of 5% between 10% and 100%. The hosted-only Background depth
slider was removed; both clients now rely on the same backend default.

The Space wrapper fails closed if the backend reports a different mode, a
selection crop, a disabled subject lock, or any context method other than
`full_scene_subject_locked_background_v1`.

## Regression gates

- The Space request test checks the exact local multipart field set, context
  mode, subject lock, inherited background-depth default, device,
  polarity/inversion, full-image dimensions, and summary provenance.
- The backend subject-lock fixture enables production top-background trimming,
  requires the selected surface to retain the whole-scene surface shape, and
  requires its finite/non-finite topology mask to match the full-scene output
  exactly. Selection therefore cannot introduce any new interior void.
- The existing fixture also requires exact subject-interior agreement with the
  unselected full-scene surface plus a watertight, consistently wound STL.

No private photo, mask, depth array, or STL is included in this evidence bundle.
