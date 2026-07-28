# STL Metric Schema v2 Replay

This bundle records the local migration from raw signed-volume repair drift to
a provenance-aware canonical fill metric, plus activation of the existing
paired Chamfer and H95 gates in every standard benchmark runner.

No GPU inference was run. The three selection replays use the exact committed
per-sample evidence from the promoted TripoSG run, the held Hunyuan d9.85 run,
and the rejected guarded TRELLIS smoke.

## Schema

Legacy fields remain unchanged:

- `raw_mesh_volume_fill_ratio`
- `repair_volume_fill_ratio_change`
- `repair_volume_fill_ratio_relative_change`
- `repair_volume_fill_ratio_relative_change_abs`

New runs also emit `repair_fill_ratio_*` as the canonical selector metric.
Its basis is explicit in `repair_fill_ratio_metric`:

- `signed-volume` is used only when raw and repaired meshes have a finite,
  closed, consistently wound, single-component signed-volume magnitude;
- `surface-component-unsigned-tetrahedra` is used when raw reliability is
  `unreliable` or `unknown`.

Raw reliability is tri-state through
`raw_mesh_volume_fill_ratio_reliability_status`: `reliable`, `unreliable`, or
`unknown`. The boolean reliability alias, reason, topology-assessed flag, and
self-intersection-assessed flag are also recorded. Dense raw meshes above the
bounded `250,000`-face topology assessment limit are `unknown`, not falsely
reported as known-bad. Self-intersection coverage is explicitly false when no
detector ran.

The surface proxy labels face components, uses each component bbox center as a
stable origin, sums unsigned triangle tetrahedron volumes, divides by the
global bbox volume, and records a clipped `[0,1]` display value. Canonical
drift uses the unclipped ratio so overlapping components cannot collapse to a
false zero-drift result. The proxy is orientation invariant, scale invariant,
deterministic, and bounded to at most `1,000,000` faces with `100,000`-face
working chunks. It is computed only for repaired rows whose raw signed-volume
reliability is not `reliable`; unsupported schema-v2 rows fail closed and are
never backfilled from legacy signed volume.

Selectors prefer canonical `repair_fill_ratio_*` values. Historical rows that
do not contain them derive the canonical value from legacy volume drift, so
old evidence remains readable. Existing gate IDs are preserved, while new
decision files declare `metric_schema_version: 2` and name both the canonical
and legacy fallback fields.

## Surface Gates

Paired worst-sample surface checks now default to `1.10x` for both Chamfer and
H95 in:

- standalone selection;
- `optimize_completion` and its resolved config;
- combined-run selection and metadata;
- the G4 orchestrator's eval and combine commands;
- generated Colab payload scripts and reports;
- the STL-first smoke runner.

The thresholds are command-line values, not experiment-list fields, so every
result records the policy that selected it. Enabled surface gates fail when
paired metrics are absent; missing coverage cannot silently remove a check.

## Fixture Results

| fixture | signed status | legacy fill | surface proxy | finding |
| --- | --- | ---: | ---: | --- |
| closed box | reliable | `1.0` | not needed | signed magnitude accepted |
| reversed closed box | reliable | `1.0` | not needed | orientation magnitude accepted |
| box missing one quad | unreliable | `0.5` | `0.833333` | open signed volume is unstable |
| perfect closure of open box | reliable | `1.0` | `1.0` forced pair | canonical drift `0.20` |
| thin closed box | reliable | `1.0` | not needed | no absolute-volume failure |
| uniformly tiny closed box | reliable | `1.0` | not needed | scale invariant |
| separated boxes | unreliable | `0.571429` | `0.571429` | fragmentation retained |
| separated boxes, mixed winding | unreliable | `0.0` | `0.571429` | signed cancellation removed |
| boxes touching at one vertex | unreliable | `0.25` | `0.25` | face components remain distinct |
| overlapping boxes | unreliable | `1.81818` | `1.0` display, `1.81818` comparison | clipping cannot erase drift |
| topology assessment over limit | unknown | `1.0` | `1.0` | unknown is distinct from unreliable |
| proxy over face limit | unreliable | n/a | unsupported | canonical metric fails closed |

`fixture_metrics.json` preserves the exact values.

## Historical Replay

| evidence | old | schema v2 | result |
| --- | --- | --- | --- |
| TripoSG inferred-bbox n10 | promote, `0` failures | hold, `10` failures | grandfathered incumbent; re-benchmark required |
| Hunyuan d9.85 n10 | hold, `3` failures | hold, `5` failures | unchanged decision; two hidden surface outliers exposed |
| guarded TRELLIS n1 | hold, `9` failures | hold, `11` failures | unchanged decision; failed output has no surface coverage |

Hunyuan's newly active worst-sample ratios are `1.68384x` Chamfer and
`1.60827x` H95, both above `1.10x`. Its original hull and fill failures remain.
TRELLIS remains a failed smoke with no candidate STL.

The promoted TripoSG archive predates repair-audit telemetry. Under schema v2
it cannot satisfy required canonical fill coverage, and its worst paired
Chamfer ratio versus mirror is `1.15871x` while H95 passes at `1.02136x`.
This does not silently remove the production incumbent: TripoSG remains
grandfathered until an exact schema-v2 confirmation run measures the missing
repair fields. It cannot be used as evidence for a new promotion without that
confirmation.

`replay_summary.csv` makes the migration classification explicit, and the six
selection files preserve every check and threshold from the schema-v2 replay.

## Validation

The focused geometry, migration, selector, package, combined-run, and G4
command tests pass. The complete benchmark regression module passes `209/209`.
The complete tracked backend suite passes `312/312` with six passing subtests;
the three warnings are pre-existing Pydantic namespace and flat-mesh warnings.
Source geometry remains diagnostic-only throughout the replay.
