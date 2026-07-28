import csv
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.benchmark.ingest_stl_results import (
    main as ingest_main,
    render_markdown,
    schema_v2_repair_telemetry_result,
    summarize_inputs,
)


CURRENT_METHOD = "triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
CANDIDATE_METHOD = "step1x3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"


def schema_v2_row(method: str, sample_id: str, *, metric: str = "signed-volume") -> dict:
    row = {
        "sample_id": sample_id,
        "method": method,
        "repair_fill_ratio_metric": metric,
        "repair_fill_ratio_surface_proxy_used": str(
            metric == "surface-component-unsigned-tetrahedra"
        ),
        "repair_fill_ratio_supported": "True",
        "repair_fill_ratio_change": "0.1",
        "repair_fill_ratio_relative_change": "0.2",
        "repair_fill_ratio_relative_change_abs": "0.2",
    }
    if metric == "signed-volume":
        row.update(
            {
                "raw_mesh_volume_fill_ratio": "0.5",
                "stl_volume_fill_ratio": "0.6",
            }
        )
    else:
        row.update(
            {
                "raw_mesh_surface_fill_ratio_comparison": "0.5",
                "stl_surface_fill_ratio_comparison": "0.6",
            }
        )
    return row


def write_result_run(
    run_dir: Path,
    *,
    include_current: bool = True,
    include_candidate: bool = True,
) -> Path:
    run_dir.mkdir(parents=True)
    methods = ["masked"]
    if include_current:
        methods.append(CURRENT_METHOD)
    if include_candidate:
        methods.append(CANDIDATE_METHOD)
    summary_fields = ["method", "success_rate", "masked_mae_median"]
    with (run_dir / "aggregate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        for index, method in enumerate(methods):
            writer.writerow(
                {
                    "method": method,
                    "success_rate": "1",
                    "masked_mae_median": str(0.5 - index * 0.1),
                }
            )
    rows = [{"sample_id": "a", "method": "masked"}]
    if include_current:
        rows.append(schema_v2_row(CURRENT_METHOD, "a"))
    if include_candidate:
        rows.append(
            schema_v2_row(
                CANDIDATE_METHOD,
                "a",
                metric="surface-component-unsigned-tetrahedra",
            )
        )
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "per_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


class SchemaV2RepairTelemetryIngestTest(unittest.TestCase):
    def test_accepts_both_canonical_schema_v2_metrics(self):
        rows = [
            schema_v2_row(CURRENT_METHOD, "a", metric="signed-volume"),
            schema_v2_row(
                CANDIDATE_METHOD,
                "a",
                metric="surface-component-unsigned-tetrahedra",
            ),
        ]

        current = schema_v2_repair_telemetry_result(rows, CURRENT_METHOD)
        candidate = schema_v2_repair_telemetry_result(rows, CANDIDATE_METHOD)

        self.assertTrue(current["passed"])
        self.assertTrue(candidate["passed"])

    def test_rejects_legacy_unsupported_and_nonfinite_telemetry(self):
        row = schema_v2_row(CURRENT_METHOD, "a")
        row.update(
            {
                "repair_fill_ratio_metric": "legacy-signed-volume-fallback",
                "repair_fill_ratio_supported": "False",
                "repair_fill_ratio_relative_change_abs": "nan",
            }
        )

        result = schema_v2_repair_telemetry_result([row], CURRENT_METHOD)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_sample_count"], 1)
        reasons = result["failed_samples"][0]["reasons"]
        self.assertIn("missing_or_noncanonical_metric", reasons)
        self.assertIn("unsupported", reasons)
        self.assertIn("nonfinite_repair_fill_ratio_relative_change_abs", reasons)

    def test_rejects_unsupported_row_even_when_canonical_values_are_finite(self):
        row = schema_v2_row(CURRENT_METHOD, "a")
        row["repair_fill_ratio_supported"] = "False"

        result = schema_v2_repair_telemetry_result([row], CURRENT_METHOD)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_samples"][0]["reasons"], ["unsupported"])

    def test_required_incumbent_and_candidate_are_checked_from_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_result_run(root / "run")
            archive = root / "results.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(run_dir, arcname="output/run")

            report = summarize_inputs(
                [str(archive)],
                output_dir=root / "ingested",
                required_schema_v2_repair_methods=(CURRENT_METHOD, CANDIDATE_METHOD),
            )
            markdown = render_markdown(report)

        self.assertTrue(report["required_schema_v2_repair_telemetry_passed"])
        checks = report["runs"][0]["required_schema_v2_repair_telemetry"]
        self.assertEqual([check["method"] for check in checks], [CURRENT_METHOD, CANDIDATE_METHOD])
        self.assertIn("Required Schema-V2 Repair Telemetry", markdown)
        self.assertIn(CURRENT_METHOD, markdown)

    def test_cli_writes_report_and_exits_two_when_required_method_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_result_run(root / "run", include_current=False)
            output_dir = root / "ingested"
            argv = [
                "ingest_stl_results",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--require-schema-v2-repair-method",
                CURRENT_METHOD,
                "--require-schema-v2-repair-method",
                CANDIDATE_METHOD,
            ]

            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
                ingest_main()

            report = json.loads(
                (output_dir / "stl_first_ingest_report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(report["required_schema_v2_repair_telemetry_passed"])
        current = report["runs"][0]["required_schema_v2_repair_telemetry"][0]
        self.assertEqual(current["failure_reasons"], {"missing_method_rows": 1})

    def test_missing_required_incumbent_fails_closed(self):
        result = schema_v2_repair_telemetry_result(
            [schema_v2_row(CANDIDATE_METHOD, "a")],
            CURRENT_METHOD,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_reasons"], {"missing_method_rows": 1})


if __name__ == "__main__":
    unittest.main()
