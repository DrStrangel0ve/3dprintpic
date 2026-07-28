from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.select_completion_candidate import json_safe


RESULTS_SUMMARY_MARKER = "---RESULTS_SUMMARY_JSON---"
RESULT_ARCHIVES_MARKER = "---RESULT_ARCHIVES_JSON---"


def parse_json_after_marker(text: str, marker: str, default):
    index = text.rfind(marker)
    if index < 0:
        return default, False
    payload = text[index + len(marker) :].lstrip()
    if not payload:
        return default, True
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return default, True
    return value, True


def parse_colab_output(text: str) -> dict:
    summary, summary_marker_found = parse_json_after_marker(text, RESULTS_SUMMARY_MARKER, {})
    archives, archives_marker_found = parse_json_after_marker(text, RESULT_ARCHIVES_MARKER, [])
    if not isinstance(summary, dict):
        summary = {"unexpected_results_summary": summary}
    if isinstance(archives, dict):
        archives = [archives]
    elif not isinstance(archives, list):
        archives = [{"unexpected_result_archives": archives}]
    return {
        "results_summary": summary,
        "result_archives": archives,
        "markers_found": {
            "results_summary": summary_marker_found,
            "result_archives": archives_marker_found,
        },
    }


def extract_colab_output(input_path: Path, output_dir: Path) -> dict:
    text = input_path.read_text(encoding="utf-8")
    parsed = parse_colab_output(text)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_output = output_dir / "colab_output.txt"
    summary_path = output_dir / "results_summary.json"
    archives_path = output_dir / "result_archives.json"
    report_path = output_dir / "colab_output_extract_report.json"

    copied_output.write_text(text, encoding="utf-8")
    summary_path.write_text(
        json.dumps(json_safe(parsed["results_summary"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archives_path.write_text(
        json.dumps(json_safe(parsed["result_archives"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = parsed["results_summary"]
    archives = parsed["result_archives"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "copied_output": str(copied_output),
        "results_summary": str(summary_path),
        "result_archives": str(archives_path),
        "markers_found": parsed["markers_found"],
        "run_name": summary.get("run_name", ""),
        "run_status": summary.get("run_status", None),
        "provider_setup_only": summary.get("provider_setup_only", None),
        "archive_count": len(archives),
        "archive_paths": [str(item.get("path", "")) for item in archives if isinstance(item, dict) and item.get("path")],
    }
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract summary JSON blocks printed by package_colab_inputs Colab launcher cells."
    )
    parser.add_argument("input", help="Text file containing copied Colab cell or terminal output.")
    parser.add_argument("--output-dir", default=None, help="Directory for extracted results_summary.json and archive metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.stem}_colab_extract")
    report = extract_colab_output(input_path, output_dir)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
