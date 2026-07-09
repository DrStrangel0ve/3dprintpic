from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_PATH_FIELDS = (
    "full_image",
    "masked_image",
    "mask",
    "gt_depth",
    "gt_silhouette",
    "silhouette",
    "mesh",
    "asset_path",
)
DEFAULT_EXTRACT_ROOT = "/content/3dprintpic_colab_inputs"
DEFAULT_REPO_REMOTE = "https://github.com/DrStrangel0ve/3dprintpic.git"
DEFAULT_REPO_REF = "codex/3d-completion-benchmark-g4"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict], start_index: int, limit: int | None) -> list[dict]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    end_index = None if limit is None else start_index + limit
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    return rows[start_index:end_index]


def clean_parts(path: Path) -> list[str]:
    parts = []
    for part in path.parts:
        cleaned = part.replace(":", "")
        if cleaned in {"", ".", "..", "\\", "/"}:
            continue
        parts.append(cleaned)
    return parts


def archive_path_for(source: Path, root: Path, prefix: str) -> Path:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(*clean_parts(source))
    return Path(prefix, *clean_parts(relative))


def resolve_input_path(value: str, manifest_dir: Path, root: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [root / raw, manifest_dir / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def add_file_once(tar: tarfile.TarFile, source: Path, archive_name: Path, added: set[Path]) -> None:
    resolved = source.resolve()
    if resolved in added:
        return
    if not source.exists():
        raise FileNotFoundError(f"Referenced file does not exist: {source}")
    tar.add(source, arcname=archive_name.as_posix())
    added.add(resolved)


def colab_path(extract_root: str, archive_name: Path) -> str:
    return f"{extract_root.rstrip('/')}/{archive_name.as_posix()}"


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def rewrite_manifest_rows(
    rows: Iterable[dict],
    *,
    manifest_dir: Path,
    root: Path,
    path_fields: Iterable[str],
    extract_root: str,
) -> tuple[list[dict], dict[Path, Path]]:
    source_to_archive: dict[Path, Path] = {}
    rewritten_rows = []
    path_field_set = set(path_fields)
    for row in rows:
        rewritten = dict(row)
        for field in path_field_set:
            value = row.get(field)
            if not value:
                continue
            source = resolve_input_path(str(value), manifest_dir, root)
            archive_name = source_to_archive.get(source.resolve())
            if archive_name is None:
                archive_name = archive_path_for(source, root, "inputs/files")
                source_to_archive[source.resolve()] = archive_name
            rewritten[field] = colab_path(extract_root, archive_name)
        rewritten_rows.append(rewritten)
    return rewritten_rows, source_to_archive


def write_rewritten_manifest(tar: tarfile.TarFile, rows: list[dict]) -> Path:
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest_info = tarfile.TarInfo("inputs/manifest.jsonl")
    manifest_bytes = manifest_text.encode("utf-8")
    manifest_info.size = len(manifest_bytes)
    tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_bytes))
    return Path("inputs/manifest.jsonl")


def add_text_file(tar: tarfile.TarFile, archive_name: str, text: str, mode: int = 0o644) -> None:
    encoded = text.encode("utf-8")
    info = tarfile.TarInfo(archive_name)
    info.mode = mode
    info.size = len(encoded)
    tar.addfile(info, fileobj=io.BytesIO(encoded))


def add_lora(tar: tarfile.TarFile, lora_path: Path, root: Path, added: set[Path]) -> Path | None:
    if not lora_path:
        return None
    if not lora_path.exists():
        raise FileNotFoundError(f"--lora-weights does not exist: {lora_path}")
    lora_base = Path("lora") / lora_path.stem if lora_path.is_file() else Path("lora") / lora_path.name
    if lora_path.is_file():
        archive_name = lora_base / lora_path.name
        add_file_once(tar, lora_path, archive_name, added)
        report_path = lora_path.with_name("training_report.json")
        if report_path.exists():
            add_file_once(tar, report_path, lora_base / "training_report.json", added)
        return archive_name

    for child in sorted(path for path in lora_path.rglob("*") if path.is_file()):
        archive_name = lora_base / child.relative_to(lora_path)
        add_file_once(tar, child, archive_name, added)
    return lora_base


def build_colab_run_script(
    *,
    archive_filename: str,
    extract_root: str,
    manifest_path: str,
    lora_path: str,
    colab_archive_path: str | None,
    colab_repo_dir: str,
    repo_remote: str,
    repo_ref: str,
    run_name: str,
    eval_starts: Iterable[int],
    eval_limit: int,
    score_profile: str,
    train_steps: int,
    require_modern_cache: bool,
    cache_download_mode: str,
    cache_max_workers: int,
    min_paired_n: int,
) -> str:
    archive_default = colab_archive_path or f"/content/{archive_filename}"
    command = [
        "python",
        "-u",
        "-m",
        "backend.benchmark.colab_g4_orchestrator",
        "--use-current-repo",
        "--run-name",
        run_name,
        "--stage",
        "cache",
        "--stage",
        "eval",
        "--stage",
        "combine",
        "--manifest",
        manifest_path,
        "--existing-lora-weights",
        lora_path,
        "--train-steps",
        str(train_steps),
        "--eval-limit",
        str(eval_limit),
        "--score-profile",
        score_profile,
        "--cache-download-mode",
        cache_download_mode,
        "--cache-max-workers",
        str(cache_max_workers),
        "--min-paired-n",
        str(min_paired_n),
    ]
    for start in eval_starts:
        command.extend(["--eval-start", str(start)])
    if require_modern_cache:
        command.append("--require-modern-cache")

    lora_adapter_path = colab_path(lora_path, Path("pytorch_lora_weights.safetensors"))
    lora_report_path = colab_path(lora_path, Path("training_report.json"))
    preflight_path = colab_path(extract_root, Path("launch_preflight.json"))
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"ARCHIVE_PATH=\"${{1:-{archive_default}}}\"\n"
        f"EXTRACT_ROOT=\"${{EXTRACT_ROOT:-{extract_root}}}\"\n"
        f"REPO_DIR=\"${{REPO_DIR:-{colab_repo_dir}}}\"\n"
        f"REPO_REMOTE=\"${{REPO_REMOTE:-{repo_remote}}}\"\n"
        f"REPO_REF=\"${{REPO_REF:-{repo_ref}}}\"\n\n"
        f"MANIFEST_PATH={shell_join([manifest_path])}\n"
        f"LORA_PATH={shell_join([lora_path])}\n"
        f"LORA_ADAPTER_PATH={shell_join([lora_adapter_path])}\n"
        f"LORA_REPORT_PATH={shell_join([lora_report_path])}\n"
        f"PREFLIGHT_PATH={shell_join([preflight_path])}\n"
        "export ARCHIVE_PATH EXTRACT_ROOT REPO_DIR REPO_REMOTE REPO_REF MANIFEST_PATH LORA_PATH LORA_ADAPTER_PATH LORA_REPORT_PATH PREFLIGHT_PATH\n"
        "mkdir -p \"$EXTRACT_ROOT\"\n"
        "if [[ -n \"${EXPECTED_SHA256:-}\" ]]; then\n"
        "  actual_sha=\"$(sha256sum \"$ARCHIVE_PATH\" | awk '{print $1}')\"\n"
        "  if [[ \"$actual_sha\" != \"$EXPECTED_SHA256\" ]]; then\n"
        "    echo \"archive sha256 mismatch: expected $EXPECTED_SHA256 got $actual_sha\" >&2\n"
        "    exit 2\n"
        "  fi\n"
        "fi\n"
        "tar -xzf \"$ARCHIVE_PATH\" -C \"$EXTRACT_ROOT\"\n"
        "test -s \"$MANIFEST_PATH\"\n"
        "test -s \"$LORA_ADAPTER_PATH\"\n"
        "test -s \"$LORA_REPORT_PATH\"\n"
        "manifest_rows=\"$(python - <<'PY'\n"
        "import os, pathlib\n"
        "manifest = pathlib.Path(os.environ['MANIFEST_PATH'])\n"
        "print(sum(1 for line in manifest.read_text().splitlines() if line.strip()))\n"
        "PY\n"
        ")\"\n"
        "if [[ \"$manifest_rows\" -lt 1 ]]; then\n"
        "  echo \"rewritten manifest has no rows\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "if [[ ! -d \"$REPO_DIR/.git\" ]]; then\n"
        "  rm -rf \"$REPO_DIR\"\n"
        "  git clone --filter=blob:none \"$REPO_REMOTE\" \"$REPO_DIR\"\n"
        "fi\n"
        "cd \"$REPO_DIR\"\n"
        "git fetch \"$REPO_REMOTE\" \"$REPO_REF\"\n"
        "git reset --hard FETCH_HEAD\n"
        "resolved_commit=\"$(git rev-parse HEAD)\"\n"
        "python - <<'PY' > \"$PREFLIGHT_PATH\"\n"
        "import json, os, pathlib, subprocess\n"
        "archive = pathlib.Path(os.environ['ARCHIVE_PATH'])\n"
        "manifest = pathlib.Path(os.environ['MANIFEST_PATH'])\n"
        "payload = {\n"
        "    'archive': str(archive),\n"
        "    'archive_sha256': subprocess.check_output(['sha256sum', str(archive)], text=True).split()[0],\n"
        "    'repo_dir': os.environ['REPO_DIR'],\n"
        "    'repo_remote': os.environ['REPO_REMOTE'],\n"
        "    'repo_ref': os.environ['REPO_REF'],\n"
        "    'resolved_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),\n"
        "    'extract_root': os.environ['EXTRACT_ROOT'],\n"
        "    'manifest': str(manifest),\n"
        "    'manifest_rows': sum(1 for line in manifest.read_text().splitlines() if line.strip()),\n"
        "    'lora_path': os.environ['LORA_PATH'],\n"
        "    'lora_adapter': os.environ['LORA_ADAPTER_PATH'],\n"
        "}\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "PY\n"
        f"{shell_join(command)}\n"
    )


def package_inputs(
    *,
    manifest: Path,
    output: Path,
    extract_root: str,
    root: Path,
    start_index: int = 0,
    limit: int | None = None,
    path_fields: Iterable[str] = DEFAULT_PATH_FIELDS,
    lora_weights: Path | None = None,
    run_script_path: Path | None = None,
    include_run_script: bool = False,
    colab_archive_path: str | None = None,
    colab_repo_dir: str = "/content/3dprintpic",
    repo_remote: str = DEFAULT_REPO_REMOTE,
    repo_ref: str = DEFAULT_REPO_REF,
    run_name: str = "g4_modelnet10_weighted_surface_eval_s20",
    eval_starts: Iterable[int] = (40, 50),
    eval_limit: int = 10,
    score_profile: str = "object-surface",
    train_steps: int = 20,
    require_modern_cache: bool = True,
    cache_download_mode: str = "snapshot",
    cache_max_workers: int = 8,
    min_paired_n: int = 5,
    report_path: Path | None = None,
) -> dict:
    if not manifest.exists():
        raise FileNotFoundError(f"--manifest does not exist: {manifest}")
    rows = load_jsonl(manifest)
    selected = select_rows(rows, start_index, limit)
    rewritten_rows, source_to_archive = rewrite_manifest_rows(
        selected,
        manifest_dir=manifest.parent,
        root=root,
        path_fields=path_fields,
        extract_root=extract_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    added: set[Path] = set()
    with tarfile.open(output, "w:gz") as tar:
        manifest_archive_path = write_rewritten_manifest(tar, rewritten_rows)
        for source_resolved, archive_name in sorted(source_to_archive.items(), key=lambda item: item[1].as_posix()):
            add_file_once(tar, source_resolved, archive_name, added)
        lora_archive_path = add_lora(tar, lora_weights, root, added) if lora_weights else None
        run_script_text = ""
        if include_run_script:
            if lora_archive_path is None:
                raise ValueError("--include-run-script requires --lora-weights")
            run_script_text = build_colab_run_script(
                archive_filename=output.name,
                extract_root=extract_root,
                manifest_path=colab_path(extract_root, manifest_archive_path),
                lora_path=colab_path(extract_root, lora_archive_path),
                colab_archive_path=colab_archive_path,
                colab_repo_dir=colab_repo_dir,
                repo_remote=repo_remote,
                repo_ref=repo_ref,
                run_name=run_name,
                eval_starts=eval_starts,
                eval_limit=eval_limit,
                score_profile=score_profile,
                train_steps=train_steps,
                require_modern_cache=require_modern_cache,
                cache_download_mode=cache_download_mode,
                cache_max_workers=cache_max_workers,
                min_paired_n=min_paired_n,
            )
            add_text_file(tar, "run_colab_eval.sh", run_script_text, mode=0o755)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "extract_root": extract_root,
        "rewritten_manifest": colab_path(extract_root, manifest_archive_path),
        "input_rows": len(rows),
        "packaged_rows": len(rewritten_rows),
        "start_index": start_index,
        "limit": limit,
        "referenced_files": len(source_to_archive),
        "archive_files": len(added),
        "path_fields": list(path_fields),
        "lora_weights": str(lora_weights) if lora_weights else "",
        "rewritten_lora_weights": colab_path(extract_root, lora_archive_path) if lora_archive_path else "",
        "run_script_in_archive": "run_colab_eval.sh" if include_run_script else "",
    }
    report_path = report_path or output.with_name(output.name + ".report.json")
    report["report"] = str(report_path)
    if run_script_path and include_run_script:
        run_script_path.write_text(run_script_text, encoding="utf-8")
        report["run_script"] = str(run_script_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package benchmark manifest assets and optional LoRA weights for Colab.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path-field", action="append", default=None)
    parser.add_argument("--lora-weights", default=None)
    parser.add_argument("--include-run-script", action="store_true")
    parser.add_argument("--run-script", default=None)
    parser.add_argument("--colab-archive-path", default=None)
    parser.add_argument("--colab-repo-dir", default="/content/3dprintpic")
    parser.add_argument("--repo-remote", default=DEFAULT_REPO_REMOTE)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--run-name", default="g4_modelnet10_weighted_surface_eval_s20")
    parser.add_argument("--eval-start", type=int, action="append", default=None)
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--score-profile", choices=("default", "object-surface"), default="object-surface")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--no-require-modern-cache", action="store_true")
    parser.add_argument("--cache-download-mode", choices=("files", "snapshot"), default="snapshot")
    parser.add_argument("--cache-max-workers", type=int, default=8)
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root) if args.repo_root else repo_root()
    report = package_inputs(
        manifest=Path(args.manifest),
        output=Path(args.output),
        extract_root=args.extract_root,
        root=root,
        start_index=args.start_index,
        limit=args.limit,
        path_fields=args.path_field or DEFAULT_PATH_FIELDS,
        lora_weights=Path(args.lora_weights) if args.lora_weights else None,
        run_script_path=Path(args.run_script) if args.run_script else None,
        include_run_script=args.include_run_script,
        colab_archive_path=args.colab_archive_path,
        colab_repo_dir=args.colab_repo_dir,
        repo_remote=args.repo_remote,
        repo_ref=args.repo_ref,
        run_name=args.run_name,
        eval_starts=args.eval_start or [40, 50],
        eval_limit=args.eval_limit,
        score_profile=args.score_profile,
        train_steps=args.train_steps,
        require_modern_cache=not args.no_require_modern_cache,
        cache_download_mode=args.cache_download_mode,
        cache_max_workers=args.cache_max_workers,
        min_paired_n=args.min_paired_n,
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
