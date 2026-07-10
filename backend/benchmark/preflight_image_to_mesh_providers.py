from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.run_image_to_mesh_provider import (
    CLI_PROVIDERS,
    HUNYUAN3D_SHAPE_PROVIDER,
    MULTIVIEW_VISUAL_HULL_PROVIDER,
    PROVIDERS,
    SOURCE_MESH_BUNDLE_ORACLE_PROVIDER,
    TRIPOSR_API_PROVIDER,
    provider_dir_config_key,
    resolve_provider_dir,
)
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_REMBG_MODEL,
    pixal3d_model_specs,
)
from backend.benchmark.triposg_models import triposg_model_specs


BUILTIN_PROVIDERS = {SOURCE_MESH_BUNDLE_ORACLE_PROVIDER, MULTIVIEW_VISUAL_HULL_PROVIDER}


def command_exists(executable: str | None) -> bool:
    text = str(executable or "").strip()
    if not text:
        return False
    path = Path(text)
    has_path_separator = any(separator and separator in text for separator in (os.sep, os.altsep))
    if path.is_absolute() or has_path_separator:
        return path.exists()
    return shutil.which(text) is not None


def safe_find_spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def flag_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return shlex.split(command, posix=False)


def parse_provider_command(command: str) -> dict | None:
    tokens = split_command(command)
    if "backend.benchmark.run_image_to_mesh_provider" not in tokens:
        return None
    provider = flag_value(tokens, "--provider")
    if provider not in PROVIDERS:
        return None
    parsed = {
        "command": command,
        "tokens": tokens,
        "provider": provider,
        "provider_dir": flag_value(tokens, "--provider-dir"),
        "provider_python": flag_value(tokens, "--provider-python"),
        "wrapper_python": tokens[0] if tokens else "",
    }
    if provider == "pixal3d":
        specs = pixal3d_model_specs(
            model_repo=flag_value(tokens, "--pixal3d-model-path") or DEFAULT_PIXAL3D_MODEL,
            model_revision=flag_value(tokens, "--pixal3d-model-revision") or "",
            moge_revision=flag_value(tokens, "--pixal3d-moge-revision") or "",
            dinov3_revision=flag_value(tokens, "--pixal3d-dinov3-revision") or "",
            rembg_repo=flag_value(tokens, "--pixal3d-rembg-model") or DEFAULT_PIXAL3D_REMBG_MODEL,
            rembg_revision=flag_value(tokens, "--pixal3d-rembg-revision") or "",
        )
        parsed["provider_models"] = {
            name: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
            for name, spec in specs.items()
        }
    elif provider == "triposg":
        parsed["provider_models"] = triposg_model_specs(
            model_revision=flag_value(tokens, "--triposg-model-revision") or "",
            rembg_revision=flag_value(tokens, "--triposg-rembg-revision") or "",
        )
    return parsed


def provider_dir_env_names(provider: str) -> list[str]:
    prefix = provider.upper().replace("-", "_")
    names = [f"{prefix}_DIR"]
    config = CLI_PROVIDERS.get(provider_dir_config_key(provider), {})
    canonical = config.get("env")
    if canonical and canonical not in names:
        names.append(str(canonical))
    if provider == HUNYUAN3D_SHAPE_PROVIDER and "HUNYUAN3D_DIR" not in names:
        names.append("HUNYUAN3D_DIR")
    return names


def provider_entrypoint(provider: str) -> Path | None:
    if provider in CLI_PROVIDERS:
        runner = CLI_PROVIDERS[provider].get("runner")
        if runner == "triposg-module":
            return Path("scripts/inference_triposg.py")
        if runner == "pixal3d-inference":
            return Path("inference.py")
        return Path("run.py")
    if provider == TRIPOSR_API_PROVIDER:
        return Path("tsr/system.py")
    return None


def provider_preflight_row(parsed: dict, experiment_names: list[str] | None = None) -> dict:
    provider = parsed["provider"]
    wrapper_python = parsed.get("wrapper_python") or ""
    provider_python = parsed.get("provider_python") or wrapper_python
    setup_errors: list[str] = []
    checks: dict[str, object] = {}

    wrapper_python_found = command_exists(wrapper_python)
    provider_python_found = command_exists(provider_python)
    checks["wrapper_python_found"] = wrapper_python_found
    checks["provider_python_found"] = provider_python_found
    if not wrapper_python_found:
        setup_errors.append("Wrapper Python executable is unavailable.")
    if not provider_python_found:
        setup_errors.append("Provider Python executable is unavailable.")

    if provider == "pixal3d":
        revisions = [
            str(spec.get("revision") or "").strip()
            for spec in (parsed.get("provider_models") or {}).values()
        ]
        revisions_present = sum(bool(revision) for revision in revisions)
        revisions_complete = revisions_present in {0, 4}
        checks["model_revisions_complete"] = revisions_complete
        checks["model_revisions_pinned"] = revisions_present == 4
        if not revisions_complete:
            setup_errors.append("Pixal3D model revisions must be supplied together for all four snapshots.")
    elif provider == "triposg":
        revisions = [
            str(spec.get("revision") or "").strip()
            for spec in (parsed.get("provider_models") or {}).values()
        ]
        revisions_present = sum(bool(revision) for revision in revisions)
        revisions_complete = revisions_present in {0, 2}
        checks["model_revisions_complete"] = revisions_complete
        checks["model_revisions_pinned"] = revisions_present == 2
        if not revisions_complete:
            setup_errors.append("TripoSG model revisions must be supplied together for both snapshots.")

    provider_dir = None
    provider_dir_resolved = False
    if provider in BUILTIN_PROVIDERS:
        provider_dir_resolved = True
    elif provider in CLI_PROVIDERS or provider == TRIPOSR_API_PROVIDER:
        try:
            provider_dir = resolve_provider_dir(provider, parsed.get("provider_dir"))
            provider_dir_resolved = True
        except FileNotFoundError:
            setup_errors.append(f"Provider repo is missing. Configure one of: {', '.join(provider_dir_env_names(provider))}.")
        entrypoint = provider_entrypoint(provider)
        if provider_dir and entrypoint:
            entrypoint_found = (provider_dir / entrypoint).exists()
            checks["entrypoint"] = entrypoint.as_posix()
            checks["entrypoint_found"] = entrypoint_found
            if not entrypoint_found:
                setup_errors.append(f"Provider repo is missing expected entrypoint: {entrypoint.as_posix()}.")
    elif provider == HUNYUAN3D_SHAPE_PROVIDER:
        provider_dir_value = parsed.get("provider_dir") or os.getenv("HUNYUAN3D_DIR")
        source_available = False
        if provider_dir_value:
            provider_dir = Path(str(provider_dir_value))
            provider_dir_resolved = provider_dir.exists()
            source_available = (provider_dir / "hy3dshape").exists()
        importable = safe_find_spec("hy3dshape")
        checks["hy3dshape_source_found"] = source_available
        checks["hy3dshape_importable"] = importable
        if not (source_available or importable):
            setup_errors.append("Hunyuan3D Shape is not importable. Configure HUNYUAN3D_DIR or install hy3dshape.")

    checks["provider_dir_resolved"] = provider_dir_resolved
    runnable = not setup_errors
    return {
        "provider": provider,
        "readiness": "ready" if runnable else "missing",
        "runnable": runnable,
        "experiment_names": experiment_names or [],
        "wrapper_python": wrapper_python,
        "provider_python": provider_python,
        "provider_dir_configured": bool(parsed.get("provider_dir")),
        "provider_dir_env_names": provider_dir_env_names(provider),
        "provider_models": parsed.get("provider_models", {}),
        "setup_errors": setup_errors,
        "checks": checks,
    }


def experiment_provider_commands(experiments: list[dict]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for experiment in experiments:
        command = experiment.get("direct_mesh_command")
        if not command:
            continue
        parsed = parse_provider_command(str(command))
        if parsed is None:
            continue
        key = json.dumps(
            {
                "provider": parsed["provider"],
                "provider_dir": parsed.get("provider_dir") or "",
                "provider_python": parsed.get("provider_python") or "",
                "wrapper_python": parsed.get("wrapper_python") or "",
                "provider_models": parsed.get("provider_models") or {},
            },
            sort_keys=True,
        )
        commands.setdefault(key, []).append(experiment.get("name") or experiment.get("method") or parsed["provider"])
    return commands


def preflight_experiments(experiments: list[dict]) -> list[dict]:
    rows = []
    for key, names in sorted(experiment_provider_commands(experiments).items()):
        parsed = json.loads(key)
        rows.append(provider_preflight_row(parsed, experiment_names=names))
    return rows


def write_preflight_report(path: Path, rows: list[dict], require_runnable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "require_runnable": require_runnable,
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight image-to-mesh provider commands without loading models.")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()

    rows = []
    for command in args.command:
        parsed = parse_provider_command(command)
        if parsed is None:
            rows.append({"command": command, "readiness": "unsupported-command", "runnable": False})
        else:
            rows.append(provider_preflight_row(parsed))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.output:
        write_preflight_report(Path(args.output), rows, require_runnable=args.require_runnable)
    if args.require_runnable:
        missing = [row for row in rows if not row.get("runnable")]
        if missing:
            details = "; ".join(f"{row.get('provider', '<unknown>')}: {', '.join(row.get('setup_errors') or [])}" for row in missing)
            raise SystemExit(f"Image-to-mesh provider preflight failed: {details}")


if __name__ == "__main__":
    main()
