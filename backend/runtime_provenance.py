"""Capture the source revision loaded by a backend process."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _detect_runtime_source_provenance() -> dict:
    repository = Path(__file__).resolve().parents[1]
    environment_revision = os.getenv("GIT_COMMIT", "").strip().lower()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        status_text = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "available": True,
            "revision": revision,
            "clean": not bool(status_text),
            "status": status_text.splitlines() if status_text else [],
            "source": "git-worktree",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": bool(environment_revision),
            "revision": environment_revision or None,
            "clean": None,
            "status": [],
            "source": "environment" if environment_revision else "unavailable",
            "error": type(exc).__name__,
        }


_RUNTIME_SOURCE_PROVENANCE = _detect_runtime_source_provenance()


def runtime_source_provenance() -> dict:
    """Return the immutable source snapshot captured when this module loaded."""
    snapshot = dict(_RUNTIME_SOURCE_PROVENANCE)
    snapshot["status"] = list(snapshot.get("status", []))
    return snapshot
