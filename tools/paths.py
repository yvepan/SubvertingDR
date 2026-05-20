from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def path_from_env(name: str, default: Path | str) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


REPO_ROOT = RELEASE_ROOT
WORKSPACE_ROOT = path_from_env("EXPERIMENT_WORKSPACE_ROOT", REPO_ROOT)
OUTPUTS_ROOT = path_from_env("EXPERIMENT_OUTPUTS_ROOT", REPO_ROOT / "outputs")
DATASET_ROOT = path_from_env("FORGE_DATASET_ROOT", REPO_ROOT / "data")


def require_repo_root() -> Path:
    if not (REPO_ROOT / "cli.py").exists() or not (REPO_ROOT / "gpt_researcher").exists():
        raise FileNotFoundError(
            "This repository must contain cli.py and the gpt_researcher package."
        )
    return REPO_ROOT
