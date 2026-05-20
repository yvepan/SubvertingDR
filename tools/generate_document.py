from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tools.paths import OUTPUTS_ROOT, require_repo_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deep-research document with an explicit defense switch."
    )
    parser.add_argument("query")
    parser.add_argument("--report-source", choices=["web", "local", "hybrid"], default="hybrid")
    parser.add_argument("--local-poison-dir", type=Path, default=None)
    parser.add_argument("--web-poison-dir", type=Path, default=None)
    parser.add_argument("--breadth", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    defense_group = parser.add_mutually_exclusive_group(required=True)
    defense_group.add_argument("--enable-defense", dest="enable_defense", action="store_true")
    defense_group.add_argument("--disable-defense", dest="enable_defense", action="store_false")
    args = parser.parse_args()

    repo_root = require_repo_root()
    command = [
        sys.executable,
        "cli.py",
        args.query,
        "--report_type",
        "deep",
        "--report_source",
        args.report_source,
        "--tone",
        "objective",
        "--deep_research_breadth",
        str(args.breadth),
        "--deep_research_depth",
        str(args.depth),
        "--no-pdf",
        "--no-docx",
        "--enable-defense" if args.enable_defense else "--disable-defense",
    ]

    if args.local_poison_dir:
        command.extend(["--doc_path_local_poison", str(args.local_poison_dir)])
    elif args.web_poison_dir and args.report_source == "hybrid":
        empty_local = OUTPUTS_ROOT / "empty_local_poison_docs"
        empty_local.mkdir(parents=True, exist_ok=True)
        command.extend(["--doc_path_local_poison", str(empty_local)])
    if args.web_poison_dir:
        command.extend(["--doc_path_web_poison", str(args.web_poison_dir)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
