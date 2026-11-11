#!/usr/bin/env python3
"""Command line entry for the original ZMP walking assignment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from walker.report import main as write_report

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "test"))
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.command == "run":
        write_report(args.output_dir)
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
