#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from motion.export import main as export_report

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "test"))
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.command == "run":
        export_report(args.output_dir)
        return 0
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
