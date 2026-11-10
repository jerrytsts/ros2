#!/usr/bin/env python3
"""Unified entry point for the original cargo balance assignment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "demo", "test"))
    args, passthrough = parser.parse_known_args()
    commands = {
        "train": SRC / "train_policy.py",
        "evaluate": SRC / "evaluate.py",
        "demo": SRC / "demo.py",
    }
    if args.command == "test":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
            cwd=ROOT,
        )
        return result.returncode
    command = [sys.executable, str(commands[args.command]), *passthrough]
    result = subprocess.run(command, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
