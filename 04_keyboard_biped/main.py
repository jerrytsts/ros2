#!/usr/bin/env python3
"""Keyboard-controlled biped demo."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from biped.control import decode_key
from biped.rig import KeyboardBiped

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")


class TerminalKeys:
    def __init__(self):
        self.platform = sys.platform
        self.fd = None
        self.original = None
        if self.platform.startswith("linux"):
            import termios
            import tty

            try:
                self.fd = os.open("/dev/tty", os.O_RDONLY)
            except OSError:
                self.fd = sys.stdin.fileno()
            self.original = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)

    def poll(self):
        if self.platform.startswith("linux"):
            import select

            ready, _, _ = select.select([self.fd], [], [], 0.02)
            if ready:
                return os.read(self.fd, 1).decode("utf-8", errors="ignore")
            return None
        if self.platform.startswith("win"):
            import msvcrt

            if msvcrt.kbhit():
                return msvcrt.getwch()
        return None

    def close(self):
        if self.original is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)
            if self.fd != sys.stdin.fileno():
                os.close(self.fd)


def save_media(frames, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(output / "biped_demo.gif", frames, duration=0.04, loop=0)
    Image.fromarray(frames[-1]).save(output / "biped_demo.png")


def demo(rig, output):
    frames = [rig.render()]
    actions = [(0.32, 0.0, 1.8), (0.0, -0.55, 1.2), (0.32, 0.0, 1.2),
               (0.0, 0.55, 1.2), (-0.25, 0.0, 1.0)]
    for forward, turn, duration in actions:
        steps = int(duration / (rig.dt * 5))
        for index in range(steps):
            rig.step(forward, turn)
            if index % 8 == 0:
                frames.append(rig.render())
    save_media(frames, output)
    print(f"[saved] {Path(output) / 'biped_demo.gif'}")


def interactive(rig, viewer_enabled):
    keys = TerminalKeys()
    viewer = mujoco.viewer.launch_passive(rig.model, rig.data) if viewer_enabled else None
    command = (0.0, 0.0)
    try:
        while viewer is None or viewer.is_running():
            key = keys.poll()
            if key is not None:
                decoded = decode_key(key)
                if decoded.quit_requested:
                    break
                command = (decoded.forward, decoded.turn)
            rig.step(*command)
            if viewer is not None:
                viewer.sync()
            time.sleep(rig.dt * 5)
    finally:
        keys.close()
        if viewer is not None:
            viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if args.test:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT
        ).returncode
    rig = KeyboardBiped()
    if args.demo:
        demo(rig, args.output_dir)
    else:
        interactive(rig, not args.no_viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
