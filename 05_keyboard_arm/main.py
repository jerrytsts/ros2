#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seven-axis keyboard-controlled table arm."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from armlab.core import KeyboardArm, command_for_key  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


HELP = """
============================================================
  七轴桌面机械臂键盘控制
------------------------------------------------------------
  1-7       对应关节正向转动 0.1 rad
  Shift+1-7 对应关节反向转动 0.1 rad
  r         复位
  q / ESC   退出
============================================================
"""


class KeyReader:
    def __init__(self):
        self.platform = sys.platform
        self.fd = None
        self.old = None
        if self.platform.startswith("linux"):
            import termios
            import tty

            try:
                self.fd = os.open("/dev/tty", os.O_RDONLY)
            except OSError:
                self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)

    def read(self, timeout=0.03):
        if self.platform.startswith("linux"):
            import select

            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                return None
            return os.read(self.fd, 1).decode("utf-8", errors="ignore")
        if self.platform.startswith("win"):
            import msvcrt

            if not msvcrt.kbhit():
                return None
            return msvcrt.getwch()
        return None

    def close(self):
        if self.fd is not None and self.old is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            if self.fd != sys.stdin.fileno():
                os.close(self.fd)


def apply_key(sim, key):
    index, direction, reset, quit_requested = command_for_key(key)
    if quit_requested:
        return False
    if reset:
        sim.reset()
        print("[INFO] 已复位")
    elif index is not None:
        value = sim.move_joint(index, direction)
        print(f"[INFO] 关节 {index + 1} 目标: {value:+.3f} rad")
    return True


def run_interactive(sim, no_viewer=False):
    print(HELP)
    reader = KeyReader()
    viewer = None
    if not no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 无法打开窗口，转为无界面模式: {exc}")
    try:
        while viewer is None or viewer.is_running():
            key = reader.read()
            if key is not None and not apply_key(sim, key):
                break
            sim.step()
            if viewer is not None:
                viewer.sync()
            time.sleep(sim.dt)
    finally:
        reader.close()
        if viewer is not None:
            viewer.close()


def save_media(frames, output_dir):
    import imageio.v2 as imageio
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "arm_keyboard_demo.gif"
    frame_path = output_dir / "arm_keyboard_final.png"
    imageio.mimsave(gif_path, frames, duration=0.05, loop=0)
    Image.fromarray(frames[-1]).save(frame_path)
    print(f"[INFO] GIF: {gif_path}")
    print(f"[INFO] 截图: {frame_path}")


def run_demo(sim, output_dir):
    sim.reset()
    sim.step(150)
    frames = [sim.render()]
    sequence = [
        (0, 1), (1, 1), (3, -1), (5, 1), (6, 1),
        (2, 1), (4, 1), (6, -1), (5, -1), (3, 1),
    ]
    for index, direction in sequence:
        sim.move_joint(index, direction, step=0.12)
        for step in range(90):
            sim.step()
            if step % 6 == 0:
                frames.append(sim.render())
    sim.reset()
    for step in range(90):
        sim.step()
        if step % 6 == 0:
            frames.append(sim.render())
    save_media(frames, output_dir)
    return frames


def run_tests(sim):
    del sim  # 保留统一入口签名；测试各自创建隔离的仿真实例
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
        cwd=ROOT,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="七轴桌面机械臂键盘控制")
    parser.add_argument("--demo", action="store_true", help="自动演示并录制 GIF")
    parser.add_argument("--test", action="store_true", help="运行自动化测试")
    parser.add_argument("--no-viewer", action="store_true", help="关闭交互窗口")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    sim = KeyboardArm()
    if args.test:
        return run_tests(sim)
    if args.demo:
        run_demo(sim, Path(args.output_dir))
        return 0
    run_interactive(sim, no_viewer=args.no_viewer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
