"""Cross-platform key mapping and command state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionCommand:
    forward: float = 0.0
    turn: float = 0.0
    quit_requested: bool = False


def decode_key(key, linear=0.32, angular=0.55):
    if key in ("w", "W", "up"):
        return MotionCommand(linear, 0.0)
    if key in ("s", "S", "down"):
        return MotionCommand(-linear, 0.0)
    if key in ("a", "A", "left"):
        return MotionCommand(0.0, angular)
    if key in ("d", "D", "right"):
        return MotionCommand(0.0, -angular)
    if key in ("q", "Q", "\x1b"):
        return MotionCommand(quit_requested=True)
    return MotionCommand()
