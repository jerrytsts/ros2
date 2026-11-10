#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seven-axis table arm for the original keyboard-control lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "keyboard_arm.xml"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))


@dataclass
class KeyboardArm:
    model_path: Path = MODEL_PATH

    def __post_init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise RuntimeError("模型缺少七关节机械臂")
        self.qpos_addr = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=int,
        )
        self.low = np.array(
            [self.model.jnt_range[joint_id, 0] for joint_id in self.joint_ids]
        )
        self.high = np.array(
            [self.model.jnt_range[joint_id, 1] for joint_id in self.joint_ids]
        )
        self.reset()

    @property
    def dt(self):
        return float(self.model.opt.timestep)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.target = self.home.copy()
        self.apply_target()
        mujoco.mj_forward(self.model, self.data)

    @property
    def home(self):
        return np.array(self.model.key_ctrl[0, :7], dtype=float)

    def joint_positions(self):
        return np.array(self.data.qpos[self.qpos_addr], dtype=float)

    def target_positions(self):
        return self.target.copy()

    def apply_target(self):
        self.data.ctrl[:7] = self.target

    def move_joint(self, index, direction, step=0.1):
        if not 0 <= index < 7:
            raise IndexError("关节索引必须在 0..6")
        if direction not in (-1, 1):
            raise ValueError("direction 必须为 +1 或 -1")
        before = float(self.target[index])
        self.target[index] = float(
            np.clip(before + direction * step, self.low[index], self.high[index])
        )
        self.apply_target()
        return self.target[index]

    def step(self, count=1):
        for _ in range(count):
            self.apply_target()
            mujoco.mj_step(self.model, self.data)

    def render(self, camera="overview", width=640, height=480):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            if camera == "overview":
                view = mujoco.MjvCamera()
                view.type = mujoco.mjtCamera.mjCAMERA_FREE
                view.lookat[:] = (0.0, 0.0, 0.55)
                view.distance = 2.0
                view.azimuth = 135.0
                view.elevation = -18.0
                renderer.update_scene(self.data, camera=view)
            else:
                renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()


def command_for_key(key):
    """把终端字符映射为 (关节索引, 方向, 是否复位, 是否退出)。"""
    reverse = {"!": 0, "@": 1, "#": 2, "$": 3, "%": 4, "^": 5, "&": 6}
    if key in tuple("1234567"):
        return int(key) - 1, 1, False, False
    if key in reverse:
        return reverse[key], -1, False, False
    if key in ("r", "R"):
        return None, 0, True, False
    if key in ("q", "Q", "\x1b"):
        return None, 0, False, True
    return None, 0, False, False
