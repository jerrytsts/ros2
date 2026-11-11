"""MuJoCo biped with a mocap-guided base and articulated limbs."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "keyboard_biped.xml"


class KeyboardBiped:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base"
        )
        if self.mocap_id < 0 or self.model.body_mocapid[self.mocap_id] < 0:
            raise RuntimeError("robot_base must be a mocap body")
        self.mocap_slot = int(self.model.body_mocapid[self.mocap_id])
        self.actuator = {
            self.model.actuator(i).name: i for i in range(self.model.nu)
        }
        self.position = np.array([0.0, 0.0], dtype=float)
        self.yaw = 0.0
        self.phase = 0.0
        self.distance = 0.0
        self.command = np.zeros(2)
        self.reset()

    @property
    def dt(self):
        return float(self.model.opt.timestep)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.position[:] = (0.0, 0.0)
        self.yaw = 0.0
        self.phase = 0.0
        self.distance = 0.0
        self.command[:] = 0.0
        self._place_base()
        self._apply_pose(0.0)
        mujoco.mj_forward(self.model, self.data)

    def _place_base(self):
        self.data.mocap_pos[self.mocap_slot] = (self.position[0], self.position[1], 0.98)
        half = self.yaw / 2.0
        self.data.mocap_quat[self.mocap_slot] = (
            math.cos(half),
            0.0,
            0.0,
            math.sin(half),
        )

    def _apply_pose(self, gait):
        swing = 0.34 * gait
        knee_l = 0.15 * gait * gait + 0.18 * max(0.0, math.sin(self.phase))
        knee_r = 0.15 * gait * gait + 0.18 * max(0.0, -math.sin(self.phase))
        values = {
            "l_hip_servo": swing * math.sin(self.phase),
            "r_hip_servo": -swing * math.sin(self.phase),
            "l_knee_servo": -knee_l,
            "r_knee_servo": -knee_r,
            "l_ankle_servo": -0.45 * swing * math.sin(self.phase),
            "r_ankle_servo": 0.45 * swing * math.sin(self.phase),
            "l_shoulder_servo": -0.55 * swing * math.sin(self.phase),
            "r_shoulder_servo": 0.55 * swing * math.sin(self.phase),
            "l_elbow_servo": -0.28,
            "r_elbow_servo": -0.28,
        }
        for name, value in values.items():
            self.data.ctrl[self.actuator[name]] = value

    def set_command(self, forward, turn):
        self.command[:] = (float(forward), float(turn))

    def step(self, forward=None, turn=None, substeps=5):
        if forward is not None or turn is not None:
            self.set_command(
                0.0 if forward is None else forward,
                0.0 if turn is None else turn,
            )
        forward, turn = self.command
        dt = self.dt * substeps
        self.yaw += turn * dt
        dx = forward * math.cos(self.yaw) * dt
        dy = forward * math.sin(self.yaw) * dt
        self.position += (dx, dy)
        self.distance += math.hypot(dx, dy)
        gait = min(1.0, abs(forward) / 0.32)
        self.phase = (self.phase + 2.0 * math.pi * 1.25 * gait * dt) % (2.0 * math.pi)
        self._place_base()
        self._apply_pose(gait)
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

    def state(self):
        return {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "yaw": float(self.yaw),
            "phase": float(self.phase),
            "distance": float(self.distance),
            "forward": float(self.command[0]),
            "turn": float(self.command[1]),
        }

    def render(self, width=640, height=480):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (self.position[0], self.position[1], 0.9)
        camera.distance = 3.0
        camera.azimuth = 135.0
        camera.elevation = -12.0
        try:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()
