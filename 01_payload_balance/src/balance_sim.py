"""Novel MuJoCo balance rig for a two-handed cargo carrying experiment."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from balance_net import pack_signals

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "balance_rig.xml"


class CarryBalanceRig:
    def __init__(self, payload_kg=0.0):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
        self.data = mujoco.MjData(self.model)
        self.pitch_joint = self._joint("root_pitch")
        self.pitch_qadr = int(self.model.jnt_qposadr[self.pitch_joint])
        self.pitch_dadr = int(self.model.jnt_dofadr[self.pitch_joint])
        self.cargo_body = self._body("cargo_box")
        self.cargo_inertia0 = self.model.body_inertia[self.cargo_body].copy()
        self.cargo_mass0 = float(self.model.body_mass[self.cargo_body])
        self.actuator = {
            self.model.actuator(i).name: i for i in range(self.model.nu)
        }
        self.stand_targets = {
            "left_hip_servo": 0.05,
            "left_knee_servo": -0.08,
            "left_ankle_servo": 0.03,
            "right_hip_servo": 0.05,
            "right_knee_servo": -0.08,
            "right_ankle_servo": 0.03,
            "left_shoulder_servo": -0.35,
            "left_elbow_servo": -0.75,
            "right_shoulder_servo": -0.35,
            "right_elbow_servo": -0.75,
        }
        self.control_dt = 0.01
        self.substeps = int(round(self.control_dt / self.model.opt.timestep))
        self.payload_kg = float(payload_kg)
        self.time = 0.0
        self.previous_action = np.zeros(2)
        self.reset(payload_kg)

    def _joint(self, name):
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise RuntimeError(f"missing joint: {name}")
        return index

    def _body(self, name):
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise RuntimeError(f"missing body: {name}")
        return index

    def reset(
        self,
        payload_kg=0.0,
        initial_pitch=0.0,
        initial_rate=0.0,
        disturbance_scale=1.0,
        disturbance_phase=0.0,
    ):
        self.payload_kg = float(payload_kg)
        self.time = 0.0
        self.disturbance_scale = float(disturbance_scale)
        self.disturbance_phase = float(disturbance_phase)
        self.previous_action[:] = 0.0
        mujoco.mj_resetData(self.model, self.data)

        mass = max(0.10, float(payload_kg))
        scale = mass / max(self.cargo_mass0, 1e-6)
        self.model.body_mass[self.cargo_body] = mass
        self.model.body_inertia[self.cargo_body] = self.cargo_inertia0 * scale

        self.data.qpos[self.pitch_qadr] = float(initial_pitch)
        self.data.qvel[self.pitch_dadr] = float(initial_rate)
        for name, value in self.stand_targets.items():
            self.data.ctrl[self.actuator[name]] = value
        self.data.ctrl[self.actuator["ankle_assist"]] = 0.0
        self.data.ctrl[self.actuator["hip_assist"]] = 0.0
        self.data.qfrc_applied[self.pitch_dadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def pitch(self):
        return float(self.data.qpos[self.pitch_qadr])

    def pitch_rate(self):
        return float(self.data.qvel[self.pitch_dadr])

    def disturbance(self):
        t = self.time + self.disturbance_phase
        mass = self.payload_kg
        return self.disturbance_scale * float(
            11.0 * math.sin(1.15 * t + 0.4 * mass)
            + 4.5 * math.sin(2.85 * t + 0.8)
            + 0.85 * mass * math.sin(0.62 * t)
            + 7.0 * math.exp(-0.5 * ((t - 7.0) / 0.18) ** 2)
            + 7.0 * math.exp(-0.5 * ((t - 17.0) / 0.18) ** 2)
        )

    def signals(self):
        phase = (0.35 * self.time) % 1.0
        return pack_signals(
            {
                "pitch": self.pitch(),
                "pitch_rate": self.pitch_rate(),
                "disturbance": self.disturbance(),
                "mass_norm": min(max(self.payload_kg / 4.0, 0.0), 1.25),
                "phase_sin": math.sin(2.0 * math.pi * phase),
                "phase_cos": math.cos(2.0 * math.pi * phase),
                "ankle_prev": self.previous_action[0],
                "hip_prev": self.previous_action[1],
            }
        )

    def step(self, controller):
        signals = self.signals()
        action = np.asarray(controller(signals), dtype=float)
        action = np.nan_to_num(np.clip(action, -150.0, 150.0))
        self.data.ctrl[self.actuator["ankle_assist"]] = action[0]
        self.data.ctrl[self.actuator["hip_assist"]] = action[1]
        self.data.qfrc_applied[self.pitch_dadr] = self.disturbance()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.time += self.control_dt
        self.previous_action = action
        return signals, action

    def fallen(self):
        return abs(self.pitch()) > math.radians(32.0)

    def render(self, width=640, height=480):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 1.0)
        camera.distance = 3.2
        camera.azimuth = 138.0
        camera.elevation = -10.0
        try:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()
