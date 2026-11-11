#!/usr/bin/env python3
"""Regression tests for the original balance homework."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from balance_net import PayloadPolicy, ReferencePD  # noqa: E402
from balance_sim import CarryBalanceRig  # noqa: E402
from evaluate import episode, no_feedback  # noqa: E402


def test_model_compiles_with_expected_dofs():
    rig = CarryBalanceRig(2.0)
    assert rig.model.nq == 14
    assert rig.model.nu == 12
    assert rig.model.njnt == 14


def test_payload_mass_override_and_reset():
    rig = CarryBalanceRig(4.0)
    assert math.isclose(float(rig.model.body_mass[rig.cargo_body]), 4.0, abs_tol=1e-6)
    rig.reset(0.0)
    assert float(rig.model.body_mass[rig.cargo_body]) >= 0.1
    assert math.isclose(rig.pitch(), 0.0, abs_tol=1e-12)


def test_teacher_corrects_positive_pitch():
    teacher = ReferencePD()
    signals = {
        "pitch": 0.1,
        "pitch_rate": 0.0,
        "disturbance": 0.0,
        "mass_norm": 0.5,
        "phase_sin": 0.0,
        "phase_cos": 1.0,
        "ankle_prev": 0.0,
        "hip_prev": 0.0,
    }
    action = teacher(signals)
    assert action[0] < 0
    assert action[1] < 0
    assert np.all(np.abs(action) <= teacher.limit)


def test_policy_roundtrip(tmp_path):
    policy = PayloadPolicy()
    path = tmp_path / "policy.npz"
    policy.save(path)
    loaded = PayloadPolicy.load(path)
    input_value = np.zeros((2, 8))
    assert np.allclose(policy.predict_many(input_value), loaded.predict_many(input_value))


def test_policy_actions_are_finite_and_bounded():
    policy = PayloadPolicy.load(ROOT / "models" / "cargo_policy.npz")
    result = policy.predict_many(np.tile(np.linspace(-1, 1, 8), (4, 1)))
    assert np.all(np.isfinite(result))
    assert np.all(np.abs(result) <= policy.limit)


def test_feedback_survives_three_payloads():
    for controller in (ReferencePD(), PayloadPolicy.load(ROOT / "models" / "cargo_policy.npz")):
        for mass in (0.0, 2.0, 4.0):
            result = episode(mass, controller, seconds=5.0)
            assert result["survived"]
            assert result["peak_deg"] < 28.0


def test_no_feedback_destabilizes_under_test_profile():
    result = episode(2.0, no_feedback, seconds=12.0)
    assert not result["survived"]


def test_render_has_visual_content():
    rig = CarryBalanceRig(2.0)
    image = rig.render(width=360, height=270)
    assert image.shape == (270, 360, 3)
    assert image.std() > 4.0
