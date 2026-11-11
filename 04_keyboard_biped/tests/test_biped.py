#!/usr/bin/env python3

import math

import numpy as np

from biped.control import decode_key
from biped.rig import KeyboardBiped


def test_model_has_ten_actuated_limb_joints():
    rig = KeyboardBiped()
    assert rig.model.nu == 10
    assert rig.model.nq == 10


def test_key_decoding():
    assert decode_key("w").forward > 0
    assert decode_key("s").forward < 0
    assert decode_key("a").turn > 0
    assert decode_key("d").turn < 0
    assert decode_key("up").forward > 0
    assert decode_key(" ").forward == 0
    assert decode_key("q").quit_requested


def test_reset_returns_origin():
    rig = KeyboardBiped()
    rig.step(0.3, 0.0, substeps=50)
    rig.reset()
    state = rig.state()
    assert state["x"] == 0
    assert state["y"] == 0
    assert state["yaw"] == 0


def test_forward_distance_matches_command():
    rig = KeyboardBiped()
    for _ in range(100):
        rig.step(0.3, 0.0)
    state = rig.state()
    assert math.isclose(state["x"], 0.3, rel_tol=1e-9, abs_tol=1e-9)
    assert abs(state["y"]) < 1e-12


def test_turn_then_move_follows_heading():
    rig = KeyboardBiped()
    for _ in range(50):
        rig.step(0.0, 0.5)
    yaw = rig.state()["yaw"]
    for _ in range(50):
        rig.step(0.3, 0.0)
    state = rig.state()
    assert state["x"] > 0
    assert state["y"] > 0
    assert abs(math.atan2(state["y"], state["x"]) - yaw) < 1e-9


def test_zero_command_stops_motion():
    rig = KeyboardBiped()
    rig.step(0.3, 0.0, substeps=100)
    before = rig.state()
    for _ in range(20):
        rig.step(0.0, 0.0)
    after = rig.state()
    assert math.isclose(before["x"], after["x"])
    assert after["distance"] >= before["distance"]


def test_joint_state_remains_finite():
    rig = KeyboardBiped()
    for _ in range(100):
        rig.step(0.32, 0.3)
    assert np.all(np.isfinite(rig.data.qpos))


def test_render_shape():
    rig = KeyboardBiped()
    image = rig.render(width=400, height=300)
    assert image.shape == (300, 400, 3)
    assert image.std() > 3
