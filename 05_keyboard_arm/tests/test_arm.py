#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np

from armlab.core import KeyboardArm, command_for_key


def test_model_structure():
    sim = KeyboardArm()
    assert sim.model.nq == 9
    assert sim.model.nu == 9
    assert len(sim.joint_ids) == 7


def test_home_keyframe_and_reset():
    sim = KeyboardArm()
    home = sim.home.copy()
    sim.move_joint(0, 1)
    sim.reset()
    assert np.allclose(sim.target, home)
    assert sim.data.ctrl[:7].shape == (7,)


def test_key_mapping_forward_reverse_and_reset():
    assert command_for_key("1") == (0, 1, False, False)
    assert command_for_key("7") == (6, 1, False, False)
    assert command_for_key("!") == (0, -1, False, False)
    assert command_for_key("&") == (6, -1, False, False)
    assert command_for_key("r") == (None, 0, True, False)
    assert command_for_key("q") == (None, 0, False, True)


def test_joint_increment_and_clamp():
    sim = KeyboardArm()
    before = sim.target[0]
    after = sim.move_joint(0, 1, step=0.1)
    assert after > before
    for _ in range(100):
        sim.move_joint(0, 1, step=1.0)
    assert math.isclose(sim.target[0], sim.high[0], abs_tol=1e-9)
    for _ in range(100):
        sim.move_joint(0, -1, step=1.0)
    assert math.isclose(sim.target[0], sim.low[0], abs_tol=1e-9)


def test_servo_tracks_new_target():
    sim = KeyboardArm()
    sim.reset()
    target = sim.target.copy()
    target[1] += 0.18
    sim.target[1] = target[1]
    sim.apply_target()
    sim.step(800)
    assert abs(sim.joint_positions()[1] - target[1]) < 0.08


def test_render_camera():
    sim = KeyboardArm()
    image = sim.render(camera="overview", width=480, height=360)
    assert image.shape == (360, 480, 3)
    assert image.std() > 5.0
