#!/usr/bin/env python3

import math

import numpy as np
import pytest

from motion.analysis import switch_times, transition_metrics
from motion.curves import hermite, hermite_velocity, smoothstep
from motion.gaits import GaitScheduler, gait_target, run, target_velocity


def test_hermite_hits_end_positions():
    q0 = np.array([0.1, -0.2])
    v0 = np.array([0.4, 0.0])
    q1 = np.array([0.8, 0.5])
    v1 = np.array([-0.2, 0.3])
    assert np.allclose(hermite(0.0, 0.0, 1.2, q0, v0, q1, v1), q0)
    assert np.allclose(hermite(1.2, 0.0, 1.2, q0, v0, q1, v1), q1)


def test_hermite_hits_end_velocities():
    q0 = np.array([0.0, 1.0])
    v0 = np.array([0.2, -0.5])
    q1 = np.array([2.0, 0.4])
    v1 = np.array([0.7, 0.1])
    assert np.allclose(hermite_velocity(0.0, 0.0, 1.0, q0, v0, q1, v1), v0)
    assert np.allclose(hermite_velocity(1.0, 0.0, 1.0, q0, v0, q1, v1), v1)


def test_hermite_derivative_matches_finite_difference():
    q0, v0 = np.array([0.1]), np.array([0.3])
    q1, v1 = np.array([1.2]), np.array([-0.4])
    h = 1e-7
    for t in np.linspace(0.1, 0.9, 12):
        numeric = (
            hermite(t + h, 0, 1, q0, v0, q1, v1)
            - hermite(t - h, 0, 1, q0, v0, q1, v1)
        ) / (2 * h)
        assert np.allclose(numeric, hermite_velocity(t, 0, 1, q0, v0, q1, v1), atol=1e-5)


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        hermite(0.0, 1.0, 1.0, [0], [0], [1], [0])


def test_smoothstep_clamps():
    assert smoothstep(-1) == 0
    assert smoothstep(2) == 1
    assert math.isclose(smoothstep(0.5), 0.5)


def test_gait_target_shapes_and_finite_velocity():
    for name in ("stand", "walk", "jog"):
        assert gait_target(name, 0.31).shape == (6,)
        assert np.all(np.isfinite(target_velocity(name, 0.31)))


def test_scheduler_covers_segments():
    scheduler = GaitScheduler()
    assert scheduler.mode_at(0.0) == "stand"
    assert scheduler.mode_at(2.1) == "walk"
    assert scheduler.mode_at(5.1) == "jog"
    assert scheduler.mode_at(8.1) == "walk"
    assert scheduler.mode_at(11.1) == "stand"


def test_scheduler_rejects_empty():
    with pytest.raises(ValueError):
        GaitScheduler(segment=())


def test_spline_is_continuous_at_boundaries():
    scheduler, samples = run("hermite")
    split_ids = [
        min(range(len(samples)), key=lambda i: abs(samples[i].time - event))
        for event in switch_times(scheduler)
    ]
    for index in split_ids:
        assert np.linalg.norm(samples[index].command - samples[index - 1].command) < 0.02
        assert np.linalg.norm(samples[index].velocity - samples[index - 1].velocity) < 0.4


def test_direct_and_smooth_metric_contrast():
    scheduler, smooth = run("hermite")
    _, direct = run("direct")
    good = transition_metrics(smooth, scheduler)
    bad = transition_metrics(direct, scheduler)
    assert good["max_velocity_step"] < bad["max_velocity_step"] * 0.2
    assert good["max_acceleration"] < bad["max_acceleration"] * 0.2


def test_schedule_has_exact_sample_count():
    scheduler, samples = run("hermite", dt=0.01)
    assert len(samples) == round(scheduler.duration / 0.01) + 1
