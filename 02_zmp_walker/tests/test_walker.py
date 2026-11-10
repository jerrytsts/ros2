#!/usr/bin/env python3

import math

import pytest

from walker.engine import (
    instability_probe,
    open_loop_state,
    periodic_state,
    simulate,
    summarize,
)
from walker.footplan import Contact, FootSchedule
from walker.model import Pendulum


def test_pendulum_frequency():
    pendulum = Pendulum(height=0.8, gravity=9.81)
    assert math.isclose(pendulum.omega, math.sqrt(9.81 / 0.8))


def test_zmp_inverse_relation():
    pendulum = Pendulum()
    assert math.isclose(pendulum.zmp(0.3, 1.7), 0.3 - 1.7 / pendulum.omega**2)


def test_constant_zmp_propagation():
    pendulum = Pendulum()
    x, v = pendulum.advance(0.0, 0.0, 0.2, 0.1)
    assert x < 0.0
    assert v < 0.0
    acceleration = pendulum.acceleration(x, 0.2)
    assert acceleration < 0.0


def test_foot_schedule_support_and_double_contact():
    schedule = FootSchedule.alternating(count=5, spacing=0.3, cadence=0.5, stance=1.1)
    assert len(schedule.active(0.55)) == 2
    low, high = schedule.support(0.55)
    assert low < high
    assert schedule.support(4.0) is None


def test_reference_stays_in_support():
    schedule = FootSchedule.alternating()
    for time in (0.2, 0.7, 1.1, 1.4, 2.0):
        support = schedule.support(time)
        reference = schedule.zmp_reference(time)
        assert support[0] - 1e-12 <= reference <= support[1] + 1e-12


def test_validation_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        Pendulum(height=0)
    with pytest.raises(ValueError):
        FootSchedule.alternating(count=1)
    with pytest.raises(ValueError):
        FootSchedule([Contact(0, 0, 0, 1, 0.1), Contact(1, 1, 2, 3, 0.1)], 0)


def test_dcm_definition():
    pendulum = Pendulum()
    assert math.isclose(pendulum.dcm(0.2, 0.7), 0.2 + 0.7 / pendulum.omega)


def test_closed_loop_stays_inside_support():
    schedule = FootSchedule.alternating()
    metrics = summarize(simulate(schedule, gain=1.05, dt=0.002), desired_speed=0.68 / 0.92)
    assert metrics["inside_support"]
    assert metrics["distance"] > 4.0
    assert metrics["speed_ratio"] > 0.95


def test_zero_gain_diverges():
    schedule = FootSchedule.alternating()
    samples = simulate(schedule, gain=0.0, dt=0.002)
    assert max(abs(sample.velocity) for sample in samples) > 1e5


def test_periodic_state_is_fixed_point():
    pendulum = Pendulum()
    schedule = FootSchedule.alternating()
    x, v, stride, start, end = periodic_state(pendulum, schedule, dt=0.002)
    x2, v2 = open_loop_state(pendulum, schedule, x, v, start, end, dt=0.002)
    assert math.isclose(x2 - x, stride, rel_tol=1e-8)
    assert math.isclose(v2, v, abs_tol=1e-8)


def test_instability_growth_is_much_greater_than_one():
    deltas, ratio = instability_probe(Pendulum(), FootSchedule.alternating(), cycles=5, dt=0.002)
    assert len(deltas) == 5
    assert abs(ratio) > 5.0
