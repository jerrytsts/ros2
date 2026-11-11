"""Simulation, periodic-state search and metrics."""

from __future__ import annotations

import dataclasses
import math

from .footplan import FootSchedule
from .model import Pendulum


@dataclasses.dataclass
class Sample:
    time: float
    x: float
    velocity: float
    zmp: float
    reference: float
    support_low: float
    support_high: float
    margin: float
    dcm_error: float


def _clip(value, lower, upper):
    return max(lower, min(value, upper))


def open_loop_state(pendulum, schedule, x, v, start, end, dt=0.001):
    time = start
    while time < end - 1e-14:
        h = min(dt, end - time)
        zmp = schedule.zmp_reference(time + 0.5 * h)
        x, v = pendulum.advance(x, v, zmp, h)
        time += h
    return x, v


def affine_cycle_map(pendulum, schedule, start, end, dt=0.001):
    x0, v0 = open_loop_state(pendulum, schedule, 0.0, 0.0, start, end, dt)
    x1, v1 = open_loop_state(pendulum, schedule, 1.0, 0.0, start, end, dt)
    x2, v2 = open_loop_state(pendulum, schedule, 0.0, 1.0, start, end, dt)
    a11, a21 = x1 - x0, v1 - v0
    a12, a22 = x2 - x0, v2 - v0
    return ((a11, a12), (a21, a22)), (x0, v0)


def periodic_state(pendulum, schedule, dt=0.001):
    start, end, stride = schedule.cycle_window()
    matrix, offset = affine_cycle_map(pendulum, schedule, start, end, dt)
    a, b = matrix
    m11, m12 = 1.0 - a[0], -a[1]
    m21, m22 = -b[0], 1.0 - b[1]
    r1 = offset[0] - stride
    r2 = offset[1]
    determinant = m11 * m22 - m12 * m21
    if abs(determinant) < 1e-12:
        raise RuntimeError("periodic state is singular")
    x = (r1 * m22 - m12 * r2) / determinant
    v = (m11 * r2 - r1 * m21) / determinant
    return x, v, stride, start, end


def simulate(schedule, gain=1.05, dt=0.001, pendulum=None, initial=(0.0, 0.0)):
    pendulum = Pendulum() if pendulum is None else pendulum
    x, v = map(float, initial)
    time = 0.0
    desired_speed = schedule.step_period and (
        schedule.contacts[2].center - schedule.contacts[0].center
    ) / (schedule.contacts[2].start - schedule.contacts[0].start)
    samples = []
    while time < schedule.end_time - 1e-12:
        h = min(dt, schedule.end_time - time)
        support = schedule.support(time)
        if support is None:
            break
        reference = schedule.zmp_reference(time)
        dcm_target = reference + desired_speed / pendulum.omega
        dcm_now = pendulum.dcm(x, v)
        command = reference + gain * (dcm_now - dcm_target)
        zmp = _clip(command, support[0], support[1])
        margin = min(zmp - support[0], support[1] - zmp)
        samples.append(
            Sample(
                time=time,
                x=x,
                velocity=v,
                zmp=zmp,
                reference=reference,
                support_low=support[0],
                support_high=support[1],
                margin=margin,
                dcm_error=abs(dcm_now - dcm_target),
            )
        )
        x, v = pendulum.advance(x, v, zmp, h)
        time += h
    return samples


def summarize(samples, desired_speed=None):
    if not samples:
        raise ValueError("empty trajectory")
    margins = [sample.margin for sample in samples]
    times = [sample.time for sample in samples]
    distance = samples[-1].x - samples[0].x
    duration = times[-1] - times[0]
    result = {
        "samples": len(samples),
        "duration": duration,
        "distance": distance,
        "mean_speed": distance / duration if duration > 0 else 0.0,
        "max_speed": max(abs(sample.velocity) for sample in samples),
        "minimum_margin": min(margins),
        "mean_margin": sum(margins) / len(margins),
        "mean_dcm_error": sum(sample.dcm_error for sample in samples) / len(samples),
        "inside_support": min(margins) >= -1e-10,
    }
    if desired_speed is not None:
        result["speed_ratio"] = result["mean_speed"] / desired_speed
    return result


def instability_probe(pendulum, schedule, cycles=5, dt=0.001):
    x, v, _stride, start, end = periodic_state(pendulum, schedule, dt)
    initial_velocity = v
    deltas = []
    for cycle in range(cycles):
        cycle_start = start + cycle * (end - start)
        cycle_end = cycle_start + (end - start)
        x, v = open_loop_state(pendulum, schedule, x, v, cycle_start, cycle_end, dt)
        deltas.append(v - initial_velocity)
    ratios = [
        deltas[i + 1] / deltas[i]
        for i in range(len(deltas) - 1)
        if abs(deltas[i]) > 1e-9 and abs(deltas[i + 1]) > 1e-9
    ]
    return deltas, (sum(ratios) / len(ratios) if ratios else math.nan)
