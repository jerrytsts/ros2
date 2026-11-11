"""Three analytic gait families and a transition scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .curves import hermite, hermite_velocity

STAND = np.array([0.12, 0.16, -0.10, 0.12, 0.16, -0.10])


@dataclass(frozen=True)
class Preset:
    frequency: float
    amplitude: float
    knee_bias: float
    speed: float


PRESETS = {
    "stand": Preset(0.0, 0.0, 0.0, 0.0),
    "walk": Preset(0.85, 0.22, 0.16, 0.38),
    "jog": Preset(1.55, 0.38, 0.28, 0.75),
}


def gait_target(name, local_time):
    if name not in PRESETS:
        raise KeyError(name)
    if name == "stand":
        return STAND.copy()
    p = PRESETS[name]
    theta = 2.0 * math.pi * p.frequency * local_time
    stride = p.amplitude * math.sin(theta)
    lift_left = p.knee_bias * max(0.0, math.sin(theta))
    lift_right = p.knee_bias * max(0.0, -math.sin(theta))
    return STAND + np.array(
        [
            stride,
            lift_left,
            -0.55 * stride,
            -stride,
            lift_right,
            0.55 * stride,
        ]
    )


def target_velocity(name, local_time, epsilon=1e-5):
    before = gait_target(name, local_time - epsilon)
    after = gait_target(name, local_time + epsilon)
    return (after - before) / (2.0 * epsilon)


@dataclass
class Sample:
    time: float
    gait: str
    command: np.ndarray
    velocity: np.ndarray
    base_x: float
    transition: float


class GaitScheduler:
    def __init__(self, segment=(("stand", 2.0), ("walk", 3.0), ("jog", 3.0),
                                ("walk", 3.0), ("stand", 2.0)),
                 transition_time=0.65):
        if not segment:
            raise ValueError("schedule cannot be empty")
        self.segments = tuple(segment)
        self.transition_time = float(transition_time)
        self.events = []
        cursor = 0.0
        for name, duration in self.segments:
            self.events.append((cursor, name))
            cursor += float(duration)
        self.duration = cursor

    def mode_at(self, time):
        mode = self.events[0][1]
        for start, name in self.events:
            if time >= start:
                mode = name
            else:
                break
        return mode


def run(mode="hermite", dt=0.005, transition_time=0.65):
    if mode not in ("hermite", "direct"):
        raise ValueError("mode must be hermite or direct")
    scheduler = GaitScheduler(transition_time=transition_time)
    samples = []
    command = gait_target(*reversed(scheduler.events[0]))
    velocity = target_velocity(scheduler.events[0][1], 0.0)
    base_x = 0.0
    active_event = scheduler.events[0]
    blend = None
    event_index = 0

    count = int(round(scheduler.duration / dt)) + 1
    for index in range(count):
        time = index * dt
        start, name = scheduler.events[event_index]
        local = time - start

        if index and scheduler.mode_at(time) != scheduler.mode_at(time - dt):
            event_index += 1
            start, name = scheduler.events[event_index]
            local = time - start
            if mode == "hermite":
                end = time + transition_time
                new_local = local + transition_time
                q1 = gait_target(name, new_local)
                v1 = target_velocity(name, new_local)
                blend = {
                    "start": time,
                    "end": end,
                    "q0": command.copy(),
                    "v0": velocity.copy(),
                    "q1": q1,
                    "v1": v1,
                }

        if blend is not None:
            if time < blend["end"]:
                command = hermite(
                    time,
                    blend["start"],
                    blend["end"],
                    blend["q0"],
                    blend["v0"],
                    blend["q1"],
                    blend["v1"],
                )
                velocity = hermite_velocity(
                    time,
                    blend["start"],
                    blend["end"],
                    blend["q0"],
                    blend["v0"],
                    blend["q1"],
                    blend["v1"],
                )
                progress = (time - blend["start"]) / transition_time
            else:
                command = gait_target(name, local)
                velocity = target_velocity(name, local)
                blend = None
                progress = 1.0
        else:
            command = gait_target(name, local)
            velocity = target_velocity(name, local)
            progress = 0.0

        base_x += PRESETS[name].speed * dt
        samples.append(
            Sample(time, name, command.copy(), velocity.copy(), base_x, progress)
        )
    return scheduler, samples
