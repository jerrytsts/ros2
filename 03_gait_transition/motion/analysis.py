"""Metrics for comparing transition strategies."""

from __future__ import annotations

import numpy as np

from .gaits import GaitScheduler


def switch_times(scheduler):
    return [start for start, _name in scheduler.events[1:]]


def transition_metrics(samples, scheduler, window=0.65):
    times = np.array([sample.time for sample in samples])
    commands = np.stack([sample.command for sample in samples])
    velocities = np.stack([sample.velocity for sample in samples])
    command_delta = np.linalg.norm(np.diff(commands, axis=0), axis=1)
    velocity_delta = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
    acceleration = np.diff(velocities, axis=0) / np.diff(times)[:, None]
    windows = [
        (times >= start) & (times <= start + window)
        for start in switch_times(scheduler)
    ]
    mask = np.zeros(len(times), dtype=bool)
    for item in windows:
        mask |= item
    pair_mask = mask[:-1]
    return {
        "samples": len(samples),
        "switch_count": len(windows),
        "max_command_step": float(command_delta[pair_mask].max()),
        "max_velocity_step": float(velocity_delta[pair_mask].max()),
        "max_acceleration": float(np.linalg.norm(acceleration[pair_mask], axis=1).max()),
        "final_x": float(samples[-1].base_x),
    }
