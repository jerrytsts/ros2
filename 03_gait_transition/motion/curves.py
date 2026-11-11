"""Cubic Hermite interpolation helpers."""

from __future__ import annotations

import numpy as np


def hermite(t, start, end, q_start, v_start, q_end, v_end):
    if end <= start:
        raise ValueError("end must be greater than start")
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_end, dtype=float)
    v0 = np.asarray(v_start, dtype=float)
    v1 = np.asarray(v_end, dtype=float)
    if not (q0.shape == q1.shape == v0.shape == v1.shape):
        raise ValueError("endpoint arrays must have equal shapes")
    duration = float(end - start)
    u = min(max((t - start) / duration, 0.0), 1.0)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * q0 + h10 * duration * v0 + h01 * q1 + h11 * duration * v1


def hermite_velocity(t, start, end, q_start, v_start, q_end, v_end):
    if end <= start:
        raise ValueError("end must be greater than start")
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_end, dtype=float)
    v0 = np.asarray(v_start, dtype=float)
    v1 = np.asarray(v_end, dtype=float)
    duration = float(end - start)
    u = min(max((t - start) / duration, 0.0), 1.0)
    d00 = 6 * u**2 - 6 * u
    d10 = 3 * u**2 - 4 * u + 1
    d01 = -6 * u**2 + 6 * u
    d11 = 3 * u**2 - 2 * u
    return (
        d00 * q0 + d10 * duration * v0 + d01 * q1 + d11 * duration * v1
    ) / duration


def smoothstep(u):
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)
