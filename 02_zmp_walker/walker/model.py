"""Core mechanics for a planar linear inverted pendulum."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pendulum:
    height: float = 0.80
    gravity: float = 9.81

    def __post_init__(self):
        if self.height <= 0 or self.gravity <= 0:
            raise ValueError("height and gravity must be positive")

    @property
    def omega(self):
        return math.sqrt(self.gravity / self.height)

    def advance(self, x, v, zmp, duration):
        """Propagate exactly under a constant zero-moment point."""
        if duration < 0:
            raise ValueError("duration cannot be negative")
        w = self.omega
        offset = x - zmp
        c = math.cosh(w * duration)
        s = math.sinh(w * duration)
        return offset * c + v / w * s + zmp, offset * w * s + v * c

    def acceleration(self, x, zmp):
        return self.omega**2 * (x - zmp)

    def zmp(self, x, acceleration):
        return x - acceleration / self.omega**2

    def dcm(self, x, v):
        return x + v / self.omega
