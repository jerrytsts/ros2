"""Footprint schedule and support polygon calculations."""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Contact:
    number: int
    center: float
    start: float
    end: float
    half_length: float

    @property
    def bounds(self):
        return self.center - self.half_length, self.center + self.half_length


class FootSchedule:
    def __init__(self, contacts, step_period):
        if len(contacts) < 2:
            raise ValueError("at least two contacts are required")
        if step_period <= 0:
            raise ValueError("step_period must be positive")
        self.contacts = tuple(contacts)
        self.step_period = float(step_period)
        self.end_time = max(contact.end for contact in contacts)
        self._starts = [contact.start for contact in contacts]

    @classmethod
    def alternating(cls, count=14, spacing=0.34, cadence=0.46, stance=1.14,
                    foot_half=0.11):
        if count < 2 or spacing <= 0 or cadence <= 0 or stance <= 0:
            raise ValueError("invalid gait dimensions")
        contacts = []
        for number in range(count):
            start = number * cadence
            contacts.append(
                Contact(
                    number=number,
                    center=number * spacing,
                    start=start,
                    end=start + stance,
                    half_length=foot_half,
                )
            )
        return cls(contacts, cadence)

    @property
    def duration(self):
        return self.end_time

    def active(self, time):
        return [contact for contact in self.contacts if contact.start <= time < contact.end]

    def support(self, time):
        active = self.active(time)
        if not active:
            return None
        lower = min(contact.bounds[0] for contact in active)
        upper = max(contact.bounds[1] for contact in active)
        return lower, upper

    def zmp_reference(self, time):
        active = self.active(time)
        if not active:
            raise ValueError("no support foot at this time")
        if len(active) == 1:
            return active[0].center
        left, right = active[0], active[1]
        overlap_start = right.start
        overlap_end = left.end
        progress = (time - overlap_start) / max(overlap_end - overlap_start, 1e-9)
        progress = min(max(progress, 0.0), 1.0)
        blend = progress * progress * (3.0 - 2.0 * progress)
        return left.center + (right.center - left.center) * blend

    def cycle_window(self, start_contact=1, contact_span=2):
        first = self.contacts[start_contact]
        last = self.contacts[start_contact + contact_span]
        return first.start, last.start, last.center - first.center
