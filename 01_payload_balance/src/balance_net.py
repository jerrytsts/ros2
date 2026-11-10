"""Original payload-balance controller and NumPy MLP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

SIGNALS = (
    "pitch",
    "pitch_rate",
    "disturbance",
    "mass_norm",
    "phase_sin",
    "phase_cos",
    "ankle_prev",
    "hip_prev",
)
ACTIONS = ("ankle_torque", "hip_torque")


class ReferencePD:
    """Feedback law used only to create supervised demonstrations."""

    def __init__(self, limit=150.0):
        self.limit = float(limit)

    def __call__(self, x: dict) -> np.ndarray:
        pitch = x["pitch"]
        rate = x["pitch_rate"]
        mass = x["mass_norm"] * 4.0
        disturbance = x["disturbance"]
        ankle = -(115.0 * pitch + 17.0 * rate + 0.18 * disturbance)
        hip = -(35.0 * pitch + 5.0 * rate) - 0.85 * mass
        return np.clip([ankle, hip], -self.limit, self.limit)


class PayloadPolicy:
    """A compact tanh network trained on the reference balances."""

    def __init__(self, weights=None, mean=None, std=None, limit=150.0):
        self.limit = float(limit)
        self.mean = np.zeros(len(SIGNALS)) if mean is None else np.asarray(mean, float)
        self.std = np.ones(len(SIGNALS)) if std is None else np.asarray(std, float)
        if weights is None:
            rng = np.random.default_rng(17)
            self.weights = []
            sizes = [len(SIGNALS), 24, 16, len(ACTIONS)]
            for left, right in zip(sizes[:-1], sizes[1:]):
                self.weights.append(rng.normal(0, np.sqrt(2 / left), (left, right)))
                self.weights.append(np.zeros(right))
        else:
            self.weights = [np.asarray(item, float) for item in weights]

    @staticmethod
    def vector(signals):
        return np.asarray([signals[name] for name in SIGNALS], dtype=float)

    def predict_many(self, matrix):
        hidden = (np.asarray(matrix, float) - self.mean) / self.std
        for index in range(0, len(self.weights), 2):
            hidden = hidden @ self.weights[index] + self.weights[index + 1]
            if index + 2 < len(self.weights):
                hidden = np.tanh(hidden)
        return hidden

    def __call__(self, signals):
        output = self.predict_many(self.vector(signals)[None, :])[0]
        return np.clip(output, -self.limit, self.limit)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **{f"layer_{i}": value for i, value in enumerate(self.weights)},
            mean=self.mean,
            std=self.std,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "signals": SIGNALS,
                    "actions": ACTIONS,
                    "hidden": [24, 16],
                    "torque_limit": self.limit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        weights = [data[key] for key in sorted(k for k in data.files if k.startswith("layer_"))]
        return cls(weights, data["mean"], data["std"])


def pack_signals(values: dict) -> dict:
    return {name: float(values[name]) for name in SIGNALS}


def flatten_episode(examples: Iterable[tuple[dict, np.ndarray]]):
    xs, ys = [], []
    for signals, action in examples:
        xs.append([signals[name] for name in SIGNALS])
        ys.append(action)
    return np.asarray(xs, float), np.asarray(ys, float)
