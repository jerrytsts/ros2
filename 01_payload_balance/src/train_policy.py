#!/usr/bin/env python3
"""Collect teacher demonstrations and train the original payload policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from balance_net import (
    ACTIONS,
    SIGNALS,
    PayloadPolicy,
    ReferencePD,
    flatten_episode,
)
from balance_sim import CarryBalanceRig

ROOT = Path(__file__).resolve().parent.parent


def collect(rng):
    teacher = ReferencePD()
    episodes = []
    episode_id = 0
    for mass in (0.0, 2.0, 4.0):
        for repeat in range(8):
            rig = CarryBalanceRig(mass)
            rig.reset(
                mass,
                initial_pitch=rng.uniform(-0.018, 0.018),
                initial_rate=rng.uniform(-0.035, 0.035),
                disturbance_scale=rng.uniform(0.82, 1.10),
                disturbance_phase=rng.uniform(0.0, 4.0),
            )
            samples = []
            for _ in range(420):
                signals, action = rig.step(teacher)
                samples.append((signals, action))
                if rig.fallen():
                    samples.clear()
                    break
            if samples:
                episodes.append((episode_id, samples))
                episode_id += 1
    return episodes


def gradients(weights, x, y, batch):
    activations = [x[batch]]
    layer_count = len(weights) // 2
    for layer in range(layer_count):
        value = activations[-1] @ weights[2 * layer] + weights[2 * layer + 1]
        if layer + 1 < layer_count:
            value = np.tanh(value)
        activations.append(value)
    prediction = activations[-1]
    delta = 2.0 * (prediction - y[batch]) / len(batch)
    grads = [None] * len(weights)
    for layer in range(layer_count - 1, -1, -1):
        grads[2 * layer] = activations[layer].T @ delta
        grads[2 * layer + 1] = delta.sum(axis=0)
        if layer:
            delta = (delta @ weights[2 * layer].T) * (1.0 - activations[layer] ** 2)
    return prediction, grads


def forward(weights, x):
    value = x
    layer_count = len(weights) // 2
    for layer in range(layer_count):
        value = value @ weights[2 * layer] + weights[2 * layer + 1]
        if layer + 1 < layer_count:
            value = np.tanh(value)
    return value


def train_network(x, y, epochs, seed):
    rng = np.random.default_rng(seed)
    sizes = [len(SIGNALS), 24, 16, len(ACTIONS)]
    weights = []
    for left, right in zip(sizes[:-1], sizes[1:]):
        weights.append(rng.normal(0, np.sqrt(2.0 / left), (left, right)))
        weights.append(np.zeros(right))
    weights = [weights[i] if i % 2 == 0 else weights[i] for i in range(len(weights))]

    first_moment = [np.zeros_like(weight) for weight in weights]
    second_moment = [np.zeros_like(weight) for weight in weights]
    history = []
    step = 0
    batch_size = 128
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            prediction, grads = gradients(weights, x, y, batch)
            losses.append(float(np.mean((prediction - y[batch]) ** 2)))
            step += 1
            for index, grad in enumerate(grads):
                first_moment[index] = 0.9 * first_moment[index] + 0.1 * grad
                second_moment[index] = 0.999 * second_moment[index] + 0.001 * grad * grad
                m_hat = first_moment[index] / (1.0 - 0.9**step)
                v_hat = second_moment[index] / (1.0 - 0.999**step)
                weights[index] -= 0.0015 * m_hat / (np.sqrt(v_hat) + 1e-8)
        if (epoch + 1) % 40 == 0 or epoch == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "optimization_mse": float(np.mean(losses)),
                }
            )
            print(f"[train] epoch {epoch + 1:3d} mse={np.mean(losses):.3e}")
    return weights, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", default=str(ROOT / "models" / "cargo_policy.npz"))
    args = parser.parse_args()

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    episodes = collect(rng)
    all_x, all_y, group = [], [], []
    for episode_id, samples in episodes:
        x, y = flatten_episode(samples)
        all_x.append(x)
        all_y.append(y)
        group.append(np.full(len(x), episode_id))
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    group = np.concatenate(group)

    episode_ids = np.unique(group)
    validation_ids = episode_ids[::5]
    validation_mask = np.isin(group, validation_ids)
    train_mask = ~validation_mask
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    xn = (x - mean) / std

    weights, history = train_network(xn[train_mask], y[train_mask], args.epochs, args.seed)
    policy = PayloadPolicy(weights=weights, mean=mean, std=std)
    policy.save(args.output)

    validation_prediction = policy.predict_many(x[validation_mask])
    final_validation = float(np.mean((validation_prediction - y[validation_mask]) ** 2))
    metrics = {
        "samples": int(len(x)),
        "episodes": int(len(episodes)),
        "signals": list(SIGNALS),
        "actions": list(ACTIONS),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "validation_mse": final_validation,
        "training_seconds": round(time.perf_counter() - started, 2),
        "history": history,
    }
    metrics_path = Path(args.output).with_name("cargo_policy_metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
