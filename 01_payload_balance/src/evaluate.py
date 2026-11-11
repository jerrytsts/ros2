#!/usr/bin/env python3
"""Compare no feedback, reference PD and the trained network."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from balance_net import PayloadPolicy, ReferencePD
from balance_sim import CarryBalanceRig

ROOT = Path(__file__).resolve().parent.parent


def no_feedback(signals):
    del signals
    return np.zeros(2)


def episode(mass, controller, seconds=20.0):
    rig = CarryBalanceRig(mass)
    rig.reset(mass)
    pitches = []
    torques = []
    fall_time = None
    steps = int(round(seconds / rig.control_dt))
    for _ in range(steps):
        _, action = rig.step(controller)
        pitches.append(math.degrees(rig.pitch()))
        torques.append(float(np.linalg.norm(action)))
        if rig.fallen():
            fall_time = rig.time
            break
    values = np.asarray(pitches)
    return {
        "mass_kg": float(mass),
        "mean_abs_deg": float(np.mean(np.abs(values))),
        "rms_deg": float(np.sqrt(np.mean(values * values))),
        "peak_deg": float(np.max(np.abs(values))),
        "mean_torque": float(np.mean(torques)),
        "fall_time_s": "" if fall_time is None else round(fall_time, 3),
        "survived": fall_time is None,
        "samples": len(values),
    }


def plot(rows, output):
    masses = ["0 kg", "2 kg", "4 kg"]
    controllers = ["open", "pd", "policy"]
    colors = ["#6b7280", "#0e7490", "#ea580c"]
    labels = ["No feedback", "Reference PD", "Neural policy"]
    values = {
        (row["controller"], row["mass_kg"]): row["mean_abs_deg"] for row in rows
    }
    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for index, (name, color, label) in enumerate(zip(controllers, colors, labels)):
        heights = [values[(name, mass)] for mass in (0.0, 2.0, 4.0)]
        bars = ax.bar(x + (index - 1) * width, heights, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, masses)
    ax.set_ylabel("Mean absolute torso pitch (deg)")
    ax.set_title("Payload balance comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--weights", default=str(ROOT / "models" / "cargo_policy.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    policy = PayloadPolicy.load(args.weights)
    controllers = [
        ("open", no_feedback),
        ("pd", ReferencePD()),
        ("policy", policy),
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mass in (0.0, 2.0, 4.0):
        for name, controller in controllers:
            row = episode(mass, controller, args.seconds)
            row["controller"] = name
            rows.append(row)
            print(
                f"{mass:>3.0f} kg {name:>6s} "
                f"mean={row['mean_abs_deg']:6.2f} rms={row['rms_deg']:6.2f} "
                f"peak={row['peak_deg']:6.2f} survived={row['survived']}"
            )

    csv_path = output / "evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot(rows, output / "comparison.png")
    summary = {
        "duration_s": args.seconds,
        "weights": str(Path(args.weights).resolve()),
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
