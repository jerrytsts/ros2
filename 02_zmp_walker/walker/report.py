"""Generate data files and a visual report for the walker lab."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import instability_probe, periodic_state, simulate, summarize
from .footplan import FootSchedule
from .model import Pendulum


def main(output_dir="results"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pendulum = Pendulum()
    schedule = FootSchedule.alternating()

    gains = [0.0, 0.35, 0.65, 0.9, 1.05, 1.3, 1.7]
    rows = []
    trajectories = {}
    for gain in gains:
        samples = simulate(schedule, gain=gain, dt=0.002, pendulum=pendulum)
        metrics = summarize(samples, desired_speed=0.68 / 0.92)
        row = {
            "gain": gain,
            "distance": metrics["distance"],
            "mean_speed": metrics["mean_speed"],
            "minimum_margin": metrics["minimum_margin"],
            "mean_margin": metrics["mean_margin"],
            "inside_support": metrics["inside_support"],
        }
        rows.append(row)
        trajectories[gain] = samples

    with (output / "gain_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = trajectories[1.05]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot([s.time for s in selected], [s.x for s in selected], label="COM")
    axes[0].plot([s.time for s in selected], [s.zmp for s in selected], label="ZMP")
    axes[0].plot(
        [s.time for s in selected],
        [s.support_low for s in selected],
        color="0.65",
        linestyle="--",
        linewidth=0.9,
        label="support bounds",
    )
    axes[0].plot(
        [s.time for s in selected],
        [s.support_high for s in selected],
        color="0.65",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_ylabel("position (m)")
    axes[0].set_title("Planar LIPM walking with DCM feedback")
    axes[0].legend(frameon=False, ncols=3)
    axes[0].grid(alpha=0.2)

    axes[1].plot([s.time for s in selected], [s.margin for s in selected], color="tab:green")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("ZMP margin (m)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "walking_profile.png", dpi=180, facecolor="white")
    plt.close(fig)

    deltas, ratio = instability_probe(pendulum, schedule, cycles=5, dt=0.002)
    summary = summarize(selected, desired_speed=0.68 / 0.92)
    summary["open_loop_velocity_errors"] = deltas
    summary["open_loop_growth_ratio"] = ratio
    summary["periodic_state"] = periodic_state(pendulum, schedule, dt=0.002)[:3]
    summary["gain_sweep"] = rows
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
