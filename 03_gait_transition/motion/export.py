"""Write trajectories, plots and a side-view animation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .analysis import switch_times, transition_metrics
from .gaits import run


def write_csv(samples, path):
    fields = ["time", "gait", "base_x"] + [f"q{i}" for i in range(6)] + [
        f"dq{i}" for i in range(6)
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for sample in samples:
            writer.writerow(
                [sample.time, sample.gait, sample.base_x,
                 *sample.command.tolist(), *sample.velocity.tolist()]
            )


def draw_frame(sample, size=(640, 360)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.line((0, height - 38, width, height - 38), fill=(70, 78, 90), width=3)
    origin_x = width / 2 - 40
    hip_y = height - 175
    shoulder_y = hip_y - 78
    draw.line((origin_x, hip_y, origin_x, shoulder_y), fill=(30, 85, 120), width=10)
    draw.ellipse((origin_x - 16, shoulder_y - 37, origin_x + 16, shoulder_y - 5),
                 fill=(42, 54, 70))

    for side, offset, color in (
        ("left", -10, (222, 112, 24)),
        ("right", 10, (10, 115, 140)),
    ):
        base = 0 if side == "left" else 3
        hip_angle = sample.command[base]
        knee_angle = sample.command[base + 1]
        thigh_length = 72
        shin_length = 70
        knee_x = origin_x + offset + thigh_length * math.sin(hip_angle)
        knee_y = hip_y + thigh_length * math.cos(hip_angle)
        ankle_angle = hip_angle + knee_angle
        ankle_x = knee_x + shin_length * math.sin(ankle_angle)
        ankle_y = knee_y + shin_length * math.cos(ankle_angle)
        draw.line((origin_x + offset, hip_y, knee_x, knee_y), fill=color, width=9)
        draw.line((knee_x, knee_y, ankle_x, ankle_y), fill=color, width=9)
        draw.line((ankle_x - 16, height - 38, ankle_x + 22, height - 38),
                  fill=(30, 34, 42), width=8)
        arm_angle = -sample.command[base] * 0.8
        hand_x = origin_x + (24 if side == "left" else -24) + 47 * math.sin(arm_angle)
        hand_y = shoulder_y + 45 * math.cos(arm_angle)
        draw.line((origin_x, shoulder_y + 8, hand_x, hand_y), fill=color, width=6)

    draw.text((18, 14), f"t={sample.time:5.2f}s  mode={sample.gait}", fill=(20, 24, 32))
    draw.text((18, 34), "orange=left leg   teal=right leg", fill=(70, 78, 90))
    return np.asarray(image)


def render(samples, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = [draw_frame(sample) for index, sample in enumerate(samples) if index % 10 == 0]
    import imageio.v2 as imageio

    imageio.mimsave(output / "gait_transition.gif", frames, duration=0.05, loop=0)
    Image.fromarray(frames[-1]).save(output / "gait_transition.png")


def main(output_dir="results"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scheduler, smooth = run("hermite")
    _, abrupt = run("direct")
    smooth_metrics = transition_metrics(smooth, scheduler)
    abrupt_metrics = transition_metrics(abrupt, scheduler)
    write_csv(smooth, output / "hermite_trajectory.csv")
    write_csv(abrupt, output / "direct_trajectory.csv")
    render(smooth, output)

    times = np.array([sample.time for sample in smooth])
    commands = np.stack([sample.command for sample in smooth])
    velocities = np.stack([sample.velocity for sample in smooth])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for joint in range(6):
        axes[0].plot(times, commands[:, joint], linewidth=1.2, label=f"q{joint + 1}")
        axes[1].plot(times, velocities[:, joint], linewidth=1.0)
    for switch in switch_times(scheduler):
        axes[0].axvspan(switch, switch + scheduler.transition_time, color="orange", alpha=0.08)
        axes[1].axvspan(switch, switch + scheduler.transition_time, color="orange", alpha=0.08)
    axes[0].set_ylabel("joint command (rad)")
    axes[0].set_title("Six-joint gait trajectory with Hermite transition windows")
    axes[0].legend(frameon=False, ncols=6, fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("joint velocity (rad/s)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "joint_trajectories.png", dpi=180, facecolor="white")
    plt.close(fig)

    summary = {
        "hermite": smooth_metrics,
        "direct": abrupt_metrics,
        "velocity_step_reduction_percent": 100.0
        * (1.0 - smooth_metrics["max_velocity_step"] / abrupt_metrics["max_velocity_step"]),
        "acceleration_reduction_percent": 100.0
        * (1.0 - smooth_metrics["max_acceleration"] / abrupt_metrics["max_acceleration"]),
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
