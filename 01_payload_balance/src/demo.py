#!/usr/bin/env python3
"""Render a short neural-policy carrying demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

from balance_net import PayloadPolicy
from balance_sim import CarryBalanceRig

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--weights", default=str(ROOT / "models" / "cargo_policy.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    policy = PayloadPolicy.load(args.weights)
    rig = CarryBalanceRig(args.mass)
    rig.reset(args.mass)
    capture_every = max(1, int(round(1.0 / (args.fps * rig.control_dt))))
    frames = []
    steps = int(args.seconds / rig.control_dt)
    for index in range(steps):
        rig.step(policy)
        if index % capture_every == 0:
            frames.append(rig.render())
        if rig.fallen():
            break

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gif = output / "cargo_balance.gif"
    png = output / "cargo_balance.png"
    imageio.mimsave(gif, frames, duration=1.0 / args.fps, loop=0)
    Image.fromarray(frames[-1]).save(png)
    print(f"[saved] {gif}")
    print(f"[saved] {png}")


if __name__ == "__main__":
    main()
