#!/usr/bin/env python3
"""Plot top and side projections of an ASCII PLY produced by Scout LiDAR tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_ascii_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Expected an ASCII PLY file")
        count: int | None = None
        ascii_format = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header ended before end_header")
            words = line.split()
            if words[:2] == ["format", "ascii"]:
                ascii_format = True
            if words[:2] == ["element", "vertex"] and len(words) == 3:
                count = int(words[2])
            if line.strip() == "end_header":
                break
        if not ascii_format or count is None:
            raise ValueError("PLY must declare ASCII format and a vertex count")
        points = np.loadtxt(stream, dtype=np.float32, max_rows=count, usecols=(0, 1, 2))
    points = np.atleast_2d(points)
    if points.shape[0] != count or points.shape[1] != 3:
        raise ValueError(f"Expected {count} XYZ points; read shape {points.shape}")
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="input ASCII PLY")
    parser.add_argument("--output", type=Path, help="PNG path; defaults beside input")
    parser.add_argument("--max-points", type=int, default=50000, help="plotting-only random sample cap")
    args = parser.parse_args()
    if args.max_points < 100:
        parser.error("--max-points must be at least 100")
    points = load_ascii_vertices(args.ply)
    input_count = len(points)
    if len(points) > args.max_points:
        rng = np.random.default_rng(7)
        points = points[rng.choice(len(points), args.max_points, replace=False)]

    output = args.output or args.ply.with_name(args.ply.stem + "_preview.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    projections = (
        (0, 1, "North (m)", "East (m)", "Top view"),
        (0, 2, "North (m)", "Up (m)", "North / height"),
        (1, 2, "East (m)", "Up (m)", "East / height"),
    )
    scatter = None
    for ax, (x, y, x_label, y_label, title) in zip(axes, projections):
        scatter = ax.scatter(
            points[:, x], points[:, y], c=points[:, 2], cmap="viridis",
            s=0.7, linewidths=0, rasterized=True,
        )
        ax.set(title=title, xlabel=x_label, ylabel=y_label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
    assert scatter is not None
    fig.colorbar(scatter, ax=axes, label="Height (m)", shrink=0.8)
    fig.suptitle(f"{args.ply.name}: {len(points):,} plotted points")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"input_points={input_count}")
    print(f"plotted_points={len(points)}")
    print("plotted_sample_bounds_xyz_m=" + ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(mins, maxs)))
    print(f"preview={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
