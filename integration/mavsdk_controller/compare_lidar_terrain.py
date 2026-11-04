#!/usr/bin/env python3
"""Overlay Scout's local LiDAR PLY with the apron terrain mesh in three views.

The point cloud is NEU relative to its first GPS pose. The terrain mesh is in
Gazebo world ENU coordinates; this script aligns them using Scout's configured
world spawn pose, so the overlay is a visual sanity check, not surveyed truth.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD = ROOT / "worlds/earthquake_landslide_apron_world.sdf"
DEFAULT_MESH = ROOT / "worlds/earthquake_landslide_assets/meshes/terrain_apron.obj"


def read_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"Not an ASCII PLY: {path}")
        count = None
        is_ascii = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            fields = line.split()
            if fields[:2] == ["format", "ascii"]:
                is_ascii = True
            if fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
            if line.strip() == "end_header":
                break
        if not is_ascii or count is None:
            raise ValueError("Expected an ASCII PLY with a vertex count")
        points = np.loadtxt(f, dtype=np.float32, max_rows=count, usecols=(0, 1, 2))
    points = np.atleast_2d(points)
    if len(points) != count or points.shape[1] != 3:
        raise ValueError(f"Expected {count} PLY points, got {points.shape}")
    return points


def read_terrain_mesh(mesh_path: Path, world_path: Path) -> np.ndarray:
    root = ET.parse(world_path).getroot()
    terrain = root.find(".//world/model[@name='terrain']")
    if terrain is None:
        raise ValueError("Could not find model named 'terrain' in world SDF")
    pose_text = terrain.findtext("pose", "0 0 0 0 0 0").split()
    x, y, z, roll, pitch, yaw = map(float, pose_text[:6])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # SDF model pose rotation: Rz(yaw) * Ry(pitch) * Rx(roll).
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                vals = line.split()
                vertices.append((float(vals[1]), float(vals[2]), float(vals[3])))
    if not vertices:
        raise ValueError(f"No OBJ vertices in {mesh_path}")
    local = np.asarray(vertices, dtype=np.float64)
    world = local @ rotation.T + np.array([x, y, z])
    return world


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="pose-registered NEU ASCII PLY")
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--spawn-x", type=float, default=-76.0, help="Scout Gazebo spawn X (east), metres")
    parser.add_argument("--spawn-y", type=float, default=-61.0, help="Scout Gazebo spawn Y (north), metres")
    parser.add_argument("--spawn-z", type=float, default=1.70, help="Scout Gazebo spawn Z, metres")
    parser.add_argument("--output", type=Path, help="output PNG; defaults beside PLY")
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()
    for file_path in (args.ply, args.world, args.mesh):
        if not file_path.is_file():
            parser.error(f"file not found: {file_path}")

    cloud = read_ply(args.ply)
    terrain_world = read_terrain_mesh(args.mesh, args.world)
    # Convert Gazebo ENU world points to the PLY's local [north,east,up] frame.
    terrain_local = np.column_stack((
        terrain_world[:, 1] - args.spawn_y,
        terrain_world[:, 0] - args.spawn_x,
        terrain_world[:, 2] - args.spawn_z,
    ))

    rng = np.random.default_rng(7)
    if len(cloud) > args.max_points:
        cloud = cloud[rng.choice(len(cloud), args.max_points, replace=False)]
    output = args.output or args.ply.with_name(args.ply.stem + "_terrain_overlay.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    views = ((0, 1, "North (m)", "East (m)", "Top view"),
             (0, 2, "North (m)", "Up (m)", "North / height"),
             (1, 2, "East (m)", "Up (m)", "East / height"))
    for ax, (i, j, xlabel, ylabel, title) in zip(axes, views):
        ax.scatter(terrain_local[:, i], terrain_local[:, j], s=1.0, c="#777777",
                   alpha=0.42, linewidths=0, label="SDF terrain mesh")
        points = ax.scatter(cloud[:, i], cloud[:, j], c=cloud[:, 2], cmap="viridis",
                             s=0.65, alpha=0.75, linewidths=0, label="LiDAR returns")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(points, ax=axes, label="LiDAR height (m above Scout spawn)", shrink=0.8)
    axes[0].legend(loc="upper right", markerscale=4, fontsize=8)
    fig.suptitle("LiDAR cloud vs. known terrain (aligned using configured Scout spawn)")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"cloud_points_plotted={len(cloud)}")
    print(f"terrain_mesh_vertices={len(terrain_local)}")
    print(f"spawn_world_enu_m=[{args.spawn_x:.3f}, {args.spawn_y:.3f}, {args.spawn_z:.3f}]")
    print(f"overlay_png={output}")
    print("alignment_note=spawn-based approximate overlay; GPS origin and mesh geometry can differ slightly")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
