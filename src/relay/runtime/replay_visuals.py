"""Side-effect-free replay rendering and media export."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from relay.envs.visibility import visibility_mask
from relay.runtime.replay_store import REPLAY_SCHEMA_VERSION

DEFAULT_OVERLAYS = {
    "grid",
    "visibility",
    "incidents",
    "actions",
    "communications",
    "rewards",
    "hash",
}

ROLE_COLORS = {
    "scout": "#2E74B5",
    "ambulance": "#2E7D32",
    "fireman": "#C44E32",
}

DRONE_ROLE_NAMES = {
    "scout": "Reconnaissance Drone",
    "ambulance": "Medical-Support Drone",
    "fireman": "Fire-Response Drone",
}


class ReplayData:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.rows: list[dict[str, Any]] = pq.read_table(self.path).to_pylist()
        if not self.rows:
            raise ValueError("replay contains no rows")
        if any(row["schema_version"] != REPLAY_SCHEMA_VERSION for row in self.rows):
            raise ValueError("unsupported replay schema")
        self.environment_config: dict[str, Any] = json.loads(
            self.rows[0]["environment_config_json"]
        )
        self.snapshots = [json.loads(row["snapshot_json"]) for row in self.rows]
        self.actions = [json.loads(row["actions_json"]) for row in self.rows]
        self.infos = [json.loads(row["infos_json"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def agents(self) -> list[str]:
        return list(self.snapshots[0]["agents"])

    def inspection(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        first_agent = self.agents[0]
        info = self.infos[index][first_agent]
        snapshot = self.snapshots[index]
        unresolved = [
            incident_id
            for incident_id, incident in snapshot["incidents"].items()
            if incident["status"] != "resolved"
        ]
        return {
            "replay": str(self.path),
            "tick": row["tick"],
            "episode": row["episode"],
            "scenario_seed": row["scenario_seed"],
            "scenario_id": row["scenario_id"],
            "state_hash": row["state_hash"],
            "run_id": row.get("run_id", "unknown"),
            "checkpoint_id": row.get("checkpoint_id", "unknown"),
            "checkpoint_step": row.get("checkpoint_step", -1),
            "actions": self.actions[index],
            "reward_components": info["reward_components"],
            "recipient_attempts": info["recipient_attempts"],
            "packet_events": info["packet_events"],
            "unresolved_incidents": unresolved,
            "terminal_reason": info["terminal_reason"],
        }


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = (
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _perspective_mask(data: ReplayData, index: int, perspective: str) -> np.ndarray | None:
    if perspective == "global":
        return None
    snapshot = data.snapshots[index]
    if perspective not in snapshot["agents"]:
        raise ValueError(f"unknown perspective {perspective!r}")
    agent = snapshot["agents"][perspective]
    walls = np.asarray(snapshot["walls"], dtype=np.bool_)
    radius = (
        int(data.environment_config["scout_sensor_radius"])
        if agent["role"] == "scout"
        else int(data.environment_config["specialist_sensor_radius"])
    )
    return visibility_mask(walls, (int(agent["x"]), int(agent["y"])), radius)


def _is_drone_replay(data: ReplayData) -> bool:
    """Return whether a replay uses the fixed-altitude drone presentation layer."""

    return bool(data.environment_config.get("drone_mode", False))


def _iso_point(
    x: float,
    y: float,
    z: float,
    *,
    origin_x: float,
    origin_y: float,
    tile_width: float,
    tile_height: float,
) -> tuple[float, float]:
    return (
        origin_x + (x - y) * tile_width / 2.0,
        origin_y + (x + y) * tile_height / 2.0 - z,
    )


def _render_drone_isometric_frame(
    data: ReplayData,
    index: int,
    *,
    perspective: str,
    overlays: set[str],
    cell_size: int,
) -> Image.Image:
    """Render the discrete 3D drone scenario as an isometric mission view.

    This changes presentation only: the recorded policy still navigates on the
    original horizontal grid. Drone altitude and service phases are replayed
    metadata derived from the existing three-tick INTERACT operation.
    """

    snapshot = data.snapshots[index]
    row = data.rows[index]
    walls = np.asarray(snapshot["walls"], dtype=np.bool_)
    height, width = walls.shape
    tile_width = float(max(30, int(cell_size * 1.55)))
    tile_height = float(max(15, int(cell_size * 0.75)))
    building_height = float(max(32, int(cell_size * 1.35)))
    drone_height = float(max(40, int(cell_size * 1.65)))
    margin = 36
    panel_width = 390
    map_width = int((width + height) * tile_width / 2.0 + margin * 2)
    map_height = int((width + height) * tile_height / 2.0 + building_height + margin * 3)
    image = Image.new("RGB", (map_width + panel_width, map_height), "#EAF2F8")
    draw = ImageDraw.Draw(image)
    origin_x = margin + width * tile_width / 2.0
    origin_y = margin + building_height + tile_height / 2.0

    def diamond(x: int, y: int, z: float = 0.0) -> list[tuple[float, float]]:
        centre_x, centre_y = _iso_point(
            x, y, z,
            origin_x=origin_x,
            origin_y=origin_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        return [
            (centre_x, centre_y - tile_height / 2.0),
            (centre_x + tile_width / 2.0, centre_y),
            (centre_x, centre_y + tile_height / 2.0),
            (centre_x - tile_width / 2.0, centre_y),
        ]

    def is_damaged_structure(x: int, y: int) -> bool:
        """Stable cosmetic classification used only by the disaster renderer."""

        return (x * 31 + y * 17) % 5 in (0, 1)

    def has_rubble(x: int, y: int) -> bool:
        if walls[y, x]:
            return False
        adjacent_building = any(
            0 <= nx < width and 0 <= ny < height and walls[ny, nx]
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        )
        return adjacent_building and (x * 19 + y * 23) % 4 == 0

    # Draw rear cells before front cells so raised buildings occlude correctly.
    for y in range(height):
        for x in range(width):
            ground = diamond(x, y)
            if walls[y, x]:
                height_scale = 0.58 if is_damaged_structure(x, y) else 1.0
                structure_height = building_height * height_scale
                top = diamond(x, y, structure_height)
                draw.polygon(
                    [top[1], ground[1], ground[2], top[2]], fill="#455A64", outline="#263238"
                )
                draw.polygon(
                    [top[2], ground[2], ground[3], top[3]], fill="#37474F", outline="#263238"
                )
                draw.polygon(
                    top,
                    fill="#78909C" if height_scale == 1.0 else "#8D6E63",
                    outline="#263238",
                )
                if height_scale < 1.0:
                    draw.line((top[0], top[2]), fill="#4E342E", width=2)
                    draw.line((top[3], top[1]), fill="#4E342E", width=2)
            else:
                fill = "#CCD9C4" if not snapshot["staging"][y][x] else "#C6DFF1"
                draw.polygon(ground, fill=fill, outline="#B1C2B0")
                if has_rubble(x, y):
                    cx, cy = _iso_point(
                        x,
                        y,
                        2.0,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        tile_width=tile_width,
                        tile_height=tile_height,
                    )
                    for offset_x, offset_y in ((-6, 1), (1, -2), (7, 2)):
                        draw.ellipse(
                            (cx + offset_x - 2, cy + offset_y - 2, cx + offset_x + 2, cy + offset_y + 2),
                            fill="#8D6E63",
                        )

    def ground_marker(x: int, y: int, color: str, label: str) -> None:
        cx, cy = _iso_point(
            x, y, 5.0,
            origin_x=origin_x,
            origin_y=origin_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        marker_radius = max(8, int(tile_height / 2.4))
        draw.ellipse(
            (cx - marker_radius, cy - marker_radius, cx + marker_radius, cy + marker_radius),
            fill=color,
            outline="white",
            width=2,
        )
        draw.text((cx - 5, cy - 8), label, fill="white", font=_font(13))

    if "incidents" in overlays:
        for incident in snapshot["incidents"].values():
            if incident["status"] == "resolved":
                continue
            ground_marker(
                int(incident["x"]),
                int(incident["y"]),
                "#D32F2F" if incident["kind"] == "victim" else "#EF6C00",
                "P" if incident["kind"] == "victim" else "F",
            )
            if incident["kind"] == "fire":
                cx, cy = _iso_point(
                    int(incident["x"]),
                    int(incident["y"]),
                    9.0,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )
                draw.polygon(
                    [(cx, cy - 10), (cx + 6, cy + 5), (cx, cy + 2), (cx - 6, cy + 5)],
                    fill="#FFB300",
                )

    positions: dict[str, tuple[float, float]] = {}
    cruise_altitude = float(data.environment_config.get("drone_cruise_altitude_m", 30.0))
    for agent_id, agent in snapshot["agents"].items():
        altitude = float(agent.get("altitude_m", cruise_altitude))
        z = drone_height * altitude / max(cruise_altitude, 1.0)
        cx, cy = _iso_point(
            float(agent["x"]),
            float(agent["y"]),
            z,
            origin_x=origin_x,
            origin_y=origin_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        positions[agent_id] = (cx, cy)
        ground_x, ground_y = _iso_point(
            float(agent["x"]), float(agent["y"]), 0.0,
            origin_x=origin_x,
            origin_y=origin_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        draw.line((ground_x, ground_y, cx, cy), fill="#90A4AE", width=1)
        rotor = max(7, int(tile_height / 2.8))
        arm = rotor + 4
        draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill="#263238", width=2)
        draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill="#263238", width=2)
        for rotor_x, rotor_y in (
            (cx - arm, cy - arm),
            (cx + arm, cy + arm),
            (cx - arm, cy + arm),
            (cx + arm, cy - arm),
        ):
            draw.ellipse(
                (rotor_x - rotor, rotor_y - 2, rotor_x + rotor, rotor_y + 2),
                outline="#263238",
                width=1,
            )
        body_radius = max(7, int(tile_height / 2.5))
        draw.ellipse(
            (cx - body_radius, cy - body_radius, cx + body_radius, cy + body_radius),
            fill=ROLE_COLORS[str(agent["role"])],
            outline="#FFD54F" if agent_id == perspective else "white",
            width=3 if agent_id == perspective else 1,
        )
        draw.text((cx - 5, cy - 8), agent_id[0].upper(), fill="white", font=_font(12))
        phase = str(agent.get("service_phase", "cruise")).upper()
        draw.text((cx - 24, cy + body_radius + 4), phase, fill="#243447", font=_font(10))

    first_agent = data.agents[0]
    info = data.infos[index][first_agent]
    if "communications" in overlays:
        event_colors = {"queued": "#7B1FA2", "delivered": "#00838F", "dropped": "#C62828"}
        for event in info["packet_events"]:
            sender = event.get("sender_id")
            recipient = event.get("recipient_id")
            if sender in positions and recipient in positions:
                draw.line(
                    (*positions[sender], *positions[recipient]),
                    fill=event_colors.get(event["event"], "#6B7280"),
                    width=3,
                )

    panel_x = map_width + 20
    draw.text((panel_x, margin), "DRONE SWARM 3D", fill="#17365D", font=_font(21))
    draw.text((panel_x, margin + 28), "Post-earthquake urban rescue replay", fill="#51606E", font=_font(12))
    lines = [
        f"Tick: {row['tick']} / {len(data)}",
        f"Cruise altitude: {cruise_altitude:.0f} m",
        f"Building height: {data.environment_config.get('building_height_m', 18.0):.0f} m",
        f"Communication: {data.environment_config['communication_mode']}",
        f"Scenario: {row['scenario_id']}",
        "",
        "Roles:",
        "  S  Reconnaissance Drone",
        "  A  Medical-Support Drone",
        "  F  Fire-Response Drone",
        "",
        "Service sequence:",
        "  DESCEND → SERVICE → ASCEND",
        "",
        "Scene: damaged buildings + rubble",
        "Flight: discrete 3D airspace model",
        f"Recipient attempts: {info['recipient_attempts']}",
    ]
    if info["terminal_reason"]:
        lines.append(f"Terminal: {info['terminal_reason']}")
    y_cursor = margin + 60
    for line in lines:
        draw.text((panel_x, y_cursor), line, fill="#27364A", font=_font(12))
        y_cursor += 20
    return image


def render_replay_frame(
    data: ReplayData,
    index: int,
    *,
    perspective: str = "global",
    overlays: set[str] | None = None,
    cell_size: int = 32,
) -> Image.Image:
    if not 0 <= index < len(data):
        raise IndexError(index)
    enabled = DEFAULT_OVERLAYS if overlays is None else overlays
    if _is_drone_replay(data):
        return _render_drone_isometric_frame(
            data,
            index,
            perspective=perspective,
            overlays=enabled,
            cell_size=cell_size,
        )
    snapshot = data.snapshots[index]
    row = data.rows[index]
    walls = np.asarray(snapshot["walls"], dtype=np.bool_)
    staging = np.asarray(snapshot["staging"], dtype=np.bool_)
    height, width = walls.shape
    panel_width = 430
    margin = 20
    image = Image.new(
        "RGB",
        (width * cell_size + panel_width + margin * 3, height * cell_size + margin * 2),
        "#F4F7FB",
    )
    draw = ImageDraw.Draw(image)
    ox = oy = margin
    visibility = _perspective_mask(data, index, perspective)
    for y in range(height):
        for x in range(width):
            box = (
                ox + x * cell_size,
                oy + y * cell_size,
                ox + (x + 1) * cell_size,
                oy + (y + 1) * cell_size,
            )
            fill = "#FFFFFF"
            if staging[y, x]:
                fill = "#DCEAF7"
            if walls[y, x]:
                fill = "#26364A"
            if visibility is not None and not visibility[y, x] and "visibility" in enabled:
                fill = "#B7BEC8" if not walls[y, x] else "#556170"
            outline = "#CBD2DA" if "grid" in enabled else fill
            draw.rectangle(box, fill=fill, outline=outline)
    if "incidents" in enabled:
        for incident in snapshot["incidents"].values():
            if incident["status"] == "resolved":
                continue
            x, y = int(incident["x"]), int(incident["y"])
            if visibility is not None and not visibility[y, x]:
                continue
            cx = ox + x * cell_size + cell_size // 2
            cy = oy + y * cell_size + cell_size // 2
            radius = max(6, cell_size // 3)
            color = "#D32F2F" if incident["kind"] == "victim" else "#F57C00"
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
            label = "V" if incident["kind"] == "victim" else "F"
            draw.text((cx - 5, cy - 8), label, fill="white", font=_font(13))
            if incident["status"] == "reserved":
                draw.ellipse(
                    (cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3),
                    outline="#8A6500",
                    width=3,
                )
    positions: dict[str, tuple[int, int]] = {}
    for agent_id, agent in snapshot["agents"].items():
        x, y = int(agent["x"]), int(agent["y"])
        positions[agent_id] = (x, y)
        if visibility is not None and not visibility[y, x] and agent_id != perspective:
            continue
        cx = ox + x * cell_size + cell_size // 2
        cy = oy + y * cell_size + cell_size // 2
        radius = max(8, cell_size // 3)
        draw.rounded_rectangle(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            radius=5,
            fill=ROLE_COLORS[str(agent["role"])],
            outline="#FFD54F" if agent_id == perspective else "white",
            width=3 if agent_id == perspective else 1,
        )
        draw.text((cx - 6, cy - 8), agent_id[0].upper(), fill="white", font=_font(13))
        if int(agent["busy_remaining"]) > 0:
            draw.text(
                (cx - 10, cy + radius + 1),
                f"{agent['busy_remaining']}t",
                fill="#6A4A00",
                font=_font(11),
            )
    if "actions" in enabled:
        direction = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
        for agent_id, action in data.actions[index].items():
            physical = int(action["physical"])
            if physical not in direction or agent_id not in positions:
                continue
            x, y = positions[agent_id]
            dx, dy = direction[physical]
            cx = ox + x * cell_size + cell_size // 2
            cy = oy + y * cell_size + cell_size // 2
            draw.line(
                (cx, cy, cx + dx * cell_size * 0.7, cy + dy * cell_size * 0.7),
                fill="#111827",
                width=3,
            )
    first_agent = data.agents[0]
    info = data.infos[index][first_agent]
    if "communications" in enabled:
        event_colors = {"queued": "#7B1FA2", "delivered": "#00838F", "dropped": "#C62828"}
        for event in info["packet_events"]:
            sender = event.get("sender_id")
            recipient = event.get("recipient_id")
            if sender not in positions or recipient not in positions:
                continue
            sx, sy = positions[sender]
            rx, ry = positions[recipient]
            draw.line(
                (
                    ox + sx * cell_size + cell_size // 2,
                    oy + sy * cell_size + cell_size // 2,
                    ox + rx * cell_size + cell_size // 2,
                    oy + ry * cell_size + cell_size // 2,
                ),
                fill=event_colors.get(event["event"], "#6B7280"),
                width=2,
            )
    panel_x = ox + width * cell_size + margin
    title_font = _font(20)
    body_font = _font(13)
    draw.text((panel_x, margin), "RELAY replay", fill="#17365D", font=title_font)
    lines = [
        f"Tick: {row['tick']} / {len(data)}",
        f"Communication: {data.environment_config['communication_mode']}",
        f"Perspective: {perspective}",
        f"Scenario: {row['scenario_id']}",
        f"Seed: {row['scenario_seed']}",
        f"Run: {row.get('run_id', 'unknown')}",
        f"Checkpoint: {row.get('checkpoint_id', 'unknown')}",
        f"Attempts: {info['recipient_attempts']}",
    ]
    if "rewards" in enabled:
        lines.append("Reward components:")
        lines.extend(f"  {key}: {value:+.3f}" for key, value in info["reward_components"].items())
    if info["terminal_reason"]:
        lines.append(f"Terminal: {info['terminal_reason']}")
    y_cursor = margin + 38
    for line in lines:
        draw.text((panel_x, y_cursor), line[:58], fill="#27364A", font=body_font)
        y_cursor += 21
    if "hash" in enabled:
        y_cursor += 8
        draw.text((panel_x, y_cursor), "State hash", fill="#17365D", font=body_font)
        y_cursor += 20
        state_hash = str(row["state_hash"])
        draw.text((panel_x, y_cursor), state_hash[:32], fill="#5F6B7A", font=_font(11))
        draw.text((panel_x, y_cursor + 16), state_hash[32:], fill="#5F6B7A", font=_font(11))
    return image


def export_replay(
    data: ReplayData,
    output: str | Path,
    *,
    start: int = 0,
    end: int | None = None,
    fps: float = 4.0,
    perspective: str = "global",
    overlays: set[str] | None = None,
    stride: int = 1,
) -> Path:
    output_path = Path(output).resolve()
    final = len(data) - 1 if end is None else end
    if start < 0 or final >= len(data) or start > final:
        raise ValueError("invalid export tick range")
    if stride < 1:
        raise ValueError("replay export stride must be positive")
    frames = [
        np.asarray(render_replay_frame(data, index, perspective=perspective, overlays=overlays))
        for index in range(start, final + 1, stride)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        Image.fromarray(frames[0]).save(output_path)
    elif suffix == ".gif":
        imageio.mimsave(output_path, cast(Any, frames), duration=1000 / fps, loop=0)
    elif suffix == ".mp4":
        writer: Any = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        with writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ValueError("export extension must be .png, .gif, or .mp4")
    return output_path
