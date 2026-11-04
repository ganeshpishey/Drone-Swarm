"""Collect one read-only PX4 telemetry snapshot per configured SITL vehicle.

The output can feed the dry-run assignment planner. This script never arms,
flies, changes parameters, or sends any action command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mavsdk_grpc import System


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def first_value(stream: Any, timeout_s: float) -> Any:
    async def receive() -> Any:
        async for value in stream:
            return value
        raise RuntimeError("telemetry stream ended before its first sample")

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def collect_one(vehicle_id: str, config: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    role = str(config.get("role", "unknown"))
    result: dict[str, Any] = {
        "role": role,
        "available": False,
        "connected": False,
        "in_air": None,
        "is_armable": None,
        "battery_remaining_percent": None,
        "sampled_at_utc": utc_now(),
    }
    try:
        # If a server address is configured, attach to that running server
        # instead of spawning a second UDP listener for the same PX4 endpoint.
        drone = System(
            mavsdk_server_address=config.get("mavsdk_server_address"),
            port=int(config["mavsdk_server_port"]),
        )
        print(
            f"{vehicle_id}: connecting to MAVSDK server "
            f"{config.get('mavsdk_server_address') or 'managed'}:{config['mavsdk_server_port']} "
            f"and PX4 UDP {config['mavsdk_port']}...",
            flush=True,
        )
        await asyncio.wait_for(
            drone.connect(system_address=f"udpin://0.0.0.0:{int(config['mavsdk_port'])}"),
            timeout=timeout_s,
        )
        print(f"{vehicle_id}: MAVSDK connect returned; waiting for telemetry...", flush=True)
        connection = await first_value(drone.core.connection_state(), timeout_s)
        result["connected"] = bool(connection.is_connected)
        if not result["connected"]:
            result["status_reason"] = "PX4_not_connected"
            return result

        health, in_air, position, battery = await asyncio.gather(
            first_value(drone.telemetry.health(), timeout_s),
            first_value(drone.telemetry.in_air(), timeout_s),
            first_value(drone.telemetry.position(), timeout_s),
            first_value(drone.telemetry.battery(), timeout_s),
        )
        battery_percent = float(battery.remaining_percent)
        result.update({
            "in_air": bool(in_air),
            "is_armable": bool(health.is_armable),
            "battery_remaining_percent": battery_percent,
            "position": {
                "latitude_deg": float(position.latitude_deg),
                "longitude_deg": float(position.longitude_deg),
                "absolute_altitude_m": float(position.absolute_altitude_m),
                "relative_altitude_m": float(position.relative_altitude_m),
            },
            "sampled_at_utc": utc_now(),
        })
        result["available"] = (
            result["connected"]
            and result["in_air"] is False
            and result["is_armable"] is True
            and battery_percent >= 30.0
        )
        result["status_reason"] = (
            "landed_armable_and_battery_ok" if result["available"]
            else "must_be_connected_landed_armable_and_at_least_30_percent"
        )
    except Exception as error:
        result["status_reason"] = f"telemetry_unavailable: {type(error).__name__}: {error}"
        result["sampled_at_utc"] = utc_now()
    return result


async def collect(config: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    results = await asyncio.gather(*(
        collect_one(vehicle_id, vehicle, timeout_s)
        for vehicle_id, vehicle in config.items()
    ))
    return {vehicle_id: result for vehicle_id, result in zip(config, results)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drones", type=Path, default=Path(__file__).with_name("drones.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/scout_coordinator/vehicle_status.json"))
    parser.add_argument("--timeout-s", type=float, default=8.0)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if not args.drones.is_file():
        parser.error(f"drone configuration not found: {args.drones}")
    config = json.loads(args.drones.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config:
        parser.error("drone configuration must be a non-empty JSON object")

    print(f"Collecting read-only vehicle status for {', '.join(config)}...", flush=True)
    status = asyncio.run(collect(config, args.timeout_s))
    document = {
        "schema_version": "swarm-vehicle-status-v1",
        "generated_at_utc": utc_now(),
        "source": "PX4_MAVSDK_read_only_telemetry",
        "flight_commands_sent": False,
        "vehicles": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for vehicle_id, sample in status.items():
        print(
            f"{vehicle_id}: connected={sample['connected']} in_air={sample['in_air']} "
            f"armable={sample['is_armable']} battery={sample['battery_remaining_percent']} "
            f"available={sample['available']} reason={sample['status_reason']}"
        )
    print(f"status_file={args.output}")
    print("Read-only telemetry only; no flight command was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
