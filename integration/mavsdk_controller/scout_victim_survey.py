"""Fly Scout from its apron pad to a survey waypoint, capture data, and RTL.

This uses PX4's high-level goto action, not Offboard setpoints. It is intended
for the existing three-vehicle SITL session; only Scout is commanded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controller import PX4Controller
from lidar_pose_mapper import LidarPoseMapper


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "scout_hover_test.json"
CAPTURE = HERE / "capture_scout_frame.py"


class SurveyInterrupted(Exception):
    """Raised when the operator requests a controlled RTL with Ctrl+C."""


async def run(args: argparse.Namespace, stop_requested: asyncio.Event) -> None:
    settings: dict[str, dict[str, Any]] = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )
    scout = settings["scout_0"]
    controller = PX4Controller(
        f"udpin://0.0.0.0:{int(scout['mavsdk_port'])}",
        mavsdk_server_port=int(scout["mavsdk_server_port"]),
    )
    airborne = False
    connected = False
    lidar_mapper: LidarPoseMapper | None = None
    lidar_started = False
    pose_tasks: list[asyncio.Task[None]] = []
    last_pose_sample: dict[str, float] = {}

    async def wait_or_operator_stop(awaitable: Any, operation: str) -> Any:
        task = asyncio.create_task(awaitable)
        stop_task = asyncio.create_task(stop_requested.wait())
        done, _ = await asyncio.wait(
            (task, stop_task), return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise SurveyInterrupted(
                f"Ctrl+C received during {operation}; requesting RTL."
            )
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        return task.result()

    async def log_pose_streams() -> None:
        if args.pose_log is None:
            return
        args.pose_log.parent.mkdir(parents=True, exist_ok=True)

        async def write_sample(sample_type: str, sample: dict[str, Any]) -> None:
            now_monotonic = time.monotonic()
            if now_monotonic - last_pose_sample.get(sample_type, -1e9) < 0.5:
                return
            last_pose_sample[sample_type] = now_monotonic
            record = {
                "schema_version": "scout-pose-sample-v1",
                "sample_type": sample_type,
                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "host_monotonic_s": time.monotonic(),
                **sample,
            }
            with args.pose_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

        async def positions() -> None:
            async for position in controller.drone.telemetry.position():
                await write_sample("position", {
                    "position": {
                        "latitude_deg": float(position.latitude_deg),
                        "longitude_deg": float(position.longitude_deg),
                        "absolute_altitude_m": float(position.absolute_altitude_m),
                        "relative_altitude_m": float(position.relative_altitude_m),
                    },
                })

        async def attitudes() -> None:
            async for attitude in controller.drone.telemetry.attitude_euler():
                await write_sample("attitude", {
                    "attitude": {
                        "roll_deg": float(attitude.roll_deg),
                        "pitch_deg": float(attitude.pitch_deg),
                        "yaw_deg": float(attitude.yaw_deg),
                    },
                })

        pose_tasks.extend((
            asyncio.create_task(positions(), name="scout-pose-position-log"),
            asyncio.create_task(attitudes(), name="scout-pose-attitude-log"),
        ))
        print(f"Logging timestamped Scout position and attitude to {args.pose_log}")

    async def log_snapshot(event: str) -> None:
        record = {"event": event, **await controller.telemetry_snapshot()}
        args.telemetry_log.parent.mkdir(parents=True, exist_ok=True)
        with args.telemetry_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))

    async def wait_for_takeoff_altitude(minimum_m: float, timeout_s: float) -> None:
        latest_altitude: float | None = None
        next_report = asyncio.get_running_loop().time()

        async def monitor() -> None:
            nonlocal latest_altitude, next_report
            async for position in controller.drone.telemetry.position():
                latest_altitude = float(position.relative_altitude_m)
                now = asyncio.get_running_loop().time()
                if now >= next_report:
                    print(
                        f"Scout altitude: {latest_altitude:.2f} m "
                        f"(target at least {minimum_m:.2f} m)"
                    )
                    next_report = now + 2.0
                if latest_altitude >= minimum_m:
                    return

        try:
            await asyncio.wait_for(monitor(), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            observed = (
                "no altitude samples received"
                if latest_altitude is None
                else f"last altitude sample was {latest_altitude:.2f} m"
            )
            raise TimeoutError(
                f"Scout did not reach {minimum_m:.2f} m within {timeout_s:.0f} s; "
                f"{observed}. Aborting survey and returning to launch."
            ) from error

    try:
        print("Connecting to Scout at UDP 14540...")
        await controller.connect(timeout_s=45.0)
        connected = True
        await log_pose_streams()
        if args.lidar_map:
            lidar_mapper = LidarPoseMapper(args.lidar_map)
            try:
                await lidar_mapper.start(controller.drone)
                lidar_started = True
                print(f"Pose-registered Scout LiDAR recording to {args.lidar_map}")
            except Exception as error:
                print(f"Could not start LiDAR mapping capture: {error}")
        print(f"Arming and taking off vertically to {args.altitude:.1f} m...")
        await controller.arm()
        await controller.takeoff(args.altitude)
        airborne = True
        await wait_for_takeoff_altitude(
            args.minimum_altitude, timeout_s=args.takeoff_timeout
        )
        await controller.hold()

        print(
            "Flying Scout to the configured survey waypoint using PX4 goto action: "
            f"north={args.north:.1f} m, east={args.east:.1f} m, "
            f"altitude={args.altitude:.1f} m, yaw={args.yaw_deg:.1f} deg relative to home."
        )
        await wait_or_operator_stop(
            controller.goto_location(
                args.north,
                args.east,
                args.altitude,
                timeout_s=args.waypoint_timeout,
                yaw_deg=args.yaw_deg,
            ),
            "waypoint navigation",
        )
        await controller.hold()
        await log_snapshot("victim_area_hover_start")
        print(
            f"Scout is holding at the survey waypoint for {args.hold_seconds:.0f} s. "
            "Keep the live detector running during this hold."
        )
        await wait_or_operator_stop(
            asyncio.sleep(args.hold_seconds), "waypoint data collection"
        )
        await log_snapshot("victim_area_hover_end")

        if args.skip_frame_capture:
            print(
                "Skipping one-frame capture; the live detector is expected to be "
                "the only receiver on UDP camera port 5600."
            )
        else:
            print("Capturing a fresh Scout camera frame...")
            capture = await asyncio.create_subprocess_exec(
                "/usr/bin/python3",
                str(CAPTURE),
                "--timeout",
                str(args.capture_timeout),
                "--output",
                str(args.output),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            capture_output, _ = await capture.communicate()
            if capture_output:
                print(capture_output.decode(errors="replace").rstrip())
            if capture.returncode != 0:
                print("Camera capture failed; Scout will still return to its launch area.")

    except Exception as error:
        print(f"Scout survey stopped: {error}")
    finally:
        if connected and airborne:
            try:
                print("Requesting PX4 return-to-launch and landing...")
                await asyncio.wait_for(
                    controller.drone.action.return_to_launch(), timeout=15.0
                )
                await controller.wait_until_landed(timeout_s=args.landing_timeout)
                print("PX4 telemetry confirms Scout landed.")
                await log_snapshot("returned_and_landed")
            except Exception as error:
                print(f"Could not confirm return and landing: {error}")
                print("Check Scout in Gazebo and use PX4/Gazebo controls if needed.")
        for task in pose_tasks:
            if not task.done():
                task.cancel()
        if pose_tasks:
            await asyncio.gather(*pose_tasks, return_exceptions=True)
        if lidar_started and lidar_mapper is not None:
            try:
                print("Stopping LiDAR mapping capture...")
                print(json.dumps(await lidar_mapper.stop(), indent=2))
            except Exception as error:
                print(f"Could not finalize LiDAR mapping output: {error}")
        print(f"Telemetry log: {args.telemetry_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--north", type=float, default=5.0)
    parser.add_argument("--east", type=float, default=8.0)
    parser.add_argument("--altitude", type=float, default=8.0)
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="PX4 heading at the survey waypoint in degrees; default 0 preserves prior behavior.",
    )
    parser.add_argument(
        "--minimum-altitude",
        type=float,
        default=7.0,
        help="Altitude gate before proceeding (lower than target to allow slow SITL climbs).",
    )
    parser.add_argument("--takeoff-timeout", type=float, default=200.0)
    parser.add_argument("--waypoint-timeout", type=float, default=180.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=35.0,
        help="Time to hover at the survey waypoint while collecting sensor data.",
    )
    parser.add_argument("--landing-timeout", type=float, default=180.0)
    parser.add_argument("--capture-timeout", type=float, default=20.0)
    parser.add_argument(
        "--lidar-map",
        type=Path,
        help="record Scout LiDAR scans and register them with MAVSDK position/attitude into a local XYZ PLY",
    )
    parser.add_argument(
        "--pose-log",
        type=Path,
        help="append UTC-stamped Scout position and attitude samples for camera/pose synchronization",
    )
    parser.add_argument(
        "--skip-frame-capture",
        action="store_true",
        help="Do not open a second receiver on camera UDP port 5600 while live detection is running.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/scout_day7_victim_area.ppm")
    )
    parser.add_argument(
        "--telemetry-log",
        type=Path,
        default=Path("/tmp/scout-day7-survey.jsonl"),
    )
    args = parser.parse_args()
    if not 0.0 < args.minimum_altitude <= args.altitude:
        parser.error("--minimum-altitude must be greater than 0 and no higher than --altitude")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds cannot be negative")
    async def run_with_signal_handling() -> None:
        loop = asyncio.get_running_loop()
        stop_requested = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, stop_requested.set)
        try:
            await run(args, stop_requested)
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    asyncio.run(run_with_signal_handling())


if __name__ == "__main__":
    main()
