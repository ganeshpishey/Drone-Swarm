"""Read PX4 battery status over the Scout's existing GCS MAVLink UDP port."""

from __future__ import annotations

import argparse
import time

from pymavlink import mavutil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Seconds to send GCS heartbeats and report MAVLink status (default: 15).",
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration cannot be negative")

    connection = mavutil.mavlink_connection(
        "udpin:0.0.0.0:14550", source_system=255, source_component=190
    )
    print(
        "Listening for PX4 MAVLink on UDP 14550; sending GCS heartbeat only "
        f"for {args.duration:g} seconds."
        if args.duration > 0
        else "Listening for PX4 MAVLink on UDP 14550; sending GCS heartbeat only until Ctrl+C."
    )
    connection.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )

    heartbeat = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=10)
    if heartbeat is None:
        print("No PX4 heartbeat received within 10 seconds.")
        return 1
    print(f"PX4 heartbeat received from system {heartbeat.get_srcSystem()}.")

    end_time = time.monotonic() + args.duration if args.duration > 0 else None
    next_gcs_heartbeat = time.monotonic() + 1.0
    battery_count = 0
    try:
        while end_time is None or time.monotonic() < end_time:
            now = time.monotonic()
            if now >= next_gcs_heartbeat:
                connection.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                next_gcs_heartbeat = now + 1.0

            message = connection.recv_match(blocking=True, timeout=0.25)
            if message is None:
                continue

            message_type = message.get_type()
            if message_type == "BATTERY_STATUS":
                battery_count += 1
                voltages = [v / 1000.0 for v in message.voltages if v != 65535]
                current_a = (
                    message.current_battery / 100.0
                    if message.current_battery != -1
                    else float("nan")
                )
                print(
                    f"BATTERY_STATUS id={message.id} "
                    f"volts={voltages} current_a={current_a:.2f} "
                    f"remaining_pct={message.battery_remaining} "
                    f"fault_bitmask=0x{message.fault_bitmask:x} "
                    f"charge_state={message.charge_state}"
                )
            elif message_type == "SYS_STATUS":
                voltage = (
                    message.voltage_battery / 1000.0
                    if message.voltage_battery != 65535
                    else float("nan")
                )
                print(
                    f"SYS_STATUS voltage_v={voltage:.2f} "
                    f"current_battery_cA={message.current_battery} "
                    f"battery_remaining_pct={message.battery_remaining} "
                    f"sensor_health=0x{message.onboard_control_sensors_health:x}"
                )
            elif message_type == "STATUSTEXT":
                print(f"PX4 {message_type}: {message.text}")
    except KeyboardInterrupt:
        print("Stopped GCS heartbeat and MAVLink monitoring.")

    if battery_count == 0:
        print(f"No BATTERY_STATUS message received during the {args.duration:g}-second window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
