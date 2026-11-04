"""Send GCS heartbeats to each local PX4 instance and report responses.

This is a no-flight diagnostic. It uses the normal PX4 MAVLink UDP ports
created by launch_three_px4.sh and binds the shared GCS port 14550.
"""

from __future__ import annotations

import argparse
import re
import select
import socket
import time

from pymavlink import mavutil


DEFAULT_TARGETS = {
    "scout_0": 18570,
    "ambulance_0": 18571,
    "fireman_0": 18572,
}

FIREMAN_SENSOR_COMMANDS = (
    "listener sensor_accel -n 5",
    "listener sensor_gyro -n 5",
    "sensors status",
)


def render_terminal(data: bytes) -> str:
    """Render the small subset of ANSI terminal control used by pxh."""
    text = data.decode("utf-8", errors="replace")
    lines: list[str] = []
    line: list[str] = []
    cursor = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\x1b" and index + 1 < len(text) and text[index + 1] == "[":
            end = index + 2
            while end < len(text) and not ("@" <= text[end] <= "~"):
                end += 1
            if end < len(text):
                sequence = text[index + 2 : end]
                if text[end] == "K":
                    if sequence == "2":
                        line.clear()
                        cursor = 0
                    else:
                        del line[cursor:]
                index = end + 1
                continue
        if char == "\r":
            cursor = 0
        elif char == "\n":
            rendered = "".join(line).strip()
            if rendered:
                lines.append(rendered)
            line.clear()
            cursor = 0
        elif char in ("\b", "\x7f"):
            cursor = max(0, cursor - 1)
        elif char >= " " and char != "\x7f":
            if cursor < len(line):
                line[cursor] = char
            else:
                line.extend(" " for _ in range(cursor - len(line)))
                line.append(char)
            cursor += 1
        index += 1
    rendered = "".join(line).strip()
    if rendered:
        lines.append(rendered)
    # pxh redraws the editable command line after each keypress. Keep only its
    # final rendering, while retaining all completed command output lines.
    cleaned: list[str] = []
    for item in lines:
        item = re.sub(r"^pxh>\s*", "", item)
        if item and (not cleaned or item != cleaned[-1]):
            cleaned.append(item)
    return "\n".join(cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument(
        "--fireman-sensor-diagnostic",
        action="store_true",
        help="Run read-only PX4 shell sensor commands on Fireman (system ID 3).",
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration cannot be negative; use 0 to run until Ctrl+C")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 14550))
    encoder = mavutil.mavlink.MAVLink(None, srcSystem=255, srcComponent=190)
    heartbeat = encoder.heartbeat_encode(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    ).pack(encoder)
    parser_mav = mavutil.mavlink.MAVLink(None)
    if args.fireman_sensor_diagnostic and args.duration == 0:
        parser.error("--fireman-sensor-diagnostic requires a finite --duration")
    print("Sending GCS heartbeats to each PX4 normal MAVLink port; no flight commands.")
    if args.fireman_sensor_diagnostic:
        print("Read-only Fireman PX4 shell diagnostic enabled (system ID 3).")
    if args.duration == 0:
        print("Listening on UDP 14550 until Ctrl+C.")
    else:
        print("Listening on UDP 14550 for PX4 responses for %.0f seconds." % args.duration)

    end = time.monotonic() + args.duration if args.duration > 0 else None
    next_send = 0.0
    seen_heartbeats: set[int] = set()
    shell_command_index = 0
    shell_command_deadline = 0.0
    shell_command_started = False
    shell_reply_bytes = 0
    next_shell_poll = 0.0
    shell_output = bytearray()
    shell_command_outputs: list[str] = []

    def send_shell_command(command: str) -> None:
        # PX4 POSIX pxh executes a remote command only after receiving '\n'.
        payload = (command + "\n").encode("ascii")
        for offset in range(0, len(payload), 70):
            chunk = payload[offset : offset + 70]
            data = list(chunk) + [0] * (70 - len(chunk))
            packet = encoder.serial_control_encode(
                mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
                mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
                | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
                0,
                0,
                len(chunk),
                data,
            ).pack(encoder)
            sock.sendto(packet, ("127.0.0.1", DEFAULT_TARGETS["fireman_0"]))

    def poll_shell() -> None:
        packet = encoder.serial_control_encode(
            mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL,
            mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
            | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
            0,
            0,
            0,
            [0] * 70,
        ).pack(encoder)
        sock.sendto(packet, ("127.0.0.1", DEFAULT_TARGETS["fireman_0"]))

    try:
        while end is None or time.monotonic() < end:
            now = time.monotonic()
            if now >= next_send:
                for role, port in DEFAULT_TARGETS.items():
                    sock.sendto(heartbeat, ("127.0.0.1", port))
                next_send = now + 1.0
            if (
                args.fireman_sensor_diagnostic
                and 3 in seen_heartbeats
                and not shell_command_started
            ):
                command = FIREMAN_SENSOR_COMMANDS[shell_command_index]
                shell_output.clear()
                send_shell_command(command)
                shell_command_started = True
                shell_command_deadline = now + 4.0
                next_shell_poll = now
                print(f"PX4 shell command sent: {command}")
            elif (
                args.fireman_sensor_diagnostic
                and shell_command_started
                and now >= shell_command_deadline
            ):
                rendered = render_terminal(bytes(shell_output))
                print(f"PX4 shell output for {FIREMAN_SENSOR_COMMANDS[shell_command_index]}:")
                print(rendered if rendered else "<no readable output>")
                shell_command_outputs.append(rendered)
                shell_command_index += 1
                if shell_command_index >= len(FIREMAN_SENSOR_COMMANDS):
                    print("PX4 shell diagnostic commands complete.")
                    break
                command = FIREMAN_SENSOR_COMMANDS[shell_command_index]
                shell_output.clear()
                send_shell_command(command)
                shell_command_deadline = now + 4.0
                next_shell_poll = now
                print(f"PX4 shell command sent: {command}")
            if (
                args.fireman_sensor_diagnostic
                and shell_command_started
                and now >= next_shell_poll
            ):
                poll_shell()
                next_shell_poll = now + 0.1
            wait_s = 0.2 if end is None else min(0.2, max(0.0, end - now))
            readable, _, _ = select.select([sock], [], [], wait_s)
            if not readable:
                continue
            packet, address = sock.recvfrom(65535)
            for byte in packet:
                message = parser_mav.parse_char(bytes([byte]))
                if message is None:
                    continue
                kind = message.get_type()
                system_id = message.get_srcSystem()
                if kind == "HEARTBEAT" and system_id not in seen_heartbeats:
                    seen_heartbeats.add(system_id)
                    print(f"PX4 heartbeat response: system={system_id} from={address[0]}:{address[1]}")
                elif kind == "STATUSTEXT":
                    print(f"PX4 system={system_id} STATUSTEXT: {message.text}")
                elif kind == "SERIAL_CONTROL":
                    if message.device == mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL:
                        output = bytes(message.data[: message.count]).decode(
                            "utf-8", errors="replace"
                        )
                        if output.strip("\x00\r\n"):
                            shell_reply_bytes += message.count
                            shell_output.extend(output.encode("utf-8", errors="replace"))
    except KeyboardInterrupt:
        print("Stopped fleet GCS heartbeat diagnostic.")
    finally:
        sock.close()

    if args.fireman_sensor_diagnostic:
        if 3 not in seen_heartbeats:
            print("No Fireman heartbeat observed; PX4 shell commands were not sent.")
            return 1
        if shell_command_index < len(FIREMAN_SENSOR_COMMANDS) - 1:
            print("PX4 shell diagnostic did not complete all commands.")
            return 1
        if shell_reply_bytes == 0:
            print("No PX4 shell output received; sensor values were not verified.")
            return 1
        combined_output = "\n".join(shell_command_outputs).lower()
        if "topic: sensor_accel" not in combined_output or "topic: sensor_gyro" not in combined_output:
            print("PX4 shell replied, but accel/gyro listener output was incomplete.")
            return 1
        print(f"Received {shell_reply_bytes} bytes of PX4 shell output.")
        return 0

    expected = {1, 2, 3}
    missing = sorted(expected - seen_heartbeats)
    if missing:
        print(f"No heartbeat response observed from PX4 system ID(s): {missing}")
        return 1
    print("Heartbeat responses observed from PX4 systems 1, 2, and 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
