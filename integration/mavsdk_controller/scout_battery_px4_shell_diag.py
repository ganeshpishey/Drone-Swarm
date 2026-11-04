"""Read PX4 battery topic and simulator-task status over MAVSDK Shell."""

from __future__ import annotations

import asyncio

from mavsdk_grpc import System


async def main() -> None:
    drone = System(port=50051)
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    await asyncio.wait_for(wait_connected(), timeout=15.0)

    async def read_shell() -> None:
        async for data in drone.shell.receive():
            print(data, end="", flush=True)

    reader = asyncio.create_task(read_shell())
    try:
        # Let the receive subscription attach before sending commands.
        await asyncio.sleep(0.5)
        for command in ("battery_simulator status", "listener battery_status"):
            print(f"\nPX4 shell > {command}", flush=True)
            try:
                # Shell::send forwards the supplied text to PX4's serial shell;
                # include the newline that executes the command at the prompt.
                await asyncio.wait_for(
                    drone.shell.send(command + "\n"), timeout=5.0
                )
            except Exception as error:
                print(f"Shell command error: {type(error).__name__}: {error}", flush=True)
            await asyncio.sleep(3.0)
    finally:
        reader.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(reader, return_exceptions=True), timeout=1.0
            )
        except asyncio.TimeoutError:
            print("PX4 shell receive stream did not close; ending diagnostic.")


if __name__ == "__main__":
    asyncio.run(main())
