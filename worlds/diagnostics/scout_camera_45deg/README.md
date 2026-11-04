# Scout 45-degree camera diagnostic experiment

This is a **disposable, scout-only diagnostic model**. It is not the production Scout model and is not a validated localization calibration.

## What changes

- Camera include and fixed-joint pitch: 45 degrees forward-down from the X500 body frame.
- Camera sensor definition: still supplied by the stock PX4 `mono_cam` model.
- X500 base, motor/propeller model, LiDAR mount, LiDAR sensor, scan geometry, range, and topic: unchanged from the production Scout model.
- Gazebo PosePublisher: added so the camera/link pose can be inspected during the experiment.

The camera's nominal vertical field of view is approximately 83.3 degrees. At a 45-degree downward pitch, its nominal world-relative vertical coverage is approximately 3.35 to 86.65 degrees below horizontal. The level LiDAR's vertical coverage is -15 to +15 degrees. This creates a nominal angular overlap, but does **not** prove that the same victim is visible to both sensors: occlusion, camera optical-axis conventions, target geometry, vehicle pose, and LiDAR returns must be measured in Gazebo.

## Preconditions

1. Work on branch `codex/scout-incident-e2e`.
2. Stop any currently running project-owned Scout/Ambulance simulation using the launcher's supported stop command. Do not kill unrelated processes.
3. Confirm no launcher lock/run is active.
4. Keep the original production model and active UAV configuration unchanged.
5. Do not arm, take off, enable Offboard, or issue movement/dispatch commands during this diagnostic.

## Run

From the repository root, use the existing launcher's explicit scout-only model override:

```bash
PX4_GZ_SCOUT_MODEL_OVERRIDE="$PWD/worlds/diagnostics/scout_camera_45deg" \
  ./integration/mavsdk_controller/launch_active_uavs.sh scout-only
```

The launcher explicitly restricts this override to `scout-only` mode. Do not use it for the two-UAV production run. Stop it with the supported command after collecting evidence:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh stop
```

## Evidence required before accepting the experiment

- Gazebo actually loads the diagnostic model (check the run's staged model and startup logs).
- The published pose chain reports the camera_link orientation and the LiDAR frame orientation at the same simulation time.
- Capture a real Gazebo camera frame containing a known victim model.
- Capture a real LiDAR scan at a matching simulation time.
- Project the measured LiDAR points into the image using empirically checked transforms and report how many returns fall inside the victim's image box.
- Repeat for more than one victim position and verify no target/ground-truth pose is used as a runtime localization input.
- Keep `temporal_pair_motion_budget_unverified` and localization dispatch gates closed until the existing synchronization, extrinsic calibration, uncertainty, and ground-truth validation requirements are actually satisfied.

A visually detected person without a validated 3D position remains an **unlocalized observation** and must not be dispatched as a located victim. This experiment does not authorize flight and does not change the production model.
