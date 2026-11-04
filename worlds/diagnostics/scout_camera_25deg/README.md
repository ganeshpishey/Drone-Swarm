# Scout 25-degree camera diagnostic experiment

This is a **disposable, scout-only diagnostic model**. It is separate from both the production Scout model and the existing 45-degree diagnostic model. It is not a validated localization calibration.

## What changes

- Camera include and fixed-joint pitch: 25 degrees forward-down from the X500 body frame (0.4363323129985824 rad).
- Camera sensor definition: still supplied by the stock PX4 `mono_cam` model.
- X500 base, motor/propeller model, LiDAR mount, LiDAR sensor, scan geometry, range, and topic: unchanged from the production Scout model and the 45-degree diagnostic fixture.
- Gazebo PosePublisher: retained for runtime pose inspection.

The camera's nominal vertical field of view is approximately 83.3 degrees. With a 25-degree downward center angle, its nominal world-relative vertical coverage is approximately 16.65 degrees above horizontal through 66.65 degrees below horizontal. This should include nearby victim models whose surfaces are close to the camera's height, unlike the prior 45-degree diagnostic view whose upper edge was approximately 3.35 degrees below horizontal. This is a geometry-based test hypothesis, not proof of visibility: optical-axis conventions, occlusion, target geometry, and actual vehicle pose still need live verification.

## Preconditions

1. Work on branch `codex/scout-incident-e2e`.
2. Stop any currently running project-owned Scout/Ambulance simulation using the launcher's supported stop command. Do not kill unrelated processes.
3. Keep the original production model, 45-degree diagnostic model, world, and active UAV configuration unchanged.
4. Do not arm, take off, enable Offboard, or issue movement/dispatch commands during this diagnostic.

## Run

From the repository root:

```bash
PX4_GZ_SCOUT_MODEL_OVERRIDE="$PWD/worlds/diagnostics/scout_camera_25deg" \
  ./integration/mavsdk_controller/launch_active_uavs.sh scout-only
```

The launcher restricts this override to `scout-only`. The existing spawn override may be used only with this diagnostic model and scout-only mode, but the measured prior run settled to approximately 1.59 m despite a 4.2 m requested spawn altitude. Therefore, do not assume the requested Z is the actual camera height; verify the live PosePublisher output.

Stop with:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh stop
```

## Evidence required

- Confirm the staged model is `scout_camera_25deg` and camera_link pitch is approximately 25 degrees.
- Capture a live Gazebo image containing at least one known victim model.
- Record the actual Scout and camera poses at the same simulation time.
- Run the existing real YOLO pipeline on captured frames and retain its raw detections.
- Do not infer 3D victim position from image detection alone. Unlocalized observations remain non-dispatchable.
- Do not claim camera/LiDAR association until synchronized scan points, transforms, and projection evidence support it.

This experiment does not authorize flight and does not change production sensors or PX4 parameters.
