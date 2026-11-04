# Day 3: MAVSDK-Python controller

> **Current handoff status (29 Oct 2026):** the Scout-only vertical
> `scout_hover_check.py` takeoff/hover/land check passed at 3 m, and an 8 m
> hover produced telemetry and a useful airborne camera frame. The
> three-vehicle launcher has also spawned and connected all three SITL
> vehicles. However, `three_drone_takeoff_demo.py` lost its MAVSDK gRPC stream
> during Offboard waypoint control, and a PX4 log reported a Hold failsafe
> after link loss. The cause is unknown. Do not run the waypoint demos as the
> next step; diagnose the disconnect and validate one Scout waypoint first.
> The current status and 14-day continuation plan are in
> [`../../docs/DRONE_SWARM_14_DAY_PLAN.md`](../../docs/DRONE_SWARM_14_DAY_PLAN.md).

> **Active scope update (1 Nov 2026):** continue Tasks 10.3–14 with Scout and
> Ambulance only. Fireman is temporarily excluded while its PX4 IMU/preflight
> issue remains unresolved. Existing three-vehicle and Fireman commands below
> are historical/reference material, not instructions for active runs. The
> general launcher still starts more than the active pair; do not assume its
> default run excludes Fireman.

This module is the first Python-to-PX4 control adapter. It uses MAVSDK-Python's
asyncio API and leaves the existing MAPPO environment and checkpoints intact.
The controller owns flight-level actions; PX4 remains responsible for low-level
attitude and motor control.

## Install

The main Drone-Swarm package requires Python 3.11.9 or newer. The current
Ubuntu session has Python 3.10, so you can run this standalone integration in a
separate Python 3.10 venv without installing or changing the MAPPO package:

```bash
cd Drone-Swarm
python3 -m venv .venv-mavsdk
source .venv-mavsdk/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mavsdk-grpc>=3.17,<4'
```

For a Python 3.11.9+ environment, install the optional project extra instead:
`python -m pip install -e '.[px4]'`. The extra installs `mavsdk-grpc`, the
established MAVSDK-Python asyncio client package, separately from the project's
normal dependencies.

## Run

Start PX4 SITL and Gazebo first. For a safe vertical-only check, keep the PX4
session open and run the Scout-only script from a second terminal:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
source .venv-mavsdk/bin/activate
python integration/mavsdk_controller/scout_hover_check.py \
  --altitude 3 --hover-seconds 10 \
  --telemetry-log /tmp/scout-hover-check.jsonl
```

This check does not test horizontal movement or Offboard waypoint control. The
older `single_drone_mission.py` and `three_drone_takeoff_demo.py` waypoint
examples are experimental; do not treat their presence as evidence that
waypoint flight has passed. See the status notice above before attempting them.

The default UDP listener is `udpin://0.0.0.0:14540`, matching the PX4 SITL
MAVSDK endpoint. Override it with `--connection-uri` if the SITL configuration
uses another port.

## Coordinate convention

`PX4Controller.goto_location(x, y, z)` accepts metres relative to PX4 home:
`x` is north, `y` is east, and `z` is altitude above home. It converts these
local offsets to the WGS84 latitude/longitude and AMSL altitude required by
MAVSDK's `goto_location` action. The waypoint call waits for telemetry to show
arrival before the mission advances.

The script is for SITL only. Do not point it at a physical vehicle.

## Active Scout and Ambulance SITL launch

For the current two-drone scope, use the additive
`launch_active_uavs.sh` launcher. It supports Scout-only, Ambulance-only and
Scout-plus-Ambulance runs. It does not call the historical three-drone launcher
or start Fireman. The selected records, PX4 instance/system IDs, model names,
spawn pads, MAVSDK ports and LiDAR topics are in `active_uavs.json`.

From the repository root, choose exactly one mode:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh scout-only
./integration/mavsdk_controller/launch_active_uavs.sh ambulance-only
./integration/mavsdk_controller/launch_active_uavs.sh scout-ambulance
```

Before a live launch, the no-simulation preflight mode can check the pair's
ports, processes, source parameter stores, model paths, and SDF XML/asset URIs:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh preflight-only
```

The default mode is `scout-ambulance`, but the explicit command is preferred.
The launcher checks for existing PX4/Gazebo processes and required ports,
stages the selected world/model resources under a unique `/tmp` run directory,
and gives each PX4 instance a private working directory containing copies of
its existing parameter BSON files. In pair mode it also starts exactly one Day
10 coordinator, one Scout perception process, the existing Scout reporting
bridge, one Ambulance suggestion-only MANET agent, terminal monitoring, and
read-only Scout/Ambulance sensor monitors. MAVSDK-Python creates its two
configured local gRPC servers from the coordinator process. Direct UDP peers
are bound to `127.0.0.1` for SITL integration; this validates localhost
transport only, not physical MANET radio operation. Scout detections stay
unlocalized and non-dispatchable unless a trusted localization policy accepts
them. No flight command path is started.

The launcher leaves source parameter stores, the PX4 checkout, production
models, world, sensors and flight-control code untouched. Per-vehicle and
runtime logs, manifests and isolated parameter copies are retained under
`/tmp/drone-swarm-active-uavs-$UID/`. Coordinator reports are created under
`reports/run_<timestamp>_<unique_id>/`. Append-only `events.jsonl`,
`communication.jsonl`, and `errors.jsonl` preserve evidence if the coordinator
cannot write its final report bundle during abrupt shutdown. All nine report
snapshots are initialized before services start, with the summary marked
`IN_PROGRESS` until graceful shutdown writes the final bundle.

Keep the launcher terminal open while the simulation runs. Stop only its
recorded PX4 process groups with Ctrl+C in that terminal, or from another
terminal run:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh stop
```

Stop sends SIGTERM only after checking each recorded PX4 or auxiliary process
group against its executable/script, session leader, working directory and
run ownership marker. It does not force-kill processes. If an identity check
fails or an owned process does not exit, it retains the run manifest and logs
for inspection. No arm, takeoff, movement, Offboard, or mission execution
commands are issued by these runtime components.

After a pair run, inspect retained logs and reports with:

```bash
RUN_ROOT="/tmp/drone-swarm-active-uavs-$(id -u)"
readlink -f "$RUN_ROOT/current"
find reports -maxdepth 2 -type f -name 'run_summary.json' -print
```

## Day 4: three PX4 vehicles

The starter swarm configuration is in `drones.json`. It assigns PX4 instances
0, 1, and 2 to Scout, Medical-Support, and Fire-Response, giving them system
IDs 1, 2, and 3 and MAVSDK UDP ports 14540, 14541, and 14542. Their Gazebo
spawn points are separated by 6 m. Each Python MAVSDK controller also uses a
distinct local gRPC server port (50051, 50052, and 50053); sharing the default
gRPC port would route the controllers through the same MAVSDK server.

Close the existing single-PX4 SITL and Gazebo sessions first. From the
Drone-Swarm root, launch the three PX4 instances:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
./integration/mavsdk_controller/launch_three_px4.sh
```

The script uses `/tmp/PX4-Autopilot` by default. Set `PX4_AUTOPILOT_DIR` if
your checkout is elsewhere. Instance 0 starts Gazebo; instances 1 and 2 join
its world. Startup logs are written under `/tmp/drone-swarm-px4-multi-<uid>/`.
The launcher refuses to start if a PX4 process is already running, to avoid
overlapping it with the single-drone session.

The three-vehicle launcher and connection have been observed to work, but the
multi-vehicle waypoint mission is currently blocked by the Offboard/gRPC
disconnect described above. Do not run the following demo until that issue has
been diagnosed and a Scout-only waypoint test has passed:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
source .venv-mavsdk/bin/activate
python integration/mavsdk_controller/three_drone_takeoff_demo.py
```

The demo connects to each configured UDP port, arms and takes off all three
vehicles, waits for each to reach takeoff altitude, then sends each to its own
nearby waypoint using local NED offboard position setpoints. It requests
landing after the waypoint run. It logs telemetry to
`three_drone_telemetry.jsonl` and waits for PX4 to report each vehicle is no
longer airborne before confirming landing.

The launcher uses the project-owned `x500_mono_cam_down_lidar` model for Scout
and stock `x500` models for Ambulance and Fireman. Scout keeps the existing
downward-facing `mono_cam` camera and adds a 5 Hz GPU LiDAR with a 360-degree
horizontal field of view, 30-degree vertical field of view (16 channels),
0.2–25 m range, and 2 cm range resolution. The scan is 180 × 16 samples per
frame and is published on `/scout/lidar/points`; visualization of individual
returns is disabled to reduce rendering load. Ambulance and Fireman keep their
original models. No PX4 parameters, airframe, motors, propellers, or
flight-control code are changed. The sensor link adds 1 g of modeled payload.
Gazebo names the vehicles `x500_mono_cam_down_lidar_0`, `x500_1`, and
`x500_2`.

The Scout camera is expected to publish a Gazebo RGB image topic under
`/world/earthquake_landslide_world/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/camera/image`.
PX4's Gazebo server configuration also discovers camera sensors and streams
RGB frames as H.264/RTP to UDP port 5600. Gazebo Transport Python bindings and
OpenCV are not present in the project's MAVSDK virtual environment. The host
system has PyGObject/GStreamer, which a Python camera consumer can use to read
the RTP stream; this camera setup does not add victim detection. To capture one
decoded RGB frame without ROS, stop any existing `gst-launch-1.0` viewer first
(only one UDP receiver should read port 5600), then run:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
/usr/bin/python3 integration/mavsdk_controller/capture_scout_frame.py \
  --output /tmp/scout_camera_frame.ppm
xdg-open /tmp/scout_camera_frame.ppm
```

Run this while the simulation and Scout camera stream are active. The capture
utility uses the host Python GStreamer bindings, not the MAVSDK virtualenv.

Gazebo currently reports `/scout/lidar/points` as `gz.msgs.LaserScan`, rather
than `PointCloudPacked`. That message can carry vertical layers and ranges; the
`points` suffix is only the configured topic name. Runtime checks observed
about 2.0–2.3 messages per second against the requested 5 Hz, so the measured
rate is below target and may reflect simulation load. Use the Python probe to
check the scan dimensions and valid ranges and optionally save a local XYZ PLY
sample (no ROS or extra Python packages are required):

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
/usr/bin/python3 integration/mavsdk_controller/scout_lidar_probe.py \
  --ply /tmp/scout_lidar_scan.ply
```

The probe reads one message through the Gazebo `gz topic` command. The PLY
output is a single scan in the sensor-local frame, not a registered map. Confirm
its reported `horizontal_count`, `vertical_count`, range count, and positive
returns before using it for mapping.

For a short moving-scan check, start the recorder in a terminal, then promptly
start the known Scout survey in a second terminal. The recorder saves first,
middle, and last scans separately; they remain in sensor-local coordinates and
are not motion-registered:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
/usr/bin/python3 integration/mavsdk_controller/scout_lidar_record.py \
  --duration 70 --output-dir /tmp/scout_lidar_moving
```

In the second terminal, run the Scout survey with camera capture disabled:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
source .venv-mavsdk/bin/activate
python integration/mavsdk_controller/scout_victim_survey.py \
  --north 3 --east 3 --altitude 6 --minimum-altitude 5 \
  --hold-seconds 30 --skip-frame-capture \
  --telemetry-log /tmp/scout_lidar_moving_telemetry.jsonl
```

For the next mapping step, the same survey script can optionally collect scans
and register them with position and Euler attitude from its existing MAVSDK
connection. Add `--lidar-map /tmp/scout_day7_pose_registered.ply` to the survey
command; it writes a neighboring JSON summary after the return-and-land step.
This is a pose-registered point cloud, not SLAM, and sensor/pose timing and
frame alignment still require validation before using it for navigation.

Response-drone LiDAR publishes `LaserScan` messages on
`/ambulance_0/lidar/points` and `/fireman_0/lidar/points`. The active scope uses
Ambulance; Fireman commands below are retained for historical reference only.
Scout's separate sensor publishes on `/scout/lidar/points`. The bounded guarded
move entry point recognizes this Scout topic, but that path has not been
validated against Scout telemetry or integrated into its survey mission. Run
the initial
analysis-only forward-sector and rate check while both vehicles are landed:

```bash
/usr/bin/python3 integration/mavsdk_controller/response_lidar_safety_monitor.py \
  --topic /ambulance_0/lidar/points --duration 20 \
  --output /tmp/response_lidar_test/ambulance/safety_scan_summary.json
/usr/bin/python3 integration/mavsdk_controller/response_lidar_safety_monitor.py \
  --topic /fireman_0/lidar/points --duration 20 \
  --output /tmp/response_lidar_test/fireman/safety_scan_summary.json
```

The provisional 1 m STOP and 2 m CAUTION distances classify returns in a
30-degree forward sector. The monitor sends no PX4 commands. Its scan-rate
check requires at least 80% of the configured 5 Hz; a failure means the sensor
stream is not ready to drive an avoidance action.

### Response-drone LiDAR guard prototype

`response_lidar_guard.py` applies the measured X500 self-return mask to live
Gazebo scans for either response drone. `CLEAR` permits the requested forward
speed, `CAUTION` scales it to 35%, and `STOP`, stale scans, missing scans, or
stale position telemetry command zero NED velocity. If a stop persists, the
bounded test ends and requests landing. A failed MAVSDK stream triggers one
reconnect attempt for landing. PX4 remains the low-level flight controller.

`response_lidar_local_planner.py` adds a short-horizon local corridor selector.
It evaluates scan returns across candidate headings up to 75 degrees from the
goal corridor, selects a clear detour at reduced speed, and returns HOLD when
the scan is stale, invalid, or all candidate corridors are blocked. The
`--replan-obstacles` mode in the guarded-move script recomputes the goal bearing
from local position and feeds that choice to PX4 Offboard velocity setpoints.
Before arming, it measures scan rate in simulation time and refuses to fly
below 4 Hz or below 0.5 Hz in wall time; stale scans also hold. This is an
experimental planar local planner, not a
global map, multi-drone deconfliction system, or validated avoidance envelope.

`response_lidar_guarded_move.py` is a SITL-only, one-drone-at-a-time flight
prototype. It is limited to a 1.5–3 m takeoff, a 0.2–2 m goal leg, and at most
0.4 m/s. The `--confirm-sitl` flag is required. Replanning mode additionally
requires at least 4 Hz simulation-time and 0.5 Hz wall-time scan rate before arming. Review the JSONL
planner-state log and PX4 landing confirmation; unit tests do not establish
flight safety. Example commands:

```bash
source .venv-mavsdk/bin/activate
python integration/mavsdk_controller/response_lidar_guarded_move.py \
  --vehicle ambulance_0 --confirm-sitl \
  --altitude-m 2 --distance-m 1 --speed-m-s 0.25 \
  --log /tmp/response_lidar_test/ambulance/guarded_move.jsonl

python integration/mavsdk_controller/response_lidar_guarded_move.py \
  --vehicle ambulance_0 --confirm-sitl --replan-obstacles \
  --altitude-m 2 --distance-m 1 --speed-m-s 0.20 \
  --minimum-scan-rate-hz 4 --minimum-wall-scan-rate-hz 0.5 \
  --max-scan-age-s 1.5 \
  --log /tmp/response_lidar_test/ambulance/replan_move.jsonl

python integration/mavsdk_controller/response_lidar_guarded_move.py \
  --vehicle fireman_0 --confirm-sitl \
  --altitude-m 2 --distance-m 1 --speed-m-s 0.25 \
  --log /tmp/response_lidar_test/fireman/guarded_move.jsonl
```

This prototype is not yet wired into autonomous dispatch/mission flow and has
not completed an in-flight replan test. It is not validated for real aircraft,
other sensor mounts, descent, landing, full 3D routing, or obstacle bypass.
Keep the manual SITL checks one at a time. Recent 8-second captures measured
over 5 Hz in Gazebo simulation time but 0.75 Hz in wall time because the world
advanced about 0.9 simulation seconds in 8 wall seconds. The minimum wall-rate
and stale-scan gates remain active; poor real-time factor still limits the
usefulness of a flight test.

The model must be loaded by restarting PX4/Gazebo after changing the launcher.
The `stop` command stops PX4/Gazebo processes system-wide; use it only after
confirming all vehicles have landed and no other simulation should remain
open. To isolate Fireman LiDAR timing from the other PX4 instances while
preserving Fireman's system ID and MAVSDK UDP port, start a Fireman-only run:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
./integration/mavsdk_controller/launch_three_px4.sh fireman-only
```

This mode starts PX4 instance 2 only (`fireman_0`, UDP 14542). Close its Gazebo
GUI after spawn if measuring server-side sensor timing; do not change the LiDAR
resolution or freshness limits for this comparison.

To check whether a PX4 instance-specific fault follows the old Fireman process,
the Fireman-only launcher also supports a temporary instance-0 run. It uses
system ID 1 and UDP 14540; it is still one Fireman vehicle and is not a fourth
fleet member. `fireman_instance0_status.json` contains the matching read-only
MAVSDK endpoint configuration. The comparison is diagnostic: it does not
guarantee that a new instance will fix a sensor or estimator fault.

```bash
PX4_GZ_FIREMAN_INSTANCE=0 PX4_GZ_HEADLESS=1 \
  ./integration/mavsdk_controller/launch_three_px4.sh fireman-only
```

For a sensor-health baseline that excludes the custom Fireman LiDAR model, set
`PX4_GZ_FIREMAN_MODEL=x500` for a Fireman-only launch. This runs the stock PX4
X500 model at the same Fireman instance and spawn pose; it is a diagnostic
comparison and does not provide the response LiDAR topic.

If the full disaster world produces PX4 IMU/barometer timeouts or unstable
height estimates, isolate SITL health in the lightweight scene before another
flight. A `fireman-only` launch now defaults to the lightweight LiDAR obstacle
validation world, uses the LiDAR equipped Fireman model, spawns it at 1.7 m,
and skips the GUI. This removes the terrain, debris, particle-effect, and GUI
load while retaining the response LiDAR and a fixed test obstacle. It is an
infrastructure and bounded obstacle-validation setup, not a replacement for
validation in the disaster scene. To select the clear lightweight scene
instead, set `PX4_GZ_WORLD_FILE` to `worlds/lidar_validation_world.sdf`.

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
./integration/mavsdk_controller/launch_three_px4.sh stop
./integration/mavsdk_controller/launch_three_px4.sh fireman-only
```

Do not override the Fireman validation spawn below 1.0 m; the launcher rejects
low test spawns so the vehicle has clearance while PX4's estimator initializes.

To inspect more terrain with the downward-facing Scout camera without testing
horizontal Offboard control, the Scout-only hover check accepts a vertical
takeoff height. For example, use `--altitude 8 --hover-seconds 30`, capture the
camera frame during the hover, and confirm it lands before trying waypoint
flight again.

## Combined earthquake and landslide world

`worlds/earthquake_landslide_world.sdf` targets Gazebo Harmonic / Gazebo Sim
8.15.0 and SDF 1.9. The world retains its original 180 x 150 m terrain footprint and layout.
Mesh assets under
`worlds/earthquake_landslide_assets/` provide the collapsed apartment, village
houses, bridge, rubble, and irregular boulders. Their original geometry is split
by existing MTL material groups, then assigned explicit SDF colors so the
materials do not depend on Gazebo importing the OBJ MTL appearance. Building
and bridge collisions, poses, terrain texture, stream segments, and object
placement remain as in the original scene. The valley, landslide slope, debris,
cracked roads, forested slopes, fog, static dust/smoke cues, incident markers,
and flattened 20 x 18 m launch apron with three pads are retained.
There are no drones in the SDF. Fire and victim objects are visual/collision
markers rather than sensor detections; smoke and dust are static visual cues,
not animated particles or simulated fluid/heat. The SDF world relies on the
PX4 launcher for its normal Gazebo systems.

Reapply the material fix without regenerating terrain or moving any objects:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
python3 worlds/generate_earthquake_landslide.py
```

The material repair script needs Python 3 and Pillow (`python -m pip install Pillow`);
afterward, the SDF and mesh assets run without Python or ROS.

Launch three external PX4 X500 drones; the script stages the SDF and its local
asset model into PX4's Gazebo world directory and starts at the three corner
pads:

```bash
./integration/mavsdk_controller/launch_three_px4.sh
```

### Apron-centered launcher default

The PX4 launcher defaults to `worlds/earthquake_landslide_apron_world.sdf`,
the apron-centered scene shown above. It keeps the 20 x 18 m launch apron and
three pads full-size, with the surrounding terrain and environment scaled to
an 18 x 15 m footprint. To use the original 180 x 150 m world instead, set
`PX4_GZ_WORLD_FILE="$PWD/worlds/earthquake_landslide_world.sdf"` when launching.

### Half-size operational-area variant

`worlds/earthquake_landslide_world_50pct.sdf` is a separate compact variant
with a 90 x 75 m terrain footprint. Landscape and obstacle footprints are
scaled horizontally around the existing launch apron; terrain elevation and
PX4's three launch coordinates remain unchanged. It references the original
heightmap without modifying it, and the original world is not overwritten.
Recreate and validate it:

```bash
python3 worlds/generate_compact_disaster_world.py
gz sdf -k worlds/earthquake_landslide_world_50pct.sdf
```

To launch the three PX4 drones in this version, stop any existing simulation
first, then set the world's source file for the normal launcher:

```bash
./integration/mavsdk_controller/launch_three_px4.sh stop
PX4_GZ_WORLD_FILE="$PWD/worlds/earthquake_landslide_world_50pct.sdf" \
  ./integration/mavsdk_controller/launch_three_px4.sh
```

For a visual-only preview, add the project's `worlds` folder to Gazebo's
resource search path and run:

```bash
source /tmp/PX4-Autopilot/build/px4_sitl_default/gz_env.sh
export GZ_SIM_RESOURCE_PATH="$PWD/worlds:$GZ_SIM_RESOURCE_PATH"
gz sim -r worlds/earthquake_landslide_world.sdf
```

### Compact box-sized disaster world

`worlds/earthquake_landslide_box_world.sdf` is a separate compact variant of
the same disaster layout. Its terrain is 90 x 75 m with half-height relief;
buildings, rubble, roads, vegetation, water features, and their collisions are
uniformly scaled to half size so they remain aligned. The launch apron and
three pads retain their original dimensions and locations, so the PX4 spawn
poses and drone models are unchanged. The original environment and assets are
not modified.

Regenerate and validate the compact world:

```bash
python3 worlds/generate_box_sized_disaster_world.py
gz sdf -k worlds/earthquake_landslide_box_world.sdf
```

Launch the three PX4 vehicles in it:

```bash
./integration/mavsdk_controller/launch_three_px4.sh stop
PX4_GZ_WORLD_FILE="$PWD/worlds/earthquake_landslide_box_world.sdf" \
  ./integration/mavsdk_controller/launch_three_px4.sh
```

Create repeatable isolated variants with an explicit integer seed. The script
writes the chosen SDF and a JSON manifest; incident marker positions are
deterministically varied. Earthquake-only terrain removes landslide debris;
landslide-only terrain retains intact settlement shells and removes the
earthquake fissures. The combined SDF remains the default:

```bash
python3 worlds/generate_earthquake_landslide.py --mode earthquake --seed 7
PX4_GZ_WORLD_FILE="$PWD/worlds/earthquake_seed_7.sdf" ./integration/mavsdk_controller/launch_three_px4.sh

python3 worlds/generate_earthquake_landslide.py --mode landslide --seed 7
PX4_GZ_WORLD_FILE="$PWD/worlds/landslide_seed_7.sdf" ./integration/mavsdk_controller/launch_three_px4.sh
```

Validate the default world and generated variants with:

```bash
gz sdf -k worlds/earthquake_landslide_world.sdf
python3 worlds/generate_earthquake_landslide.py --mode earthquake --seed 7
gz sdf -k worlds/earthquake_seed_7.sdf
python3 worlds/generate_earthquake_landslide.py --mode landslide --seed 7
gz sdf -k worlds/landslide_seed_7.sdf
```

For operation, maintain terrain clearance and route around ridges, treetops,
building faces, boulders, and bridge cables/deck. Static collision shapes let
Gazebo's physics report contacts; the planner still needs a map or onboard
range/vision sensing to detect obstacles before contact. Water and smoke are
visual-only, and the X500 model's available sensors depend on the PX4 model
configuration. The old compact `worlds/earthquake_world.sdf` remains available
as a separate reference scene.

## Earlier compact earthquake world

`worlds/earthquake_world.sdf` is a self-contained SDF 1.9 world for Gazebo
Harmonic. It has a 70 m square ground area and a 14 m clear launch apron in the
southwest corner. The three marked pads are spaced 6 m apart. Compact road
corridors, damaged multi-storey shells with exposed beams and fallen roof slabs,
grouped concrete/brick rubble, road cracks and barriers, three abandoned
vehicles, a 5 m canal and a segmented damaged bridge make up the rescue area.
The scene follows the supplied reference composition, with an exposed-frame
collapsed tower, a diagonal cracked street, rubble on the road shoulders, six
inspection pads, victim cues and fire/smoke visual cues. The world contains no
drone models. Its earlier PX4 pad coordinates were (-31,-31), (-25,-31), and
(-31,-25) metres. Ground truth coordinates are in
`worlds/incident_manifest.yaml`.

This is a primitive-geometry SDF approximation of the reference. It does not
provide photorealistic building textures, animated smoke, or physically
simulated fire.

Fire and smoke are static visual markers, not simulated heat, fluid, or
particle effects. All environment collision geometry uses simple boxes,
cylinders or spheres; visual-only road markings, water and incident cues do not
create collision surfaces. There are no external model assets or ROS
dependencies. The PX4 launcher supplies its normal Gazebo system plugins.

This compact legacy world can still be previewed without PX4:

```bash
cd "/home/ganesh/Drone Swarm/Drone-Swarm"
source /tmp/PX4-Autopilot/build/px4_sitl_default/gz_env.sh
export GZ_SIM_RESOURCE_PATH="$PWD/worlds:$GZ_SIM_RESOURCE_PATH"
gz sim -r worlds/earthquake_world.sdf
```

The PX4 launcher uses the apron-centered disaster scene by default; this
compact legacy world is not selected by that launcher.

The scene provides multiple approach routes but has no autonomy or collision
avoidance controller built into SDF. Keep the initial flight within the clear
launch apron, then use conservative speed/altitude limits and route around the
bridge piers, buildings, and road obstacles. Visual target markers are not
sensor detections; those are implemented in a later milestone.
# Fireman PX4 sensor-path diagnostic

To collect PX4-side accelerometer, gyro, and sensor-voter output over the
MAVLink shell during a Fireman-only SITL run, set
`PX4_GZ_SENSOR_DIAGNOSTIC=1`. This mode owns UDP 14550 for the diagnostic and
stops after issuing only `listener sensor_accel -n 5`, `listener sensor_gyro -n 5`,
and `sensors status`; it does not send flight commands. Read
`/tmp/drone-swarm-px4-multi-<uid>/fireman_sensor_diagnostic.log` afterward.
Use a Fireman spawn altitude of at least 1.0 m in the LiDAR validation worlds.
