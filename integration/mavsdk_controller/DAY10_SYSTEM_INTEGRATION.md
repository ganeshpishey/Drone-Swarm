# Day 10 final integration status

This is an incremental, software-only integration for the active Scout and
Ambulance vehicles. Fireman remains excluded by `active_uavs.json` and the
coordinator rejects a configuration containing Fireman. No PX4 parameter,
model, sensor, world, flight-control or production launcher changes were made
for this integration. No vehicle was armed or sent a flight command.

## Status summary

| Area | Status | Evidence / limit |
| --- | --- | --- |
| 10.3 obstacle avoidance | PARTIALLY COMPLETED | Live-scan parser, 3D corridor policy, mission adapter, stale-data HOLD, guidance exchange and loopback tests are connected. No Offboard executor is active. No live Gazebo scan was collected in this pass. |
| 10.4 MANET | PARTIALLY COMPLETED | Direct configured-peer UDP, envelope validation, HMAC option, transport ACK/retry, duplicate suppression, application ACK, heartbeat/liveness, incident, mission, progress, battery and guidance paths are implemented and loopback-tested. Physical radio/IP-link tests and multi-hop discovery are not done. |
| 10.5 battery | PARTIALLY COMPLETED | MAVSDK sampling, voltage/current when present, history, measured drain, stale/unknown, LOW and CRITICAL state feed coordinator and terminal/report state. No live PX4 battery stream was collected in this pass. |
| 10.6 battery-aware management | PARTIALLY COMPLETED | Assignment requires fresh landed/armable/available vehicle state, observed drain and operator-supplied sortie durations plus reserve. LOW/CRITICAL are advisory; critical battery cancels an advisory mission, but sends no RTL or land command. Scout is not a second response vehicle, so reassignment is explicitly unavailable. |
| Terminal monitoring | PARTIALLY COMPLETED | The coordinator prints observed operational events and status changes; run reports are written as nine JSON files. Camera/YOLO and Scout LiDAR are not connected to this service. |
| End-to-end software | PARTIALLY COMPLETED | Loopback test covers Scout report → coordinate checks/transform → battery/availability gate → mission delivery → Ambulance application ACK and progress → cancellation. Current camera detector emits image-space, unlocalized observations, so operational victim reports are retained but cannot dispatch until a validated localization producer exists. |

## What is connected

### 10.3 — Ambulance mission and obstacle guidance

`day10_coordinate_frames.py` transforms an explicitly identified Scout local-NED
point into Ambulance local-NED through WGS84 geodetic/ECEF coordinates. It
requires fresh PX4/MAVSDK home-origin records, matching home IDs, the declared
common altitude datum, explicit vehicle/frame identity, finite coordinates,
and a configured maximum separation. A transform hash and geodetic result are
retained as provenance. Missing, stale, image-space, ambiguous, malformed, or
unlocalized coordinates cannot become mission targets.

`day10_coordinator_service.py` owns the suggestion-only integration: read-only
MAVSDK state, home origins, battery history, incoming Scout observations,
mission eligibility, Ambulance LaserScan input, adapter decisions, terminal
events, and JSON report state. `ambulance_mission_adapter.py` receives an acknowledged mission and
fresh pose/attitude, coordinator liveness and scan; it produces a velocity
suggestion or HOLD/landing handoff. Clear-path, caution/slow-down, detour,
three-clear-scan resume, blocked-corridor HOLD and stale-scan HOLD remain in
`ambulance_local_avoidance.py`. No module sends Offboard or Action commands.

`day10_vehicle_agent.py` persists proposals as `accepted_suggestion_only`,
rejects concurrent nonterminal assignments, reports validated mission progress
records, and persists cancellation. `--progress-log` accepts only status
transitions for the active mission. The optional guidance JSONL relay sends
bounded adapter suggestions as Ambulance-originated MANET messages. Progress
records must come from a validated monitor or operator observation; they are
not proof of vehicle movement. `SetpointOwnership` remains a same-process
future-executor guard, not a cross-process lock. Any later flight executor
must be reviewed as a single owner of Offboard setpoint streaming.

The current `scout_live_person_detection.py` records bounding boxes and camera
timestamps, but sets `incident_world_location` to null. The separate candidate
location helper is not a validated world-localization system. The coordinator
therefore refuses to dispatch those observations. No localization accuracy or
autonomous victim response is claimed.

### 10.4 — direct MANET messages

`swarm_manet.py` validates schema/version, identities, timestamps, sequence,
TTL, payload types and finite coordinates; supports optional shared-key HMAC;
uses reliable UDP ACK/retry and duplicate suppression; and tracks heartbeat
liveness. The Scout bridge tails new camera-observation records and forwards
them with reliable delivery. The coordinator validates the sender and full
report. Ambulance mission proposal delivery is followed by a separate
application-level ACK. Mission progress, cancellation, battery status and
bounded obstacle suggestions use the same direct-peer transport.

This is direct IP between configured peers, with no internet service, cellular
network, broker or required external router. A radio must provide reachable IP
addresses/interface configuration. Route discovery, multi-hop relaying and
physical MANET operation are outside the validated scope. Loopback UDP is
network emulation only.

### 10.5 and 10.6 — battery policy

`day10_battery.py` stores actual MAVSDK battery samples and maps unavailable
sentinels to unknown. Drain and remaining time are estimates only when the
history contains an observed positive percentage drop over the configured
minimum span. It does not invent consumption values. `day10_coordinator_service`
collects battery along with the other read-only MAVSDK telemetry and includes
freshness, voltage/current when available, measured drain, warnings and policy
results in coordinator state. `day10_battery_monitor.py` remains available as
a standalone JSONL recorder/optional peer publisher.

The policy defaults are project-level assumptions, not PX4 settings: low 30%,
critical 20%, reserve 25%. Sortie outbound/task/return durations are required
CLI inputs rather than guessed by the service. No PX4 failsafe setting is
changed. CRITICAL cancels an advisory mission and reports that operator
recovery is needed; it never commands RTL or landing. Since Scout is not
response-capable, a reassign request is rejected and recorded instead of
silently sending a medical mission to Scout.

### Terminal monitor and JSON reports

`day10_coordinator_service.py` prints observed coordinator, telemetry, LiDAR,
mission, battery, and MANET events to stdout. It writes a unique report bundle
under `reports/run_<timestamp>_<unique_id>/` when the coordinator session exits.
The bundle contains `run_summary.json`, `vehicles.json`, `incidents.json`,
`missions.json`, `communication.json`, `battery.json`, `sensor_health.json`,
`events.json`, and `errors.json`. Reports distinguish measured UDP events from
physical radio validation and mark camera/YOLO and Scout LiDAR as unmonitored.

For a separate read-only terminal watch of the coordinator state file:

```bash
python integration/mavsdk_controller/day10_ground_station.py \
  --coordinator-state /tmp/day10_coordinator/state.json
```

This compatibility entrypoint is a terminal JSON-state watcher. It does not
serve HTML, bind an HTTP port, or expose a browser/API endpoint. The pure
`world_map_from_sdf` and `snapshot` projection helpers remain for existing
command-free test workflows.

## Tests and evidence

`tests/test_day10_system.py` covers battery history/sentinels/thresholds,
observed-energy reserve checks, invalid/expired message envelopes, HMAC,
duplicate suppression, heartbeat loss/recovery, simulated dropped ACK retry,
mission freshness, adapter HOLD/caution behavior, and terminal state/report fallback.
`tests/test_day10_runtime.py` covers coordinate provenance and transforms,
unlocalized Scout reports, invalid coordinates, transformed report-to-mission
delivery, battery/availability denial, application ACK, progress and obstacle
status, cancellation, mission rejection, critical-battery cancellation,
reassignment refusal, terminal state staleness, and suggestion-only behavior.
`tests/test_ambulance_local_avoidance.py` covers scan conversion, all six
sectors, safe/blocked corridors, slowing, detours, HOLD, stale data and
three-clear-scan mission resumption.

The protocol tests use local UDP sockets and synthetic fixtures; they validate
software loopback only, not radio/MANET performance. A disarmed Scout–Ambulance
run on 2026-11-01 confirmed both MAVSDK connections, `in_air=false`,
`armable=true`, and 100% reported battery at the read-only status sample.
The existing log captures eight live Ambulance LiDAR scans at 5 Hz; processing
those scans through the pure XYZ/sector converter yielded 1,894–1,897 valid
points per scan. No mission was assigned and no obstacle guidance was sent to
a flight controller. The run was stopped through the supported launcher,
which reported both PX4 processes and Gazebo stopped; the original parameter
hash checks were `OK`.

The Day 10 coordinator and terminal monitor remain suggestion-only/read-only. A
validated victim location is not available, so the end-to-end dispatch path
was not exercised. Battery freshness and eligibility are covered by tests;
the live battery evidence is a single 100% telemetry sample, not a measured
flight drain curve. No physical radio, packet-loss, multi-hop, or reconnection
test has been run.

Focused Day 10/localization tests passed. Full repository unittest discovery
ran 89 tests successfully but could not import four MAPPO acceptance modules:
`relay` is missing for `test_drone_3d`, and `numpy` is missing from this
MAVSDK virtualenv for `test_environment`, `test_release_acceptance`, and
`test_runtime`. These dependency errors do not affect the Day 10 focused
modules, which passed with localhost socket access.

## Repeatable commands

### 1. Check the supported active-UAV launcher

From the repository root:

```bash
source .venv-mavsdk/bin/activate
./integration/mavsdk_controller/launch_active_uavs.sh preflight-only
```

The command above was run after the 2026-11-01 capture and passed. The live
disarmed Scout–Ambulance invocation used during validation was:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh scout-ambulance
```

Stop that run from the owning launcher terminal with Ctrl+C or, from another
terminal, use its supported cleanup:

```bash
./integration/mavsdk_controller/launch_active_uavs.sh stop
```

### 2. Configure direct MANET peers

Use the real addresses assigned by the radio/IP setup. Do not copy placeholder
addresses as real deployment values. Example peer-map shape:

```json
{
  "scout_0": {"host": "<scout-radio-ip>", "port": 8765},
  "ambulance_0": {"host": "<ambulance-radio-ip>", "port": 8766}
}
```

Coordinator local port is 8767. On the corresponding vehicle, run
`day10_vehicle_agent.py` on its listed peer port and point it at the
coordinator radio IP/8767. On Scout, run `day10_scout_manet_bridge.py` against
the `detections.jsonl` written by the existing camera detector. The detector's
current reports remain unlocalized; they are acknowledged and retained as
incidents, but no mission is dispatched from them.

Example component entrypoints (replace all angle-bracket values; provide the
three sortie-duration estimates and battery thresholds from reviewed policy):

```bash
python integration/mavsdk_controller/day10_coordinator_service.py \
  --peers /path/to/direct_peers.json --bind 0.0.0.0 --port 8767 \
  --outbound-minutes <reviewed-value> --task-minutes <reviewed-value> \
  --return-minutes <reviewed-value>
```

```bash
python integration/mavsdk_controller/day10_vehicle_agent.py \
  --vehicle ambulance_0 --bind 0.0.0.0 --port 8766 \
  --coordinator-host <coordinator-radio-ip> --coordinator-port 8767 \
  --progress-log /path/to/validated_mission_status.jsonl \
  --guidance-log /path/to/ambulance_guidance.jsonl
```

```bash
python integration/mavsdk_controller/day10_scout_manet_bridge.py \
  --input /tmp/scout_live_perception/detections.jsonl \
  --bind 0.0.0.0 --port 8765 \
  --coordinator-host <coordinator-radio-ip> --coordinator-port 8767
```

The guidance log can be produced by the read-only Ambulance mission monitor
with a reviewed, acknowledged local-NED mission document and coordinator
heartbeat. Do not run multiple competing applications that own an active
Offboard stream; this implementation owns none.

### 3. Terminal monitor

In another terminal, watch the persisted coordinator state without starting a server:

```bash
python integration/mavsdk_controller/day10_ground_station.py \
  --coordinator-state /tmp/day10_coordinator/state.json
```

### 4. Regression commands

```bash
python3 -m py_compile \
  integration/mavsdk_controller/day10_battery.py \
  integration/mavsdk_controller/day10_coordinate_frames.py \
  integration/mavsdk_controller/day10_coordinator_service.py \
  integration/mavsdk_controller/day10_ground_station.py \
  integration/mavsdk_controller/day10_terminal_monitor.py \
  integration/mavsdk_controller/day10_mission_coordinator.py \
  integration/mavsdk_controller/day10_scout_manet_bridge.py \
  integration/mavsdk_controller/day10_vehicle_agent.py \
  integration/mavsdk_controller/swarm_manet.py \
  integration/mavsdk_controller/ambulance_local_avoidance.py \
  integration/mavsdk_controller/ambulance_mission_adapter.py \
  integration/mavsdk_controller/ambulance_mission_monitor.py
python3 -m unittest tests.test_day10_system tests.test_day10_runtime \
  tests.test_ambulance_local_avoidance tests.test_active_uav_config \
  tests.test_response_lidar_guard tests.test_response_lidar_local_planner -v
bash -n integration/mavsdk_controller/launch_active_uavs.sh
git diff --check
```

The unittest command needs permission to bind localhost UDP sockets in this
environment. These tests do not launch Gazebo/PX4 and do not send flight
commands.

## Remaining limitations

1. The current Scout RTP camera pipeline does not preserve a verified mapping
   from its frame PTS to Gazebo Image acquisition time. It does not provide
   validated victim world coordinates or a verified origin ID. The production
   report-to-dispatch path therefore correctly stops before mission assignment.
2. Mission progress reports are accepted only from an explicit validated
   monitor/operator status input; no vehicle navigation executor is enabled.
3. Live disarmed Scout/Ambulance startup, telemetry, camera topic, and LiDAR
   streams have been observed. No autonomous response or movement has been
   tested.
4. Direct UDP requires provisioned IP connectivity. Radio route discovery,
   multi-hop operation, packet loss on actual radios, security-key management,
   and reconnect behavior on hardware require deployment tests.
5. The in-memory coordinator state is emitted as a live state snapshot but is
   not a durable incident database; restart persistence and operational audit
   storage remain future work.
6. Thresholds and sortie times require operator review against measured
   vehicle battery data and the intended scenario before any live mission use.

No autonomous flight, obstacle-avoidance flight, victim response, physical
MANET, or operational readiness is claimed by these software tests.
