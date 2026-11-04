# Drone-Swarm 14-Day Roadmap and Handoff

This is the current consolidated plan for the existing disaster-response
simulation. Days 1–6 preserve the original setup and completed work; Days 7–14
follow the updated roadmap covering camera AI, 3D LiDAR, dynamic mapping,
communication, mission coordination, battery simulation, and the ground
station. Status notes distinguish demonstrated results from plans.

### Active scope update — 2026-11-01

Tasks 10.3–14 now target **Scout and Ambulance only**. Fireman is temporarily
excluded from active simulation, integration, and acceptance runs while its
PX4 IMU/preflight issue remains unresolved. Earlier Fireman experiments below
are retained as historical evidence; they are not current acceptance results.
The platform remains PX4 SITL, Gazebo Harmonic, and MAVSDK-Python with an
infrastructure-independent MANET as the communication goal. ROS and ROS 2 are
out of scope. Preserve the current project and add capabilities incrementally.

The existing general launcher and three-vehicle configuration have not been
converted into a Scout/Ambulance-only launch workflow by this scope note. Before
integrated flight work, identify a safe existing launch path or prepare an
additive, reviewable two-vehicle workflow; do not start the three-vehicle
launcher and assume Fireman is excluded.

## Project configuration and boundaries

- Ubuntu 24.04, PX4 SITL, Gazebo Harmonic, MAVSDK-Python, and Python perception.
- Three existing quadcopters: Scout, Medical-Response, and Fire-Response.
- No ROS or ROS 2.
- Preserve the existing disaster world, terrain, water, debris, buildings,
  victims, fire/smoke visuals, launch pads, drone dimensions, mass, motors,
  propellers, PX4 flight controls, waypoint navigation, and landing.
- Planned LiDAR additions must be sensor-only model changes. They must not
  alter vehicle mass/dynamics or motor and propeller settings.
- Preserve the discrete MAPPO research work and checkpoints. Start coordination
  with explicit rule-based Python logic; MAPPO remains a possible later
  high-level policy and never replaces PX4 flight control.
- Operational person, fire, and smoke detections must come only from Scout's
  live sensor observations and AI models. Never synthesize them from hardcoded
  coordinates, Gazebo model poses, or simulator world state.
- Estimate incident locations from actual camera detections, calibrated camera
  geometry, timestamped PX4 Scout pose, and available sensor-built 3D data.
  Reports must include confidence, timestamps, and uncertainty.
- Keep `earthquake_landslide_apron_ground_truth.json` strictly for offline
  evaluation after inference. Never expose its positions to operational
  detection or coordinator inputs.
- Clearly label sensor-derived detections separately from simulated ground
  truth in messages, logs, and metrics.
- Do not infer medical severity from ordinary RGB images.
- Added models, sensor data, or low-confidence model outputs are not proof of
  detection, localization, avoidance, dispatch, or real-world capability.

## Existing project status

### Verified or user-reported

- The selected world is `worlds/earthquake_landslide_apron_world.sdf`.
- Three PX4/Gazebo vehicles have spawned in earlier sessions.
- Scout completed vertical takeoff, hover, and landing checks. It also flew to
  the victim area at about 8 m, returned to the launch area on earlier runs,
  and produced 1280×960 camera frames.
- Gazebo Transport publishes the Scout image topic. PX4 streams its RGB camera
  as H.264/RTP on UDP 5600. The Python tools consume it without ROS.
- Five victim entities and world poses are documented in
  `worlds/earthquake_landslide_apron_ground_truth.json`.
- A YOLO CPU live-inference script exists. A pretrained COCO model produced no
  useful detections at practical thresholds in the victim scene. At
  `conf=0.001`, it produced seven very-low-confidence boxes on one frame; visual
  inspection showed these were false positives, not confirmed victims.
- Person-model fine-tuning and hard-negative evaluation have been attempted.
  One later hard-negative run used 31 training images and one validation image,
  then its checkpoint produced no detections on 30 independent Scout frames;
  the COCO baseline mostly labeled drone-like objects as airplanes. Do not use
  this person checkpoint as a validated detector. More varied labeled captures
  and a new independent evaluation are needed.
- A fire/smoke model was trained on 27 frames and validated on four frames from
  the same flight. Best recorded metrics were P=0.762, R=0.450, mAP50=0.559,
  and mAP50-95=0.245. This is exploratory evidence only, not cross-run
  performance.

### Not yet verified or implemented

- Reliable camera-based person/fire/smoke detection across independent runs;
  current detections and training metrics are preliminary.
- Empirical camera calibration and accurate incident world-coordinate
  estimation; current location outputs are candidate surface proxies.
- Quantitatively validated registered point-cloud mapping, semantic obstacle
  representation, and explored/unexplored coverage.
- 3D LiDAR on Medical-Response and Fire-Response, local obstacle avoidance,
  and safe obstacle-aware descent.
- Runtime acceptance of the new incident registry, mission dispatch,
  battery-aware assignment, and live ground-station dashboard. Local UDP
  coordinator and map transfer have been exercised; robust transport remains
  pending.
- Infrastructure-independent MANET operation. Localhost UDP is not evidence
  of a wireless MANET or multi-hop AODV routing.
- A realistic current-based battery model. The proposed 4S, 5 Ah, 14.8 V
  specification below is an initial simulation assumption, not a validated
  physical battery specification.
- The experimental three-drone Offboard waypoint mission has not passed an
  end-to-end test; a previous MAVSDK gRPC disconnect caused PX4 to enter Hold.
  Diagnose and validate Scout-only waypoint control before retrying the full
  swarm waypoint mission.

## Repository handoff map

| Path | Purpose |
| --- | --- |
| `worlds/earthquake_landslide_apron_world.sdf` | Existing disaster-response world |
| `worlds/earthquake_landslide_apron_ground_truth.json` | Victim IDs, models, and poses for offline evaluation only; not detection or coordinator input |
| `worlds/disaster_assets/` | Local terrain, victim, fire, and smoke assets |
| `integration/mavsdk_controller/launch_three_px4.sh` | Existing three-vehicle SITL launcher |
| `integration/mavsdk_controller/controller.py` | MAVSDK controller wrapper |
| `integration/mavsdk_controller/scout_victim_survey.py` | Scout victim-area flight, frame capture, and telemetry logging |
| `integration/mavsdk_controller/scout_live_person_detection.py` | CPU YOLO inference on Scout's RTP camera stream |
| `integration/mavsdk_controller/scout_lidar_probe.py` | Reads and summarizes one Gazebo LaserScan; can export one local-frame PLY |
| `integration/mavsdk_controller/scout_lidar_record.py` | Records a short scan stream and saves representative local-frame PLY samples |
| `integration/mavsdk_controller/lidar_pose_mapper.py` | Registers scans against timestamp-near MAVSDK position and attitude samples |
| `integration/mavsdk_controller/plot_lidar_ply.py` | Plots top and side views of an ASCII PLY using the perception venv |
| `integration/mavsdk_controller/compare_lidar_terrain.py` | Overlays a local NEU PLY with the known terrain mesh using Scout's configured spawn pose |
| `integration/mavsdk_controller/label_victim_frames.py` | Manual YOLO-format person-box labeling tool |
| `integration/mavsdk_controller/prepare_victim_yolo_dataset.py` | Label-preserving appearance augmentation and dataset preparation |
| `perception/datasets/victim_detection/` | Current image, label, and exploratory training data |
| `src/relay/viewer.py` | Existing MAPPO replay viewer; not the planned live drone ground station |
| `docs/RELAY_LEGACY_README.md` | Preserved legacy MAPPO/RELAY setup guide |

The launcher defines Gazebo spawn poses in its own shell script; `spawn_pose`
fields in `drones.json` do not set them. Its `stop` command affects PX4/Gazebo
processes system-wide, so check process ownership first.

## Day-by-day roadmap

### Days 1–5 — Base setup, flight, swarm, and world

**Status: completed per project history, with three-drone waypoint validation
still incomplete.** Set up Ubuntu, PX4 SITL, Gazebo Harmonic, MAVSDK-Python,
and the existing earthquake/landslide world. Verify single-drone basic flight,
then bring up all three vehicles. Preserve the current world and its three
launch pads. Keep the external PX4 checkout outside this repository.

**Acceptance:** existing project launches; basic Scout takeoff/hover/landing
works; all three vehicle instances can spawn. A successful spawn is not proof
of three-drone mission coordination.

### Day 6 — Scout camera and scene ground truth

**Status: sensor stream and frame capture demonstrated.** Verify the downward
RGB camera topic and UDP 5600 stream, capture airborne frames, and maintain a
separate evaluation-only ground-truth file. Ground truth must never generate
live detections or incident reports.

**Acceptance:** usable RGB frames and a separate, documented scene ground truth.

### Day 7 — Camera-based AI detection

**Status: in progress.** Scout must detect people, visible fire, and smoke
dynamically from its live RGB feed. The current pretrained-person baseline is
not reliable on this scene. An initial fine-tuning run completed early, but its
one-image validation is too small to establish detection performance. Fire and
smoke AI and independent evaluation remain unimplemented. No detector may read
predefined incident coordinates, Gazebo model poses, or world state.

Tasks:

1. Consume Scout RGB frames through Gazebo Transport or the existing H.264/RTP
   stream using Python, without ROS.
2. Evaluate a suitable pretrained person detector on live Scout frames;
   record boxes, confidence, source-frame timestamp, and model identity. Do not
   treat weak false positives as confirmed incidents.
3. Fine-tune a person detector using labeled simulator captures. Keep the
   current six-image dataset explicitly exploratory; collect more distinct
   images/runs before claiming generalization.
4. Select and evaluate AI models for visible fire and smoke separately. A
   scene asset or simulator model pose is not a detection.
5. Save annotated frames and structured, timestamped sensor-inference records.
6. Measure missed detections and false positives across distances, image
   locations, occlusion, and lighting.

**Acceptance:** live-frame detections are independently evaluated against
separate ground truth, with false positives and misses documented. Ground truth
is used only after inference to calculate evaluation metrics. Do not infer
medical severity from ordinary RGB imagery.

### Day 7.2 — Add 3D LiDAR to Scout

**Status:** Scout's sensor model publishes 180 × 16 `gz.msgs.LaserScan`
ranges. A stationary scan had 2,459 positive returns; a 70-second moving
recording captured 149 scans (2.13 Hz versus the configured 5 Hz). A
pose-registered survey later captured 200 scans and 202,728 points over
106.78 seconds (1.87 Hz); telemetry confirmed the victim-area hover and return
to landing. The scan cloud has been plotted and overlaid with the known terrain
mesh using the configured Scout spawn pose. This is an approximate visual
alignment only. MAVSDK slow-callback warnings were still reported during the
latest capture, so callback/stream load and map repeatability need more work.
No autonomous avoidance is implemented.

**Goal:** give Scout 3D environmental sensing.

1. Inspect the existing Scout SDF and sensor stack before editing. **Done.**
2. Add a simulated 3D LiDAR with documented range, horizontal/vertical field
   of view, resolution, and update rate. **Done; settings are in the
   controller README and Scout model.**
3. Inspect LaserScan dimensions/range values and convert a scan to local XYZ
   points through Gazebo Transport and Python, without ROS. **Verified by the
   user:** 180 × 16 ranges; 2,459 positive returns in a stationary sample.
4. Visualize scans and check buildings, debris, open space, and obstacles.
   **Initial scan plot and terrain-mesh overlay completed; alignment and
   obstacle correspondence still need validation.**
5. Check scan quality while Scout moves. A 70-second stream produced 149
   messages, with 264–2,475 positive returns per scan. The survey telemetry
   shows Scout reached and held at the victim-area waypoint. Pose-registered
   mapping output has been captured once; repeatability remains pending.

**Acceptance:** Scout produces usable 3D scans with documented sensor settings.
Do not implement autonomous avoidance in this substage.

### Day 7.3 — Basic 3D landslide-environment mapping

**Status: in progress.** The optional pose-registration path in
`scout_victim_survey.py` produced `/tmp/scout_day7_pose_registered_v2.ply`:
200 scans, 202,728 registered points, and 106.78 seconds of recording. The
mission telemetry confirms waypoint hover and landing. A top/side plot and an
approximate overlay against `terrain_apron.obj` were generated with
`compare_lidar_terrain.py`. The overlay shows broad terrain agreement, but uses
the configured spawn pose rather than an independently calibrated transform;
it is not a quantitative accuracy result. Slow callback warnings also
appeared during the capture. This is pose-based registration, not SLAM.

**Goal:** make Scout generate a 3D map while exploring the existing landslide
environment.

1. **Done:** read Scout LiDAR scans in Python through Gazebo Transport.
2. **Done once:** collect Scout position and orientation from PX4 telemetry.
3. **Prototype:** pair scans with nearby host-time pose samples. Validate
   timestamp synchronization and callback load before treating it as reliable.
4. **Prototype:** transform scan points to a shared local North-East-Up frame.
   Quantitative alignment validation is pending.
5. **Done once:** combine scans collected during the waypoint survey.
6. **Done once:** save a 202,728-point pose-registered cloud from 200 scans.
7. **Done:** generate top/side plots and a terrain-mesh overlay.
8. **Partial:** broad terrain agreement is visible in the approximate overlay;
   buildings, rocks, and debris have not been individually validated against
   known geometry.
9. **Done once:** save `/tmp/scout_day7_pose_registered_v2.ply` and its JSON
   summary. Repeatable generation is pending.

**Expected result:** a usable 3D map of the landslide environment. **Not yet
met:** current output is a single pose-registered point cloud; its alignment,
repeatability, obstacle identification, and coverage need validation.

### Day 7.4 — 3D map storage and environmental representation

**Status: transfer and reload path demonstrated; reusable mapping semantics
remain partial.** The 202,728-point PLY and JSON summary were sent to the local
TCP receiver, which verified the byte count and SHA-256 and saved the map under
`/tmp/scout_coordinator/maps`. The map remains a local NEU frame relative to
the first telemetry pose. It has no semantic obstacle classes or coverage map.

**Goal:** make the generated map reusable by other components.

1. **Prototype done:** save the point cloud as an ASCII PLY with a JSON summary.
   It currently lives in `/tmp`; choose a durable project data location for
   maps intended for reuse.
2. **Partial:** the PLY is labeled local North-East-Up relative to the first
   telemetry pose. Store a formal frame definition, origin, units, and pose
   transform with each map.
3. **Pending:** identify and represent major obstacles such as buildings,
   rocks, and debris; raw returns are not semantic obstacle labels.
4. **Pending:** represent explored/unexplored regions if feasible.
5. **Prototype done:** a documented TCP sender/receiver transfers the PLY and
   metadata; receiver verified size and checksum on one transfer.
6. **Partially demonstrated:** the PLY was loaded for plots and transferred
   with metadata. Repeatable reload/visualization and frame-consistency checks
   remain pending.

**Expected result:** a saved, reusable 3D representation of the landslide
environment. Basic coordinator access is demonstrated once; obstacle
extraction, coverage representation, and repeatability remain incomplete.

### Day 7.5 — Share detection and mapping information

**Goal:** send Scout's information to the central coordinator.

1. **Demonstrated experimentally:** live Scout RGB inference has run for
   person and fire/smoke models, with frames and JSONL logs saved. Outputs are
   intermittent and not validated as reliable incidents.
2. **Demonstrated experimentally:** fire and smoke detections were produced by
   the supplied AI model on camera frames; labels and scores remain
   preliminary.
3. **Partial:** an offline location pipeline combines detections, synchronized
   pose, nominal camera geometry, and the Scout LiDAR map to produce candidate
   surfaces. It has not been calibrated or validated for dispatch.
4. **Partial:** JSONL observations include confidence, model/class labels,
   frame-receive UTC, inference-completion UTC, optional GStreamer PTS, pixel
   boxes, and an explicit non-ground-truth source marker. Calibrated exposure
   timestamps and synchronization with PX4 pose remain pending.
5. **Demonstrated once:** camera detections were paired with PX4 pose; the
   paired report replay reached the coordinator. Location was unlocalized in
   that initial transport check.
6. **Demonstrated once:** the pose-registered PLY and metadata reached the
   coordinator's map receiver and passed size/checksum verification.
7. **Pending:** encode obstacle information only after map obstacle
   representation is implemented.
8. **Partial:** versioned JSON report and map-transfer formats exist;
   obstacle updates are not implemented because the map has no semantic
   obstacle representation.
9. **Demonstrated once:** local UDP receiver accepted pose-paired detection
   reports; TCP receiver verified a map transfer. Duplicate handling and the
   persistent incident registry are being added under Day 8.2.

**Expected result:** the central coordinator receives Scout's experimental
detection reports and one verified map transfer. This is a prototype only;
reliable detections, validated localization, semantic obstacle messages, and
robust communications remain work.

### Day 8 — Incident location estimation and registry

#### 8.1 Camera-based location estimation

**Status: preliminary pipeline demonstrated; acceptance is not complete.** On
the Oct 31 Scout run, the reviewed fire/smoke model was applied to 31 recorded
RGB frames. Its training run stopped after three epochs; the four-image
validation result was precision 0.762, recall 0.450, mAP50 0.559, and mAP50-95
0.245. This split is too small and comes from one flight, so it is not a
generalization claim. Pairing found pose samples within 1 s for 26/31
observations (maximum matched position/attitude delta 0.765 s). The map
projection produced 19 unvalidated surface candidates and left 11 detections
unlocalized. Two fire candidates were compared with scene references only
after estimation and had about 0.31 m and 0.33 m horizontal error. This is an
offline, same-scene preliminary check, not independent calibration or a
reliable accuracy bound. Person-location accuracy has not been measured.

1. Calibrate RGB intrinsics and camera-to-body transform/orientation.
   Intrinsics currently come from the simulator SDF; optical-axis convention
   and distortion have not been empirically calibrated.
2. Synchronize image timestamps with PX4 position and attitude.
   Recorded synchronization was within 1 s for 26/31 observations; improve
   pose sampling/alignment before treating locations as accurate.
3. Estimate coordinates from live detections and calibrated camera geometry,
   using sensor-built 3D environmental data where available. State terrain and
   visibility assumptions; preserve confidence and uncertainty.
   Existing output is a map-surface proxy near the box footpoint, not a
   validated incident position. Low-confidence and edge-clipped boxes remain
   unlocalized.
4. Only after generating estimates, compare them with known Gazebo victim
   poses and measure error. Ground truth must not be an inference input.
   Two fire references were used for this offline comparison only; no GT is
   used by inference, location generation, or coordinator ingestion.
5. Record uncertainty due to slope, smoke, occlusion, and viewing angle.
   A calibrated uncertainty model and person/smoke location evaluation remain
   pending.

**Acceptance:** partially met for two fire examples only. Do not claim 8.1
complete until camera geometry and synchronization are empirically checked and
multiple independent runs evaluate person, fire, and smoke localization.

#### 8.2 Central incident registry

Track a unique incident ID, type (victim/fire/smoke), estimated coordinates,
confidence, detection timestamp, reporting drone, assigned drone, status, and
history. Deduplicate repeated reports and update existing incidents.

**Status: basic creation, spatial merge, and replay idempotency verified.** The
coordinator received 14 unique reports from the recorded candidate file and
kept 14 observations after replaying the same file again. Nineteen
map-supported detections were grouped into four provisional clusters (two
fire, two smoke); eleven detections remained unlocalized. A second replay
created or updated no incidents. Registry records include confidence,
timestamps, reporting drones, assignment field, status, and observation
history. Clusters remain unvalidated and cannot be dispatched. Specialist
assignment and operator status transitions are handled by the Day 9 tools.

**Acceptance:** basic report ingestion, spatial candidate clustering, and
duplicate-report handling passed on recorded data. Full incident lifecycle and
assignment remain pending and must use validated locations only.

### Day 9 — Central Python coordinator and basic communication

**Status: prototype acceptance verified; limitations recorded below.**
The user run connected to all three PX4 vehicles; each reported landed,
armable, and 100% battery, and the collector sent no flight commands. After
operator review of camera/LiDAR overlays, three incidents were operator-
attested and one weak smoke candidate was rejected. The planner proposed one
Fire-Response task and kept the remaining validated fire/smoke incidents
pending because only one Fire-Response vehicle was available in that plan.
`collect_swarm_status.py` records connection, armability, landed/in-air state,
position, and battery. `plan_incident_assignments.py` proposes a role only
when incident and location are explicitly `validated`, with fresh available
vehicle status and sufficient battery. It does not connect to MAVSDK or send
flight commands. `review_incident_registry.py` lists contributing report IDs
and records audited operator status changes. Validation requires an explicit
location-confirmation flag, a sensor-evidence file, and a reason; it records
operator attestation without changing coordinates or consulting ground truth.
These attestations are not independent metric localization accuracy results.
`manage_incident_assignment_queue.py` records operator-accepted schedules and
later operator-confirmed completions in a separate ledger. The planner can
consult that ledger to reserve a vehicle until its task is completed. Completion
is not inferred from telemetry, and no flight command is sent. Queue behavior
was verified on the user's live status snapshot and a copied ledger: the
scheduled Fire-Response vehicle was reserved, and recording completion in the
copy made it eligible for the next validated fire. The production ledger still
has the original item scheduled. Live dispatch and end-to-end assignment
persistence remain future work.

Start with deterministic, rule-based allocation. Receive Scout's genuine
sensor-derived detection reports and map updates; track drone position,
battery, and availability; assign validated victim incidents to
Medical-Response and validated fire incidents to Fire-Response; prevent
duplicate assignments and leave incidents pending if a vehicle is unavailable.
Do not read Gazebo model poses or world state to make or populate assignments.

`scout_udp_protocol.py` now provides bounded report retries awaiting a matching
coordinator acknowledgement. Live camera reports and recorded replays include
session IDs and sequence numbers. The coordinator acknowledges accepted,
duplicate, stale, and rejected reports; it persists per-session sequence state,
and duplicate report IDs are not ingested or archived twice. These changes are
implemented. A localhost replay test received `accepted` for all 14 initial
reports and `duplicate` for all 14 on replay, with 14/14 acknowledgements on
both passes. With `--test-drop-first-ack`, the first report was ingested once,
its first acknowledgement was dropped, and its retry received a duplicate
acknowledgement; the sender completed with 14/14 reports acknowledged. This
validates one deterministic localhost retry, not actual network packet-loss,
delay, or MANET behavior. Keep the fault hook limited to isolated test
registries. UDP is transport, not a complete MANET.

**Acceptance:** read-only vehicle status, a positive role proposal for a
validated record, and the no-flight-command boundary were verified by the
user. Queue scheduling, vehicle reservation across planning rounds, and
completion/release behavior passed a dry-run lifecycle check. UDP acceptance,
duplicate acknowledgement, idempotent replay, and one deterministic
ACK-loss/retry passed on localhost. Actual network delay/loss, telemetry
freshness edge cases, and network limitations still need validation. No
flight command is enabled by these tools.

The live Scout detector was also run against the coordinator during flight 02:
fire/smoke and person detections were produced, and reports shown in the
terminal were acknowledged. Pose pairing was present for 11 of 13 inference
records; the two unmatched records had no detections. PX4 telemetry confirmed
the Scout returned and landed. This verifies the localhost camera-to-coordinator
software path for that run, not localization accuracy or specialist dispatch.

**Day 9 acceptance is complete for this prototype.** Actual lossy-network
behavior, MANET operation, telemetry freshness edge cases, and mission dispatch
remain outside this acceptance: MANET testing is Day 10.4 and live mission
dispatch is Day 11.

### Day 10 — Response sensors, safety, MANET, and battery work

#### 10.1 and 10.2 — Add 3D LiDAR to response drones (historical setup)

Inspect each existing SDF first. Add one 3D LiDAR to Medical-Response and one
to Fire-Response with documented parameters. Publish/read data via Gazebo
Transport and Python, without ROS. Verify independent scans while hovering and
descending. Response drones must retain local sensing; do not make them wholly
dependent on Scout's map.

**Implementation status:** separate sensor-only X500 wrappers now add a
5 Hz, 180-by-16 GPU LiDAR to Medical-Response and Fire-Response with distinct
Gazebo topics. The launcher stages and selects these wrappers while leaving
PX4's stock X500 model untouched. XML parsing and launcher syntax checks pass.
Runtime probes received 2,880 ranges per scan, with 2,113 positive returns on
Medical-Response (0.201–15.188 m) and 2,479 on Fire-Response (0.201–15.769 m).
Independent scan publication is verified while the simulation is running;
Medical-Response then completed a vertical 3 m hover, held for 15 seconds, and
landed. Its 55-second recording contained 79 scans (1.44 Hz observed), each
with 2,880 ranges and 939–2,127 positive returns. This observed rate is below
the configured 5 Hz and needs investigation. Fire-Response produced 62 scans
in an initial 55-second recording (1.13 Hz). A later 55-second recording
contained 33 scans (0.60 Hz), each with 2,880 ranges and 2,470–2,480 positive
returns. The first Fire-Response hover attempt timed out before hover telemetry
while its GCS link was lost. A fleet heartbeat diagnostic then confirmed GCS
recovery for PX4 system IDs 1, 2, and 3. On retry, Fire-Response armed, took
off, held near 3 m for 15 seconds with near-zero horizontal velocity, and PX4
telemetry confirmed landing. The observed LiDAR rates remain below the
configured 5 Hz, and descent-specific sensing and obstacle avoidance remain
pending.

**Follow-up verification:** each response drone then completed a vertical 2 m
hover followed by PX4 landing while its independent Gazebo LiDAR topic was
subscribed continuously. Ambulance published 30 scans during its 13.09-second
landing interval (2.292 scans/s wall time); Fire-Response published 30 during
14.68 seconds (2.043 scans/s). PX4 confirmed both landed, and read-only
telemetry afterward showed both armable. This closes the hover/descent sensing
check for 10.1–10.2. The measured wall-clock scan rates remain below the
configured 5 Hz and are affected by Gazebo runtime performance; this is a
remaining performance limitation, not a sensor-topic absence.

An analysis-only `response_lidar_safety_monitor.py` evaluates the forward
LaserScan sector and reports provisional CLEAR/CAUTION/STOP labels plus a scan
rate check. A 20-second run observed 29 Medical scans (1.45 Hz) and 28 Fire
scans (1.40 Hz), both below the 4 Hz acceptance threshold. All scans reported a
fixed nearest forward return: 0.212 m at -15.08 degrees for Medical and 0.323 m
at +29.16 degrees for Fire. These could be vehicle geometry or nearby world
surfaces; their source is unresolved. The tool cannot connect to PX4 or command
motion, and its distance thresholds are not a validated safety envelope. In a
later 8-second capture associated with the Medical 3 m hover (telemetry: hover
start 07:21:57, hover end 07:22:23, then landed), 9 scans arrived (1.125 Hz):
the first 5 were CLEAR around 5.3 m and the last 4 were STOP at 0.20–0.27 m.
Individual scan timestamps were not saved, so returns cannot be aligned exactly
to vehicle altitude. A Gazebo stats sample reported real-time-factor values
from 0.0058 to 0.037, indicating severe simulator slowdown during that sample.

#### 10.3 — Local obstacle avoidance for Scout and Ambulance

Process Scout's and Ambulance's existing 180-by-16 Gazebo LaserScan streams,
identify obstacles along each intended path, use defined vehicle
clearance/stopping margins, and implement local stop/slow/replan behavior.
Test approach, descent, and landing, including sensor blind spots. Preserve
PX4 failsafes and stability. Connect avoidance to active mission flow only
after each vehicle's sensor frame, freshness, and bounded planner behavior are
validated.

**Acceptance:** tested response to obstacles in bounded scenarios. A LiDAR
sensor alone does not provide collision avoidance.

**Active-scope status:** `response_lidar_guarded_move.py` now recognizes
Ambulance, Scout (`/scout/lidar/points`), and the retained historical Fireman
topic. Scout's sensor is mounted level on the X500 frame, but the shared
guard/planner has not been validated against Scout telemetry or its survey
mission. Next verify scan-to-pose alignment and self-mask assumptions in a
read-only run, then perform bounded SITL validation before connecting avoidance
to autonomous survey/navigation. Fireman results in the historical progress
record below do not count toward this active acceptance criterion.

**Progress:** a shared PX4/MAVSDK guarded-move prototype applies CLEAR,
CAUTION, STOP, and stale-data HOLD behavior to both response-drone LiDAR topics.
Both drones completed bounded clear-scene tests and tests with an unmapped
static obstacle; when no safe local corridor was available, the planner held
position, timed out, and PX4 confirmed landing. A SITL-only stale-scan injection
also produced HOLD and landing on both drones. The injection verifies the
controller's stale-data branch; it does not simulate loss of the Gazebo
publisher. A local planar corridor selector can choose a limited-angle detour
or HOLD behind `--replan-obstacles`; Fire-Response selected DETOUR in one
pre-injection run, so detour selection still needs investigation and validation.
During follow-up, a stale temporary Fireman obstacle was found in Gazebo and
removed; with the clear lane restored, Fireman returned GOAL on all 11 hover
scans and all 30 landing scans. A new narrow-obstacle detour attempt then
initially aborted before arming because the attached MAVSDK server timed out
waiting for battery telemetry; after restarting the server and simulation,
read-only status confirmed Fireman landed, armable, and at 100% battery. The
next bounded obstacle run selected DETOUR but changed its scan-relative heading
from -45 to -75 degrees, made only 0.36 m of 1.5 m goal progress, then failed
closed to HOLD and PX4-confirmed landing. Review found the conversion between
counter-clockwise Gazebo scan angles and clockwise PX4 NED yaw was reversed in
both goal-bearing and movement-heading calculations. The conversion is fixed
and covered by unit tests. The corrected Fireman obstacle run selected DETOUR
(-75 to -65 degrees) intermittently, but repeatedly alternated with HOLD due to
stale scans and made effectively no progress (about 0.033 m regressing to
0.003 m of 1.5 m). It hit the 35-second flight limit and PX4 confirmed landing.
This is a failed detour validation, not an avoidance pass. The reported scan
rate was 5.00 Hz in simulation time and 1.59 Hz wall-clock before takeoff. The
guard now logs scan age/rate and checks the largest observed scan gap against
two-thirds of the freshness limit before arming, preserving one-third of that
window as runtime jitter reserve. The in-flight stale-scan HOLD threshold is
unchanged. The read-only
`response_lidar_scan_readiness.py` check reports rate, latest-scan age, and the
largest scan gap without connecting to PX4; use it before any further flight.
Eleven unit tests cover guard and planner decisions.
After the Fireman server recovered, another bounded 1.5 m attempt passed
preflight (5.00 Hz simulation rate, 1.61 Hz wall rate, 0.65 s largest warmup
gap), but in-flight scans never showed an obstacle in the commanded corridor:
the planner stayed GOAL with 2.5 m lookahead clearance, progressed 1.227 m,
then reached the 35-second limit and PX4 confirmed landing. This run verifies
clear-path motion and landing only; it did not validate obstacle detouring. The
test obstacle pose relative to Fireman's actual forward path must be verified
before another detour attempt. The subsequent Gazebo model-list/pose check found
Fireman at approximately (-74.667, -56.990, 1.687) m, yaw -0.003 rad, and did
not list `fireman_detour_test_obstacle`; the missing obstacle explains why this
run remained GOAL. Recreate and verify the obstacle before another detour test.
The obstacle was then recreated at (-72.820, -56.990, 2.800) m. Fireman saw
nearest forward returns decrease from 1.58 m to 0.81 m, switched from GOAL to
DETOUR (-65 degrees) around 0.81 m of progress, and intermittently held on
stale scans (scan age 1.53-2.22 s). It reached 0.88 m of the 1.5 m leg, hit the
35-second limit, and PX4 confirmed landing. This confirms obstacle detection,
DETOUR selection, stale-scan HOLD, and bounded landing; it does not show a
completed bypass. The next test must allow a goal beyond the obstacle and
address scan gaps while preserving the stale-data HOLD. The post-run pose check
put Fireman at (-73.743, -57.089, 1.607) m, yaw -0.014 rad, with the box center
at (-72.820, -56.990, 2.800) m: its near face is about 0.67 m ahead and 0.10 m
off the forward line. A 2.0 m leg from this new start would extend beyond the
box, but another flight should wait until scan freshness is acceptable under
flight load.
The follow-up 2.0 m attempt passed its 20 s preflight (5.00 Hz simulation
rate, 1.66 Hz wall rate, 0.64 s largest gap), then selected DETOUR repeatedly.
In flight, scan ages periodically reached 1.55-2.94 s, triggering HOLD; the
detour bearing changed from -75 to -25 degrees and progress reached only 0.28 m
before the 75 s limit. PX4 confirmed landing. This is another safe abort, not a
completed bypass. The planner now prefers the previous safe detour heading,
while still rejecting it if that corridor is blocked and preferring GOAL when
the direct corridor clears; unit tests cover those transitions. This code
change has not yet been flight-validated. The in-flight scan delays remain a
Gazebo/runtime issue to diagnose; keep stale scans fail-closed and do not retry
until scan timing is addressed. Fourteen guard/planner unit tests pass.
An idle three-vehicle Gazebo stats sample then advanced from sim time 71.340 s
to 72.172 s over real time 240.915 s to 244.367 s, an interval RTF of about
0.24. This supports investigating simulator load rather than relaxing the
scan-age guard. The launcher now has a Fireman-only mode (PX4 instance 2,
system ID 3, UDP 14542) to compare LiDAR timing without the other PX4 instances;
this mode has passed shell syntax validation but has not yet been run in SITL.
Approach success around an obstacle, descent/landing with obstacles, sensor
blind spots, 3D route planning, multi-drone deconfliction, and
incident-dispatch integration remain unfinished. This is a SITL prototype, not
a validated collision-avoidance system.

#### 10.4 — Infrastructure-independent MANET

Investigate compatible radios, Linux drivers, Wi-Fi ad-hoc/mesh support, and
routing. Test Scout-to-Ambulance and coordinator-to-drone communication,
heartbeat, link failure, recovery, range, packet loss, and latency without
internet or an external router. Test multi-hop only if available hardware
provides an eligible relay that is inside the active scope. Localhost UDP is
only software-path testing, not MANET evidence.

**Active participants:** Scout, Ambulance, and the central coordinator. Test
direct drone-to-drone and coordinator-to-drone paths; test multi-hop only when
the available radio topology supports it. Do not use Fireman as a relay while
it is excluded.

**Acceptance:** document actual radios/drivers and demonstrate measured
communication without internet or an external router. If hardware is
unavailable, mark MANET operation as not validated; localhost UDP remains
software-protocol testing only.

#### 10.5 — Realistic battery simulation

Initial assumption to evaluate (not a confirmed physical pack): 4S LiPo model,
5,000 mAh, 14.8 V nominal, 16.8 V full, and 13.2 V suggested low operating
point (about 74 Wh nominal). Configure each vehicle independently. Replace any
timer-only model with a documented current/power estimate integrated over
simulation time. Compare state of charge, voltage, and current with PX4 and
MAVSDK readings over takeoff, hover, waypoint travel, and landing.

Preserve mass, motors, propellers, and flight behavior. Gazebo's motor model
may not expose measured electrical current; document the estimate and
assumptions rather than presenting it as a physical measurement.

**Active-scope acceptance:** independent, repeatable battery states for Scout
and Ambulance, with explained differences between estimated and reported
energy. Fireman is excluded until it returns to the active scope.

#### 10.6 — Battery-aware mission management

Estimate remaining flight time and mission/return energy, reserve landing
energy, block assignments that exceed available energy, warn on low charge,
and request safe return/landing. Show battery state at the ground station.
Thresholds must be based on measured simulation behavior, not an arbitrary
percentage alone.

Apply the policy to Scout and Ambulance mission roles only. In particular,
reject an Ambulance assignment when available energy cannot support the
outbound leg, response, and safe return/landing reserve.

### Day 11 — Mission dispatch and swarm coordination

Use MAVSDK-Python for validated waypoint execution. Dispatch Ambulance using
coordinates from validated sensor-derived Scout incident reports; track
status and completion, handle cancellation/unavailability, and define safe
hold/return behavior during communication failures. Add multi-hop forwarding only after
Day 10 MANET tests. Do not retry the full two-drone waypoint mission until the
earlier Offboard/gRPC failure is diagnosed and Scout-only waypoint control is
validated.

**Acceptance:** one incident is assigned, executed, tracked, and landed safely;
then test concurrent incidents and failures.

### Day 12 — Ground-station dashboard

Display Scout and Ambulance positions, altitude, battery, availability, and
mission state;
incidents and confidence; victim/fire/smoke locations; map and changed
obstacles; explored regions; MANET links, delay/loss, multi-hop status; pending,
assigned, and completed missions. The existing `src/relay/viewer.py` remains a
MAPPO replay viewer unless deliberately extended and validated.

**Acceptance:** one operator view presents live, timestamped Scout and
Ambulance state from the coordinator with clear stale/disconnected indicators.

### Day 13 — Integrated scenarios and measurements

Test one/multiple victims, fire, simultaneous incidents, changed debris,
blocked routes, obstacles during approach/descent, low battery, unavailable
drone, communication interruption, loss of direct link, multi-hop recovery,
and message synchronization after reconnect.

Measure person and fire/smoke precision/recall, victim/fire localization
error, map coverage and change detection, obstacle avoidance, assignment time,
mission completion, energy/endurance, communication latency/loss, and recovery.
Keep failed runs and limitations with the results.

Run the integrated Scout-to-coordinator-to-Ambulance workflow. Exclude Fireman
from active scenarios and acceptance until its IMU issue is separately
resolved and the scope is explicitly changed.

**Acceptance:** scenarios are reproducible and each metric has recorded data;
unimplemented or untested capabilities are marked plainly.

### Day 14 — Demonstration and handoff documentation

Fix integration issues; record the validated Scout-and-Ambulance demonstration;
prepare architecture, sensor layout, MANET, role, model, mapping, and battery diagrams;
document commands, measured results, limitations, and required hardware and
real-world validation. Do not claim diagnosis, reliable autonomy, or readiness
for real disaster response without appropriate evidence.

**Acceptance:** a teammate can reproduce the demonstrated two-drone workflow
from this repository and the documented environment. Document Fireman as
temporarily excluded and keep its unresolved PX4 health issue visible.

## Current perception workflow and training status

Run tools from the repository root. The prepared YOLO data is reproducible with:

```bash
.venv-perception/bin/python integration/mavsdk_controller/prepare_victim_yolo_dataset.py \
  --source perception/datasets/victim_detection/images/unassigned \
  --labels perception/datasets/victim_detection/labels/unassigned \
  --output perception/datasets/victim_detection/prepared --overwrite
```

The exploratory CPU fine-tuning command is:

```bash
.venv-perception/bin/yolo detect train \
  model=./yolo26n.pt \
  data=perception/datasets/victim_detection/prepared/data.yaml \
  epochs=30 imgsz=640 batch=4 workers=0 device=cpu patience=8 \
  project=/tmp/swarm_day7 name=victim_finetune_fixed
```

Current run output is in `/tmp/swarm_day7/victim_finetune_fixed/`; the selected
weights are `weights/best.pt`. It stopped at epoch 11 with best checkpoint at
epoch 3. The reported validation scores (P=1.0, R=0.99, mAP50=0.995,
mAP50-95=0.378) came from a single image with three people and were used for
checkpoint selection. They are not an independent performance result. The
photometric variants do not add independent scenes, labels, people, or viewing
geometry.

## Simulation launch and safety notes

From the repository root:

```bash
./integration/mavsdk_controller/launch_three_px4.sh
```

The default world is `worlds/earthquake_landslide_apron_world.sdf`; logs go to
`/tmp/drone-swarm-px4-multi-<uid>/`. Do not start another session while one is
already running. Confirm Scout is landed after each flight before sending
another flight command. Keep camera receivers from competing for UDP 5600.

## Handoff rules

- Read this roadmap, root `README.md`, and
  `integration/mavsdk_controller/README.md` before launching or editing.
- Back up the active world before any world edit; preserve the current scene and
  all launch pads.
- Inspect existing SDF/model resources before adding sensors or assets.
- Do not alter PX4 parameters, flight controllers, motor/propeller setup, or
  vehicle dynamics to work around unexplained behavior.
- Keep ROS/ROS 2 out of this project.
- Do not call model assets detections, boxes coordinates, or plans execution.
- Use rule-based coordination first; preserve MAPPO and its checkpoints.
- Separate verified results, proposed assumptions, and untested work in every
  report.
