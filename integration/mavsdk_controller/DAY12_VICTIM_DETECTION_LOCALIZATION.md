+# Day 12 — Human Victim Detection and 3D Localization Integration

## Outcome

Day 12 is **partially implemented and not complete**. The project now has an independent point-clustering and conservative person-box/LiDAR-cluster association module, plus a coordinator-compatible provisional report adapter. Synthetic regression tests pass. Existing camera images do not establish human detection performance, and no synchronized human/LiDAR observation with empirical camera calibration and surveyed human geometry is available. No victim world coordinates are considered validated or dispatchable.

Scout, Medical, PX4 parameters, production models/worlds, launch scripts, and flight-control code were not changed. No simulation was launched, and no arming or flight command was sent.

## Existing implementation inspected

- scout_live_person_detection.py accepts direct Gazebo Image messages or legacy RTP. Direct Gazebo mode preserves image simulation time and sequence; RTP does not map PTS back to Gazebo acquisition time. Inference emits only exact supported labels. "kite" and other unrelated classes are ignored. Its current output deliberately leaves detections unlocalized.
- scout_gz_image_source.py subscribes to the direct image, LaserScan, and Gazebo clock topics and retains bounded scan metadata. The latest-image path does not itself supply a synchronized Scout pose and empirical camera/LiDAR calibration.
- scout_3d_localization.py converts the multi-vertical LaserScan and fails closed unless it receives a calibration explicitly marked empirically_verified, common host-monotonic timestamps, fresh local-NED pose, and an origin ID. It estimates a LiDAR return cluster in a box; that alone does not prove the human body centre or ground-contact point.
- scout_incident_coordinator.py validates the Scout report format and retains unlocalized detections as observations. Day10Coordinator keeps them provisional; mission proposal requires a validated location and later an Ambulance-origin transform.
- Day 11G's temporary day11g_association.py achieved 30/30 correct unique matches for two stationary, high-contrast, color-coded calibration targets. Its own report excludes human recognition, human geometry, and dispatchable localization. The temporary Day 11G code was inspected and not copied into production as if those results generalized.

## Human models and available evidence

The configured world includes survivor_female, survivor_male, standing_person_small, and rescue_randy_sitting_small. Their actual SDFs and mesh references were inspected. The Day 12 collection metadata marks the victim pose categories as unverified; the saved metadata also has no resolved camera world pose. No model in the current world/asset list was independently verified as a prone/lying pose. The standing_person_small mesh is scaled by 0.2 and its position vertices span about 0.38 m vertically, so it is a small visual target, not a validated adult-height model. The SDF/model names alone were not used to assert pose.

The two existing Gazebo collection sessions contain six 1280×960 images. All six files have the same SHA256:

51a9bff5bc327d661e5ccea3014edbbc5c76b5f79baebd4bed3051a400613864

Visual inspection shows only a flat green ground view, not a visible victim. Both available person checkpoints returned zero detections on these six images. This is a capture/setup limitation, not a measured detector false negative on a visible human.

## Detector inspection and bounded offline check

- Local yolo26n.pt contains COCO class ID 0 named person; it is a valid general person detector by class metadata. The separate project checkpoint perception/models/fire_smoke_r3_e6_final.pt contains only fire and smoke; it is not a victim detector.
- The available prior fine-tuned person checkpoint is outside the repository at /tmp/swarm_day7/victim_finetune_fixed/weights/best.pt, with class 0 person. It was evaluated only as an offline diagnostic and was not copied into the production pipeline.
- On the single existing manually labeled capture_view2.jpg (three labels, confidence 0.10, 960 image size), COCO YOLO26n produced zero person boxes. The prior fine-tuned checkpoint produced 27 person boxes; a greedy one-to-one IoU ≥ 0.5 comparison matched three to the three labels, leaving 24 unmatched predictions. This one-image result is not a precision/recall estimate and is insufficient to establish reliable detection.
- On additional old Day 7 frames, the fine-tuned model also produced numerous boxes, but there are no independent labels for those frames. Those outputs are not counted as true positives.

No training was run. The current prepared data are variants of a few images from a single capture plus one held-out image, which is not enough for a reliable performance claim or additional training.

## Implementation added

### scout_human_lidar_association.py

- person_class_ids() accepts only a class whose name is exactly person case-insensitively. It rejects the fire/smoke model class list.
- euclidean_clusters() forms 3D Euclidean clusters independently of camera boxes and labels using a spatial hash and union-find.
- clusters_to_world() applies a supplied LiDAR world pose.
- associate_person_boxes_to_clusters() projects each cluster's points with the supplied camera world pose and intrinsics, then requires one unique cluster per person box and one box per cluster. It rejects no-candidate, multi-candidate, and shared-cluster cases.
- A unique result reports only the LiDAR visible-surface cluster centroid, explicitly labeled as not a victim centre or ground-contact estimate.
- build_provisional_victim_report() records simulation timestamps and association status but always emits location_status: unlocalized, a null world location and dispatchable: false. A camera/LiDAR timestamp delta above its bound rejects the association.
- ingest_provisional_report() calls the existing Day10Coordinator ingest_incident interface only. It does not send a MANET packet, assign a mission, or issue flight commands.

### Tests

tests/test_scout_human_lidar_association.py checks exact class selection, independent clustering, two synthetic separated box/cluster matches, ambiguous/shared-cluster rejection, timestamp mismatch rejection, and coordinator acceptance only as a provisional non-dispatchable incident.

These are synthetic algorithm tests. They do not validate Gazebo sensor geometry or human detection/localization accuracy.

## Coordinator integration status

The provisional report schema passed through the in-memory existing Day10Coordinator.ingest_incident() interface. It was accepted as an observation with status: provisional and dispatchable: false; mission proposal returned incident_location_not_validated.

This validates the coordinator's in-process rejection path, not the UDP coordinator service, MANET ACK/retry path, or an end-to-end live detector run. No Medical dispatch was attempted.

## Localization and ground-truth comparison

No victim 3D position or localization error is reported. The checked scout_camera_calibration.json identifies its intrinsics as SDF-derived and states that the optical-axis mapping/extrinsics are not empirically verified. The Day 11G camera/LiDAR pose and bearing experiment does not close that gap for human targets. The latest Day 12 captures also lack visible human pixels and a resolved camera world pose.

The new adapter's cluster centroid is a measured surface-reflection summary. Without a validated body pose/shape model and an empirically validated transform, it cannot be equated to a victim centre, body reference point, or ground-contact coordinate. Ground-truth model poses have not been passed into the association algorithm and no estimate-vs-ground-truth error was fabricated.

## Validation results

- New targeted unit tests: **5 passed**.
- Broader relevant test run using the perception virtual environment and the system pytest module: **75 passed, 3 failed**. All three failures occur before test behavior, when the sandbox denies creation of UDP sockets (PermissionError: Operation not permitted) in existing MANET/runtime loopback tests. Those network tests need to be rerun on the user's host to verify the existing UDP paths.
- An earlier test invocation named a nonexistent tests/test_scout_incident_coordinator.py; no tests ran in that invocation.
- Offline YOLO checks completed on the saved images described above.
- No live Gazebo capture, human/LiDAR association, world-position estimate, localization-error evaluation, or UDP service integration was run.
- No production files were edited, so no production-file backups were necessary. Existing user modifications and prior captures were preserved.

Commands used for code validation:

    PYTHONPATH=/usr/lib/python3/dist-packages .venv-perception/bin/python -m pytest -q \
      tests/test_scout_human_lidar_association.py \
      tests/test_scout_3d_localization.py \
      tests/test_scout_gz_image_source.py \
      tests/test_scout_calibration_sync.py \
      tests/test_day10_system.py tests/test_day10_runtime.py \
      tests/test_victim_dataset_workflow.py tests/test_scout_perception_runtime.py

    .venv-perception/bin/python -m py_compile \
      integration/mavsdk_controller/scout_human_lidar_association.py \
      tests/test_scout_human_lidar_association.py
    git diff --check

## Current runnable perception command

This runs camera detection only; it does not localize or dispatch:

    .venv-perception/bin/python integration/mavsdk_controller/scout_live_person_detection.py \
      --input-source gazebo-image \
      --gazebo-world earthquake_landslide_world \
      --model yolo26n.pt \
      --interval 1 --max-inferences 10 --conf 0.10 \
      --output-dir /tmp/day12_scout_detection_run

The output directory must be new or empty. To use the old fine-tuned checkpoint for diagnostic comparison only, pass --model /tmp/swarm_day7/victim_finetune_fixed/weights/best.pt; do not treat its present outputs as reliable victim detections.

The association helper is currently a library adapter, not yet wired into the live detector process. Its inputs must be an independently built cluster list with points_world_m, a timestamp-matched camera world pose, and runtime CameraInfo intrinsics. It should not be driven with model ground truth.

## Files created

- integration/mavsdk_controller/scout_human_lidar_association.py
- tests/test_scout_human_lidar_association.py
- integration/mavsdk_controller/DAY12_VICTIM_DETECTION_LOCALIZATION.md

No existing production file was modified and no backup was made for these new files.

## Remaining work and next milestone

1. Create a disposable camera view with visible actual human models while Scout is stationary/disarmed. Verify the live camera pose and avoid the identical ground-only frames observed in the existing collections.
2. Capture multiple independent sessions with exact image/LiDAR simulation timestamps, measured transforms, and surveyed model geometry. Verify each candidate model's pose visually; do not label a prone case unless a real prone model/pose is present.
3. Run COCO and a correctly trained person checkpoint on those images, record all predictions, and score against reviewed labels split by capture session. Collect additional independent views before any training or performance claim.
4. Feed independently clustered human-target scans into the new association helper. Verify point support and ambiguity rejection on actual observations.
5. Empirically validate camera/LiDAR axes and transforms, then define what physical point is estimated (body centre, a body surface, or ground contact). Only after repeated ground-truth error tests should a trusted calibration ID and coordinator policy permit validated locations.
6. Wire the live detector, synchronized sensor pair, localization gate, and coordinator transport together. Keep unlocalized and invalid observations provisional and non-dispatchable.

Day 12's definition of success has not been met: actual human detection, unique human/LiDAR association, accurate 3D victim location, and end-to-end coordinator validation on live data remain outstanding.

