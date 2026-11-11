# Scout camera–LiDAR calibration workflow

The current `scout_camera_calibration.json` is nominal simulator/SDF evidence,
not a calibration accepted by `validate_calibration()`. Do not copy its mount
transforms into an empirical result.

## Capture observations

Use the normal Scout-only or Scout–Ambulance simulation workflow and keep
Scout stationary, landed, and disarmed. The calibration target must be a
checkerboard visible from above to both the nadir camera and the LiDAR. A
single horizontal board pose is not sufficient: its plane normal does not
constrain all three translation components. Use at least eight synchronized
captures spanning distinct board tilts, with a minimum normal-design singular
value ratio of 0.20. Avoid the ground plane and other board-sized planar
surfaces in the same LiDAR scan; ambiguous plane matches are rejected.

For a fixed board pose, capture a short session, reposition/tilt the target
without moving Scout, then capture another short session. Keep the simulation
running so all samples share the same Gazebo clock epoch. `target-id` must
identify the same physical target in every session. Use the actual inner-corner
count and measured square size.

Example for each pose (choose a new output path each time):

```bash
.venv-perception/bin/python integration/mavsdk_controller/scout_calibration_capture.py \
  --world earthquake_landslide_world --duration 1.5 \
  --output-dir /tmp/scout-calibration-captures \
  --target-id checkerboard-01 --checkerboard-inner-corners 8 6 \
  --square-size-m 0.04
```

The capture tool writes lossless PNG images, finite XYZ point arrays, sensor
simulation timestamps, image and cloud host-monotonic mappings, clock-fit
residuals, frame/sequence identities, pose provenance, target identity, and a
JSONL manifest. Each capture is marked non-dispatchable.

## Estimate and validate

Pass at least eight valid observations from multiple poses. The estimator
detects the complete checkerboard with OpenCV, calibrates a zero-distortion
camera model, runs `solvePnP` for each image, and matches the board-sized
LiDAR plane by robust RANSAC. It solves the camera-from-LiDAR rotation from
paired normals and translation from plane offsets. It rejects timestamp
mismatches over 0.10 s, clock mapping residuals over 0.01 s, fewer than 20
plane returns, ambiguous planes, reprojection RMS over 1 px, normal alignment
RMS over 2 degrees, translation residual or standard uncertainty over 0.02 m,
rotation standard uncertainty over 2 degrees, intrinsic parameter standard
deviation over 1 px, and inadequate normal diversity. OpenCV calibration
parameter deviations and per-view reprojection errors are retained; bootstrap
variation supplies the reported extrinsic uncertainty.

```bash
.venv-perception/bin/python integration/mavsdk_controller/scout_calibration_estimator.py \
  --manifest /tmp/scout-calibration-captures/capture_*/capture_manifest.jsonl \
  --output /tmp/scout-calibration-estimate.json \
  --pattern 8 6 --square-size-m 0.04 --calibration-id scout-cal-YYYYMMDD \
  --verified-by 'reviewer name'
```

The estimator also needs an independently measured `body_frd_from_lidar`
matrix and a documented evidence JSON (method must be `mechanical_metrology`,
`surveyed_fixture`, or `independent_tracker`) before it can emit a
`empirically_verified` runtime calibration. A nominal SDF transform or
Gazebo entity pose is not valid evidence. Without that independent datum, the
report contains a camera–LiDAR candidate, `body_frd_from_lidar: null`, and
dispatch remains disabled. To write a root-schema calibration for the live
detector, specify a new `--calibration-output` path; it is written only after
all quality gates and `validate_calibration()` pass, and existing files are
never overwritten.

The live detector already accepts this schema via `--calibration`; its
localization wrapper maps the image and PointCloudPacked acquisition times
from Gazebo simulation time to host monotonic, interpolates Scout local-NED
pose at the image timestamp, and calls the fail-closed localizer. A nominal
projection report remains diagnostic and cannot authorize dispatch.
