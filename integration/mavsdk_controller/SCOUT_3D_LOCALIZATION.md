# Scout RGB/LiDAR 3D localization

## Current status

The pure association/localization core is `scout_3d_localization.py`. Its
`laser_scan_to_xyz` adapter validates the actual multi-vertical Gazebo
`LaserScan` dimensions/frame and converts ranges using the project's existing
Scout probe angle convention. The association API then accepts camera bounding
boxes, XYZ LiDAR returns in metres, capture timestamps on one monotonic clock,
and a fresh PX4 local-NED pose. It projects LiDAR points into
the image, associates in-box points, selects the nearest 0.75 m depth cluster,
uses component-wise medians, estimates a conservative scalar uncertainty, and
transforms the representative point through LiDAR, Scout body FRD, and Scout
local NED. All outputs are suggestion-only.

The active `scout_camera_calibration.json` is **not** accepted for localization:
its intrinsics are SDF-derived and its optical-axis mapping is explicitly
unverified. No empirical camera-to-LiDAR calibration or shared image/LiDAR
capture clock is present. The live RGB detector therefore records an explicit
unlocalized reason. Existing coordinator mission gates require `validated`
Scout-local NED coordinates, a matching PX4 home-origin ID, a fresh Ambulance,
and battery reserve; this work does not bypass those gates.

The coordinator also requires a separate trusted policy passed with
`day10_coordinator_service.py --localization-policy POLICY.json`. It must list
the approved calibration ID and coordinator-owned limits for confidence,
uncertainty, LiDAR support, image/LiDAR timestamp delta, and pose age. The
coordinator compares reported measurements against those limits instead of
trusting thresholds included in an incoming report. Without this file, any
claimed validated location remains provisional. Do not populate it until the
matching empirical calibration and independent validation are complete.

The Scout SDF reports a 1280x960 camera with 1.74 rad horizontal FOV at 30 Hz.
Its LiDAR is `gpu_lidar`, topic `/scout/lidar/points`, `gz.msgs.LaserScan`,
180x16 samples, 5 Hz, 0.20-25 m, frame `lidar_link`. The detector currently
receives H.264/RTP on UDP 5600 and preserves a GStreamer PTS, but no verified
mapping from that PTS to Gazebo simulation/capture time exists. A decoded-frame
host receive timestamp is not treated as sensor exposure time.

The PX4 GStreamer source confirms why RTP PTS cannot identify the Gazebo
image: `GstCameraSystem::onImage` retains the latest protobuf image, while its
worker pushes buffers with `GST_BUFFER_PTS` set from
`gst_system_clock_get_time()`. The source does not carry Gazebo's image sequence
or header stamp into RTP, and its single latest-frame slot can replace
intermediate images before encoding. RTP PTS therefore cannot be mapped to an
original Image acquisition timestamp from this stream. The detector now has an
opt-in direct `gazebo-image` input mode, which consumes one `gz.msgs.Image`
object and sends that same object's pixels and header identity to inference. It
records the Gazebo simulation timestamp, sequence, frame ID, receive monotonic
time, separate `gstreamer_pts_ns: null`, callback sequence gaps, detector queue
overwrites, nearest LaserScan delta/ages, and clock-fit quality. Duplicate or
reordered Image identities are rejected. Legacy RTP mode remains available and
records its association status as unproven. Neither mode makes an incident
dispatchable without the existing empirical-calibration, sensor-sync, and pose
gates.

Run direct image capture during a disarmed Scout simulation with:

```bash
.venv-mavsdk/bin/python integration/mavsdk_controller/scout_live_person_detection.py \
  --input-source gazebo-image \
  --gazebo-world earthquake_landslide_world \
  --lidar-topic /scout/lidar/points \
  --interval 1 --max-inferences 5 \
  --output-dir "/tmp/scout_direct_gz_detection_$(date -u +%Y%m%dT%H%M%SZ)"
```

The direct subscriber receives the full 1280x960 RGB payload (3,686,400 bytes)
at up to 30 Hz. The detector coalesces callbacks to the latest frame, decodes
only when inference is due, and sends JSON/image output through a bounded
background writer. A full writer queue applies backpressure; sequence gaps and
queue wait are reported. One metadata observer measured WorldStatistics median
RTF 0.996 without an Image subscriber and 0.847 with one, while a later direct
capture endpoint ratio was 0.69. These are sequential windows with inconsistent
RTF estimates, so the observer's exact performance cost remains unisolated.
Benchmark with matched windows before operational use. Temporal nearest-scan
gating uses half the configured 5 Hz LiDAR period (0.1 s) as a matching bound,
not a motion-error tolerance; motion-induced spatial error has not been
measured, so every pair remains explicitly non-dispatchable.

## SDF and timestamp inspection tooling

`scout_calibration_sync.py` is a read-only inspector and synchronization core.
Run it from the repository root:

```bash
.venv-mavsdk/bin/python -m integration.mavsdk_controller.scout_calibration_sync \
  --output /tmp/scout_sdf_calibration_inspection.json
```

The current SDF computes nominal pinhole intrinsics as `fx=fy=539.936368 px`,
`cx=640`, `cy=480`, with derived vertical FOV `1.453402 rad`. A live Gazebo
`CameraInfo` at simulation time 53.1 s reported 1280x960, frame `camera_link`,
the matching K matrix and zero distortion coefficients. The saved nominal
calibration was corrected from `(639.5, 479.5)` to `(640, 480)` after that
comparison. This verifies simulator intrinsics and frame metadata, not
empirical optical-axis or camera–LiDAR calibration.

The vehicle SDF places `camera_link` at `(0, 0, 0.10)` with pitch `1.5707 rad`
and `lidar_link` at `(0, 0, 0.06)` in `base_link`. `CameraJoint` has the same
orientation but its joint origin is at base z=0, so the report preserves that
0.10 m link-to-joint-origin offset instead of conflating the two frames. The
nominal optical transform applies Gazebo camera +X-forward axes to image
optical +Z-forward, +X-right, +Y-down axes. The derived matrices are explicitly
marked SDF-derived and cannot satisfy the coordinator's empirical calibration
gate.

Gazebo `Image` and `LaserScan` messages both define `header.stamp`; the
`CameraInfo` schema describes its header timestamp as image acquisition time.
The synchronizer requires these Gazebo simulation-time stamps and never
substitutes callback arrival time or RTP PTS. It keeps a bounded, time-sorted
scan buffer, selects the nearest scan, enforces age and delta limits, and uses
the world clock to clear buffers on simulation reset. `SimulationClockMapper`
can map simulation time to monotonic callback time for interpolating PX4 pose;
it refuses unstable fits and mappings outside its observed sample interval.
`localize_gazebo_measurements` combines these gates with a supplied empirical
calibration and a bracketing local-NED pose history. The opt-in direct detector
path records the bounded simulation-to-monotonic clock fit for analysis, but
does not call the localizer: current survey pose samples are GPS and Euler data
rather than timestamped local-NED callback samples with a PX4 origin ID, and
empirical calibration is absent. Keep reports non-dispatchable until those
inputs are measured and connected.

The timestamp probe can be built outside the repository:

```bash
g++ -std=c++17 -O2 -pthread \
  integration/mavsdk_controller/scout_gz_timestamp_probe.cc \
  -o /tmp/scout_gz_timestamp_probe \
  $(pkg-config --cflags --libs gz-transport13)
```

During an already-approved disarmed run, record CameraInfo, LaserScan, and
world-clock stamps with:

```bash
/tmp/scout_gz_timestamp_probe /tmp/scout_camera_lidar_timestamps.csv \
  earthquake_landslide_world 15
```

To include direct Image headers in a short capture, add `--image-metadata`:

```bash
/tmp/scout_gz_timestamp_probe /tmp/scout_camera_lidar_image_metadata.csv \
  earthquake_landslide_world 8 --image-metadata
```

By default it records CameraInfo metadata, scan dimensions, Gazebo simulation
stamps, sequence/frame metadata where available, and monotonic callback-arrival
times. Add `--image-metadata` to subscribe to the actual Gazebo Image topic and
record its acquisition stamp, frame, sequence, dimensions, pixel format, row
step and payload size. It does not save the image bytes, but that option still
receives full image messages and can increase transport and memory load; use a
short capture and monitor RTF. In the disarmed Scout–Ambulance
capture on 2026-11-01, the probe observed 391 CameraInfo messages at 30 Hz and
65 Scout LiDAR scans at 5 Hz. Their simulation timestamps advanced regularly:
median/max gaps were 0.032/0.036 s for CameraInfo and 0.2/0.2 s for LiDAR. The
nearest CameraInfo-to-scan stamp delta had median 0.064 s and maximum 0.2 s
(the maximum includes the capture's final unmatched camera samples). This is
not a camera-image-to-scan association measurement. The host-monotonic clock
fit had maximum residual about 0.333 s, so the pose-time mapper correctly
rejects this run; no pose interpolation or localization was performed. This
full-capture fit does not establish a persistent clock-domain defect. The probe
timestamps callbacks before doing synchronous CSV formatting and writes, so
callback and scheduler latency can perturb this arrival-time fit.

A follow-up 8 s disarmed capture enabled `--image-metadata` and recorded
`gz.msgs.Image` headers directly to
`/tmp/scout_camera_lidar_image_metadata_20261101.csv`. It received 184 image
messages (sequence 44–227), 183 CameraInfo messages, 31 LiDAR scans, and 1,525
world-clock updates. Image and CameraInfo timestamps matched exactly for 183
of 184 image samples; the remaining first image was at the capture boundary.
Image simulation-time rate was 30 Hz with 0.032 s median and 0.036 s maximum
gaps. LiDAR was 5.12 Hz by simulation time, with regular 0.2 s gaps. The
nearest image-to-scan timestamp difference was 0.064 s median, 0.1 s p95, and
0.2 s maximum; 181/184 image stamps were within 0.1 s of a captured scan, with
the remaining boundary samples lacking a later scan. The world clock advanced
at 250 Hz simulation time but only 0.765x wall time over this capture. Each
delivered image payload was 3,686,400 bytes; the metadata-only observer still
received those full payloads. A preceding no-Image observer capture measured
0.868x over its run. These were not duration-matched controlled runs and both
observers synchronously wrote CSV rows in callbacks, so the 0.103x difference
does not isolate the Image subscription as the cause. The Image run recorded a
maximum world-clock callback gap of 0.220 s (the no-Image run had a 0.303 s
maximum); this is evidence of callback/scheduling delays, not a proven cause.
A matched no-Image versus direct-image run with lightweight asynchronous
logging is still required. No raw image data was written by this probe.

## Calibration contract

Runtime calibration must use `scout-rgb-lidar-calibration-v1` and include:

- `calibration_status: empirically_verified`, a unique `calibration_id`, and
  verifier/procedure provenance.
- Camera optical-frame intrinsics (fx, fy, cx, cy), image dimensions and a
  supported measured distortion model. Current implementation accepts only
  `distortion_model: none`; other models fail closed until implemented.
- `camera_from_lidar` and `body_frd_from_lidar`, homogeneous 4x4 rigid transforms
  following the module's frame names/conventions.
- Measured uncertainty bounds for translation, rotation and intrinsics, plus a
  configured maximum dispatch uncertainty and explicit minimum detection
  confidence / supporting-return thresholds for dispatch consideration.
- `synchronization.timestamp_domain: host_monotonic_capture` and a measured
  camera/LiDAR exposure offset bound no greater than 0.5 s.

Do not copy nominal SDF values into this schema and mark them verified. To
calibrate, use a rigid target with surveyed dimensions at multiple ranges and
view angles; calibrate camera intrinsics from multiple board poses, estimate
camera-to-LiDAR extrinsics from the same target observed by both sensors, and
measure their clock offset using a shared event or simulator sensor timestamps
converted to a common monotonic clock. Verify on withheld target poses and
report position error and uncertainty coverage. Then validate body-FRD mounting
against the SDF geometry and a known orientation before using PX4 local-NED pose.

`localize_detections(...)` refuses absent/unverified calibration, mismatched or
stale sensor times, insufficient depth returns, stale/wrong-frame pose, missing
PX4 origin ID, and uncertainty above the configured dispatch limit. Synthetic
unit tests exercise geometry only; they do not establish calibration accuracy.
