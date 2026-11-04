# Day 14 command-free integration and terminal monitoring

## What the demonstration uses

`day14_command_free_demo.py` replays the preserved Day 12 COCO evaluation and
validates its sensor-only Scout report through `Day10Coordinator.ingest_incident`.
Those camera detections stay `unlocalized` and `dispatchable=false`. It separately
hash-checks a surveyed Day 12 disposable-world manifest and exposes its named
target as `GROUND_TRUTH_TEST`; the surveyed point is the SDF model origin, not a
human body-centre or ground-contact estimate. The test overlay is never marked
field-dispatchable and does not enter the operational mission proposal gate.

The harness exercises the existing MANET envelope codec and `VehicleAgent`
assignment handler with an in-process ACK double. It drives the existing
`MedicalResponseMachine` with clearly labelled mocked scenario telemetry,
including a rejected unverified-clearance descent, a full virtual-delivery
sequence, and a separate stale-telemetry `SAFETY_HOLD`. No UDP socket, PX4,
Gazebo, MAVSDK session, or flight command is started. The mock target NED origin
is not a PX4 home-origin transform and is not usable for navigation.

## Reproduce the command-free demonstration

From the repository root, using the preserved Day 12 evidence paths available
on the original host:

```bash
OUT="$(mktemp -d /tmp/day14-command-free-XXXXXX)"
python3 integration/mavsdk_controller/day14_command_free_demo.py \
  --evaluation /tmp/day12_human_validation_20261101_TLyxA9/yolo_evaluation.json \
  --ground-truth-manifest /tmp/day12_human_validation_20261101_TLyxA9/manifest_v3.json \
  --target day12_sitting_randy_01 \
  --output-dir "$OUT"
```

The harness refuses an evaluation-image hash mismatch, a world hash mismatch,
an unknown target, or a non-empty output directory. It writes
`day14-demo-state.json`, `events.jsonl`, `source-evidence.json`, and isolated
test-agent persistence under `$OUT`. These are generated evidence, not
production coordinator state.

## Run the read-only terminal monitor

Use the same shell and `$OUT` value from the preceding command:

```bash
python3 integration/mavsdk_controller/day10_ground_station.py \
  --coordinator-state "$OUT/day14-demo-state.json" \
  --interval-s 1
```

This prints changed JSON state summaries in the terminal. It starts no HTTP
server and opens no browser. For operational read-only monitoring, point
`--coordinator-state` at the running Day 10 coordinator state file. The pure
snapshot helper can also read detector logs and a world SDF for offline reports.

## Terminal semantics

- Live connection, armed/landed state, battery, mission and peer status are
  shown only when present in coordinator state; missing values remain unknown.
- Camera detections remain explicitly unlocalized unless a validated localization
  producer supplies coordinates and provenance.
- Reaching a target is not a verified rescue completion event.
- The watcher has no mission, arming, or flight-control endpoint.


## Verified scope and limitations

The preserved COCO evaluation is for one unique synchronized image: two
reviewed target matches (IoU >= 0.5), zero missed targets, and one additional
person false positive. This is an integration replay, not robust detector
validation. It does not prove a live camera stream in this run.

The production Scout camera and LiDAR have no validated shared victim-surface
coverage. Camera/LiDAR association and validated victim localization remain
unavailable. No autonomous Medical navigation/descent, flight execution,
physical MANET/radio performance, payload release mechanism, or medical
treatment was demonstrated. All state-machine flight-like inputs in the
command-free run are mocked solely to exercise software gates.
