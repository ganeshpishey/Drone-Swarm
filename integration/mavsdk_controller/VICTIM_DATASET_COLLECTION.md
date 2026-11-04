# Scout victim dataset collection

This workflow prepares data only. It never launches Gazebo/PX4, changes a
vehicle state, or trains a model. Keep Scout landed and disarmed; do not start
Ambulance or Fireman for this milestone. Every new capture goes into a unique
session directory. Existing datasets and weights remain untouched.

## Existing assets and data

The existing source set is `perception/datasets/victim_detection/images/unassigned`
with matching YOLO files under `labels/unassigned`: six 1280x960 images and 15
class-0 boxes. The `prepared` and `run03_prepared` trees are derived data. Their
31 and 216 training-image counts represent five and six appearance versions per
source frame respectively, not distinct viewpoints. The run03 tree has 36
source frames and 132 boxes after augmentation, with 156 empty-label variants.
The prepared sets duplicate source/augmented files, and their validation split
is one repeated scene. None is a held-out test set. The existing README and
audit report remain authoritative for that historical material.

Available local human assets include `survivor_female`, `survivor_male`,
`standing_person_small`, and `rescue_randy_sitting_small`. The apron world
generator includes those assets under `incident_victim_*` names. Do not infer
pose solely from an asset name or a world pose: visually verify each rendered
model, its orientation, and visibility before writing `pose_category` or a
person box. `standing_person_small` and `rescue_randy_sitting_small` indicate
intended stand/sit variants; verify their actual appearance in the selected
world. A model pose transform does not turn an upright mesh into a prone mesh.

## Controlled capture matrix

Use one immutable `scene_id` per exact world/asset arrangement and one new
capture session per camera/configuration run. Record the true camera position
and view in the world pose snapshot and capture config. Suggested matrix:

| Factor | Capture levels |
|---|---|
| Person asset / pose | Available, visually verified upright, sitting, prone, and at least two distinct meshes where available |
| Camera height / distance | Near, middle, far; record actual camera pose and image box size |
| View | Downward default, plus a changed view only in a temporary copy/configuration that leaves production SDF unchanged |
| Target position | Image center and each quadrant |
| Environment | At least two terrain/background arrangements and two lighting settings, if available in temporary world copies |
| Occlusion | Clear, partial rubble/structure occlusion, and clutter without hiding the target completely |
| Negatives | Rubble-only, buildings-only, vehicle, vegetation/terrain, and shadows/confusing objects; manually confirm no visible person |

Only claim a category after the rendered frame has been reviewed. Capture
multiple frames per setup at the configured interval, but treat all frames from
one arrangement/run as one split group. Do not create train/val/test copies
until at least three independent, reviewed session groups remain after
near-duplicate grouping.

## Prepare the session configuration

Inspect the live topic and pose topic first:

```bash
gz topic -l | rg '/world/.*/model/.*/link/camera_link/sensor/imager/image$'
gz topic -l | rg '/world/.*/pose/info$'
```

Create a config for the actual world in `/tmp` (replace every example value
with a visually checked fact for the run):

```bash
cat > /tmp/victim_capture_config.json <<'JSON'
{
  "scene_id": "apron-rubble-layout-a",
  "scene_class": "positive",
  "world_sdf": "worlds/earthquake_landslide_apron_world.sdf",
  "environment": {
    "terrain": "record actual terrain/background",
    "lighting": "record actual light condition",
    "occlusion": "clear",
    "debris_level": "record actual scene"
  },
  "camera_pose_entity": "x500_mono_cam_down_lidar_0::camera_link",
  "camera_setup": {
    "requested_altitude_m": null,
    "view_description": "record actual orientation and setup",
    "target_positions": ["record actual image region"]
  },
  "victims": [
    {
      "entity_name": "incident_victim_5",
      "asset_model": "standing_person_small",
      "pose_category": "standing",
      "orientation_description": "visually verify",
      "occlusion_description": "visually verify"
    }
  ]
}
JSON
```

For a negative scene set `scene_class` to `negative` and `victims` to `[]`.
Do not list a target as a victim if it is not visually represented in the
scene. World pose information is recorded as a geometric aid only; it does not
generate or certify image-space boxes.

## Capture from an already-running Scout simulation

The script discovers the unique image topic under the requested world. If
there is more than one, inspect `gz topic -l` and pass the exact `--topic`.
The topic must be advertised by Gazebo. The reused `GazeboScoutImageSource`
provides bounded latest-frame delivery, sequence/timestamp parsing and exact
RGB/BGR decoding. The capture also subscribes to the bounded world pose topic
for camera/victim world poses. Images are lossless PNGs with per-image JSON
metadata and a JSONL manifest.

```bash
cd "$HOME/Drone Swarm/Drone-Swarm"
.venv-perception/bin/python integration/mavsdk_controller/capture_victim_dataset.py \
  --world earthquake_landslide_world \
  --capture-config /tmp/victim_capture_config.json \
  --duration-s 90 --interval-s 2 --max-frames 32 \
  --scout-landed-disarmed-confirmed
```

The output is under
`perception/datasets/victim_detection/collections/gazebo/<unique-session-id>/`.
It contains `images/`, initially empty `labels/`, `metadata/`, `session.json`,
and `manifest.jsonl`. The script refuses to reuse a session ID. It records image
simulation time, sequence/frame ID when provided by Gazebo, local monotonic
receive time, dimensions, format, SHA256, selected topic, session/world
metadata, and the nearest available pose snapshot. A missing camera or victim
pose match is recorded as unavailable rather than invented. This capture is
not an annotation and does not run while Gazebo is absent.

## Manual boxes and negative review

Review each original image. Use the existing UI labeler; it writes class 0
(`person`) YOLO boxes. Empty files are valid only after confirming that no
person is visible.

```bash
SESSION='perception/datasets/victim_detection/collections/gazebo/<session-id>'
.venv-perception/bin/python integration/mavsdk_controller/label_victim_frames.py \
  "$SESSION/images" --labels "$SESSION/labels"
```

Then visually verify every saved box against the original frame and record the
review. This command validates pairings/coordinates and writes reviewer and
review-time provenance; it cannot prove that a human actually performed the
visual review, so run it only after doing so:

```bash
.venv-perception/bin/python integration/mavsdk_controller/confirm_victim_annotations.py \
  "$SESSION" --reviewer 'your name' --visual-review-complete
```

For reviewed negatives, save an empty `.txt` for every image before confirming.
Do not use generated bounding boxes unless a separately verified camera
projection accounts for sensor intrinsics/extrinsics, target pose and
occlusion. This workflow deliberately leaves boxes manual.

## Dataset quality report

```bash
.venv-perception/bin/python integration/mavsdk_controller/validate_victim_dataset.py \
  --images perception/datasets/victim_detection/images/unassigned \
  --labels perception/datasets/victim_detection/labels/unassigned \
  --metadata /tmp/no-legacy-metadata \
  --report /tmp/victim_unassigned_audit.json

.venv-perception/bin/python integration/mavsdk_controller/validate_victim_dataset.py \
  --images "$SESSION/images" --labels "$SESSION/labels" --metadata "$SESSION/metadata" \
  --report "$SESSION/quality_report.json"
```

The report checks readable images, matching labels, class IDs, finite normalized
coordinates, positive box sizes, image-boundary corners, dimensions, exact
SHA256 duplicates, practical dHash near duplicates, image hashes against
metadata, session provenance, and whether annotations are marked reviewed.
Missing metadata is reported as an explicit warning for the legacy dataset.
Malformed/missing labels and unreadable images are errors. Empty label files
are counted as reviewed negatives only after their metadata says `reviewed`.
No automatic repair is performed.

## Leakage-resistant split after collection

Keep each capture session intact and augment only after splitting. Split the
Gazebo collection separately from any natural-person collection. The splitter
requires complete labels, matching image metadata, visually reviewed
annotations, and at least three independent sessions. It merges sessions that
share exact or very-near dHash images before allocation, then copies them into
a new, non-existing output directory; it never overwrites its input/output.

```bash
.venv-perception/bin/python integration/mavsdk_controller/split_victim_dataset.py \
  --collection-root perception/datasets/victim_detection/collections/gazebo \
  --output /tmp/victim_gazebo_split_2026_11_01 \
  --ratios 0.70 0.15 0.15 --seed 11
```

The output has `train/`, `val/`, `test/`, and `split_manifest.json`. For a
natural-person evaluation set, use a separate root and call the splitter
separately; do not combine its metrics with Gazebo metrics. Current historical
folders have only one validation scene and cannot produce a valid three-way
split.

## Natural-person data and licensing

No suitable natural-person evaluation fixture is currently present locally.
Open Images is a candidate: the official dataset documentation identifies
images as CC BY 2.0 and annotations as CC BY 4.0. Preserve image-level author,
source URL, license and attribution for every selected image; verify each
image's actual metadata and applicable terms before use. COCO annotations are
CC BY 4.0, but COCO says it does not own the image copyrights and directs users
to Flickr terms, so the annotation license alone does not clear image rights.
No data was downloaded for this milestone.

Sources:

- [Open Images V7 official facts and figures](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [COCO official terms of use](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)

Before training, require independent session-held-out Gazebo and natural-person
tests with reviewed boxes and hard negatives. Choose confidence thresholds on
validation only. Report precision, recall, mAP50, mAP50-95, confusion matrix,
false positives/negatives, and inference latency separately by source and pose.
No new detector is trained by this workflow.
