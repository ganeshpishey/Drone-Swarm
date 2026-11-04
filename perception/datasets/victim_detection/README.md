# Scout victim detector data

This folder starts the custom victim detector dataset. `images/unassigned/`
contains five camera frames from one 8 m hover at the victim area. They are
near-duplicate samples, so they are for labeling and inspection only; collect
additional views before training or claiming detection performance.

## Label the current frames

From the repository root, activate `.venv-perception` and run:

```bash
.venv-perception/bin/python integration/mavsdk_controller/label_victim_frames.py \
  perception/datasets/victim_detection/images/unassigned \
  --labels perception/datasets/victim_detection/labels/unassigned
```

In the window, drag a tight box around each visible human model (include the
whole visible body, exclude debris), press `S` to save that frame, then `N` for
the next. `U` removes the last box; `C` clears the current image; `P` goes to
the previous image; `Q` exits. Existing labels load when the tool is rerun.
Saved text files use YOLO format with class `0` (`person`). Empty label files
mean that an image was reviewed and contains no visible person.

Before training, gather many distinct frames with victims at different image
locations, distances, and occlusion levels. Split by flight/run, not adjacent
frames, so near-identical images do not leak between training and validation.

## Current preliminary training set

`integration/mavsdk_controller/prepare_victim_yolo_dataset.py` generates five
mild appearance variants per training image. It preserves the original boxes;
it does not invent camera viewpoints or new victim examples. The current
prepared set has 30 training images derived from five near-duplicate frames and
one held-out validation image from a second viewpoint. This is only enough for
an initial fine-tuning experiment, not a reliable performance claim.

After labeling the source frames, rebuild the prepared set with:

```bash
.venv-perception/bin/python integration/mavsdk_controller/prepare_victim_yolo_dataset.py \
  --source perception/datasets/victim_detection/images/unassigned \
  --labels perception/datasets/victim_detection/labels/unassigned \
  --output perception/datasets/victim_detection/prepared --overwrite
```

Initial CPU fine-tuning command:

```bash
.venv-perception/bin/yolo detect train \
  model=./yolo26n.pt \
  data=perception/datasets/victim_detection/prepared/data.yaml \
  epochs=30 imgsz=640 batch=4 workers=0 device=cpu patience=8 \
  project=/tmp/swarm_day7 name=victim_finetune
```

The trained weights will be under
`/tmp/swarm_day7/victim_finetune/weights/`. Evaluate them on additional
independent Scout flights before using them for live victim reporting.
