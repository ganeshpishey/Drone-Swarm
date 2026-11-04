from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.mavsdk_controller.scout_perception_runtime import (
    BoundedPerceptionOutputWriter,
    PtsOrderTracker,
)


class PerceptionRuntimeTests(unittest.TestCase):
    def test_pts_tracker_marks_order_only_and_does_not_invent_gazebo_identity(self) -> None:
        tracker = PtsOrderTracker()
        self.assertEqual(tracker.update(10)["status"], "first_pts")
        result = tracker.update(20)
        self.assertEqual(result["status"], "in_order_pts")
        self.assertEqual(result["identity_limit"], "PTS_order_only_no_Gazebo_image_identity")

    def test_bounded_writer_records_detection_and_saves_previews_off_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"detections": [{"type": "person", "confidence": 0.9,
                                      "xyxy_pixels": [1, 1, 3, 3]}]}
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            writer = BoundedPerceptionOutputWriter(queue_depth=1)
            writer.submit(
                record=record,
                detections_path=root / "detections.jsonl",
                frame_bgr=frame,
                raw_frame_path=root / "raw_frames/frame.jpg",
                annotated_frame_path=root / "frames/frame.jpg",
                latest_frame_path=root / "latest.jpg",
                colors={"person": "#42a5f5"},
            )
            writer.close()
            saved = json.loads((root / "detections.jsonl").read_text().splitlines()[0])
            self.assertIn("output_queue_wait_ms", saved["processing"])
            self.assertTrue((root / "raw_frames/frame.jpg").is_file())
            self.assertTrue((root / "frames/frame.jpg").is_file())
            self.assertTrue((root / "latest.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
