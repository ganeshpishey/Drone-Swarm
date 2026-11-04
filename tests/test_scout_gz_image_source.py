from __future__ import annotations

import unittest
import threading
import time
from types import SimpleNamespace

import numpy as np

from integration.mavsdk_controller.scout_gz_image_source import (
    GazeboScoutImageSource, image_message_to_bgr, temporal_pair_within_bound,
)
from integration.mavsdk_controller.scout_perception_runtime import PtsOrderTracker


def image_message(*, pixel_format: int = 3, step: int = 6, height: int = 1,
                  data: bytes = b"\x01\x02\x03\x04\x05\x06"):
    return SimpleNamespace(
        width=2,
        height=height,
        step=step,
        data=data,
        pixel_format_type=pixel_format,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nsec=345_000_000),
            data=[
                SimpleNamespace(key="seq", value=["42"]),
                SimpleNamespace(key="frame_id", value=["camera_link"]),
            ],
        ),
    )


class GazeboImageSourceTests(unittest.TestCase):
    def test_rgb_message_preserves_exact_header_identity_and_converts_to_bgr(self) -> None:
        frame, metadata = image_message_to_bgr(image_message())
        self.assertEqual(frame.shape, (1, 2, 3))
        np.testing.assert_array_equal(frame, [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(metadata["image_acquisition_sim_s"], 12.345)
        self.assertEqual(metadata["image_sequence"], 42)
        self.assertEqual(metadata["image_frame_id"], "camera_link")
        self.assertEqual(metadata["frame_identity_status"], "exact_direct_message")

    def test_bgr_input_is_not_channel_swapped(self) -> None:
        frame, metadata = image_message_to_bgr(image_message(pixel_format=8))
        np.testing.assert_array_equal(frame, [[[1, 2, 3], [4, 5, 6]]])
        self.assertEqual(metadata["image_pixel_format"], "BGR_INT8")

    def test_rgb_input_preserves_nonuniform_pixels_and_ignores_row_padding(self) -> None:
        # Two RGB pixels per row followed by two padding bytes. Padding must
        # never become pixels or overwrite scene variation during decoding.
        message = image_message(
            step=8,
            height=2,
            data=bytes((10, 20, 30, 40, 50, 60, 255, 254,
                        70, 80, 90, 100, 110, 120, 253, 252)),
        )

        frame, metadata = image_message_to_bgr(message)

        np.testing.assert_array_equal(
            frame,
            [[[30, 20, 10], [60, 50, 40]],
             [[90, 80, 70], [120, 110, 100]]],
        )
        self.assertEqual(metadata["image_width"], 2)
        self.assertEqual(metadata["image_height"], 2)
        self.assertGreater(np.unique(frame.reshape(-1, 3), axis=0).shape[0], 1)

    def test_temporal_pair_bound_handles_float_rounding_without_widening_meaningfully(self) -> None:
        self.assertTrue(temporal_pair_within_bound(0.1, 0.1))
        self.assertTrue(temporal_pair_within_bound(0.1000005, 0.1))
        self.assertFalse(temporal_pair_within_bound(0.10001, 0.1))

    def test_rtp_pts_tracker_reports_order_duplicates_and_reordering_without_claiming_identity(self) -> None:
        tracker = PtsOrderTracker()
        self.assertEqual(tracker.update(100)["status"], "first_pts")
        self.assertEqual(tracker.update(200)["status"], "in_order_pts")
        self.assertEqual(tracker.update(200)["status"], "duplicate_pts")
        reordered = tracker.update(150)
        self.assertEqual(reordered["status"], "reordered_pts")
        self.assertEqual(reordered["identity_limit"], "PTS_order_only_no_Gazebo_image_identity")
        self.assertEqual(reordered["frames_seen"], 4)
        self.assertEqual(tracker.update(300)["status"], "in_order_pts")

    def test_invalid_format_dimensions_and_truncated_payload_are_rejected(self) -> None:
        for message, reason in (
            (image_message(pixel_format=1), "pixel_format_unsupported"),
            (image_message(step=5), "dimensions_or_step_invalid"),
            (image_message(data=b"\x00"), "payload_truncated"),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                image_message_to_bgr(message)

    def test_lidar_capture_keeps_simulation_identity_and_monotonic_receipt(self) -> None:
        source = GazeboScoutImageSource.__new__(GazeboScoutImageSource)
        source._condition = threading.Condition()
        source._lidar_messages_received = 0
        source._last_lidar_sequence = None
        source._last_lidar_stamp = None
        source._lidar_sequence_gaps = 0
        source._lidar_duplicates = 0
        source._lidar_reordered = 0
        from collections import deque
        source._lidar = deque(maxlen=64)
        message = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=9, nsec=250_000_000),
                data=[SimpleNamespace(key="seq", value=["17"])],
            ),
            frame="lidar_link",
        )
        before = time.monotonic()
        source._on_lidar(message)
        after = time.monotonic()
        self.assertEqual(source._lidar_messages_received, 1)
        scan = source._lidar[0]
        self.assertEqual(scan["sim_time_s"], 9.25)
        self.assertEqual(scan["sequence"], 17)
        self.assertEqual(scan["frame_id"], "lidar_link")
        self.assertEqual(scan["timestamp_domain"], "gazebo_simulation_time")
        self.assertEqual(scan["message_type"], "gz.msgs.LaserScan")
        self.assertGreaterEqual(scan["receive_monotonic_s"], before)
        self.assertLessEqual(scan["receive_monotonic_s"], after)

    def test_lidar_timestamp_reordering_is_marked_even_without_sequence(self) -> None:
        source = GazeboScoutImageSource.__new__(GazeboScoutImageSource)
        source._condition = threading.Condition()
        source._lidar_messages_received = 0
        source._last_lidar_sequence = None
        source._last_lidar_stamp = None
        source._lidar_sequence_gaps = 0
        source._lidar_duplicates = 0
        source._lidar_reordered = 0
        from collections import deque
        source._lidar = deque(maxlen=64)
        for sec in (12, 11):
            source._on_lidar(SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=0), data=[]),
                frame="lidar_link",
            ))
        self.assertEqual(source._lidar[1]["sequence_status"], "out_of_order_timestamp")
        self.assertEqual(source._lidar_reordered, 1)


if __name__ == "__main__":
    unittest.main()
