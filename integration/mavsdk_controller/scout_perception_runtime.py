"""Bounded perception-output worker and frame timestamp diagnostics."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class _OutputTask:
    record: dict[str, Any]
    detections_path: Path
    frame_bgr: np.ndarray
    raw_frame_path: Path
    annotated_frame_path: Path
    latest_frame_path: Path
    colors: dict[str, str]
    ready: threading.Event = field(default_factory=threading.Event)


class BoundedPerceptionOutputWriter:
    """Write JSON and preview images off the inference thread using a bounded queue.

    The queue applies backpressure rather than silently dropping observation records.
    Camera callbacks remain independent and keep only their latest frame.
    """

    def __init__(self, queue_depth: int = 2):
        if queue_depth < 1:
            raise ValueError("queue_depth must be positive")
        self._queue: queue.Queue[_OutputTask | None] = queue.Queue(maxsize=queue_depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="scout-perception-writer", daemon=True)
        self._thread.start()

    def submit(self, *, record: dict[str, Any], detections_path: Path,
               frame_bgr: np.ndarray, raw_frame_path: Path,
               annotated_frame_path: Path, latest_frame_path: Path,
               colors: dict[str, str]) -> float:
        task = _OutputTask(record, detections_path, frame_bgr, raw_frame_path,
                           annotated_frame_path, latest_frame_path, colors)
        started = time.monotonic()
        while True:
            if self._error is not None:
                raise RuntimeError("perception_output_writer_failed") from self._error
            try:
                self._queue.put(task, timeout=0.1)
                break
            except queue.Full:
                continue
        waited_ms = (time.monotonic() - started) * 1000.0
        record.setdefault("processing", {})["output_queue_wait_ms"] = waited_ms
        task.ready.set()
        return waited_ms

    def _write(self, task: _OutputTask) -> None:
        task.ready.wait()
        task.detections_path.parent.mkdir(parents=True, exist_ok=True)
        with task.detections_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(task.record, separators=(",", ":")) + "\n")
        rgb = task.frame_bgr[:, :, ::-1]
        raw = Image.fromarray(rgb)
        annotated = raw.copy()
        draw = ImageDraw.Draw(annotated)
        for detection in task.record.get("detections", []):
            box = detection.get("xyxy_pixels")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            category = str(detection.get("type", "person"))
            color = task.colors.get(category, "#ffffff")
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            confidence = float(detection.get("confidence", 0.0))
            draw.text((x1, max(0, y1 - 16)), f"{category} {confidence:.2f}", fill=color)
        task.raw_frame_path.parent.mkdir(parents=True, exist_ok=True)
        task.annotated_frame_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(task.raw_frame_path, quality=92)
        annotated.save(task.annotated_frame_path, quality=90)
        latest_tmp = task.latest_frame_path.with_suffix(task.latest_frame_path.suffix + ".tmp")
        annotated.save(latest_tmp, format="JPEG", quality=90)
        latest_tmp.replace(task.latest_frame_path)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._write(task)
            except BaseException as error:  # surface worker failures to submit/close
                self._error = error
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("perception_output_writer_failed") from self._error


@dataclass
class PtsOrderTracker:
    """Track decoded-frame PTS ordering without claiming Gazebo identity."""
    previous_pts_ns: int | None = None
    frames_seen: int = 0
    duplicate_count: int = 0
    reordered_count: int = 0

    def update(self, pts_ns: int | None) -> dict[str, Any]:
        self.frames_seen += 1
        if pts_ns is None:
            return {"status": "pts_unavailable", "delta_ns": None}
        if self.previous_pts_ns is None:
            status, delta = "first_pts", None
        else:
            delta = pts_ns - self.previous_pts_ns
            if delta == 0:
                status = "duplicate_pts"
                self.duplicate_count += 1
            elif delta < 0:
                status = "reordered_pts"
                self.reordered_count += 1
            else:
                status = "in_order_pts"
        if self.previous_pts_ns is None or pts_ns > self.previous_pts_ns:
            self.previous_pts_ns = pts_ns
        return {"status": status, "delta_ns": delta,
                "frames_seen": self.frames_seen,
                "duplicate_total": self.duplicate_count,
                "reordered_total": self.reordered_count,
                "identity_limit": "PTS_order_only_no_Gazebo_image_identity"}
