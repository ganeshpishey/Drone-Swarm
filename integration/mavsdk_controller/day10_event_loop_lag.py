"""Low-overhead asyncio scheduling-lag measurements for Day 10 runtime."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Any


THRESHOLDS_S = (0.020, 0.050, 0.100, 0.500, 1.000)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Linearly interpolated percentile over a finite, bounded sample window."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_lags(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "sample_count": len(samples),
        "median_s": percentile(samples, 0.5),
        "p90_s": percentile(samples, 0.9),
        "p99_s": percentile(samples, 0.99),
        "max_s": max(samples) if samples else None,
        "threshold_counts": {
            f"over_{int(threshold * 1000)}ms": sum(value > threshold for value in samples)
            for threshold in THRESHOLDS_S
        },
    }


class EventLoopLagMonitor:
    """Sample asyncio scheduling lateness and emit bounded time-window records."""

    def __init__(self, *, run_id: str, interval_s: float = 0.1,
                 report_interval_s: float = 10.0,
                 record_sink: Callable[[dict[str, Any]], None]) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("event-loop lag interval must be finite and positive")
        if not math.isfinite(report_interval_s) or report_interval_s <= 0:
            raise ValueError("event-loop report interval must be finite and positive")
        self.run_id = run_id
        self.interval_s = interval_s
        self.report_interval_s = report_interval_s
        self.record_sink = record_sink
        self._window: list[float] = []
        self._window_start_mono_ns: int | None = None
        self._window_start_utc: str | None = None
        self._cumulative_samples = 0
        self._cumulative_threshold_counts = {f"over_{int(t * 1000)}ms": 0 for t in THRESHOLDS_S}
        self._task_count: int | None = None

    async def run(self) -> None:
        import asyncio

        loop = asyncio.get_running_loop()
        next_deadline = loop.time() + self.interval_s
        window_deadline = loop.time() + self.report_interval_s
        try:
            while True:
                await asyncio.sleep(max(0.0, next_deadline - loop.time()))
                actual = loop.time()
                self._task_count = len(asyncio.all_tasks(loop))
                lag = max(0.0, actual - next_deadline)
                if self._window_start_mono_ns is None:
                    self._window_start_mono_ns = time.monotonic_ns()
                    self._window_start_utc = datetime.now(timezone.utc).isoformat()
                self._window.append(lag)
                self._cumulative_samples += 1
                for threshold, key in zip(THRESHOLDS_S, self._cumulative_threshold_counts):
                    if lag > threshold:
                        self._cumulative_threshold_counts[key] += 1
                next_deadline += self.interval_s
                while next_deadline <= actual:
                    next_deadline += self.interval_s
                if actual >= window_deadline:
                    self._emit_window()
                    window_deadline = actual + self.report_interval_s
        finally:
            self._emit_window()

    def _emit_window(self) -> None:
        if not self._window:
            return
        now_utc = datetime.now(timezone.utc).isoformat()
        now_mono_ns = time.monotonic_ns()
        record = {
            "schema": "day10-event-loop-lag-v1",
            "run_id": self.run_id,
            "clock_domain": "host_monotonic_scheduling_delay",
            "window_started_at_utc": self._window_start_utc,
            "window_ended_at_utc": now_utc,
            "window_started_monotonic_ns": self._window_start_mono_ns,
            "window_ended_monotonic_ns": now_mono_ns,
            "recorded_at_utc": now_utc,
            "recorded_monotonic_ns": now_mono_ns,
            "sample_interval_s": self.interval_s,
            "report_interval_s": self.report_interval_s,
            "asyncio_task_count": self._task_count,
            **summarize_lags(self._window),
            "cumulative_samples": self._cumulative_samples,
            "cumulative_threshold_counts": dict(self._cumulative_threshold_counts),
        }
        self.record_sink(record)
        self._window.clear()
        self._window_start_mono_ns = None
        self._window_start_utc = None
