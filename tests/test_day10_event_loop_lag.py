from __future__ import annotations

import asyncio
import unittest

from integration.mavsdk_controller.day10_event_loop_lag import EventLoopLagMonitor, percentile, summarize_lags


class EventLoopLagTests(unittest.TestCase):
    def test_percentiles_and_thresholds_are_correct(self) -> None:
        self.assertEqual(percentile([0.0, 1.0, 2.0, 3.0], 0.5), 1.5)
        stats = summarize_lags([0.01, 0.03, 0.08, 0.2, 0.75, 1.2])
        self.assertEqual(stats["sample_count"], 6)
        self.assertAlmostEqual(stats["median_s"], 0.14)
        self.assertAlmostEqual(stats["p90_s"], 0.975)
        self.assertAlmostEqual(stats["p99_s"], 1.1775)
        self.assertEqual(stats["max_s"], 1.2)
        self.assertEqual(stats["threshold_counts"], {
            "over_20ms": 5, "over_50ms": 4, "over_100ms": 3,
            "over_500ms": 2, "over_1000ms": 1,
        })
        self.assertIsNone(percentile([], 0.9))

    def test_monitor_measures_monotonic_lag_and_flushes_on_cancellation(self) -> None:
        async def exercise() -> list[dict]:
            records = []
            monitor = EventLoopLagMonitor(
                run_id="test-run", interval_s=0.005, report_interval_s=0.02,
                record_sink=records.append,
            )
            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.012)
            # Deliberately block this test loop briefly to create a measurable
            # scheduling delay without using wall-clock time for the metric.
            import time
            time.sleep(0.025)
            await asyncio.sleep(0.008)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return records

        records = asyncio.run(exercise())
        self.assertTrue(records)
        self.assertTrue(all(row["run_id"] == "test-run" for row in records))
        self.assertTrue(all(row["clock_domain"] == "host_monotonic_scheduling_delay" for row in records))
        self.assertTrue(all(row["window_started_monotonic_ns"] <= row["window_ended_monotonic_ns"]
                            for row in records))
        self.assertTrue(any(row["max_s"] >= 0.02 for row in records))
        self.assertTrue(all(row["asyncio_task_count"] is not None for row in records))


if __name__ == "__main__":
    unittest.main()
