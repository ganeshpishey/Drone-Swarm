"""Print recent raw battery_status samples from a PX4 ULog file."""

from __future__ import annotations

import argparse
import math

from pyulog import ULog


FIELDS = (
    "timestamp",
    "id",
    "source",
    "connected",
    "is_required",
    "voltage_v",
    "current_a",
    "remaining",
    "warning",
    "faults",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog", help="Path to the Scout .ulg file")
    args = parser.parse_args()

    ulog = ULog(args.ulog)
    datasets = [dataset for dataset in ulog.data_list if dataset.name == "battery_status"]
    if not datasets:
        print("No battery_status topic was recorded in this ULog.")
        print("Recorded battery-related topics:")
        for dataset in ulog.data_list:
            if "battery" in dataset.name.lower():
                print(f"  {dataset.name}[{dataset.multi_id}]")
        return 1

    for dataset in datasets:
        print(f"battery_status[{dataset.multi_id}]: {len(dataset.data.get('timestamp', []))} samples")
        available = [field for field in FIELDS if field in dataset.data]
        print("fields=" + ", ".join(available))
        timestamps = dataset.data.get("timestamp", [])
        count = len(timestamps)
        if count:
            first_us = int(timestamps[0])
            last_us = int(timestamps[-1])
            print(f"recorded_span_s={(last_us - first_us) / 1e6:.3f}")
            if count > 1:
                gaps_s = [
                    (int(timestamps[i]) - int(timestamps[i - 1])) / 1e6
                    for i in range(1, count)
                ]
                print(f"max_sample_gap_s={max(gaps_s):.3f}")
        for field in ("connected", "is_required", "warning", "faults"):
            if field in dataset.data:
                values = sorted({int(value) for value in dataset.data[field]})
                print(f"observed_{field}_values={values}")
        for index in range(max(0, count - 10), count):
            row = []
            for field in available:
                value = dataset.data[field][index]
                if field == "timestamp":
                    value = f"{int(value)} us"
                elif field == "voltage_v":
                    value = f"{float(value):.3f} V"
                elif field == "current_a":
                    value = "nan" if math.isnan(float(value)) else f"{float(value):.3f} A"
                elif field == "remaining":
                    value = f"{float(value) * 100:.1f}%"
                row.append(f"{field}={value}")
            print("  " + " | ".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
