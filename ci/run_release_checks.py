"""Provider-agnostic release gate for the RELAY environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PYTHON = (3, 11, 9)
RUNTIME_TMP = ROOT / "tmp" / "release-check-runtime"
PYTEST_CACHE = ROOT / "tmp" / "release-check-cache"


def run(arguments: list[str]) -> None:
    printable = " ".join(arguments)
    print(f"\n> {printable}", flush=True)
    subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        env=os.environ
        | {
            "QT_QPA_PLATFORM": "offscreen",
            "TEMP": str(RUNTIME_TMP),
            "TMP": str(RUNTIME_TMP),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="skip exhaustive validation of the two 10,000-scenario manifests",
    )
    args = parser.parse_args()
    if sys.version_info[:3] != REQUIRED_PYTHON:
        actual = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(f"Python 3.11.9 is required; running {actual}")

    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    PYTEST_CACHE.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "-o",
            f"cache_dir={PYTEST_CACHE}",
        ]
    )
    run([python, "-m", "ruff", "check", "src", "tests", "ci"])
    run([python, "-m", "mypy", "src"])
    if not args.skip_manifests:
        for manifest in (
            "configs/manifests/relay-grid-v1.json",
            "configs/manifests/relay-grid-v1-p3.json",
        ):
            run([python, "-m", "relay.scenarios", "validate", manifest, "--workers", "4"])
    print("\nRELAY release gate: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
