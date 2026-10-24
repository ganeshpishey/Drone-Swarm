"""Inspect or verify immutable RELAY replay artifacts."""

from __future__ import annotations

import argparse
import json

from relay.runtime.replay_store import replay_summary, verify_replay
from relay.runtime.replay_visuals import ReplayData, export_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("summary", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("replay")
    view = subparsers.add_parser("view")
    view.add_argument("replay")
    export = subparsers.add_parser("export")
    export.add_argument("replay")
    export.add_argument("--output", required=True)
    export.add_argument("--start", type=int, default=0)
    export.add_argument("--end", type=int)
    export.add_argument("--fps", type=float, default=4.0)
    export.add_argument("--stride", type=int, default=1)
    export.add_argument("--perspective", default="global")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--episodes", type=int, default=1)
    generate.add_argument("overrides", nargs="*")
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--nocomm-config", default="configs/experiments/v1_nocomm.yaml")
    baseline.add_argument("--comm-config", default="configs/experiments/v1_always_comm.yaml")
    baseline.add_argument("--episodes", type=int, default=1)
    baseline.add_argument("--split", choices=("train", "validation", "test"), default="test")
    baseline.add_argument("--start-index", type=int, default=0)
    baseline.add_argument("--runs-dir", default="output/baselines")
    baseline.add_argument("--no-export", action="store_true")
    baseline.add_argument("--fps", type=float, default=8.0)
    baseline.add_argument("--export-stride", type=int, default=4)
    baseline.add_argument("--view", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        from relay.baseline import main as baseline_main

        baseline_arguments = [
            "--nocomm-config",
            args.nocomm_config,
            "--comm-config",
            args.comm_config,
            "--episodes",
            str(args.episodes),
            "--split",
            args.split,
            "--start-index",
            str(args.start_index),
            "--runs-dir",
            args.runs_dir,
            "--fps",
            str(args.fps),
            "--export-stride",
            str(args.export_stride),
        ]
        if args.no_export:
            baseline_arguments.append("--no-export")
        if args.view:
            baseline_arguments.append("--view")
        return baseline_main(baseline_arguments)
    if args.command == "view":
        from relay.viewer import launch_viewer

        return launch_viewer(args.replay)
    if args.command == "generate":
        from relay.evaluate import main as evaluate_main

        return evaluate_main(
            [
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                f"evaluation.episodes={args.episodes}",
                *args.overrides,
            ]
        )
    if args.command == "export":
        output = export_replay(
            ReplayData(args.replay),
            args.output,
            start=args.start,
            end=args.end,
            fps=args.fps,
            perspective=args.perspective,
            stride=args.stride,
        )
        result = {"output": str(output)}
    else:
        result = (
            verify_replay(args.replay) if args.command == "verify" else replay_summary(args.replay)
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
