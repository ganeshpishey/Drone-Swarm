"""RELAY recurrent MAPPO training command."""

from __future__ import annotations

import argparse
import json
import sys

from relay.learning.trainer import train
from relay.runtime.configuration import config_yaml, override_diff, resolve_config
from relay.runtime.logging import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="experiment YAML path")
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate only")
    parser.add_argument("--resume", help="exact trainer checkpoint to resume")
    parser.add_argument(
        "--init-checkpoint",
        help="initialize model weights from a checkpoint, but start a fresh training run",
    )
    parser.add_argument("overrides", nargs="*", help="typed key=value overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint cannot be used together")
    resolved = resolve_config(args.config, args.overrides)
    print(config_yaml(resolved))
    print(f"config_checksum: {resolved.checksum}")
    if args.overrides:
        print("overrides:")
        print(json.dumps(override_diff(resolved), indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    command = " ".join(sys.argv if argv is None else ["relay-train", *argv])
    run = (
        RunContext.resume(args.resume, resolved, command=command)
        if args.resume
        else RunContext.create(resolved, command=command)
    )
    checkpoint = train(
        resolved,
        run,
        resume_checkpoint=args.resume,
        init_checkpoint=args.init_checkpoint,
    )
    print(f"run_id: {run.run_id}")
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
