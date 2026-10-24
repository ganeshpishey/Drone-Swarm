"""Generate/validate frozen seed manifests and calibration datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from relay.envs.environment import RelayParallelEnv
from relay.runtime.manifests import generate_manifest, load_manifest, validate_manifest
from relay.runtime.policies import RandomPolicy, ScriptedOraclePolicy


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    seeds = [int(seed) for seed in manifest["splits"][args.split]][: args.episodes]
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for policy_name in ("random", "oracle"):
        episode_records = []
        for episode, seed in enumerate(seeds):
            env = RelayParallelEnv(preset=str(manifest["preset"]), communication_mode="nocomm")
            observations, infos = env.reset(seed=seed)
            policy = (
                RandomPolicy(seed ^ 0x5EED) if policy_name == "random" else ScriptedOraclePolicy()
            )
            total_reward = 0.0
            tick = 0
            while env.agents:
                actions = policy.act(env, observations)
                observations, rewards, terminations, truncations, step_infos = env.step(actions)
                tick += 1
                reward = float(rewards[env.possible_agents[0]])
                total_reward += reward
                info = step_infos[env.possible_agents[0]]
                rows.append(
                    {
                        "schema_version": "relay-calibration-v1",
                        "policy": policy_name,
                        "episode": episode,
                        "seed": seed,
                        "scenario_id": infos[env.possible_agents[0]]["scenario_id"],
                        "tick": tick,
                        "reward": reward,
                        "recipient_attempts": int(info["recipient_attempts"]),
                        "state_hash": info["state_hash"],
                        "terminal_reason": info["terminal_reason"],
                    }
                )
            success = step_infos[env.possible_agents[0]]["terminal_reason"] == "completed"
            episode_records.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "return": total_reward,
                    "length": tick,
                    "success": success,
                }
            )
        summaries[policy_name] = {
            "episodes": len(episode_records),
            "success_rate": float(np.mean([item["success"] for item in episode_records])),
            "mean_return": float(np.mean([item["return"] for item in episode_records])),
            "mean_length": float(np.mean([item["length"] for item in episode_records])),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return {"dataset": str(output), "summary": str(summary_path), "policies": summaries}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--output", default="configs/manifests/relay-grid-v1.json")
    generate.add_argument("--preset", default="pilot_core")
    generate.add_argument("--master-seed", type=int, default=20260902)
    generate.add_argument("--workers", type=int, default=1)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("--workers", type=int, default=1)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("manifest")
    calibrate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    calibrate.add_argument("--episodes", type=int, default=100)
    calibrate.add_argument("--output", default="output/calibration/relay-grid-v1.parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_manifest(
            args.output,
            preset=args.preset,
            master_seed=args.master_seed,
            workers=args.workers,
        )
        output = {
            "output": args.output,
            "counts": result["counts"],
            "checksum": result["manifest_checksum"],
        }
    elif args.command == "validate":
        output = validate_manifest(args.manifest, workers=args.workers)
    else:
        output = _calibrate(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
