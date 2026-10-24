"""Atomic run lifecycle and structured Parquet event logging."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from relay.envs.hashing import canonical_json
from relay.runtime.configuration import ResolvedConfig, config_yaml, override_diff

SCHEMA_VERSION = "relay-events-v1"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("relay-marl", "numpy", "torch", "gymnasium", "pettingzoo", "pyarrow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


@dataclass(slots=True)
class RunContext:
    run_id: str
    path: Path
    manifest: dict[str, Any]

    @classmethod
    def create(
        cls,
        resolved: ResolvedConfig,
        *,
        command: str,
        cwd: Path | None = None,
    ) -> RunContext:
        working_directory = (cwd or Path.cwd()).resolve()
        timestamp = datetime.now(UTC)
        slug = re.sub(r"[^a-z0-9-]+", "-", resolved.config.experiment_name.lower()).strip("-")
        run_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{slug}-{resolved.checksum[:8]}"
        runs_dir = Path(resolved.config.logging.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = working_directory / runs_dir
        run_path = runs_dir / run_id
        if run_path.exists():
            suffix = 1
            while (runs_dir / f"{run_id}-{suffix:02d}").exists():
                suffix += 1
            run_id = f"{run_id}-{suffix:02d}"
            run_path = runs_dir / run_id
        for directory in ("checkpoints", "metrics", "events", "replays", "errors"):
            (run_path / directory).mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": "relay-run-v1",
            "run_id": run_id,
            "status": "running",
            "command": command,
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
            "config_source": str(resolved.source),
            "config_checksum": resolved.checksum,
            "environment_version": resolved.config.environment.environment_version,
            "preset": resolved.config.environment.preset,
            "communication_mode": resolved.config.environment.communication.mode,
            "git": _git_metadata(working_directory),
            "versions": _versions(),
        }
        atomic_json(run_path / "manifest.json", manifest)
        atomic_text(run_path / "resolved_config.yaml", config_yaml(resolved))
        atomic_json(run_path / "config_diff.json", override_diff(resolved))
        return cls(run_id, run_path, manifest)

    def update(self, **changes: Any) -> None:
        self.manifest.update(changes)
        self.manifest["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(self.path / "manifest.json", self.manifest)

    @classmethod
    def resume(
        cls,
        checkpoint: str | Path,
        resolved: ResolvedConfig,
        *,
        command: str,
    ) -> RunContext:
        checkpoint_path = Path(checkpoint).resolve()
        run_path = checkpoint_path.parent.parent
        manifest_path = run_path / "manifest.json"
        if not checkpoint_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("resume checkpoint or parent run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resumes = list(manifest.get("resumes", []))
        timestamp = datetime.now(UTC)
        resumes.append(
            {
                "timestamp": timestamp.isoformat(),
                "checkpoint": str(checkpoint_path),
                "command": command,
                "config_checksum": resolved.checksum,
            }
        )
        context = cls(str(manifest["run_id"]), run_path, manifest)
        suffix = timestamp.strftime("%Y%m%dT%H%M%SZ")
        atomic_text(run_path / f"resume_config_{suffix}.yaml", config_yaml(resolved))
        context.update(
            status="running",
            command=command,
            active_config_checksum=resolved.checksum,
            resumes=resumes,
        )
        return context

    def complete(self, **summary: Any) -> None:
        self.update(status="complete", completed_at=datetime.now(UTC).isoformat(), **summary)

    def fail(self, error: BaseException) -> None:
        record = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        atomic_json(self.path / "errors" / "failure.json", record)
        self.update(status="failed", failure=record)


class EventWriter:
    """Bounded, synchronous Parquet event writer; writer failures are fatal."""

    _schema = pa.schema(
        [
            ("schema_version", pa.string()),
            ("phase", pa.string()),
            ("vector_env_index", pa.int32()),
            ("episode", pa.int32()),
            ("tick", pa.int32()),
            ("event_family", pa.string()),
            ("payload_json", pa.large_string()),
        ]
    )

    def __init__(self, path: Path, *, buffer_size: int = 1024) -> None:
        self.path = path
        self.buffer_size = buffer_size
        self._rows: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event_family: str,
        payload: Any,
        *,
        phase: str,
        vector_env_index: int,
        episode: int,
        tick: int,
    ) -> None:
        self._rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "vector_env_index": vector_env_index,
                "episode": episode,
                "tick": tick,
                "event_family": event_family,
                "payload_json": canonical_json(payload).decode("utf-8"),
            }
        )
        if len(self._rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="zstd")
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class PartitionedEventWriter:
    """Write Hive-style event partitions by phase, vector index, and episode."""

    def __init__(self, root: Path, *, buffer_size: int = 1024) -> None:
        self.root = root
        self.buffer_size = buffer_size
        self._writers: dict[tuple[str, int, int], EventWriter] = {}

    def _writer(self, phase: str, vector_env_index: int, episode: int) -> EventWriter:
        key = (phase, vector_env_index, episode)
        if key not in self._writers:
            path = (
                self.root
                / f"phase={phase}"
                / f"vector_env_index={vector_env_index}"
                / f"episode={episode}"
                / f"part-{uuid.uuid4().hex}.parquet"
            )
            self._writers[key] = EventWriter(path, buffer_size=self.buffer_size)
        return self._writers[key]

    def record(
        self,
        event_family: str,
        payload: Any,
        *,
        phase: str,
        vector_env_index: int,
        episode: int,
        tick: int,
    ) -> None:
        self._writer(phase, vector_env_index, episode).record(
            event_family,
            payload,
            phase=phase,
            vector_env_index=vector_env_index,
            episode=episode,
            tick=tick,
        )

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
