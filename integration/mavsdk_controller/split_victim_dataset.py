"""Create immutable, capture-session-separated train/val/test copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

try:
    from .validate_victim_dataset import IMAGE_SUFFIXES, dhash
except ImportError:
    from validate_victim_dataset import IMAGE_SUFFIXES, dhash


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_records(root: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    types = set()
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            continue
        session = json.loads(session_file.read_text(encoding="utf-8"))
        session_id = session.get("session_id")
        if not session_id:
            raise ValueError(f"session_id missing: {session_file}")
        dataset_type = session.get("dataset_type")
        if dataset_type not in {"gazebo_synthetic", "natural_person"}:
            raise ValueError(f"dataset_type must be gazebo_synthetic or natural_person in {session_file}")
        types.add(dataset_type)
        image_root = session_dir / "images"
        label_root = session_dir / "labels"
        metadata_root = session_dir / "metadata"
        images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"no images in {image_root}")
        digests = set()
        image_hashes = []
        for image in images:
            rel = image.relative_to(image_root)
            label = (label_root / rel).with_suffix(".txt")
            metadata_path = (metadata_root / rel).with_suffix(".json")
            if not label.is_file():
                raise ValueError(f"annotation missing for {session_id}/{rel}")
            if not metadata_path.is_file():
                raise ValueError(f"capture metadata missing for {session_id}/{rel}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("session_id") != session_id or metadata.get("dataset_type") != dataset_type:
                raise ValueError(f"metadata provenance mismatch for {session_id}/{rel}")
            if metadata.get("annotation_status") != "reviewed":
                raise ValueError(f"annotation not visually reviewed for {session_id}/{rel}")
            review = metadata.get("annotation_review")
            if not isinstance(review, dict) or not review.get("reviewer"):
                raise ValueError(f"annotation review provenance missing for {session_id}/{rel}")
            if dataset_type == "natural_person" and not all(metadata.get(field) for field in ("source_url", "license", "attribution")):
                raise ValueError(f"natural-person source/license/attribution missing for {session_id}/{rel}")
            actual_hash = _sha(image)
            if metadata.get("image_sha256") and metadata["image_sha256"] != actual_hash:
                raise ValueError(f"image hash mismatch for {session_id}/{rel}")
            digests.add(actual_hash)
            loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if loaded is None:
                raise ValueError(f"corrupt image {image}")
            image_hashes.append((actual_hash, dhash(loaded)))
        scene_id = session.get("capture_config", {}).get("scene_id")
        if not scene_id:
            raise ValueError(f"scene_id missing in capture_config: {session_file}")
        records.append({"id": session_id, "scene_id": str(scene_id), "dir": session_dir, "type": dataset_type,
                        "images": images, "hashes": image_hashes, "exact_hashes": digests})
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate session_id values found")
    if len(types) != 1:
        raise ValueError(f"split one dataset type at a time; found {sorted(types)}")
    if len(records) < 3:
        raise ValueError(f"need at least three complete reviewed sessions, found {len(records)}")
    return records, next(iter(types))


def _session_groups(records: list[dict[str, Any]], near_threshold: int) -> list[list[dict[str, Any]]]:
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owners: dict[str, int] = {}
    scene_owners: dict[str, int] = {}
    for i, record in enumerate(records):
        if record["scene_id"] in scene_owners:
            union(i, scene_owners[record["scene_id"]])
        else:
            scene_owners[record["scene_id"]] = i
        for digest in record["exact_hashes"]:
            if digest in owners:
                union(i, owners[digest])
            owners[digest] = i
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            if any((hash_a ^ hash_b).bit_count() <= near_threshold
                   for _, hash_a in left["hashes"] for _, hash_b in right["hashes"]):
                union(i, j)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, record in enumerate(records):
        grouped[find(i)].append(record)
    return list(grouped.values())


def split_sessions(collection_root: Path, output: Path, ratios: tuple[float, float, float], seed: int,
                   near_threshold: int = 3) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) <= 0:
        raise ValueError("three positive split ratios must sum to 1")
    records, dataset_type = _session_records(collection_root)
    groups = _session_groups(records, near_threshold)
    if len(groups) < 3:
        raise ValueError(f"only {len(groups)} independent groups remain after exact/near-duplicate grouping")

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    names = ["train", "val", "test"]
    raw = [len(groups) * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[:len(groups) - sum(counts)]:
        counts[index] += 1
    if min(counts) < 1:
        raise ValueError("split ratios do not allocate at least one independent group to every split")
    assignment: dict[str, str] = {}
    cursor = 0
    for split_name, count in zip(names, counts):
        for group_index in order[cursor:cursor + count]:
            for record in groups[group_index]:
                assignment[record["id"]] = split_name
        cursor += count

    output.mkdir(parents=True, exist_ok=False)
    session_rows = []
    for record in records:
        split_name = assignment[record["id"]]
        target = output / split_name / record["id"]
        shutil.copytree(record["dir"], target)
        session_rows.append({"session_id": record["id"], "dataset_type": dataset_type,
                             "scene_id": record["scene_id"], "split": split_name,
                             "image_count": len(record["images"])})
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": dataset_type,
        "seed": seed,
        "requested_ratios": dict(zip(names, ratios)),
        "session_group_count_after_scene_and_duplicate_merge": len(groups),
        "near_duplicate_dhash_threshold": near_threshold,
        "split_counts": {name: sum(row["split"] == name for row in session_rows) for name in names},
        "sessions": session_rows,
        "image_level_random_split": False,
        "augmentation_performed": False,
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--near-duplicate-threshold", type=int, default=3)
    args = parser.parse_args()
    try:
        manifest = split_sessions(args.collection_root, args.output, tuple(args.ratios), args.seed,
                                  args.near_duplicate_threshold)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"created {manifest['dataset_type']} session split at {args.output}: {manifest['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
