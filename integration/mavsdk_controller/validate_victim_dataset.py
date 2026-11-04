"""Validate victim image/YOLO annotation pairs and capture provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Any) -> int:
    gray = cv2.cvtColor(cv2.resize(image, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def validate_yolo_text(text: str, allowed_classes: set[int]) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {line_number}: nonnumeric field")
            continue
        if not math.isfinite(class_value) or class_value != int(class_value):
            errors.append(f"line {line_number}: invalid class id")
            continue
        class_id = int(class_value)
        if class_id not in allowed_classes:
            errors.append(f"line {line_number}: class {class_id} is not allowed")
        if any(not math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: nonfinite coordinate")
            continue
        cx, cy, width, height = values
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width and height must be positive")
        if not all(0 <= value <= 1 for value in values):
            errors.append(f"line {line_number}: normalized values must be in [0, 1]")
        if cx - width / 2 < 0 or cy - height / 2 < 0 or cx + width / 2 > 1 or cy + height / 2 > 1:
            errors.append(f"line {line_number}: box corners exceed image boundaries")
    return errors


def audit_dataset(images_root: Path, labels_root: Path, metadata_root: Path,
                  allowed_classes: set[int], near_duplicate_threshold: int = 4) -> dict[str, Any]:
    images = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    exact: dict[str, list[str]] = defaultdict(list)
    hashes: list[tuple[Path, int]] = []
    classes: Counter[int] = Counter()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    rows = []
    empty_label_count = 0
    dimension_counts: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    session_dataset_types: Counter[str] = Counter()

    for image_path in images:
        relative = image_path.relative_to(images_root)
        label_path = (labels_root / relative).with_suffix(".txt")
        metadata_path = (metadata_root / relative).with_suffix(".json")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"image": str(relative), "error": "image_corrupt_or_unsupported"})
            continue
        height, width = image.shape[:2]
        dimension_counts[f"{width}x{height}"] += 1
        digest = sha256_file(image_path)
        exact[digest].append(str(relative))
        hashes.append((relative, dhash(image)))
        if not label_path.is_file():
            errors.append({"image": str(relative), "error": "missing_annotation_file"})
            label_text = None
        else:
            label_text = label_path.read_text(encoding="utf-8")
            box_errors = validate_yolo_text(label_text, allowed_classes)
            for error in box_errors:
                errors.append({"image": str(relative), "error": error})
            if not label_text.strip():
                empty_label_count += 1
            else:
                for line in label_text.splitlines():
                    fields = line.split()
                    if len(fields) == 5:
                        try:
                            c = int(fields[0])
                            classes[c] += 1
                        except ValueError:
                            pass
        metadata = None
        if not metadata_path.is_file():
            warnings.append({"image": str(relative), "warning": "missing_capture_metadata"})
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                errors.append({"image": str(relative), "error": f"invalid_metadata_json:{error}"})
            if isinstance(metadata, dict):
                session_id = metadata.get("session_id")
                if session_id:
                    sessions[str(session_id)] += 1
                else:
                    warnings.append({"image": str(relative), "warning": "metadata_missing_session_id"})
                dataset_type = metadata.get("dataset_type")
                if dataset_type:
                    session_dataset_types[str(dataset_type)] += 1
                else:
                    warnings.append({"image": str(relative), "warning": "metadata_missing_dataset_type"})
                recorded_hash = metadata.get("image_sha256")
                if recorded_hash and recorded_hash != digest:
                    errors.append({"image": str(relative), "error": "metadata_image_hash_mismatch"})
                if metadata.get("annotation_status") != "reviewed":
                    warnings.append({"image": str(relative), "warning": "annotation_not_marked_reviewed"})
                elif not isinstance(metadata.get("annotation_review"), dict) or not metadata["annotation_review"].get("reviewer"):
                    warnings.append({"image": str(relative), "warning": "review_provenance_incomplete"})
                if metadata.get("dataset_type") == "natural_person":
                    for field in ("source_url", "license", "attribution"):
                        if not metadata.get(field):
                            errors.append({"image": str(relative), "error": f"natural_person_metadata_missing_{field}"})
        rows.append({"image": str(relative), "width": width, "height": height, "sha256": digest,
                     "label_file": str(label_path.relative_to(labels_root)) if label_path.is_file() else None,
                     "annotation_count": 0 if label_text is None or not label_text.strip() else sum(1 for x in label_text.splitlines() if x.strip()),
                     "metadata_file": str(metadata_path.relative_to(metadata_root)) if metadata_path.is_file() else None,
                     "session_id": metadata.get("session_id") if isinstance(metadata, dict) else None,
                     "dataset_type": metadata.get("dataset_type") if isinstance(metadata, dict) else None})

    exact_groups = [paths for paths in exact.values() if len(paths) > 1]
    near_groups = []
    if len(hashes) <= 5000:
        for index, (path_a, hash_a) in enumerate(hashes):
            for path_b, hash_b in hashes[index + 1:]:
                distance = (hash_a ^ hash_b).bit_count()
                if distance <= near_duplicate_threshold:
                    near_groups.append({"image_a": str(path_a), "image_b": str(path_b), "dhash_distance": distance})
    else:
        warnings.append({"image": "*", "warning": "near_duplicate_scan_skipped_over_5000_images"})

    return {
        "schema_version": 1,
        "images_root": str(images_root), "labels_root": str(labels_root), "metadata_root": str(metadata_root),
        "image_count": len(images), "dimensions": dict(dimension_counts), "class_distribution": dict(classes),
        "empty_label_image_count": empty_label_count, "session_count": len(sessions),
        "images_per_session": dict(sessions), "dataset_types": dict(session_dataset_types),
        "exact_duplicate_groups": exact_groups, "near_duplicate_pairs": near_groups,
        "errors": errors, "warnings": warnings, "ok": not errors,
        "images": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allowed-class", type=int, action="append", default=None)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f"images directory does not exist: {args.images}")
    report = audit_dataset(args.images, args.labels, args.metadata,
                           set(args.allowed_class or [0]), args.near_duplicate_threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"images={report['image_count']} sessions={report['session_count']} classes={report['class_distribution']}")
    print(f"exact_duplicate_groups={len(report['exact_duplicate_groups'])} near_duplicate_pairs={len(report['near_duplicate_pairs'])}")
    print(f"errors={len(report['errors'])} warnings={len(report['warnings'])} report={args.report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
