"""Record a human visual review after labeling every image in one session."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .validate_victim_dataset import IMAGE_SUFFIXES, validate_yolo_text
except ImportError:
    from validate_victim_dataset import IMAGE_SUFFIXES, validate_yolo_text


def confirm_session(session_dir: Path, reviewer: str) -> int:
    session_file = session_dir / "session.json"
    if not session_file.is_file():
        raise ValueError(f"missing session.json: {session_file}")
    session = json.loads(session_file.read_text(encoding="utf-8"))
    images_root, labels_root, metadata_root = (session_dir / key for key in ("images", "labels", "metadata"))
    images = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("session contains no images")
    problems = []
    metadata_rows = []
    for image in images:
        relative = image.relative_to(images_root)
        label = (labels_root / relative).with_suffix(".txt")
        metadata_path = (metadata_root / relative).with_suffix(".json")
        if not label.is_file():
            problems.append(f"missing annotation: {relative}")
            continue
        errors = validate_yolo_text(label.read_text(encoding="utf-8"), {0})
        problems.extend(f"{relative}: {error}" for error in errors)
        if not metadata_path.is_file():
            problems.append(f"missing metadata: {relative}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("session_id") != session.get("session_id"):
            problems.append(f"metadata session mismatch: {relative}")
        metadata_rows.append((metadata_path, metadata))
    if problems:
        raise ValueError("session is not ready for visual-review confirmation: " + "; ".join(problems[:20]))
    reviewed_at = datetime.now(timezone.utc).isoformat()
    for path, metadata in metadata_rows:
        metadata["annotation_status"] = "reviewed"
        metadata["annotation_review"] = {
            "reviewer": reviewer,
            "reviewed_utc": reviewed_at,
            "method": "manual_visual_review_of_original_image_and_yolo_boxes",
            "negative_label_means_reviewed_no_visible_person": True,
        }
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    session["label_status"] = "reviewed"
    session["annotation_review"] = {"reviewer": reviewer, "reviewed_utc": reviewed_at, "image_count": len(images)}
    session_file.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    print(f"marked {len(images)} images reviewed in {session_dir}; verify visually before running this command")
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--visual-review-complete", action="store_true", required=True,
                        help="assert each original frame and annotation was actually reviewed")
    args = parser.parse_args()
    try:
        confirm_session(args.session, args.reviewer)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
