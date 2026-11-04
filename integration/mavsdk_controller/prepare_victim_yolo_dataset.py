"""Build a small YOLO dataset with label-preserving appearance augmentations.

No new people or viewpoints are fabricated. Training variants are derived from
the labeled Scout images with mild brightness, contrast, gamma, color, and blur
changes, so the existing bounding boxes remain valid.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def write_sample(image: np.ndarray, image_path: Path, label_path: Path,
                 out_images: Path, out_labels: Path, stem: str) -> None:
    if not cv2.imwrite(str(out_images / f"{stem}.jpg"), image,
                       [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write image {stem}")
    shutil.copyfile(label_path, out_labels / f"{stem}.txt")


def variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    output = []
    output.append(("dim", cv2.convertScaleAbs(image, alpha=0.88, beta=-10)))
    output.append(("bright", cv2.convertScaleAbs(image, alpha=1.08, beta=10)))

    gamma = 0.88
    table = np.array([round(((i / 255.0) ** gamma) * 255) for i in range(256)],
                     dtype=np.uint8)
    output.append(("gamma", cv2.LUT(image, table)))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 2.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.88, 0, 255)
    output.append(("color", cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)))

    output.append(("soft_blur", cv2.GaussianBlur(image, (3, 3), 0.45)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Folder containing labeled JPG/PNG images")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Folder containing YOLO label files")
    parser.add_argument("--output", type=Path, required=True,
                        help="Prepared dataset output directory")
    parser.add_argument("--validation-image", default="capture_view2.jpg",
                        help="Keep this distinct capture unaugmented in validation")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only the specified output directory")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            parser.error(f"Output already exists: {args.output}; use --overwrite to rebuild")
        shutil.rmtree(args.output)

    images = sorted(p for p in args.source.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 2:
        parser.error("Need at least two labeled images to create train and validation sets")
    validation = [p for p in images if p.name == args.validation_image]
    if len(validation) != 1:
        parser.error(f"Validation image not found exactly once: {args.validation_image}")

    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            parser.error(f"Missing labels for {image_path.name}: {label_path}")
        for row_number, row in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = row.split()
            if len(fields) != 5 or fields[0] != "0":
                parser.error(f"Invalid YOLO person label: {label_path}:{row_number}")
            try:
                values = [float(value) for value in fields[1:]]
            except ValueError:
                parser.error(f"Non-numeric YOLO label: {label_path}:{row_number}")
            if any(value < 0 or value > 1 for value in values):
                parser.error(f"Out-of-range YOLO label: {label_path}:{row_number}")

    train_images = args.output / "images/train"
    train_labels = args.output / "labels/train"
    val_images = args.output / "images/val"
    val_labels = args.output / "labels/val"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True, exist_ok=True)

    train_count = 0
    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            parser.error(f"Could not read image: {image_path}")
        if image_path in validation:
            write_sample(image, image_path, label_path, val_images, val_labels,
                         image_path.stem)
            continue
        write_sample(image, image_path, label_path, train_images, train_labels,
                     image_path.stem)
        train_count += 1
        for suffix, augmented in variants(image):
            write_sample(augmented, image_path, label_path, train_images, train_labels,
                         f"{image_path.stem}_{suffix}")
            train_count += 1

    yaml = args.output / "data.yaml"
    dataset_root = Path.cwd().resolve()
    relative_output = args.output.resolve().relative_to(dataset_root)
    yaml.write_text(
        f"path: {relative_output.as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    print(f"Prepared {train_count} training images and 1 held-out validation image.")
    print(f"Dataset YAML: {yaml}")
    print("Validation uses one capture only; its score is a smoke check, not a reliable evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
