"""Manually label Scout RGB images with YOLO Fire (0) and Smoke (1) boxes.

Left-drag draws a box. Press 1 to select Fire, 2 to select Smoke, S to save,
U to undo, C to clear the current image, N/P to navigate, and Q/Esc to quit.
Labels come from visual review of the supplied image pixels, never simulator
world state or model poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


CLASS_NAMES = {0: "fire", 1: "smoke"}
CLASS_COLORS = {0: (0, 100, 255), 1: (210, 210, 210)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Folder containing JPG / PNG images")
    parser.add_argument("--labels", type=Path, required=True, help="Output YOLO labels folder")
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image folder does not exist: {args.images}")
    images = sorted(
        p for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        parser.error(f"No JPG/PNG images found in {args.images}")
    args.labels.mkdir(parents=True, exist_ok=True)

    boxes_by_image: dict[Path, list[tuple[int, int, int, int, int]]] = {}
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = args.labels / f"{image_path.stem}.txt"
        loaded: list[tuple[int, int, int, int, int]] = []
        if label_path.is_file():
            for row in label_path.read_text(encoding="utf-8").splitlines():
                fields = row.split()
                if len(fields) != 5 or fields[0] not in {"0", "1"}:
                    continue
                class_id = int(fields[0])
                cx, cy, bw, bh = map(float, fields[1:])
                x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
                x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
                loaded.append((class_id, x1, y1, x2, y2))
        boxes_by_image[image_path] = loaded

    index = 0
    selected_class = 0
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None
    scale = 1.0
    current_size = (0, 0)
    window = "Scout fire and smoke labeling"

    def mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal drag_start, drag_end
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (round(x / scale), round(y / scale))
            drag_end = drag_start
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_end = (round(x / scale), round(y / scale))
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            end = (round(x / scale), round(y / scale))
            x1, x2 = sorted((drag_start[0], end[0]))
            y1, y2 = sorted((drag_start[1], end[1]))
            width, height = current_size
            if x2 - x1 >= 3 and y2 - y1 >= 3:
                boxes_by_image.setdefault(images[index], []).append(
                    (selected_class, max(0, x1), max(0, y1),
                     min(width - 1, x2), min(height - 1, y2))
                )
            drag_start = drag_end = None

    def save(image_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not open image: {image_path}")
        height, width = image.shape[:2]
        rows = []
        for class_id, x1, y1, x2, y2 in boxes_by_image.get(image_path, []):
            cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            rows.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out = args.labels / f"{image_path.stem}.txt"
        out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"Saved {len(rows)} fire/smoke box(es): {out}")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            image_path = images[index]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not open image: {image_path}")
            height, width = image.shape[:2]
            current_size = (width, height)
            scale = min(1.0, 1200 / width, 800 / height)
            preview = cv2.resize(image, (round(width * scale), round(height * scale)))
            for class_id, x1, y1, x2, y2 in boxes_by_image.get(image_path, []):
                color = CLASS_COLORS[class_id]
                cv2.rectangle(preview, (round(x1 * scale), round(y1 * scale)),
                              (round(x2 * scale), round(y2 * scale)), color, 2)
                cv2.putText(preview, CLASS_NAMES[class_id],
                            (round(x1 * scale), max(14, round(y1 * scale) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if drag_start is not None and drag_end is not None:
                cv2.rectangle(preview,
                              (round(drag_start[0] * scale), round(drag_start[1] * scale)),
                              (round(drag_end[0] * scale), round(drag_end[1] * scale)),
                              CLASS_COLORS[selected_class], 2)
            help_text = (
                f"{index + 1}/{len(images)} {image_path.name} | "
                f"selected: {CLASS_NAMES[selected_class]} | 1 fire; 2 smoke; "
                "drag box; S save; U undo; C clear; N/P next/prev; Q quit"
            )
            cv2.putText(preview, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 0, 0), 3)
            cv2.putText(preview, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("1"):
                selected_class = 0
            elif key == ord("2"):
                selected_class = 1
            elif key == ord("s"):
                save(image_path)
            elif key == ord("u") and boxes_by_image.get(image_path):
                boxes_by_image[image_path].pop()
            elif key == ord("c"):
                boxes_by_image[image_path] = []
            elif key == ord("n"):
                index = (index + 1) % len(images)
            elif key == ord("p"):
                index = (index - 1) % len(images)
    finally:
        cv2.destroyAllWindows()
    print("Labeling closed. Save each image before moving on or quitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
