"""Draw YOLO person boxes on saved Scout frames using an OpenCV window.

Left-drag to draw a box, U to undo, S to save the current image, N/P to move
between images, and Q/Esc to quit. Boxes are written as YOLO class 0 (person).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Folder containing JPG / PNG frames")
    parser.add_argument(
        "--labels", type=Path, required=True,
        help="Output labels folder; each image gets a same-stem .txt file",
    )
    args = parser.parse_args()

    images = sorted(
        p for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        parser.error(f"No JPG/PNG images found in {args.images}")

    args.labels.mkdir(parents=True, exist_ok=True)
    index = 0
    boxes_by_image: dict[Path, list[tuple[int, int, int, int]]] = {}
    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        loaded = []
        for row in label_path.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if len(fields) != 5 or fields[0] != "0":
                continue
            cx, cy, bw, bh = map(float, fields[1:])
            x1 = round((cx - bw / 2) * width)
            y1 = round((cy - bh / 2) * height)
            x2 = round((cx + bw / 2) * width)
            y2 = round((cy + bh / 2) * height)
            loaded.append((x1, y1, x2, y2))
        boxes_by_image[image_path] = loaded
    drag_start: tuple[int, int] | None = None
    drag_end: tuple[int, int] | None = None
    scale = 1.0
    current_size = (0, 0)
    window = "Scout victim labeling (YOLO)"

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
                    (max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))
                )
            drag_start = drag_end = None

    def save(image_path: Path) -> None:
        width, height = cv2.imread(str(image_path)).shape[1::-1]
        out = args.labels / f"{image_path.stem}.txt"
        rows = []
        for x1, y1, x2, y2 in boxes_by_image.get(image_path, []):
            cx = (x1 + x2) / (2 * width)
            cy = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"Saved {len(rows)} person box(es): {out}")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    while True:
        image_path = images[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not open image: {image_path}")
        height, width = image.shape[:2]
        current_size = (width, height)
        scale = min(1.0, 1200 / width, 800 / height)
        preview = cv2.resize(image, (round(width * scale), round(height * scale)))
        for x1, y1, x2, y2 in boxes_by_image.get(image_path, []):
            cv2.rectangle(preview, (round(x1 * scale), round(y1 * scale)),
                          (round(x2 * scale), round(y2 * scale)), (0, 255, 0), 2)
        if drag_start is not None and drag_end is not None:
            cv2.rectangle(preview,
                          (round(drag_start[0] * scale), round(drag_start[1] * scale)),
                          (round(drag_end[0] * scale), round(drag_end[1] * scale)),
                          (0, 255, 255), 2)
        cv2.putText(preview, f"{index + 1}/{len(images)} {image_path.name} | boxes: "
                    f"{len(boxes_by_image.get(image_path, []))} | drag victim; S save; "
                    "U undo; C clear; N/P next/prev; Q quit",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(preview, f"{index + 1}/{len(images)} {image_path.name} | boxes: "
                    f"{len(boxes_by_image.get(image_path, []))} | drag victim; S save; "
                    "U undo; C clear; N/P next/prev; Q quit",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(window, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save(image_path)
        elif key == ord("u"):
            if boxes_by_image.get(image_path):
                boxes_by_image[image_path].pop()
        elif key == ord("c"):
            boxes_by_image[image_path] = []
        elif key == ord("n"):
            index = (index + 1) % len(images)
        elif key == ord("p"):
            index = (index - 1) % len(images)
    cv2.destroyAllWindows()
    print("Labeling closed. Press S on each image whose labels should be kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
