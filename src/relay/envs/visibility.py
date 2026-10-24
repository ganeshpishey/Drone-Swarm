"""North-up local sensing with deterministic grid line-of-sight occlusion."""

from __future__ import annotations

import numpy as np


def supercover_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return grid cells touched by the segment between two cell centers."""

    dx, dy = x1 - x0, y1 - y0
    nx, ny = abs(dx), abs(dy)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    x, y = x0, y0
    ix = iy = 0
    cells = [(x, y)]
    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif decision < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        cells.append((x, y))
    return cells


def is_visible(walls: np.ndarray, origin: tuple[int, int], target: tuple[int, int]) -> bool:
    """A first blocking wall is visible, but cells behind a wall are not."""

    ox, oy = origin
    tx, ty = target
    height, width = walls.shape
    if not (0 <= tx < width and 0 <= ty < height):
        return False
    line = supercover_line(ox, oy, tx, ty)
    return not any(bool(walls[y, x]) for x, y in line[1:-1])


def visibility_mask(
    walls: np.ndarray,
    origin: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Return a full-grid mask for a Chebyshev-radius square sensor."""

    height, width = walls.shape
    ox, oy = origin
    mask = np.zeros((height, width), dtype=np.bool_)
    for y in range(max(0, oy - radius), min(height, oy + radius + 1)):
        for x in range(max(0, ox - radius), min(width, ox + radius + 1)):
            if is_visible(walls, origin, (x, y)):
                mask[y, x] = True
    return mask
