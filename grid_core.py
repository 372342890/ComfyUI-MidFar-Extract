"""Deterministic asset-sheet cutting. No model calls, redraws or resizing."""
from collections import deque
import math

import numpy as np


def _runs(mask):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def _rgb(image):
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected H×W×3 RGB or H×W×4 RGBA image")
    result = image[..., :3].astype(np.float32)
    if image.shape[2] == 4:
        alpha = image[..., 3:4].astype(np.float32) / 255
        result = result * alpha + 255 * (1 - alpha)
    return result


def _background(rgb):
    # Most of an asset sheet is background; a channel median is robust to
    # grid lines, corner assets and small shadows.
    return np.median(rgb[::3, ::3].reshape(-1, 3), axis=0)


def _axis_lines(rgb, background, axis):
    spread = np.ptp(rgb, axis=2)
    darkening = np.mean(background - rgb, axis=2)
    gray_line = (spread < 10) & (darkening > 5) & (darkening < 100)
    coverage = gray_line.mean(axis=axis)
    size = coverage.size
    bands = _runs(coverage >= 0.65)
    centers = [
        (lo + hi) // 2 for lo, hi in bands
        if hi - lo <= max(8, size * 0.02)
        and lo >= size * 0.06 and hi <= size * 0.94
    ]
    if not centers:
        return None
    bounds = [0, *centers, size]
    if len(centers) > 7 or min(np.diff(bounds)) < size * 0.055:
        raise ValueError("Ambiguous grid lines; set rows/columns explicitly")
    return bounds


def _axis_gaps(rgb, background, axis, tolerance):
    foreground = np.max(np.abs(rgb - background), axis=2) > tolerance
    profile = foreground.mean(axis=axis)
    size = profile.size
    if not np.any(profile):
        return None
    # A broad, fully empty strip across all occupied rows is a grid gutter.
    # Require a substantial width; a thin hole inside one object is not a cut.
    bands = _runs(profile <= 0.002)
    inner = [
        (lo, hi) for lo, hi in bands
        if lo > size * 0.05 and hi < size * 0.95
        and hi - lo >= size * 0.018
    ]
    if not inner:
        return None
    bounds = [0, *[(a + b) // 2 for a, b in inner], size]
    widths = np.diff(bounds)
    if len(inner) > 4 or min(widths) < size * 0.12:
        return None
    if max(widths) / min(widths) > 1.7:
        return None
    return bounds


def grid_for_count(count):
    if not 1 <= count <= 20:
        raise ValueError("asset_count must be between 1 and 20")
    for limit, shape in [
        (1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)),
        (6, (2, 3)), (8, (2, 4)), (9, (3, 3)), (12, (3, 4)),
        (15, (3, 5)), (20, (4, 5)),
    ]:
        if count <= limit:
            return shape


def _component_at_least(mask, minimum):
    """Stop as soon as a real connected region is found."""
    if int(mask.sum()) < minimum:
        return False
    try:
        import cv2
    except ImportError:
        # Optional fast path. NumPy-only flood fill works in standard ComfyUI.
        seen = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            queue = deque([(int(y), int(x))])
            seen[y, x] = True
            area = 0
            while queue:
                py, px = queue.popleft()
                area += 1
                if area >= minimum:
                    return True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ny, nx = py + dy, px + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return bool(np.any(stats[1:, cv2.CC_STAT_AREA] >= minimum))


def split_sheet(image, rows=0, columns=0, asset_count=0,
                background_tolerance=24, min_content_ratio=0.0003,
                border_ignore_ratio=0.025, trim_grid_lines=True):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("Core image must use uint8 pixels")
    if not (0 <= rows <= 20 and 0 <= columns <= 20):
        raise ValueError("rows and columns must be 0 (auto) or 1–20")
    if not (1 <= background_tolerance <= 128):
        raise ValueError("background_tolerance must be 1–128")
    if not (0 <= min_content_ratio <= 0.1
            and 0 <= border_ignore_ratio <= 0.1):
        raise ValueError("Content/border ratios must be between 0 and 0.1")
    rgb = _rgb(image)
    height, width = image.shape[:2]
    if min(height, width) < 8:
        raise ValueError("Image is too small to split")
    background = _background(rgb)
    x_lines = _axis_lines(rgb, background, axis=0)
    y_lines = _axis_lines(rgb, background, axis=1)
    x_gaps = y_gaps = None
    hinted_rows, hinted_columns = grid_for_count(asset_count) if asset_count else (0, 0)

    def bounds(size, explicit, detected, hint, axis):
        nonlocal x_gaps, y_gaps
        if explicit:
            return [int(round(k * size / explicit)) for k in range(explicit + 1)], "manual"
        if detected:
            return detected, "lines"
        if hint:
            return [int(round(k * size / hint)) for k in range(hint + 1)], "asset_list"
        gaps = _axis_gaps(rgb, background, axis, background_tolerance)
        if axis == 0:
            x_gaps = gaps
        else:
            y_gaps = gaps
        return (gaps, "gaps") if gaps else ([0, size], "single")

    xb, xm = bounds(width, columns, x_lines, hinted_columns, 0)
    yb, ym = bounds(height, rows, y_lines, hinted_rows, 1)
    actual_rows, actual_columns = len(yb) - 1, len(xb) - 1
    if actual_rows * actual_columns > 100:
        raise ValueError("Too many cells; verify the grid")
    tiles, records, kept, skipped = [], [], [], []
    for row in range(actual_rows):
        for col in range(actual_columns):
            index = row * actual_columns + col + 1
            x0, x1, y0, y1 = xb[col], xb[col + 1], yb[row], yb[row + 1]
            # Remove only the separator strokes, not surrounding whitespace.
            if trim_grid_lines and xm == "lines":
                x0 += 2 if col else 0
                x1 -= 2 if col < actual_columns - 1 else 0
            if trim_grid_lines and ym == "lines":
                y0 += 2 if row else 0
                y1 -= 2 if row < actual_rows - 1 else 0
            crop = image[y0:y1, x0:x1].copy()
            view = rgb[y0:y1, x0:x1]
            bh = max(1, int(round(view.shape[0] * border_ignore_ratio)))
            bw = max(1, int(round(view.shape[1] * border_ignore_ratio)))
            inner = view[bh:-bh, bw:-bw]
            if not inner.size:
                inner = view
            difference = np.max(np.abs(inner - background), axis=2)
            foreground = difference > background_tolerance
            minimum = max(12, int(math.ceil(foreground.size * min_content_ratio)))
            has_content = _component_at_least(foreground, minimum)
            if has_content:
                tiles.append(crop)
                kept.append(index)
            else:
                skipped.append(index)
            records.append({
                "index": index, "row": row + 1, "column": col + 1,
                "box": [x0, y0, x1, y1], "kept": has_content,
                "foreground_pixels": int(foreground.sum()),
                "foreground_ratio": float(foreground.mean()),
            })
    warnings = []
    if "gaps" in (xm, ym):
        warnings.append("Grid inferred from blank strips; connect asset_list for extra certainty")
    if not rows and not columns and xm == ym == "single" and not asset_count:
        warnings.append("No separators detected; treated as a single image")
    return tiles, {
        "rows": actual_rows, "columns": actual_columns,
        "x_boundaries": xb, "y_boundaries": yb,
        "detection": {"x": xm, "y": ym},
        "background_rgb": background.astype(int).tolist(),
        "kept_count": len(kept), "skipped_count": len(skipped),
        "kept_indices": kept, "skipped_indices": skipped,
        "cells": records, "warnings": warnings,
    }
