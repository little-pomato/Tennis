from pathlib import Path
from typing import List, Optional, Tuple
import json

import cv2
import numpy as np


def _to_poly_array(obj, scale: float = 1.0) -> Optional[np.ndarray]:
    try:
        poly = np.array(obj, dtype=np.float32)
    except Exception:
        return None
    if poly.ndim != 2 or poly.shape[1] != 2:
        return None
    poly *= float(scale)
    return poly.astype(np.int32)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pick_polygon(data: dict, keys: List[str], scale: float) -> Optional[np.ndarray]:
    for k in keys:
        if k in data:
            poly = _to_poly_array(data[k], scale=scale)
            if poly is not None:
                return poly
    return None


def pick_polygons(data: dict, keys: List[str], scale: float) -> List[np.ndarray]:
    polys = []
    for k in keys:
        if k not in data:
            continue
        obj = data[k]
        poly = _to_poly_array(obj, scale=scale)
        if poly is not None:
            polys.append(poly)
            continue
        if isinstance(obj, list):
            for item in obj:
                poly = _to_poly_array(item, scale=scale)
                if poly is not None:
                    polys.append(poly)
    return polys


def _disk_kernel(radius_px: int) -> np.ndarray:
    radius_px = max(1, int(round(radius_px)))
    k = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _directional_expand(mask: np.ndarray, dy_values: List[int], dx_values: Optional[List[int]] = None) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape[:2]
    dx_values = dx_values or [0]
    for dy in dy_values:
        for dx in dx_values:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            out = cv2.bitwise_or(out, shifted)
    return out


def build_search_mask(
    valid_mask: np.ndarray,
    roi_json: Optional[Path],
    scale: float,
    erode_iter: int = 1,
    roi_expand_px: int = 0,
    valid_mask_expand_px: int = 0,
    far_expand_px: int = 0,
    far_expand_dx: int = 0,
    net_y: Optional[float] = None,
) -> np.ndarray:
    H, W = valid_mask.shape[:2]
    base_valid = valid_mask.copy()
    if valid_mask_expand_px > 0:
        base_valid = cv2.dilate(base_valid, _disk_kernel(valid_mask_expand_px), iterations=1)

    out = base_valid.copy()
    data = load_json(roi_json) if roi_json is not None else {}

    search_poly = pick_polygon(
        data,
        [
            "bounce_roi", "bounce_polygon", "bounce_poly",
            "court_polygon", "court_poly", "play_area_polygon", "valid_polygon"
        ],
        scale,
    )
    if search_poly is not None:
        poly_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(poly_mask, [search_poly], 255)

        if roi_expand_px > 0:
            poly_mask = cv2.dilate(poly_mask, _disk_kernel(roi_expand_px), iterations=1)

        if far_expand_px > 0:
            dx_vals = [0]
            if far_expand_dx > 0:
                dx_vals = list(range(-far_expand_dx, far_expand_dx + 1))
            far_expanded = _directional_expand(poly_mask, dy_values=list(range(-far_expand_px, 1)), dx_values=dx_vals)
            if net_y is not None:
                split_y = int(np.clip(round(net_y), 0, H))
                merged = poly_mask.copy()
                merged[:split_y, :] = far_expanded[:split_y, :]
                poly_mask = merged
            else:
                poly_mask = far_expanded

        out = cv2.bitwise_and(out, poly_mask)

    exclude_polys = pick_polygons(
        data,
        [
            "bounce_exclude", "bounce_exclude_polygons", "exclude_polygons",
            "no_bounce_polygons", "net_polygon", "net_roi", "net_polygons"
        ],
        scale,
    )
    if exclude_polys:
        ex = np.zeros((H, W), np.uint8)
        cv2.fillPoly(ex, exclude_polys, 255)
        out = cv2.bitwise_and(out, cv2.bitwise_not(ex))

    if erode_iter > 0:
        out = cv2.erode(out, np.ones((5, 5), np.uint8), iterations=erode_iter)

    if far_expand_px > 0 and net_y is not None:
        split_y = int(np.clip(round(net_y), 0, H))
        far_mask = out[:split_y, :]
        far_mask = cv2.dilate(far_mask, _disk_kernel(max(1, far_expand_px // 2)), iterations=1)
        out[:split_y, :] = far_mask

    return out


def build_net_distance_map(roi_json: Optional[Path], target_hw: Tuple[int, int], scale: float) -> Optional[np.ndarray]:
    if roi_json is None or not roi_json.exists():
        return None
    data = load_json(roi_json)
    net_polys = pick_polygons(data, ["NET_LINE"], scale)
    if not net_polys:
        return None

    H, W = target_hw
    net_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(net_mask, net_polys, 255)
    inv = cv2.bitwise_not(net_mask)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def estimate_net_y(roi_json: Optional[Path], target_hw: Tuple[int, int], scale: float) -> Optional[float]:
    if roi_json is None or not roi_json.exists():
        return None
    data = load_json(roi_json)
    net_polys = pick_polygons(data, ["NET_LINE"], scale)
    if not net_polys:
        return None
    ys = []
    for poly in net_polys:
        ys.extend(poly[:, 1].astype(float).tolist())
    if not ys:
        return None
    return float(np.median(ys))
