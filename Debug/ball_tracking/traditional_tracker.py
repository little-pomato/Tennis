#!/usr/bin/env python3
"""
Classical tennis ball tracking and bounce debug pipeline.

This is designed for quick iteration on dataset/v01/frames. It deliberately
keeps dependencies to OpenCV, NumPy, and pandas, which are already listed in the
project requirements.
"""

from __future__ import annotations

import argparse
import math
import random
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    frame_idx: int
    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int
    area: float
    circularity: float
    mean_motion: float
    color_score: float
    score: float
    flow_mag: float = 0.0


@dataclass
class TrackPoint:
    frame_idx: int
    x: float
    y: float
    pred_x: float
    pred_y: float
    vx: float
    vy: float
    ax: float
    ay: float
    measurement_score: float
    distance_to_prediction: float
    used_measurement: int
    interpolated: int


@dataclass
class BounceCandidate:
    frame_idx: int
    x: float
    y: float
    score: float
    dy_before: float
    dy_after: float
    angle_change_deg: float
    split_fit_error: float
    support_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug classical tennis ball tracking and bounce detection."
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("dataset/v01/frames"))
    parser.add_argument("--mask", type=Path, default=Path("dataset/v01/valid_mask.png"))
    parser.add_argument(
        "--person-boxes-csv",
        type=Path,
        default=Path("dataset/v01/bounce_detector/yolo_person_boxes.csv"),
        help="Optional cached person boxes used only for exclusion.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Debug/ball_tracking/output"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end frame index.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--diff-thresh", type=int, default=18, help="Threshold for near-field motion mask (bottom half).")
    parser.add_argument("--far-diff-thresh", type=int, default=10, help="Threshold for far-field motion mask (top half). Lower = more sensitive but more noise.")
    parser.add_argument("--min-area", type=float, default=2.0)
    parser.add_argument("--max-area", type=float, default=95.0)
    parser.add_argument("--max-wh", type=int, default=24)
    # Reduced max-association-dist to prevent the tracker from "teleporting" to distant noise
    parser.add_argument("--max-association-dist", type=float, default=25.0)
    parser.add_argument("--roi-pad-x", type=int, default=60, help="Expand the ROI mask horizontally by this many pixels.")
    parser.add_argument("--roi-pad-y", type=int, default=40, help="Expand the ROI mask vertically by this many pixels.")
    parser.add_argument("--hm-bin-size", type=int, default=10, help="Heatmap bin size for static noise filtering.")
    parser.add_argument("--hm-threshold", type=float, default=0.35, help="Heatmap static noise threshold ratio (0.0 to 1.0).")
    parser.add_argument("--momentum-dir-penalty", type=float, default=60.0, help="Multiplier for directional change penalty.")
    parser.add_argument("--momentum-vel-penalty", type=float, default=25.0, help="Multiplier for velocity magnitude change penalty.")
    parser.add_argument("--kalman-proc-noise", type=float, default=0.15, help="Kalman process noise for acceleration (lower = stiffer track).")
    parser.add_argument("--kalman-meas-noise", type=float, default=18.0, help="Kalman measurement noise (higher = trust physics more).")
    parser.add_argument("--min-track-measurements", type=int, default=4, help="Minimum real detections required to keep a track segment.")
    parser.add_argument("--min-track-displacement", type=float, default=15.0, help="Minimum pixel displacement for a valid track segment.")
    parser.add_argument("--roi-json", type=Path, default=Path("dataset/v01/roi_config.json"), help="Path to roi_config.json to mask out painted court lines.")
    parser.add_argument("--line-mask-thickness", type=int, default=5, help="Thickness of the line exclusion mask.")
    parser.add_argument("--debug-every", type=int, default=15)
    parser.add_argument(
        "--debug-preprocess",
        action="store_true",
        help="Dump labelled preprocessing stage sheets and per-stage pixel counts.",
    )
    parser.add_argument(
        "--debug-preprocess-every",
        type=int,
        default=1,
        help="Frame interval for preprocessing sheets. Default dumps every processable frame.",
    )
    parser.add_argument(
        "--debug-preprocess-frames",
        default="",
        help="Comma-separated original frame indices to dump regardless of interval.",
    )
    parser.add_argument("--no-clahe", action="store_true", help="Use raw grayscale instead of CLAHE gray.")
    parser.add_argument("--no-roi-mask", action="store_true", help="Disable valid_mask.png ROI filtering.")
    parser.add_argument("--no-player-exclusion", action="store_true", help="Disable cached person-box exclusion.")
    parser.add_argument("--no-manual-exclusion", action="store_true", help="Disable manual --exclude-rect masking.")
    parser.add_argument("--no-mog", action="store_true", help="Disable MOG2 in motion fusion.")
    parser.add_argument("--no-morphology", action="store_true", help="Disable morphology cleanup on final motion mask.")
    parser.add_argument(
        "--exclude-rect",
        action="append",
        default=["0,285,285,360"],
        help=(
            "Manual exclusion rectangle x1,y1,x2,y2 in original frame coordinates. "
            "Can be repeated. Default masks the v01 lower-left broadcast scoreboard."
        ),
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def list_frames(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {frames_dir}")
    return frames


def read_frame(path: Path, scale: float) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Cannot read frame: {path}")
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return frame


def load_mask(mask_path: Path, shape_hw: Tuple[int, int], scale: float, pad_x: int = 0, pad_y: int = 0) -> np.ndarray:
    h, w = shape_hw
    if not mask_path.exists():
        return np.full((h, w), 255, dtype=np.uint8)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.full((h, w), 255, dtype=np.uint8)
    if scale != 1.0:
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    mask = ((mask > 0).astype(np.uint8) * 255)
    
    if pad_x > 0 or pad_y > 0:
        # Create an elliptical structural element for independent X/Y dilation
        kernel_w = max(1, pad_x * 2 + 1)
        kernel_h = max(1, pad_y * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
        mask = cv2.dilate(mask, kernel)
        
    return mask


def load_court_lines_mask(roi_json_path: Path, shape_hw: Tuple[int, int], scale: float, thickness: int = 5) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if not roi_json_path.exists():
        return mask

    try:
        with open(roi_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {roi_json_path}: {e}")
        return mask

    # List of keys that contain line segments (list of [x, y] pairs)
    line_keys = [
        "FAR_BASELINE", "NEAR_BASELINE", "FAR_SERVICE_LINE", "NEAR_SERVICE_LINE",
        "NET_BOTTOM_LINE", "NET_LINE", "NET_TOP_LINE", "MID_LINE", "SINGLES_LINES",
        "DOUBLES_LINES"
    ]

    for key in line_keys:
        val = data.get(key)
        if val is None:
            continue
        
        # Some keys might be a single line (list of 2 points), others might be multiple lines
        # Let's normalize it to a list of lines
        if isinstance(val, list) and len(val) > 0:
            # If the first element is a coordinate pair (list of 2 numbers)
            if isinstance(val[0], list) and len(val[0]) == 2 and isinstance(val[0][0], (int, float)):
                lines = [val]
            else:
                lines = val
                
            for line_pts in lines:
                if not isinstance(line_pts, list) or len(line_pts) < 2:
                    continue
                # Draw lines connecting the points
                for i in range(len(line_pts) - 1):
                    pt1 = (int(line_pts[i][0] * scale), int(line_pts[i][1] * scale))
                    pt2 = (int(line_pts[i+1][0] * scale), int(line_pts[i+1][1] * scale))
                    cv2.line(mask, pt1, pt2, 255, thickness)

    return mask


def load_person_boxes(csv_path: Path, scale: float) -> Dict[int, List[Tuple[int, int, int, int]]]:
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"frame_idx", "x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        return {}

    boxes: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for row in df.itertuples(index=False):
        frame_idx = int(getattr(row, "frame_idx"))
        x1 = int(round(float(getattr(row, "x1")) * scale))
        y1 = int(round(float(getattr(row, "y1")) * scale))
        x2 = int(round(float(getattr(row, "x2")) * scale))
        y2 = int(round(float(getattr(row, "y2")) * scale))
        boxes.setdefault(frame_idx, []).append((x1, y1, x2, y2))
    return boxes


def parse_exclude_rects(items: Sequence[str], scale: float) -> List[Tuple[int, int, int, int]]:
    rects: List[Tuple[int, int, int, int]] = []
    for item in items:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid --exclude-rect value: {item}")
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in parts]
        rects.append((x1, y1, x2, y2))
    return rects


def exclusion_mask_for_boxes(
    shape_hw: Tuple[int, int],
    boxes: Sequence[Tuple[int, int, int, int]],
    pad_ratio: float = 0.25,
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # Reduced padding to avoid masking the ball near the player
        px = int(round(bw * pad_ratio))
        py = int(round(bh * pad_ratio))
        xx1 = max(0, x1 - px)
        yy1 = max(0, y1 - py)
        xx2 = min(w - 1, x2 + px)
        yy2 = min(h - 1, y2 + py)
        if xx2 > xx1 and yy2 > yy1:
            cv2.rectangle(mask, (xx1, yy1), (xx2, yy2), 255, -1)
    return mask


def add_rects_to_mask(mask: np.ndarray, rects: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    h, w = mask.shape[:2]
    out = mask.copy()
    for x1, y1, x2, y2 in rects:
        xx1 = max(0, min(w - 1, x1))
        yy1 = max(0, min(h - 1, y1))
        xx2 = max(0, min(w - 1, x2))
        yy2 = max(0, min(h - 1, y2))
        if xx2 > xx1 and yy2 > yy1:
            cv2.rectangle(out, (xx1, yy1), (xx2, yy2), 255, -1)
    return out


def preprocess_gray(frame: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri + 1e-6))


def tennis_color_score(frame: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    pixels = mask > 0
    if not np.any(pixels):
        return 0.0

    # Tennis balls often appear yellow-green, but motion blur and broadcast
    # compression make this a weak cue. Keep it as a soft bonus only.
    hue = ys[pixels].astype(np.float32)
    s = sat[pixels].astype(np.float32)
    v = val[pixels].astype(np.float32)
    hue_bonus = np.mean((hue >= 18) & (hue <= 45))
    sat_bonus = float(np.clip(np.mean(s) / 140.0, 0.0, 1.0))
    val_bonus = float(np.clip(np.mean(v) / 190.0, 0.0, 1.0))
    return float(0.50 * hue_bonus + 0.25 * sat_bonus + 0.25 * val_bonus)


def perspective_area_limits(
    y: int,
    image_h: int,
    min_area: float,
    max_area: float,
    max_wh: int,
) -> Tuple[float, float, float]:
    # Far court pixels are smaller. This keeps tiny far-side blobs from being
    # filtered out while still rejecting near-side player/racket fragments.
    y_ratio = float(y) / float(max(1, image_h - 1))
    scale = 0.30 + 0.90 * y_ratio
    return max(1.0, min_area * scale), max_area * scale, max(5.0, max_wh * math.sqrt(scale))


class CandidateHeatmap:
    """Tracks detection frequency in spatial bins to identify static noise hotspots."""
    def __init__(self, shape_hw: Tuple[int, int], bin_size: int = 10):
        self.h, self.w = shape_hw
        self.bin_size = bin_size
        self.grid_h = math.ceil(self.h / bin_size)
        self.grid_w = math.ceil(self.w / bin_size)
        self.counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.total_frames = 0

    def update(self, candidates: Sequence[Candidate]):
        for c in candidates:
            gy = int(c.cy // self.bin_size)
            gx = int(c.cx // self.bin_size)
            if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w:
                self.counts[gy, gx] += 1
        self.total_frames += 1

    def is_static_noise(self, cx: float, cy: float, threshold_ratio: float = 0.35) -> bool:
        if self.total_frames < 20:
            return False
        gy = int(cy // self.bin_size)
        gx = int(cx // self.bin_size)
        if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w:
            # High frequency detections in a small bin -> static flicker
            return (self.counts[gy, gx] / self.total_frames) > threshold_ratio
        return False


def edge_density_score(edge_mag: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Calculates the density of strong edges within the bounding box."""
    if w <= 0 or h <= 0:
        return 0.0
    roi = edge_mag[y : y + h, x : x + w]
    # Count pixels with strong gradient (edges)
    strong_edges = np.count_nonzero(roi > 50)
    density = strong_edges / float(w * h)
    # Return normalized score (0.0 to 1.0)
    return float(np.clip(density / 0.4, 0.0, 1.0))


def extract_candidates(
    frame_idx: int,
    frame: np.ndarray,
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    gray_next: np.ndarray,
    mog_mask: np.ndarray,
    valid_mask: np.ndarray,
    exclude_mask: np.ndarray,
    edge_mag: np.ndarray,
    args: argparse.Namespace,
    heatmap: Optional[CandidateHeatmap] = None,
) -> Tuple[List[Candidate], np.ndarray, Dict[str, np.ndarray]]:
    # Far-field awareness: top of frame usually has smaller, faster balls with lower contrast
    height, width = gray_curr.shape[:2]
    far_zone_h = int(height * 0.45)
    
    diff_prev = cv2.absdiff(gray_curr, gray_prev)
    diff_next = cv2.absdiff(gray_next, gray_curr)
    diff_mix = cv2.max(diff_prev, diff_next)

    # Robust motion intersection: 
    # Use a softer approach than bitwise_and to catch small balls with 1-pixel overlap
    diff_min = cv2.min(diff_prev, diff_next)
    
    # Adaptive thresholding for far vs near
    thresh_near = args.diff_thresh
    thresh_far = args.far_diff_thresh
    
    _, bin_near = cv2.threshold(diff_min, thresh_near, 255, cv2.THRESH_BINARY)
    _, bin_far = cv2.threshold(diff_min, thresh_far, 255, cv2.THRESH_BINARY)
    
    diff_inter = bin_near.copy()
    diff_inter[:far_zone_h, :] = bin_far[:far_zone_h, :]

    # For union (fall-back), we also use adaptive thresholding
    _, union_near = cv2.threshold(diff_mix, thresh_near, 255, cv2.THRESH_BINARY)
    _, union_far = cv2.threshold(diff_mix, thresh_far, 255, cv2.THRESH_BINARY)
    diff_union = union_near.copy()
    diff_union[:far_zone_h, :] = union_far[:far_zone_h, :]

    _, mog_bin = cv2.threshold(mog_mask, 180, 255, cv2.THRESH_BINARY)
    if args.no_mog:
        motion = diff_inter.copy()
    else:
        motion = cv2.bitwise_or(diff_inter, cv2.bitwise_and(diff_union, mog_bin))
        
    # If the intersection is too weak, cautiously use union but prioritize mog
    if int(np.count_nonzero(motion)) < 10:
        # Keep bits that appear in at least one difference AND are supported by MOG
        motion = cv2.bitwise_or(motion, cv2.bitwise_and(diff_union, mog_bin))
        # If still empty, use a very tight union
        if int(np.count_nonzero(motion)) < 5:
            motion = cv2.bitwise_or(motion, diff_inter)

    motion_fused = motion.copy()
    motion_roi = cv2.bitwise_and(motion_fused, valid_mask)
    motion_excluded = cv2.bitwise_and(motion_roi, cv2.bitwise_not(exclude_mask))
    
    if args.no_morphology:
        motion = motion_excluded.copy()
    else:
        # Denoise 'stars' before adaptive morphology
        motion_denoised = cv2.medianBlur(motion_excluded, 3)
        # Adaptive morphology: don't open far-field to preserve 1-2 pixel balls
        motion_near = cv2.morphologyEx(motion_denoised[far_zone_h:, :], cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        motion = motion_denoised.copy()
        motion[far_zone_h:, :] = motion_near
        # Gentle close to bridge gaps
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    debug_stages = {
        "gray": gray_curr,
        "valid_mask": valid_mask,
        "exclude_mask": exclude_mask,
        "diff_prev": diff_prev,
        "diff_next": diff_next,
        "diff_union": diff_union,
        "diff_inter": diff_inter,
        "mog": mog_bin,
        "motion_fused": motion_fused,
        "after_roi": motion_roi,
        "after_exclusion": motion_excluded,
        "motion_clean": motion,
    }

    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Candidate] = []
    pre_candidates: List[dict] = []
    image_h = frame.shape[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            area = float(w * h) # fallback for single pixel
            
        min_area_p, max_area_p, max_wh_p = perspective_area_limits(
            y, image_h, args.min_area, args.max_area, args.max_wh
        )
        
        # More lenient area checks for far balls
        if y < far_zone_h:
            if area < 0.8 or area > max_area_p * 1.5:
                continue
            if max(w, h) > max_wh_p * 1.5:
                continue
        else:
            if area < min_area_p or area > max_area_p:
                continue
            if w > max_wh_p or h > max_wh_p:
                continue
                
        if max(w, h) / max(1.0, min(w, h)) > 4.5:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            cx = x + 0.5 * w
            cy = y + 0.5 * h
        else:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])

        # Filter static noise hotspots (flickering highlights)
        if heatmap is not None and heatmap.is_static_noise(cx, cy):
            continue

        circ = circularity(contour)
        circ_score = float(np.clip(circ, 0.0, 1.0))
        local_motion = float(np.mean(diff_mix[y : y + h, x : x + w]))
        color = tennis_color_score(frame, contour)
        edge_score = edge_density_score(edge_mag, int(x), int(y), int(w), int(h))
        y_ratio = float(y) / float(max(1, image_h - 1))
        
        pre_candidates.append({
            "cx": cx, "cy": cy, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": area, "circ": circ, "circ_score": circ_score,
            "local_motion": local_motion, "color": color, "edge_score": edge_score,
            "y_ratio": y_ratio
        })

    if not pre_candidates:
        return [], motion, debug_stages

    # Calculate Lucas-Kanade Optical Flow for all candidate centers
    pts_curr = np.array([[[c["cx"], c["cy"]]] for c in pre_candidates], dtype=np.float32)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    pts_prev, status_bwd, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_prev, pts_curr, None, **lk_params)
    pts_next, status_fwd, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_next, pts_curr, None, **lk_params)

    for i, c in enumerate(pre_candidates):
        flow_bwd = math.hypot(pts_prev[i][0][0] - c["cx"], pts_prev[i][0][1] - c["cy"]) if status_bwd is not None and status_bwd[i] else 0.0
        flow_fwd = math.hypot(pts_next[i][0][0] - c["cx"], pts_next[i][0][1] - c["cy"]) if status_fwd is not None and status_fwd[i] else 0.0
        flow_mag = max(flow_bwd, flow_fwd)
        
        # A real moving ball has significant flow. Static flicker has ~0 flow.
        flow_score = float(np.clip((flow_mag - 1.0) / 10.0, 0.0, 1.0))
        
        if c["y_ratio"] < 0.45:
            area_score = max(0.0, 1.0 - abs(c["area"] - 4.0) / 20.0)
            motion_score = float(np.clip(c["local_motion"] / 35.0, 0.0, 1.0))
            # Blending in flow_score to heavily reward real physical motion
            score = 0.40 * motion_score + 0.15 * area_score + 0.15 * c["edge_score"] + 0.10 * c["circ_score"] + 0.20 * flow_score
        else:
            area_score = max(0.0, 1.0 - abs(c["area"] - 10.0) / 50.0)
            motion_score = float(np.clip(c["local_motion"] / 55.0, 0.0, 1.0))
            score = 0.35 * motion_score + 0.15 * area_score + 0.15 * c["edge_score"] + 0.10 * c["circ_score"] + 0.05 * c["color"] + 0.20 * flow_score

        candidates.append(
            Candidate(
                frame_idx=frame_idx,
                cx=c["cx"],
                cy=c["cy"],
                x=c["x"],
                y=c["y"],
                w=c["w"],
                h=c["h"],
                area=c["area"],
                circularity=c["circ"],
                mean_motion=c["local_motion"],
                color_score=c["color"],
                score=float(score),
                flow_mag=float(flow_mag),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, motion, debug_stages



def create_kalman(dt: float = 1.0, proc_noise: float = 0.15, meas_noise: float = 18.0) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(6, 2)
    kf.transitionMatrix = np.array(
        [
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32
    )
    # Stiffen the filter: reduce acceleration noise
    # Increase measurement noise to trust the physical model more than noisy detections
    kf.processNoiseCov = np.diag([0.05, 0.05, proc_noise, proc_noise, proc_noise*2, proc_noise*2]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([meas_noise, meas_noise]).astype(np.float32)
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
    return kf


def initialize_kalman(kf: cv2.KalmanFilter, candidate: Candidate) -> None:
    kf.statePost = np.array([[candidate.cx], [candidate.cy], [0], [0], [0], [0]], dtype=np.float32)


def associate_candidate(
    candidates: Sequence[Candidate],
    pred_x: float,
    pred_y: float,
    prev_x: float,
    prev_y: float,
    max_dist: float,
    dir_penalty_mult: float = 60.0,
    vel_penalty_mult: float = 25.0,
) -> Tuple[Optional[Candidate], float]:
    best: Optional[Candidate] = None
    best_cost = float("inf")
    best_dist = float("inf")
    
    # Expected displacement vector (momentum from Kalman)
    evx = pred_x - prev_x
    evy = pred_y - prev_y
    v_norm = math.hypot(evx, evy)
    
    # If the ball is moving fast, it cannot "turn on a dime" unless it bounces.
    # Therefore, the search area should be an ellipse heavily biased forward along the velocity vector, 
    # not a perfect circle (`max_dist`). We enforce this by heavily penalizing perpendicular distance.
    
    for candidate in candidates[:40]:
        dist = math.hypot(candidate.cx - pred_x, candidate.cy - pred_y)
        
        # Absolute hard limit: A tennis ball rarely moves more than 40-50 pixels in 1/30th of a second
        if dist > max_dist:
            continue
            
        cost = dist - 18.0 * candidate.score
            
        # Momentum consistency check (only applies if we already have some speed)
        if v_norm > 5.0:
            # 1. Trajectory Vector (Where did it move from last frame?)
            cvx = candidate.cx - prev_x
            cvy = candidate.cy - prev_y
            cv_norm = math.hypot(cvx, cvy)
            
            # 2. Flow Vector (Where do the pixels *think* they are moving?)
            # If LK flow contradicts the trajectory, it's likely a background object.
            
            # Directional dot product (Trajectory vs Kalman Prediction)
            dot = (cvx * evx + cvy * evy) / (v_norm * cv_norm + 1e-6)
            
            dir_penalty = 0.0
            if dot < 0.6:
                # Extremely strict penalty for turning more than ~50 degrees
                # This prevents "snapping" to a noise point next to the ball
                dir_penalty = dir_penalty_mult * (0.6 - dot) * 2.0
            
            # Velocity magnitude penalty: prevent sudden speed jumps or stops
            v_ratio = cv_norm / v_norm
            v_penalty = vel_penalty_mult * abs(math.log(v_ratio + 1e-6))
            
            # If the candidate has NO optical flow (flow_mag < 1.0) but we expect it to move fast,
            # it is almost certainly a flickering noise dot (like a star or reflection).
            flow_penalty = 0.0
            if candidate.flow_mag < 1.5 and cv_norm > 8.0:
                flow_penalty = 50.0 # Massive penalty for teleporting to a stationary object
                
            cost += dir_penalty + v_penalty + flow_penalty
            
        if cost < best_cost:
            best = candidate
            best_cost = cost
            best_dist = dist
            
    return best, best_dist


def run_kalman_tracker(
    candidates_by_frame: Dict[int, List[Candidate]],
    frame_indices: Sequence[int],
    args: argparse.Namespace,
    image_shape_hw: Tuple[int, int],
) -> List[TrackPoint]:
    kf = create_kalman(
        proc_noise=args.kalman_proc_noise,
        meas_noise=args.kalman_meas_noise
    )
    initialized = False
    lost_count = 0
    all_points: List[TrackPoint] = []
    current_segment: List[TrackPoint] = []
    image_h, image_w = image_shape_hw
    margin = 40.0
    
    # Keep track of last valid position for association
    last_x, last_y = 0.0, 0.0

    for frame_idx in frame_indices:
        candidates = candidates_by_frame.get(frame_idx, [])
        if not initialized:
            if not candidates:
                continue
            # Seed must be a decent score candidate
            if candidates[0].score < 0.35:
                continue
            seed = candidates[0]
            initialize_kalman(kf, seed)
            initialized = True
            last_x, last_y = seed.cx, seed.cy
            current_segment = [
                TrackPoint(
                    frame_idx=frame_idx,
                    x=seed.cx,
                    y=seed.cy,
                    pred_x=seed.cx,
                    pred_y=seed.cy,
                    vx=0.0,
                    vy=0.0,
                    ax=0.0,
                    ay=0.0,
                    measurement_score=seed.score,
                    distance_to_prediction=0.0,
                    used_measurement=1,
                    interpolated=0,
                )
            ]
            continue

        pred = kf.predict()
        pred_x = float(pred[0, 0])
        pred_y = float(pred[1, 0])
        
        # Adaptive search radius: tighter if recently found, wider if lost
        current_max_dist = args.max_association_dist
        if lost_count > 0:
            current_max_dist += min(50.0, lost_count * 12.0)
            
        candidate, dist = associate_candidate(
            candidates, pred_x, pred_y, last_x, last_y, current_max_dist,
            dir_penalty_mult=args.momentum_dir_penalty,
            vel_penalty_mult=args.momentum_vel_penalty
        )

        if candidate is not None:
            corrected = kf.correct(np.array([[candidate.cx], [candidate.cy]], dtype=np.float32))
            state = corrected
            lost_count = 0
            x = float(candidate.cx)
            y = float(candidate.cy)
            last_x, last_y = x, y
            used = 1
            interpolated = 0
            score = float(candidate.score)
        else:
            state = pred
            lost_count += 1
            x = pred_x
            y = pred_y
            # Don't update last_x, last_y with prediction to avoid drift chain
            used = 0
            interpolated = 1 if lost_count <= 5 else 0
            score = 0.0
            dist = float("nan")

        outside = (
            x < -margin
            or x > image_w + margin
            or y < -margin
            or y > image_h + margin
        )
        # If lost for too long, reset. 10 frames is ~0.33s.
        if lost_count > 10 or (candidate is None and outside):
            # Validate the segment before adding to all_points
            if len(current_segment) > 5:
                dx = current_segment[-1].x - current_segment[0].x
                dy = current_segment[-1].y - current_segment[0].y
                total_dist = math.hypot(dx, dy)
                measurements = sum(1 for p in current_segment if p.used_measurement)
                
                # Stationary flicker or too few real detections -> ignore this segment
                if total_dist >= args.min_track_displacement and measurements >= args.min_track_measurements:
                    all_points.extend(current_segment)
            
            initialized = False
            lost_count = 0
            current_segment = []
            continue

        current_segment.append(
            TrackPoint(
                frame_idx=frame_idx,
                x=x,
                y=y,
                pred_x=pred_x,
                pred_y=pred_y,
                vx=float(state[2, 0]),
                vy=float(state[3, 0]),
                ax=float(state[4, 0]),
                ay=float(state[5, 0]),
                measurement_score=score,
                distance_to_prediction=float(dist),
                used_measurement=used,
                interpolated=interpolated,
            )
        )

    # Add final segment if valid
    if len(current_segment) > 5:
        dx = current_segment[-1].x - current_segment[0].x
        dy = current_segment[-1].y - current_segment[0].y
        if math.hypot(dx, dy) >= args.min_track_displacement and sum(1 for p in current_segment if p.used_measurement) >= args.min_track_measurements:
            all_points.extend(current_segment)

    return all_points


def fit_quadratic_ransac(
    points: Sequence[TrackPoint],
    iterations: int = 80,
    threshold: float = 8.0,
) -> Tuple[Optional[np.ndarray], float, int]:
    valid = [p for p in points if p.used_measurement or p.interpolated]
    if len(valid) < 5:
        return None, float("inf"), 0

    xs = np.array([p.frame_idx for p in valid], dtype=np.float64)
    ys = np.array([p.y for p in valid], dtype=np.float64)
    best_coef: Optional[np.ndarray] = None
    best_inliers = 0
    best_error = float("inf")

    rng = random.Random(7)
    for _ in range(iterations):
        idx = rng.sample(range(len(valid)), 3)
        try:
            coef = np.polyfit(xs[idx], ys[idx], 2)
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(np.polyval(coef, xs) - ys)
        inliers = residuals < threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 5:
            continue
        try:
            refined = np.polyfit(xs[inliers], ys[inliers], 2)
        except np.linalg.LinAlgError:
            continue
        refined_residuals = np.abs(np.polyval(refined, xs[inliers]) - ys[inliers])
        error = float(np.mean(refined_residuals))
        if inlier_count > best_inliers or (inlier_count == best_inliers and error < best_error):
            best_coef = refined
            best_inliers = inlier_count
            best_error = error

    if best_coef is None:
        try:
            best_coef = np.polyfit(xs, ys, 2)
            best_error = float(np.mean(np.abs(np.polyval(best_coef, xs) - ys)))
            best_inliers = len(valid)
        except np.linalg.LinAlgError:
            return None, float("inf"), 0

    return best_coef, best_error, best_inliers


def local_slope(points: Sequence[TrackPoint]) -> Tuple[float, float]:
    xs = np.array([p.frame_idx for p in points], dtype=np.float64)
    ys = np.array([p.y for p in points], dtype=np.float64)
    if len(xs) < 2:
        return 0.0, float("inf")
    try:
        coef = np.polyfit(xs, ys, 1)
    except np.linalg.LinAlgError:
        return 0.0, float("inf")
    residual = float(np.mean(np.abs(np.polyval(coef, xs) - ys)))
    return float(coef[0]), residual


def detect_bounces(track: Sequence[TrackPoint], fps: float) -> List[BounceCandidate]:
    by_frame = {p.frame_idx: p for p in track}
    frames = sorted(by_frame)
    bounces: List[BounceCandidate] = []

    for frame_idx in frames:
        pre = [by_frame[f] for f in range(frame_idx - 5, frame_idx) if f in by_frame]
        post = [by_frame[f] for f in range(frame_idx + 1, frame_idx + 6) if f in by_frame]
        if len(pre) < 3 or len(post) < 3:
            continue

        p = by_frame[frame_idx]
        if not p.used_measurement:
            continue
        if sum(q.used_measurement for q in pre) < 2 or sum(q.used_measurement for q in post) < 2:
            continue
        local_y_values = [q.y for q in pre[-3:]] + [q.y for q in post[:3]]
        if local_y_values and p.y < max(local_y_values) - 2.0:
            continue

        dy_before, err_before = local_slope(pre)
        dy_after, err_after = local_slope(post)

        # Image y grows downward. A bounce usually changes from downward motion
        # (positive y velocity) to upward motion (negative y velocity).
        reversal = dy_before > 0.4 and dy_after < -0.2
        if not reversal:
            continue

        vx_before, _ = local_slope_x(pre)
        vx_after, _ = local_slope_x(post)
        angle_before = math.atan2(dy_before, vx_before)
        angle_after = math.atan2(dy_after, vx_after)
        angle_change = abs(math.degrees(angle_after - angle_before))
        angle_change = min(angle_change, 360.0 - angle_change)

        window = [by_frame[f] for f in range(frame_idx - 8, frame_idx + 9) if f in by_frame]
        left = [q for q in window if q.frame_idx <= frame_idx]
        right = [q for q in window if q.frame_idx >= frame_idx]
        _, left_error, left_n = fit_quadratic_ransac(left, iterations=60, threshold=10.0)
        _, right_error, right_n = fit_quadratic_ransac(right, iterations=60, threshold=10.0)
        split_error = float(np.mean([left_error, right_error]))
        if not np.isfinite(split_error):
            split_error = 99.0
        if split_error > 18.0:
            continue

        support = left_n + right_n
        if support < 12:
            continue
        measurement_bonus = 0.20 if p.used_measurement else 0.0
        fit_score = float(np.clip(1.0 - split_error / 18.0, 0.0, 1.0))
        angle_score = float(np.clip(angle_change / 110.0, 0.0, 1.0))
        speed_score = float(np.clip((abs(dy_before) + abs(dy_after)) / 16.0, 0.0, 1.0))
        support_score = float(np.clip(support / 14.0, 0.0, 1.0))
        score = 0.30 * angle_score + 0.25 * speed_score + 0.25 * fit_score + 0.20 * support_score
        score += measurement_bonus

        if score >= 0.55:
            bounces.append(
                BounceCandidate(
                    frame_idx=frame_idx,
                    x=p.x,
                    y=p.y,
                    score=float(min(score, 1.0)),
                    dy_before=float(dy_before * fps),
                    dy_after=float(dy_after * fps),
                    angle_change_deg=float(angle_change),
                    split_fit_error=float(split_error),
                    support_points=int(support),
                )
            )

    return suppress_nearby_bounces(bounces, min_gap=8)


def local_slope_x(points: Sequence[TrackPoint]) -> Tuple[float, float]:
    xs = np.array([p.frame_idx for p in points], dtype=np.float64)
    vals = np.array([p.x for p in points], dtype=np.float64)
    if len(xs) < 2:
        return 0.0, float("inf")
    try:
        coef = np.polyfit(xs, vals, 1)
    except np.linalg.LinAlgError:
        return 0.0, float("inf")
    residual = float(np.mean(np.abs(np.polyval(coef, xs) - vals)))
    return float(coef[0]), residual


def suppress_nearby_bounces(
    bounces: Sequence[BounceCandidate],
    min_gap: int,
) -> List[BounceCandidate]:
    kept: List[BounceCandidate] = []
    for candidate in sorted(bounces, key=lambda b: b.score, reverse=True):
        if all(abs(candidate.frame_idx - existing.frame_idx) >= min_gap for existing in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda b: b.frame_idx)


def draw_debug_frame(
    frame: np.ndarray,
    frame_idx: int,
    candidates: Sequence[Candidate],
    track_point: Optional[TrackPoint],
    bounces: Sequence[BounceCandidate],
    motion_mask: Optional[np.ndarray],
) -> np.ndarray:
    out = frame.copy()
    for candidate in candidates[:20]:
        color = (0, int(120 + 135 * min(candidate.score, 1.0)), 255)
        cv2.rectangle(
            out,
            (candidate.x, candidate.y),
            (candidate.x + candidate.w, candidate.y + candidate.h),
            color,
            1,
        )
        cv2.circle(out, (int(round(candidate.cx)), int(round(candidate.cy))), 2, color, -1)

    if track_point is not None:
        cv2.circle(out, (int(round(track_point.pred_x)), int(round(track_point.pred_y))), 6, (255, 0, 0), 1)
        cv2.circle(out, (int(round(track_point.x)), int(round(track_point.y))), 5, (0, 255, 0), -1)

    for bounce in bounces:
        if abs(bounce.frame_idx - frame_idx) <= 2:
            cv2.drawMarker(
                out,
                (int(round(bounce.x)), int(round(bounce.y))),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.putText(
                out,
                f"bounce {bounce.score:.2f}",
                (int(round(bounce.x)) + 8, int(round(bounce.y)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(out, f"frame {frame_idx}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, f"frame {frame_idx}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)

    if motion_mask is not None:
        small = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        mh, mw = small.shape[:2]
        thumb_w = max(120, frame.shape[1] // 4)
        thumb_h = int(round(mh * (thumb_w / mw)))
        small = cv2.resize(small, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        out[0:thumb_h, out.shape[1] - thumb_w : out.shape[1]] = small
    return out


def parse_frame_set(value: str) -> set[int]:
    frames: set[int] = set()
    if not value.strip():
        return frames
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        frames.add(int(item))
    return frames


def should_dump_preprocess(
    frame_idx: int,
    args: argparse.Namespace,
    explicit_frames: set[int],
) -> bool:
    if not args.debug_preprocess:
        return False
    if frame_idx in explicit_frames:
        return True
    interval = args.debug_preprocess_every
    return interval > 0 and frame_idx % interval == 0


def to_bgr_panel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def mask_overlay(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    active = mask > 0
    out[active] = cv2.addWeighted(out, 0.35, colored, 0.65, 0)[active]
    return out


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    out = to_bgr_panel(image)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(panels: Sequence[Tuple[str, np.ndarray]], cols: int = 3) -> np.ndarray:
    if not panels:
        raise ValueError("No panels for contact sheet.")
    labelled = [label_panel(img, label) for label, img in panels]
    panel_h, panel_w = labelled[0].shape[:2]
    resized = [
        cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        if panel.shape[:2] != (panel_h, panel_w)
        else panel
        for panel in labelled
    ]
    rows = int(math.ceil(len(resized) / cols))
    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    lines = []
    for row_idx in range(rows):
        row = resized[row_idx * cols : (row_idx + 1) * cols]
        while len(row) < cols:
            row.append(blank.copy())
        lines.append(np.hstack(row))
    return np.vstack(lines)


def save_preprocess_debug_sheet(
    debug_dir: Path,
    frame_idx: int,
    frame: np.ndarray,
    stages: Dict[str, np.ndarray],
    candidates: Sequence[Candidate],
) -> None:
    candidate_overlay = frame.copy()
    for candidate in candidates[:30]:
        cv2.rectangle(
            candidate_overlay,
            (candidate.x, candidate.y),
            (candidate.x + candidate.w, candidate.y + candidate.h),
            (0, 255, 255),
            1,
        )
        cv2.circle(candidate_overlay, (int(round(candidate.cx)), int(round(candidate.cy))), 2, (0, 0, 255), -1)

    panels = [
        ("raw", frame),
        ("gray_preprocessed", stages["gray"]),
        ("valid_roi_green", mask_overlay(frame, stages["valid_mask"], (0, 180, 0))),
        ("exclude_red", mask_overlay(frame, stages["exclude_mask"], (0, 0, 220))),
        ("diff_prev", stages["diff_prev"]),
        ("diff_next", stages["diff_next"]),
        ("diff_union", stages["diff_union"]),
        ("diff_inter", stages["diff_inter"]),
        ("mog", stages["mog"]),
        ("motion_fused", stages["motion_fused"]),
        ("after_roi", stages["after_roi"]),
        ("after_exclusion", stages["after_exclusion"]),
        ("motion_clean", stages["motion_clean"]),
        ("candidates", candidate_overlay),
    ]
    sheet = make_contact_sheet(panels, cols=3)
    cv2.imwrite(str(debug_dir / f"{frame_idx:06d}_preprocess.jpg"), sheet)


def preprocess_stats_row(
    frame_idx: int,
    stages: Dict[str, np.ndarray],
    candidate_count: int,
) -> dict:
    row = {"frame_idx": frame_idx, "candidate_count": int(candidate_count)}
    for key, value in stages.items():
        if value.ndim == 2:
            row[f"{key}_pixels"] = int(np.count_nonzero(value))
    before_roi = max(1, row.get("motion_fused_pixels", 0))
    before_exclusion = max(1, row.get("after_roi_pixels", 0))
    row["roi_keep_ratio"] = row.get("after_roi_pixels", 0) / before_roi
    row["exclusion_keep_ratio"] = row.get("after_exclusion_pixels", 0) / before_exclusion
    row["morphology_keep_ratio"] = row.get("motion_clean_pixels", 0) / max(1, row.get("after_exclusion_pixels", 0))
    return row


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    debug_dir = output_dir / "debug_frames"
    preprocess_debug_dir = output_dir / "preprocess_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_preprocess:
        preprocess_debug_dir.mkdir(parents=True, exist_ok=True)

    all_frames = list_frames(args.frames_dir)
    end = len(all_frames) if args.end < 0 else min(args.end, len(all_frames))
    frame_paths = all_frames[max(0, args.start) : end]
    if len(frame_paths) < 3:
        raise ValueError("Need at least 3 frames for adjacent-frame differencing.")

    frames = [read_frame(p, args.scale) for p in frame_paths]
    grays = [preprocess_gray(f, use_clahe=not args.no_clahe) for f in frames]
    height, width = grays[0].shape[:2]
    if args.no_roi_mask:
        valid_mask = np.full((height, width), 255, dtype=np.uint8)
    else:
        valid_mask = load_mask(args.mask, (height, width), args.scale, pad_x=args.roi_pad_x, pad_y=args.roi_pad_y)
    person_boxes = {} if args.no_player_exclusion else load_person_boxes(args.person_boxes_csv, args.scale)
    manual_exclude_rects = [] if args.no_manual_exclusion else parse_exclude_rects(args.exclude_rect or [], args.scale)
    
    # Pre-compute the court lines exclusion mask
    court_lines_mask = load_court_lines_mask(args.roi_json, (height, width), args.scale, args.line_mask_thickness)
    
    explicit_preprocess_frames = parse_frame_set(args.debug_preprocess_frames)

    mog = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=18, detectShadows=False)
    mog_masks = [mog.apply(frame, learningRate=0.02) for frame in frames]

    heatmap = CandidateHeatmap((height, width))
    candidates_by_frame: Dict[int, List[Candidate]] = {}
    motion_by_frame: Dict[int, np.ndarray] = {}
    preprocess_stats: List[dict] = []
    frame_indices = [args.start + i for i in range(1, len(frames) - 1)]

    for local_idx in range(1, len(frames) - 1):
        global_idx = args.start + local_idx
        exclude_mask = exclusion_mask_for_boxes((height, width), person_boxes.get(global_idx, []))
        exclude_mask = add_rects_to_mask(exclude_mask, manual_exclude_rects)
        
        # Add court lines to the exclusion mask
        exclude_mask = cv2.bitwise_or(exclude_mask, court_lines_mask)
        
        # Calculate Sobel edge magnitude for the current frame

        gray_c = grays[local_idx]
        grad_x = cv2.Sobel(gray_c, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_c, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(grad_x, grad_y)
        
        candidates, motion, stages = extract_candidates(
            global_idx,
            frames[local_idx],
            grays[local_idx - 1],
            grays[local_idx],
            grays[local_idx + 1],
            mog_masks[local_idx],
            valid_mask,
            exclude_mask,
            edge_mag,
            args,
            heatmap=heatmap,
        )
        heatmap.update(candidates)
        candidates_by_frame[global_idx] = candidates
        motion_by_frame[global_idx] = motion
        if args.debug_preprocess:
            preprocess_stats.append(preprocess_stats_row(global_idx, stages, len(candidates)))
            if should_dump_preprocess(global_idx, args, explicit_preprocess_frames):
                save_preprocess_debug_sheet(
                    preprocess_debug_dir,
                    global_idx,
                    frames[local_idx],
                    stages,
                    candidates,
                )

    track = run_kalman_tracker(
        candidates_by_frame,
        frame_indices,
        args,
        (height, width),
    )
    track_by_frame = {p.frame_idx: p for p in track}
    bounces = detect_bounces(track, args.fps)

    write_csv(output_dir / "candidates.csv", (asdict(c) for values in candidates_by_frame.values() for c in values))
    write_csv(output_dir / "tracks.csv", (asdict(p) for p in track))
    write_csv(output_dir / "bounces.csv", (asdict(b) for b in bounces))
    if args.debug_preprocess:
        write_csv(output_dir / "preprocess_stats.csv", preprocess_stats)

    video_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_dir / "overlay_tracking.mp4"),
            fourcc,
            args.fps,
            (width, height),
        )

    for local_idx, frame in enumerate(frames):
        global_idx = args.start + local_idx
        debug = draw_debug_frame(
            frame,
            global_idx,
            candidates_by_frame.get(global_idx, []),
            track_by_frame.get(global_idx),
            bounces,
            motion_by_frame.get(global_idx),
        )
        if video_writer is not None:
            video_writer.write(debug)
        if args.debug_every > 0 and global_idx % args.debug_every == 0:
            cv2.imwrite(str(debug_dir / f"{global_idx:06d}.jpg"), debug)

    if video_writer is not None:
        video_writer.release()

    print(f"frames: {len(frame_paths)}")
    print(f"candidate frames: {sum(1 for v in candidates_by_frame.values() if v)}")
    print(f"track points: {len(track)}")
    print(f"bounces: {len(bounces)}")
    print(f"output: {output_dir}")
    if args.debug_preprocess:
        print(f"preprocess debug: {preprocess_debug_dir}")
        print(f"preprocess stats: {output_dir / 'preprocess_stats.csv'}")


if __name__ == "__main__":
    main()
