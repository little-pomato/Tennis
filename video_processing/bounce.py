import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from player_detector import PlayerDetector
from player_tracker import TwoPlayerTracker

from mask import (
    build_net_distance_map,
    build_search_mask,
    estimate_net_y,
    load_json,
)
from hit import (
    build_ball_y_signal_from_candidates,
    detect_hit_intervals_by_clear_high_low_segments,
)


# ============================================================
# Bounce detector
# ------------------------------------------------------------
# 核心邏輯：
# 1. 全場搜尋，不使用 y 方向「局部最低點 / curvature」假設。
# 2. 用網子把候選點切成 far / near 兩側，各自使用不同位移尺度。
# 3. 事件核心改為 displacement dip：
#       在某一幀附近，前後一步 / 兩步位移都偏小，且前後支撐足夠。
# 4. 保留球員排除、net 附近懲罰、5 幀支撐、peak + refine + NMS 流程。
#
# 輸入：
#   dataset/<video_name>/frames/*.jpg|png
#   dataset/<video_name>/valid_mask.png
#   dataset/<video_name>/roi_config.json  (可選)
#
# 輸出：
#   dataset/<video_name>/bounce_detector/
#       - frame_scores.csv
#       - bounce_candidates.csv
#       - bounce_events.csv
#       - scan_stats.csv
#       - debug/*.png
# ============================================================


@dataclass
class BlobCandidate:
    frame_idx: int
    cx: float
    cy: float
    x: int
    y: int
    w: int
    h: int
    area: float
    circ: float
    mean_diff: float
    score_blob: float
    dist_to_net: float = 9999.0
    net_penalty: float = 0.0

    court_side: str = "unknown"
    side_neighbor_radius: float = 0.0
    side_support_radius: float = 0.0

    has_prev: int = 0
    has_next: int = 0
    has_prev2: int = 0
    has_next2: int = 0

    prev_dx: float = 9999.0
    prev_dy: float = 9999.0
    next_dx: float = 9999.0
    next_dy: float = 9999.0
    prev2_dx: float = 9999.0
    prev2_dy: float = 9999.0
    next2_dx: float = 9999.0
    next2_dy: float = 9999.0

    prev_step: float = 9999.0
    next_step: float = 9999.0
    prev2_step: float = 9999.0
    next2_step: float = 9999.0

    symmetry_bonus1: float = 0.0
    symmetry_bonus2: float = 0.0
    compact_bonus1: float = 0.0
    compact_bonus2: float = 0.0
    speed_dip_1: float = 0.0
    speed_dip_2: float = 0.0
    dip_ratio: float = 9999.0
    dip_ratio_bonus: float = 0.0
    support_5f: int = 0
    support_local1: float = 0.0
    support_local2: float = 0.0
    long_track_gate: float = 0.0
    track_score_1: float = 0.0
    track_score_2: float = 0.0
    score_event: float = 0.0

    dist_to_human: float = 9999.0
    human_penalty: float = 0.0

    is_interpolated: int = 0
    interp_source: str = ""


@dataclass
class BounceEvent:
    peak_frame: int
    peak_score: float
    cx: float
    cy: float
    court_side: str
    support_5f: int
    has_prev: int
    has_next: int
    prev_step: float
    next_step: float
    speed_dip_1: float
    speed_dip_2: float
    dip_ratio_bonus: float
    source_blob_score: float
    window_start: int
    window_end: int
    coarse_cx: float = 0.0
    coarse_cy: float = 0.0
    refined_cx: float = 0.0
    refined_cy: float = 0.0
    refined_frame: int = -1
    refine_score: float = 0.0
    track_fit_cx: float = 0.0
    track_fit_cy: float = 0.0
    track_fit_frame: float = -1.0
    motion_peak_cx: float = 0.0
    motion_peak_cy: float = 0.0
    motion_score: float = 0.0
    track_score_refine: float = 0.0
    local_track_count_pre: int = 0
    local_track_count_post: int = 0


@dataclass
class SideParams:
    name: str
    neighbor_radius_1: float
    neighbor_radius_2: float
    support_radius: float
    step_ref_1: float
    step_ref_2: float
    compact_ref_1: float
    compact_ref_2: float
    symmetry_ref_1: float
    symmetry_ref_2: float


@dataclass
class VoteAnchor:
    frame_idx: int
    cx: float
    cy: float
    court_side: str
    score: float
    support_frames: int
    member_count: int
    
@dataclass
class WindowPoint:
    rel_idx: int
    frame_idx: int
    x: float
    y: float
    score_blob: float
    score_event: float
    track_score_1: float
    track_score_2: float
    support_5f: int
    court_side: str
    distance_to_pred: float
    picked_by: str      # "direct" / "pred" / "fallback" / "missing"
    is_valid: int

@dataclass
class PostCheckResult:
    original_peak_frame: int
    refined_peak_frame: int
    chosen_side: str
    valid_count: int
    side_consistency: float
    quality_mean: float
    smoothness_score: float
    gap_penalty: float
    sequence_score: float
    best_bounce_score: float
    refined_x: float
    refined_y: float
    keep_event: int
    points: List[WindowPoint]
    
@dataclass
class Stage2CheckResult:
    original_peak_frame: int
    refined_peak_frame: int
    chosen_side: str

    valid_count: int
    side_consistency: float
    quality_mean: float
    smoothness_score: float
    gap_penalty: float

    direction_change_score: float
    speed_dip_score: float
    event_likeness_score: float
    split_fit_score: float
    final_score: float

    refined_x: float
    refined_y: float
    keep_event: int
    points: List[WindowPoint]


def set_video_path(video_path: Path):
    video_name = video_path.stem
    base = Path("dataset") / video_name
    return {
        "VIDEO_PATH": video_path,
        "VIDEO_NAME": video_name,
        "BASE": base,
        "FRAMES_DIR": base / "frames",
        "MASK_PATH": base / "valid_mask.png",
        "ROI_JSON": base / "roi_config.json",
        "OUT_DIR": base / "bounce_detector",
    }


def load_frames_gray(frames_dir: Path, scale: float) -> Tuple[List[Path], List[np.ndarray], Tuple[int, int]]:
    frame_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    grays: List[np.ndarray] = []
    target_hw = None
    for fp in frame_paths:
        im = cv2.imread(str(fp))
        if im is None:
            raise ValueError(f"Cannot read frame: {fp}")
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        if target_hw is None:
            target_hw = gray.shape[:2]

    return frame_paths, grays, target_hw


def load_valid_mask(mask_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"valid_mask.png not found at {mask_path}")
    H, W = target_hw
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)


def circularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri + 1e-6))


def scale_yolo_boxes_for_bounce(
    boxes: List[Tuple[int, int, int, int, float]],
    scale: float,
    target_hw: Tuple[int, int],
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert original-frame YOLO boxes to the coordinate system used by bounce.py.

    vote_action should use original-frame boxes for crop geometry.
    bounce.py may run with --scale, so masks/tracker need scaled boxes.
    """
    if scale == 1.0:
        return [(int(x1), int(y1), int(x2), int(y2), float(conf)) for x1, y1, x2, y2, conf in boxes]

    H, W = target_hw
    out = []
    for x1, y1, x2, y2, conf in boxes:
        sx1 = int(round(float(x1) * float(scale)))
        sy1 = int(round(float(y1) * float(scale)))
        sx2 = int(round(float(x2) * float(scale)))
        sy2 = int(round(float(y2) * float(scale)))
        sx1 = max(0, min(sx1, W - 1))
        sy1 = max(0, min(sy1, H - 1))
        sx2 = max(0, min(sx2, W - 1))
        sy2 = max(0, min(sy2, H - 1))
        if sx2 > sx1 and sy2 > sy1:
            out.append((sx1, sy1, sx2, sy2, float(conf)))
    return out


def append_yolo_raw_box_rows(
    rows: List[dict],
    frame_idx: int,
    boxes_orig: List[Tuple[int, int, int, int, float]],
    img_w: int,
    img_h: int,
    scale: float,
) -> None:
    """Store original-frame YOLO boxes for vote_action shared cache."""
    for det_id, (x1, y1, x2, y2, conf) in enumerate(boxes_orig):
        rows.append({
            "frame_idx": int(frame_idx),
            "det_id": int(det_id),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "conf": float(conf),
            "img_w": int(img_w),
            "img_h": int(img_h),
            "scale_used_by_bounce": float(scale),
        })

def build_player_exclusion_masks_with_tracker(
    frame_bgr,
    search_mask,
    detector,
    tracker,
    net_y=None,
    raw_boxes=None,
    near_core_pad_ratio=0.08,
    near_soft_pad_ratio=0.28,
    far_core_pad_ratio=0.03,
    far_soft_pad_ratio=0.16,
):
    H, W = search_mask.shape[:2]
    core_mask = np.zeros((H, W), np.uint8)
    soft_mask = np.zeros((H, W), np.uint8)

    # raw_boxes can be supplied by a shared YOLO cache.
    # If not supplied, keep the old behavior and run detector here.
    if raw_boxes is None:
        if detector is None:
            raise ValueError("detector is None but raw_boxes was not provided")
        raw_boxes = detector.detect(frame_bgr)

    tracked_players = tracker.update(raw_boxes)

    kept_boxes = []
    for item in tracked_players:
        name, box = item
        x1, y1, x2, y2 = box[:4]

        bw = x2 - x1
        bh = y2 - y1
        cy_box = 0.5 * (y1 + y2)

        is_far = (net_y is not None and cy_box < net_y)

        if is_far:
            core_ratio = far_core_pad_ratio
            soft_ratio = far_soft_pad_ratio
        else:
            core_ratio = near_core_pad_ratio
            soft_ratio = near_soft_pad_ratio

        core_pad_w = max(2, int(bw * core_ratio))
        core_pad_h = max(2, int(bh * core_ratio))
        soft_pad_w = max(4, int(bw * soft_ratio))
        soft_pad_h = max(4, int(bh * soft_ratio))

        cx1 = max(0, int(x1) - core_pad_w)
        cy1 = max(0, int(y1) - core_pad_h)
        cx2 = min(W - 1, int(x2) + core_pad_w)
        cy2 = min(H - 1, int(y2) + core_pad_h)

        sx1 = max(0, int(x1) - soft_pad_w)
        sy1 = max(0, int(y1) - soft_pad_h)
        sx2 = min(W - 1, int(x2) + soft_pad_w)
        sy2 = min(H - 1, int(y2) + soft_pad_h)

        if sx2 > sx1 and sy2 > sy1:
            cv2.rectangle(soft_mask, (sx1, sy1), (sx2, sy2), 255, -1)
        if cx2 > cx1 and cy2 > cy1:
            cv2.rectangle(core_mask, (cx1, cy1), (cx2, cy2), 255, -1)

        kept_boxes.append((sx1, sy1, sx2, sy2, 1.0))

    soft_mask = cv2.bitwise_and(soft_mask, cv2.bitwise_not(core_mask))
    core_mask = cv2.bitwise_and(core_mask, search_mask)
    soft_mask = cv2.bitwise_and(soft_mask, search_mask)

    return core_mask, soft_mask, kept_boxes, tracked_players, raw_boxes


def build_motion_triplet(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    gray_next: np.ndarray,
    search_mask: np.ndarray,
    diff_th: int,
    blur_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diff_prev = cv2.absdiff(gray_curr, gray_prev)
    diff_next = cv2.absdiff(gray_next, gray_curr)

    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        diff_prev = cv2.GaussianBlur(diff_prev, (blur_ksize, blur_ksize), 0)
        diff_next = cv2.GaussianBlur(diff_next, (blur_ksize, blur_ksize), 0)

    _, fg_prev = cv2.threshold(diff_prev, diff_th, 255, cv2.THRESH_BINARY)
    _, fg_next = cv2.threshold(diff_next, diff_th, 255, cv2.THRESH_BINARY)

    fg_prev = cv2.bitwise_and(fg_prev, search_mask)
    fg_next = cv2.bitwise_and(fg_next, search_mask)

    fg_prev = cv2.morphologyEx(fg_prev, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    fg_next = cv2.morphologyEx(fg_next, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    fg_prev = cv2.morphologyEx(fg_prev, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    fg_next = cv2.morphologyEx(fg_next, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    fg_union = cv2.bitwise_or(fg_prev, fg_next)
    fg_inter = cv2.bitwise_and(fg_prev, fg_next)
    return diff_prev, diff_next, fg_union, fg_inter

def extract_blob_candidates(
    frame_idx: int,
    fg_union: np.ndarray,
    fg_inter: np.ndarray,
    diff_prev: np.ndarray,
    diff_next: np.ndarray,
    min_area: int,
    max_area: int,
    max_wh: int,
    inter_weight: float = 0.25,
    net_dist_map: Optional[np.ndarray] = None,
    net_penalty_radius: float = 26.0,
    net_penalty_weight: float = 0.35,
    human_dist_map: Optional[np.ndarray] = None,
    human_penalty_radius: float = 18.0,
    human_penalty_weight: float = 0.28,
) -> List[BlobCandidate]:
    candidate_mask = fg_inter.copy()

    inter_pixels = int(np.count_nonzero(fg_inter))
    union_pixels = int(np.count_nonzero(fg_union))

    # fg_inter 太稀疏時，混入 fg_union
    if inter_pixels <= 6 or (union_pixels > 0 and inter_pixels / max(1, union_pixels) < 0.18):
        candidate_mask = cv2.bitwise_or(candidate_mask, fg_union)

    cnts, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[BlobCandidate] = []

    diff_mix = ((diff_prev.astype(np.float32) + diff_next.astype(np.float32)) * 0.5).astype(np.uint8)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0 or w > max_wh or h > max_wh:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            cx = x + w / 2.0
            cy = y + h / 2.0
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        circ = circularity(cnt)
        diff_local = diff_mix[y:y + h, x:x + w]
        mean_diff = float(np.mean(diff_local)) if diff_local.size > 0 else 0.0

        inter_local = fg_inter[y:y + h, x:x + w]
        inter_ratio = float(np.count_nonzero(inter_local)) / float(max(1, w * h))

        area_score = max(0.0, 1.0 - abs(area - 9.0) / 24.0)
        circ_score = np.clip(circ, 0.0, 1.0)
        diff_score = np.clip(mean_diff / 255.0, 0.0, 1.0)
        inter_score = min(1.0, inter_ratio * 4.0)

        blob_score = (
            0.48 * area_score
            + 0.32 * circ_score
            + 0.12 * diff_score
            + 0.08 * inter_score
        )

        dist_to_net = 9999.0
        net_penalty = 0.0
        if net_dist_map is not None:
            ix = int(np.clip(round(cx), 0, net_dist_map.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, net_dist_map.shape[0] - 1))
            dist_to_net = float(net_dist_map[iy, ix])
            if net_penalty_radius > 0:
                net_penalty = max(0.0, 1.0 - dist_to_net / float(net_penalty_radius))
                blob_score *= (1.0 - net_penalty_weight * net_penalty)
                
        dist_to_human = 9999.0
        human_penalty = 0.0
        if human_dist_map is not None:
            ix = int(np.clip(round(cx), 0, human_dist_map.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, human_dist_map.shape[0] - 1))
            dist_to_human = float(human_dist_map[iy, ix])
            if human_penalty_radius > 0:
                human_penalty = max(0.0, 1.0 - dist_to_human / float(human_penalty_radius))
                blob_score *= (1.0 - human_penalty_weight * human_penalty)

        blob_score = min(1.0, max(0.0, blob_score))

        out.append(
            BlobCandidate(
                frame_idx=frame_idx,
                cx=float(cx), cy=float(cy),
                x=int(x), y=int(y), w=int(w), h=int(h),
                area=float(area), circ=float(circ),
                mean_diff=float(mean_diff),
                score_blob=float(blob_score),
                dist_to_net=float(dist_to_net),
                net_penalty=float(net_penalty),
                dist_to_human=float(dist_to_human),
                human_penalty=float(human_penalty),
            )
        )
    return out


def nearest_candidate(cands: List[BlobCandidate], cx: float, cy: float, radius: float) -> Optional[BlobCandidate]:
    best = None
    best_d2 = radius * radius
    for cand in cands:
        dx = cand.cx - cx
        dy = cand.cy - cy
        d2 = dx * dx + dy * dy
        if d2 <= best_d2:
            best_d2 = d2
            best = cand
    return best

def _best_cand_score_for_interp(c: BlobCandidate) -> float:
    return float(
        0.55 * c.score_blob
        + 0.25 * c.track_score_1
        + 0.20 * c.track_score_2
    )


def _inside_mask(mask: np.ndarray, x: float, y: float) -> bool:
    H, W = mask.shape[:2]
    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or ix >= W or iy < 0 or iy >= H:
        return False
    return bool(mask[iy, ix] > 0)


def interpolate_single_frame_gaps(
    candidates_by_frame: List[List[BlobCandidate]],
    search_mask: np.ndarray,
    net_y: Optional[float],
    near_neighbor_radius: float = 22.0,
    far_neighbor_radius: float = 8.0,
    max_interp_per_frame: int = 1,
    debug: bool = False,
) -> List[List[BlobCandidate]]:
    n = len(candidates_by_frame)

    for i in range(1, n - 1):
        # 只有完全沒 candidate 的幀才補
        if len(candidates_by_frame[i]) > 0:
            continue

        prev_list = candidates_by_frame[i - 1]
        next_list = candidates_by_frame[i + 1]

        if len(prev_list) == 0 or len(next_list) == 0:
            continue

        proposals = []

        for p in prev_list:
            p_side = classify_court_side(p.cy, net_y)
            sp = get_side_params(p_side, near_neighbor_radius, far_neighbor_radius)

            # 在 i+1 找同 side 的對應點
            q = nearest_candidate(next_list, p.cx, p.cy, sp.neighbor_radius_2)
            if q is None:
                continue

            q_side = classify_court_side(q.cy, net_y)
            if q_side != p_side:
                continue

            # 前後兩點如果差太遠，不補
            d = float(np.hypot(q.cx - p.cx, q.cy - p.cy))
            max_bridge = sp.neighbor_radius_2 * 1.15
            if d > max_bridge:
                continue

            # 線性中點
            cx = 0.5 * (p.cx + q.cx)
            cy = 0.5 * (p.cy + q.cy)

            # 必須落在 search mask 內
            if not _inside_mask(search_mask, cx, cy):
                continue

            # side 再檢查一次，避免跨 net 補錯
            interp_side = classify_court_side(cy, net_y)
            if interp_side != p_side:
                continue

            # 避免補到人附近：如果前後兩點都已經很靠近人，這種不要補
            if min(p.dist_to_human, q.dist_to_human) < 8.0:
                continue

            # 如果前後兩點都很靠近 net，也不要亂補
            if min(p.dist_to_net, q.dist_to_net) < 4.0:
                continue

            # 補點分數要低，只拿來支撐，不要搶主角
            score_blob = 0.5 * min(p.score_blob, q.score_blob)
            score_blob = float(np.clip(score_blob * 0.55, 0.08, 0.28))

            # 估一個小 bbox
            w = max(2, int(round(0.5 * (p.w + q.w))))
            h = max(2, int(round(0.5 * (p.h + q.h))))
            x = int(round(cx - w / 2.0))
            y = int(round(cy - h / 2.0))

            proposal_score = (
                0.45 * min(p.score_blob, q.score_blob)
                + 0.25 * (1.0 - d / max(max_bridge, 1e-6))
                + 0.15 * min(p.support_5f, q.support_5f) / 5.0
                + 0.15 * min(_best_cand_score_for_interp(p), _best_cand_score_for_interp(q))
            )

            new_cand = BlobCandidate(
                frame_idx=i,
                cx=float(cx),
                cy=float(cy),
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=float(max(3.0, 0.5 * (p.area + q.area))),
                circ=float(min(p.circ, q.circ)),
                mean_diff=float(0.5 * (p.mean_diff + q.mean_diff)),
                score_blob=float(score_blob),
                dist_to_net=float(min(p.dist_to_net, q.dist_to_net)),
                net_penalty=float(max(p.net_penalty, q.net_penalty)),
                dist_to_human=float(min(p.dist_to_human, q.dist_to_human)),
                human_penalty=float(max(p.human_penalty, q.human_penalty)),
                court_side=str(interp_side),
                is_interpolated=1,
                interp_source=f"{i-1}->{i+1}",
            )

            proposals.append((proposal_score, new_cand))

        if not proposals:
            continue

        proposals.sort(key=lambda z: z[0], reverse=True)

        kept = []
        for _, cand in proposals:
            too_close = False
            for old in kept:
                if np.hypot(cand.cx - old.cx, cand.cy - old.cy) <= 5.0:
                    too_close = True
                    break
            if not too_close:
                kept.append(cand)
            if len(kept) >= max_interp_per_frame:
                break

        candidates_by_frame[i].extend(kept)

        if debug and kept:
            print(
                f"[interp] frame={i} added={len(kept)} "
                + ", ".join(
                    f"({c.cx:.1f},{c.cy:.1f},side={c.court_side},src={c.interp_source})"
                    for c in kept
                )
            )

    return candidates_by_frame

def count_support_5f(candidates_by_frame: List[List[BlobCandidate]], idx: int, cx: float, cy: float, radius: float) -> int:
    count = 0
    r2 = radius * radius
    start = max(0, idx - 2)
    end = min(len(candidates_by_frame) - 1, idx + 2)
    for j in range(start, end + 1):
        found = False
        for cand in candidates_by_frame[j]:
            dx = cand.cx - cx
            dy = cand.cy - cy
            if dx * dx + dy * dy <= r2:
                found = True
                break
        if found:
            count += 1
    return count


def get_side_params(side: str, near_neighbor_radius: float, far_neighbor_radius: float) -> SideParams:
    if side == "near":
        return SideParams(
            name="near",
            neighbor_radius_1=near_neighbor_radius,
            neighbor_radius_2=near_neighbor_radius * 1.85,
            support_radius=near_neighbor_radius + 5.0,
            step_ref_1=max(2.0, near_neighbor_radius * 0.60),
            step_ref_2=max(3.0, near_neighbor_radius * 1.15),
            compact_ref_1=max(2.0, near_neighbor_radius * 1.00),
            compact_ref_2=max(4.0, near_neighbor_radius * 1.90),
            symmetry_ref_1=max(2.0, near_neighbor_radius * 0.55),
            symmetry_ref_2=max(3.0, near_neighbor_radius * 0.90),
        )
    return SideParams(
        name="far",
        neighbor_radius_1=far_neighbor_radius,
        neighbor_radius_2=far_neighbor_radius * 2.30,
        support_radius=far_neighbor_radius + 7.0,
        step_ref_1=max(0.8, far_neighbor_radius * 0.45),
        step_ref_2=max(1.4, far_neighbor_radius * 0.95),
        compact_ref_1=max(2.0, far_neighbor_radius * 1.25),
        compact_ref_2=max(3.5, far_neighbor_radius * 2.30),
        symmetry_ref_1=max(1.8, far_neighbor_radius * 0.80),
        symmetry_ref_2=max(2.4, far_neighbor_radius * 1.30),
    )


def classify_court_side(cy: float, net_y: Optional[float]) -> str:
    if net_y is None:
        return "near"
    return "near" if cy >= net_y else "far"


def _step_bonus(max_step: float, ref: float) -> float:
    if ref <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - max_step / ref, 0.0, 1.0))


def _symmetry_bonus(a: float, b: float, ref: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    if ref <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - abs(a - b) / ref, 0.0, 1.0))


def _compact_bonus(max_step: float, ref: float) -> float:
    if ref <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - max_step / ref, 0.0, 1.0))

def is_likely_human_motion_artifact(c: BlobCandidate) -> bool:
    """
    只抓「像人身體雜訊撐出來的假 bounce」。
    不抓一般靠近人的真球。

    條件設計：
    - human_penalty 很高：代表在人的 soft mask 裡或非常靠近人
    - score_blob 低：代表它本身不像清楚的球 blob
    - score_event 明顯高於 score_blob：代表分數主要是靠 track/support 撐起來
    """
    return (
        c.human_penalty >= 0.85
        and c.score_blob < 0.55
        and c.score_event >= c.score_blob + 0.25
    )

def score_bounce_events(
    candidates_by_frame: List[List[BlobCandidate]],
    net_y: Optional[float],
    near_neighbor_radius: float = 22.0,
    far_neighbor_radius: float = 12.0,
) -> Tuple[List[List[BlobCandidate]], np.ndarray]:
    frame_scores = np.zeros(len(candidates_by_frame), dtype=np.float32)
    n_frames = len(candidates_by_frame)

    for i in range(n_frames):
        if not candidates_by_frame[i]:
            continue

        prev_list = candidates_by_frame[i - 1] if i - 1 >= 0 else []
        next_list = candidates_by_frame[i + 1] if i + 1 < n_frames else []
        prev2_list = candidates_by_frame[i - 2] if i - 2 >= 0 else []
        next2_list = candidates_by_frame[i + 2] if i + 2 < n_frames else []
        event_scores = []

        for cand in candidates_by_frame[i]:
            cand.court_side = classify_court_side(cand.cy, net_y)
            sp = get_side_params(cand.court_side, near_neighbor_radius, far_neighbor_radius)
            cand.side_neighbor_radius = float(sp.neighbor_radius_1)
            cand.side_support_radius = float(sp.support_radius)

            prev_c = nearest_candidate(prev_list, cand.cx, cand.cy, sp.neighbor_radius_1)
            next_c = nearest_candidate(next_list, cand.cx, cand.cy, sp.neighbor_radius_1)
            prev2_c = nearest_candidate(prev2_list, cand.cx, cand.cy, sp.neighbor_radius_2)
            next2_c = nearest_candidate(next2_list, cand.cx, cand.cy, sp.neighbor_radius_2)

            cand.has_prev = int(prev_c is not None)
            cand.has_next = int(next_c is not None)
            cand.has_prev2 = int(prev2_c is not None)
            cand.has_next2 = int(next2_c is not None)

            if prev_c is not None:
                cand.prev_dx = float(cand.cx - prev_c.cx)
                cand.prev_dy = float(cand.cy - prev_c.cy)
                cand.prev_step = float(np.hypot(cand.prev_dx, cand.prev_dy))
            if next_c is not None:
                cand.next_dx = float(next_c.cx - cand.cx)
                cand.next_dy = float(next_c.cy - cand.cy)
                cand.next_step = float(np.hypot(cand.next_dx, cand.next_dy))
            if prev2_c is not None:
                cand.prev2_dx = float(cand.cx - prev2_c.cx)
                cand.prev2_dy = float(cand.cy - prev2_c.cy)
                cand.prev2_step = float(np.hypot(cand.prev2_dx, cand.prev2_dy))
            if next2_c is not None:
                cand.next2_dx = float(next2_c.cx - cand.cx)
                cand.next2_dy = float(next2_c.cy - cand.cy)
                cand.next2_step = float(np.hypot(cand.next2_dx, cand.next2_dy))

            if prev_c is not None and next_c is not None:
                max_step1 = max(cand.prev_step, cand.next_step)
                cand.speed_dip_1 = _step_bonus(max_step1, sp.step_ref_1)
                cand.symmetry_bonus1 = _symmetry_bonus(cand.prev_step, cand.next_step, sp.symmetry_ref_1)
                cand.compact_bonus1 = _compact_bonus(max_step1, sp.compact_ref_1)
                cand.support_local1 = float(min(1.0, 0.5 * cand.has_prev + 0.5 * cand.has_next))
                cand.track_score_1 = float(
                    0.52 * cand.speed_dip_1
                    + 0.24 * cand.symmetry_bonus1
                    + 0.14 * cand.compact_bonus1
                    + 0.10 * cand.support_local1
                )

            if prev2_c is not None and next2_c is not None:
                max_step2 = max(cand.prev2_step, cand.next2_step)
                cand.speed_dip_2 = _step_bonus(max_step2, sp.step_ref_2)
                cand.symmetry_bonus2 = _symmetry_bonus(cand.prev2_step, cand.next2_step, sp.symmetry_ref_2)
                cand.compact_bonus2 = _compact_bonus(max_step2, sp.compact_ref_2)
                denom = max(1e-6, cand.prev2_step + cand.next2_step)
                numer = cand.prev_step + cand.next_step if np.isfinite(cand.prev_step) and np.isfinite(cand.next_step) else denom
                cand.dip_ratio = float(numer / denom)
                cand.dip_ratio_bonus = float(np.clip(1.0 - cand.dip_ratio, 0.0, 1.0))
                cand.support_local2 = float(min(1.0, 0.5 * cand.has_prev2 + 0.5 * cand.has_next2))
                cand.track_score_2 = float(
                    0.40 * cand.speed_dip_2
                    + 0.18 * cand.symmetry_bonus2
                    + 0.14 * cand.compact_bonus2
                    + 0.18 * cand.dip_ratio_bonus
                    + 0.10 * cand.support_local2
                )

            cand.support_5f = count_support_5f(candidates_by_frame, i, cand.cx, cand.cy, radius=sp.support_radius)
            cand.long_track_gate = float(min(1.0, 0.5 * cand.has_prev2 + 0.5 * cand.has_next2))

            if cand.court_side == "far":
                linger_bonus = 0.0
                if cand.support_5f >= 4 and (cand.has_prev or cand.has_next):
                    linger_bonus = 0.06
                cand.score_event = float(
                    0.22 * cand.score_blob
                    + 0.08 * cand.has_prev
                    + 0.08 * cand.has_next
                    + 0.05 * cand.has_prev2
                    + 0.05 * cand.has_next2
                    + 0.14 * cand.track_score_1
                    + 0.12 * cand.track_score_2
                    + 0.16 * min(1.0, cand.support_5f / 5.0)
                    + 0.08 * cand.long_track_gate
                    + linger_bonus
                    - 0.02 * cand.net_penalty
                )
            else:
                cand.score_event = float(
                    0.17 * cand.score_blob
                    + 0.07 * cand.has_prev
                    + 0.07 * cand.has_next
                    + 0.09 * cand.has_prev2
                    + 0.09 * cand.has_next2
                    + 0.18 * cand.track_score_1
                    + 0.20 * cand.track_score_2
                    + 0.09 * min(1.0, cand.support_5f / 5.0)
                    + 0.08 * cand.long_track_gate
                    - 0.04 * cand.net_penalty
                )
            cand.score_event = max(0.0, cand.score_event)

            if cand.is_interpolated == 1:
                cand.score_event *= 0.72

            event_scores.append(cand.score_event)

        if event_scores:
            frame_scores[i] = float(max(event_scores))

    return candidates_by_frame, frame_scores


def smooth_1d(x: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize <= 1:
        return x.copy()
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones(ksize, np.float32) / float(ksize)
    return np.convolve(x, kernel, mode="same")


def find_peaks_1d(x: np.ndarray, min_score: float, min_gap: int) -> List[int]:
    peaks: List[int] = []
    last_peak = -10**9
    for i in range(1, len(x) - 1):
        if x[i] < min_score:
            continue
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if i - last_peak >= min_gap:
                peaks.append(i)
                last_peak = i
            elif x[i] > x[last_peak]:
                peaks[-1] = i
                last_peak = i
    return peaks


def build_temporal_spatial_vote_signal(
    candidates_by_frame: List[List[BlobCandidate]],
    near_neighbor_radius: float,
    far_neighbor_radius: float,
    vote_radius: int = 3,
) -> Tuple[np.ndarray, List[Optional[VoteAnchor]], np.ndarray]:
    n_frames = len(candidates_by_frame)
    vote_raw = np.zeros(n_frames, dtype=np.float32)
    vote_support = np.zeros(n_frames, dtype=np.float32)
    anchors: List[Optional[VoteAnchor]] = [None] * n_frames

    for t in range(n_frames):
        start = max(0, t - vote_radius)
        end = min(n_frames - 1, t + vote_radius)
        pool: List[Tuple[int, BlobCandidate]] = []
        for j in range(start, end + 1):
            for cand in candidates_by_frame[j]:
                pool.append((j, cand))
        if not pool:
            continue

        best_anchor = None
        best_score = -1.0
        best_support = 0

        for _, center in pool:
            # 不讓「靠人 + 低 blob + 靠 track 撐高分」的點當 vote anchor
            # 但注意：只是不能當中心，不是從 pool 刪掉。
            if is_likely_human_motion_artifact(center):
                continue

            radius = (near_neighbor_radius + 6.0) if center.court_side == "near" else (far_neighbor_radius + 5.0)
            sigma_space = max(2.5, radius * 0.55)
            sigma_time = max(0.9, vote_radius * 0.55)

            total = 0.0
            member_count = 0
            distinct_frames = set()
            spatial_dists = []

            for j, cand in pool:
                if center.court_side != "unknown" and cand.court_side != "unknown" and center.court_side != cand.court_side:
                    continue
                dist = float(np.hypot(cand.cx - center.cx, cand.cy - center.cy))
                if dist > radius:
                    continue

                dt = abs(j - t)
                w_space = float(np.exp(-(dist * dist) / (2.0 * sigma_space * sigma_space)))
                w_time = float(np.exp(-(dt * dt) / (2.0 * sigma_time * sigma_time)))
                base = float(
                    0.54 * cand.score_event
                    + 0.18 * cand.track_score_1
                    + 0.12 * cand.track_score_2
                    + 0.08 * min(1.0, cand.support_5f / 5.0)
                    + 0.08 * cand.score_blob
                )
                total += base * w_space * w_time
                member_count += 1
                distinct_frames.add(j)
                spatial_dists.append(dist)

            support_frames = len(distinct_frames)
            if member_count == 0:
                continue

            support_bonus = 0.45 + 0.55 * min(1.0, support_frames / 4.0)
            time_center_bonus = 0.70 + 0.30 * float(np.exp(-((center.frame_idx - t) ** 2) / (2.0 * 1.2 * 1.2)))
            avg_dist = float(np.mean(spatial_dists)) if spatial_dists else radius
            tight_bonus = 0.65 + 0.35 * float(np.clip(1.0 - avg_dist / max(1e-6, radius), 0.0, 1.0))
            center_self_score = float(
                0.65 * center.score_event
                + 0.20 * center.track_score_1
                + 0.10 * center.track_score_2
                + 0.05 * center.score_blob
            )

            cluster_score = (
                0.55 * center_self_score
                + 0.45 * total / max(1, member_count)
            ) * support_bonus * time_center_bonus * tight_bonus

            if support_frames >= 2 and cluster_score > best_score:
                best_score = float(cluster_score)
                best_support = int(support_frames)
                best_anchor = VoteAnchor(
                    frame_idx=int(t),
                    cx=float(center.cx),
                    cy=float(center.cy),
                    court_side=str(center.court_side),
                    score=float(cluster_score),
                    support_frames=int(support_frames),
                    member_count=int(member_count),
                )

        if best_anchor is not None:
            vote_raw[t] = float(best_anchor.score)
            vote_support[t] = float(best_support)
            anchors[t] = best_anchor

    scale = float(np.percentile(vote_raw[vote_raw > 0], 95)) if np.any(vote_raw > 0) else 1.0
    scale = max(scale, 1e-6)
    vote_raw = np.clip(vote_raw / scale, 0.0, 1.5)
    vote_raw = np.minimum(vote_raw, 1.0)
    return vote_raw.astype(np.float32), anchors, vote_support.astype(np.float32)

def infer_peak_search_direction(
    peak_idx: int,
    candidates_by_frame: List[List[BlobCandidate]],
    vote_anchors: List[Optional[VoteAnchor]],
    look_forward: int = 4,
    link_radius_near: float = 28.0,
    link_radius_far: float = 16.0,
) -> str:
    """
    判斷 saturated peak 應該往前找還是往後找。

    規則：
    - 看 peak 後面幾幀的短軌跡趨勢
    - 如果後面幾幀整體往 near 方向走（cy 變大），代表真 bounce 應該更早 -> backward
    - 如果後面幾幀整體往 far 方向走（cy 變小），代表真 bounce 應該更晚 -> forward

    回傳:
      "forward"  -> 往後跳出 plateau，再往後找
      "backward" -> 往前跳出 plateau，再往前找
    """
    n = len(candidates_by_frame)
    if peak_idx < 0 or peak_idx >= n:
        return "forward"

    anchor = vote_anchors[peak_idx] if 0 <= peak_idx < len(vote_anchors) else None

    # 中心參考點
    if anchor is not None:
        ref_x = float(anchor.cx)
        ref_y = float(anchor.cy)
        ref_side = str(anchor.court_side)
    elif len(candidates_by_frame[peak_idx]) > 0:
        best = max(candidates_by_frame[peak_idx], key=lambda c: c.score_event)
        ref_x = float(best.cx)
        ref_y = float(best.cy)
        ref_side = str(best.court_side)
    else:
        return "forward"

    link_radius = link_radius_near if ref_side == "near" else link_radius_far

    pts = [(peak_idx, ref_x, ref_y)]
    last_x, last_y = ref_x, ref_y

    # 只往後追
    for j in range(peak_idx + 1, min(n, peak_idx + look_forward + 1)):
        cands = candidates_by_frame[j]
        if not cands:
            continue

        same_side = [c for c in cands if c.court_side == ref_side]
        search_list = same_side if same_side else cands

        cand = nearest_candidate(search_list, last_x, last_y, link_radius)
        if cand is None:
            cand = min(search_list, key=lambda c: np.hypot(c.cx - last_x, c.cy - last_y))

        pts.append((j, float(cand.cx), float(cand.cy)))
        last_x, last_y = float(cand.cx), float(cand.cy)

    if len(pts) < 2:
        return "forward"

    # 只看後方整體 y 變化
    y0 = pts[0][2]
    y1 = pts[-1][2]
    dy = float(y1 - y0)

    # dy > 0 : 往 near
    # dy < 0 : 往 far
    if dy > 0.8:
        return "backward"
    elif dy < -0.8:
        return "forward"

    # 若很平，再看線性 slope
    t = np.array([p[0] for p in pts], dtype=np.float32)
    y = np.array([p[2] for p in pts], dtype=np.float32)
    if len(t) >= 2 and float(np.max(t) - np.min(t)) > 1e-6:
        slope = float(np.polyfit(t, y, 1)[0])
        if slope > 0:
            return "backward"
        elif slope < 0:
            return "forward"

    return "forward"

def relocate_saturated_vote_peak(
    peak_idx: int,
    vote_scores_raw: np.ndarray,
    vote_anchors: List[Optional[VoteAnchor]],
    candidates_by_frame: List[List[BlobCandidate]],
    search_window: int = 5,
    one_th: float = 0.999,
    debug: bool = False,
) -> int:
    n = len(vote_scores_raw)
    if peak_idx < 0 or peak_idx >= n:
        return peak_idx

    if float(vote_scores_raw[peak_idx]) < one_th:
        return peak_idx

    direction = infer_peak_search_direction(
        peak_idx=peak_idx,
        candidates_by_frame=candidates_by_frame,
        vote_anchors=vote_anchors,
        look_forward=4,
        link_radius_near=28.0,
        link_radius_far=16.0,
    )

    if direction == "forward":
        j = peak_idx
        while j < n and float(vote_scores_raw[j]) >= one_th:
            j += 1
        if j >= n:
            return peak_idx
        start = j
        end = min(n - 1, j + search_window)

    else:  # backward
        j = peak_idx
        while j >= 0 and float(vote_scores_raw[j]) >= one_th:
            j -= 1
        if j < 0:
            return peak_idx
        start = max(0, j - search_window)
        end = j

    best_idx = peak_idx
    best_score = -1.0

    for k in range(start, end + 1):
        v = float(vote_scores_raw[k])
        if v >= one_th:
            continue

        cand_bonus = 0.0
        if 0 <= k < len(candidates_by_frame) and len(candidates_by_frame[k]) > 0:
            cand_bonus = max(float(c.score_event) for c in candidates_by_frame[k])

        score = 0.84 * v + 0.16 * cand_bonus
        if score > best_score:
            best_score = score
            best_idx = k

    if debug and best_idx != peak_idx:
        print(
            f"[vote_relocate] peak={peak_idx} dir={direction} "
            f"-> {best_idx} "
            f"(vote_raw {vote_scores_raw[peak_idx]:.3f} -> {vote_scores_raw[best_idx]:.3f})"
        )

    return best_idx

def get_anchor_for_peak(
    peak_idx: int,
    vote_anchors: List[Optional[VoteAnchor]],
    candidates_by_frame: List[List[BlobCandidate]],
) -> Optional[VoteAnchor]:
    anchor = vote_anchors[peak_idx] if 0 <= peak_idx < len(vote_anchors) else None
    if anchor is not None:
        return anchor

    if 0 <= peak_idx < len(candidates_by_frame) and len(candidates_by_frame[peak_idx]) > 0:
        best_cand = max(candidates_by_frame[peak_idx], key=lambda c: c.score_event)
        return VoteAnchor(
            frame_idx=int(peak_idx),
            cx=float(best_cand.cx),
            cy=float(best_cand.cy),
            court_side=str(best_cand.court_side),
            score=float(best_cand.score_event),
            support_frames=int(best_cand.support_5f),
            member_count=1,
        )

    return None


def detect_dual_side_conflict(
    candidates_by_frame: List[List[BlobCandidate]],
    peak_idx: int,
    window_radius: int = 1,
    min_score_event: float = 0.45,
    min_score_blob: float = 0.30,
    min_support_5f: int = 2,
) -> bool:
    """
    只有在 peak 附近真的同時出現 near / far 兩側都像樣的候選，才觸發 side arbitration。
    這樣避免每個 peak 都去看前後五幀。
    """
    start = max(0, peak_idx - window_radius)
    end = min(len(candidates_by_frame) - 1, peak_idx + window_radius)

    near_found = False
    far_found = False

    for j in range(start, end + 1):
        for cand in candidates_by_frame[j]:
            strong_enough = (
                cand.score_event >= min_score_event
                or (cand.score_blob >= min_score_blob and cand.support_5f >= min_support_5f)
            )
            if not strong_enough:
                continue

            if cand.court_side == "near":
                near_found = True
            elif cand.court_side == "far":
                far_found = True

            if near_found and far_found:
                return True

    return False


def choose_side_by_temporal_context(
    candidates_by_frame: List[List[BlobCandidate]],
    peak_idx: int,
    look_radius: int = 5,
    near_neighbor_radius: float = 22.0,
    far_neighbor_radius: float = 8.0,
    min_score_event: float = 0.35,
) -> Tuple[str, dict]:
    start = max(0, peak_idx - look_radius)
    end = min(len(candidates_by_frame) - 1, peak_idx + look_radius)

    side_stats = {
        "near": {
            "score_sum": 0.0,
            "track_sum": 0.0,
            "blob_sum": 0.0,
            "support_frames": set(),
            "count": 0,
            "best_score": 0.0,
        },
        "far": {
            "score_sum": 0.0,
            "track_sum": 0.0,
            "blob_sum": 0.0,
            "support_frames": set(),
            "count": 0,
            "best_score": 0.0,
        },
    }

    # 先做 side score
    for j in range(start, end + 1):
        best_by_side = {"near": None, "far": None}

        for cand in candidates_by_frame[j]:
            side = cand.court_side
            if side not in ("near", "far"):
                continue
            if cand.score_event < min_score_event:
                continue

            if best_by_side[side] is None or cand.score_event > best_by_side[side].score_event:
                best_by_side[side] = cand

        for side in ("near", "far"):
            cand = best_by_side[side]
            if cand is None:
                continue

            s = side_stats[side]
            s["score_sum"] += float(cand.score_event)
            s["track_sum"] += float(0.55 * cand.track_score_1 + 0.45 * cand.track_score_2)
            s["blob_sum"] += float(cand.score_blob)
            s["support_frames"].add(j)
            s["count"] += 1
            s["best_score"] = max(s["best_score"], float(cand.score_event))

    near_frames = len(side_stats["near"]["support_frames"])
    far_frames = len(side_stats["far"]["support_frames"])

    near_final = (
        0.34 * side_stats["near"]["score_sum"]
        + 0.18 * side_stats["near"]["track_sum"]
        + 0.08 * side_stats["near"]["blob_sum"]
        + 0.40 * near_frames
    )
    far_final = (
        0.34 * side_stats["far"]["score_sum"]
        + 0.18 * side_stats["far"]["track_sum"]
        + 0.08 * side_stats["far"]["blob_sum"]
        + 0.40 * far_frames
    )

    chosen_side = "near" if near_final >= far_final else "far"

    debug_info = {
        "near_score": near_final,
        "far_score": far_final,
        "near_frames": near_frames,
        "far_frames": far_frames,
        "near_count": side_stats["near"]["count"],
        "far_count": side_stats["far"]["count"],
        "near_best": side_stats["near"]["best_score"],
        "far_best": side_stats["far"]["best_score"],
    }

    return chosen_side, debug_info

def make_event_from_vote_peak(
    peak_idx: int,
    anchor: VoteAnchor,
    candidates_by_frame: List[List[BlobCandidate]],
    near_refine_radius: float = 26.0,
    far_refine_radius: float = 16.0,
    window_radius: int = 3,
    enable_side_arbitration: bool = True,
    side_conflict_check_radius: int = 1,
    side_look_radius: int = 5,
) -> Optional[BounceEvent]:
    start = max(0, peak_idx - window_radius)
    end = min(len(candidates_by_frame) - 1, peak_idx + window_radius)

    chosen_side = anchor.court_side
    has_conflict = False
    if enable_side_arbitration:
        has_conflict = detect_dual_side_conflict(
            candidates_by_frame,
            peak_idx=peak_idx,
            window_radius=side_conflict_check_radius,
            min_score_event=0.30,
            min_score_blob=0.20,
            min_support_5f=1,
        )

        if has_conflict:
            chosen_side, _ = choose_side_by_temporal_context(
                candidates_by_frame,
                peak_idx=peak_idx,
                look_radius=side_look_radius,
                near_neighbor_radius=near_refine_radius,
                far_neighbor_radius=far_refine_radius,
                min_score_event=0.35,
            )

    # 關鍵：只回到 peak_idx 這一幀挑 candidate
    curr_cands = candidates_by_frame[peak_idx] if 0 <= peak_idx < len(candidates_by_frame) else []

    # 只保留該側候選
    side_cands = [c for c in curr_cands if c.court_side == chosen_side]

    def cand_total_score(c: BlobCandidate) -> float:
        return float(
            0.78 * c.score_event
            + 0.14 * c.track_score_1
            + 0.05 * c.track_score_2
            + 0.03 * c.score_blob
        )

    best_cand = None

    # 規則：
    # 先只在 chosen_side 裡找
    # 但只有當這一側存在至少一顆 score_event >= 0.75，才真的信這一側
    strong_side_cands = [
        c for c in side_cands
        if c.score_event >= 0.75
        and not is_likely_human_motion_artifact(c)
    ]

    if len(strong_side_cands) > 0:
        best_cand = max(strong_side_cands, key=cand_total_score)
    else:
        # chosen_side 當幀沒有夠強候選，fallback 回該幀所有候選裡找最高分
        if len(curr_cands) > 0:
            safe_cands = [
                c for c in curr_cands
                if not is_likely_human_motion_artifact(c)
            ]

            if len(safe_cands) > 0:
                best_cand = max(safe_cands, key=cand_total_score)
            else:
                # 如果整幀真的只有靠近人的候選，那才允許使用原本候選
                # 這是為了避免真球在人旁邊彈跳時被完全漏掉。
                best_cand = max(curr_cands, key=cand_total_score)
        
    # 如果這一幀完全沒 candidate，才用 anchor fallback
    if best_cand is None:
        return BounceEvent(
            peak_frame=int(peak_idx),
            peak_score=float(anchor.score),
            cx=float(anchor.cx),
            cy=float(anchor.cy),
            court_side=str(chosen_side),
            support_5f=int(anchor.support_frames),
            has_prev=0,
            has_next=0,
            prev_step=9999.0,
            next_step=9999.0,
            speed_dip_1=0.0,
            speed_dip_2=0.0,
            dip_ratio_bonus=0.0,
            source_blob_score=0.0,
            window_start=int(start),
            window_end=int(end),
            coarse_cx=float(anchor.cx),
            coarse_cy=float(anchor.cy),
            refined_cx=float(anchor.cx),
            refined_cy=float(anchor.cy),
            refined_frame=int(peak_idx),
            track_fit_cx=float(anchor.cx),
            track_fit_cy=float(anchor.cy),
            track_fit_frame=float(peak_idx),
            motion_peak_cx=float(anchor.cx),
            motion_peak_cy=float(anchor.cy),
        )

    peak_score = float(
        0.72 * best_cand.score_event
        + 0.18 * anchor.score
        + 0.06 * best_cand.track_score_1
        + 0.04 * best_cand.track_score_2
    )

    return BounceEvent(
        peak_frame=int(peak_idx),
        peak_score=peak_score,
        cx=float(best_cand.cx),
        cy=float(best_cand.cy),
        court_side=str(chosen_side),
        support_5f=int(max(best_cand.support_5f, anchor.support_frames)),
        has_prev=int(best_cand.has_prev),
        has_next=int(best_cand.has_next),
        prev_step=float(best_cand.prev_step),
        next_step=float(best_cand.next_step),
        speed_dip_1=float(best_cand.speed_dip_1),
        speed_dip_2=float(best_cand.speed_dip_2),
        dip_ratio_bonus=float(best_cand.dip_ratio_bonus),
        source_blob_score=float(best_cand.score_blob),
        window_start=int(start),
        window_end=int(end),
        coarse_cx=float(best_cand.cx),
        coarse_cy=float(best_cand.cy),
        refined_cx=float(best_cand.cx),
        refined_cy=float(best_cand.cy),
        refined_frame=int(peak_idx),
        track_fit_cx=float(best_cand.cx),
        track_fit_cy=float(best_cand.cy),
        track_fit_frame=float(peak_idx),
        motion_peak_cx=float(best_cand.cx),
        motion_peak_cy=float(best_cand.cy),
    )


def _post_cand_quality(c: BlobCandidate) -> float:
    return float(
        0.50 * c.score_event
        + 0.20 * c.score_blob
        + 0.15 * c.track_score_1
        + 0.10 * c.track_score_2
        + 0.05 * min(1.0, c.support_5f / 5.0)
    )


def _pick_post_local_candidate(
    cands: List[BlobCandidate],
    ref_x: float,
    ref_y: float,
    radius: float,
    allowed_side: Optional[str] = None,
) -> Tuple[Optional[BlobCandidate], float, str]:
    if not cands:
        return None, 9999.0, "missing"

    best = None
    best_score = -1e18
    best_dist = 9999.0

    for cand in cands:
        if allowed_side in ("near", "far") and cand.court_side != allowed_side:
            continue

        dist = float(np.hypot(cand.cx - ref_x, cand.cy - ref_y))
        if dist > radius:
            continue

        score = (
            0.58 * _post_cand_quality(cand)
            + 0.30 * (1.0 - dist / max(radius, 1e-6))
            + 0.12 * cand.score_blob
        )
        if score > best_score:
            best_score = score
            best = cand
            best_dist = dist

    if best is not None:
        return best, float(best_dist), "pred"

    # fallback：同側整幀最高分
    side_cands = [c for c in cands if (allowed_side not in ("near", "far") or c.court_side == allowed_side)]
    if side_cands:
        best = max(side_cands, key=_post_cand_quality)
        best_dist = float(np.hypot(best.cx - ref_x, best.cy - ref_y))
        return best, best_dist, "fallback"

    return None, 9999.0, "missing"


def build_postcheck_window_track(
    candidates_by_frame: List[List[BlobCandidate]],
    center_frame: int,
    center_x: float,
    center_y: float,
    chosen_side: str,
    near_neighbor_radius: float,
    far_neighbor_radius: float,
    window_radius: int = 3,
) -> List[WindowPoint]:
    n = len(candidates_by_frame)
    link_radius = (near_neighbor_radius + 6.0) if chosen_side == "near" else (far_neighbor_radius + 5.0)

    points_dict = {}

    # 中心幀：優先選靠近 event 位置的候選
    center_cands = candidates_by_frame[center_frame] if 0 <= center_frame < n else []
    center_cand, center_dist, center_pick = _pick_post_local_candidate(
        center_cands,
        center_x,
        center_y,
        radius=link_radius * 1.35,
        allowed_side=chosen_side,
    )

    if center_cand is not None:
        cx, cy = float(center_cand.cx), float(center_cand.cy)
        points_dict[center_frame] = WindowPoint(
            rel_idx=0,
            frame_idx=center_frame,
            x=cx,
            y=cy,
            score_blob=float(center_cand.score_blob),
            score_event=float(center_cand.score_event),
            track_score_1=float(center_cand.track_score_1),
            track_score_2=float(center_cand.track_score_2),
            support_5f=int(center_cand.support_5f),
            court_side=str(center_cand.court_side),
            distance_to_pred=float(center_dist),
            picked_by="direct",
            is_valid=1,
        )
    else:
        cx, cy = float(center_x), float(center_y)
        points_dict[center_frame] = WindowPoint(
            rel_idx=0,
            frame_idx=center_frame,
            x=cx,
            y=cy,
            score_blob=0.0,
            score_event=0.0,
            track_score_1=0.0,
            track_score_2=0.0,
            support_5f=0,
            court_side=str(chosen_side),
            distance_to_pred=0.0,
            picked_by="missing",
            is_valid=0,
        )

    # backward
    prev_valid_x, prev_valid_y = cx, cy
    back_ref_x, back_ref_y = cx, cy
    back_vx, back_vy = 0.0, 0.0

    for j in range(center_frame - 1, max(-1, center_frame - window_radius - 1), -1):
        pred_x = back_ref_x + back_vx
        pred_y = back_ref_y + back_vy

        cand, dist, picked_by = _pick_post_local_candidate(
            candidates_by_frame[j],
            pred_x,
            pred_y,
            radius=link_radius,
            allowed_side=chosen_side,
        )

        if cand is not None:
            curr_x, curr_y = float(cand.cx), float(cand.cy)
            back_vx = prev_valid_x - curr_x
            back_vy = prev_valid_y - curr_y
            prev_valid_x, prev_valid_y = curr_x, curr_y
            back_ref_x, back_ref_y = curr_x, curr_y

            points_dict[j] = WindowPoint(
                rel_idx=j - center_frame,
                frame_idx=j,
                x=curr_x,
                y=curr_y,
                score_blob=float(cand.score_blob),
                score_event=float(cand.score_event),
                track_score_1=float(cand.track_score_1),
                track_score_2=float(cand.track_score_2),
                support_5f=int(cand.support_5f),
                court_side=str(cand.court_side),
                distance_to_pred=float(dist),
                picked_by=picked_by,
                is_valid=1,
            )
        else:
            points_dict[j] = WindowPoint(
                rel_idx=j - center_frame,
                frame_idx=j,
                x=float(pred_x),
                y=float(pred_y),
                score_blob=0.0,
                score_event=0.0,
                track_score_1=0.0,
                track_score_2=0.0,
                support_5f=0,
                court_side=str(chosen_side),
                distance_to_pred=9999.0,
                picked_by="missing",
                is_valid=0,
            )
            back_ref_x, back_ref_y = pred_x, pred_y

    # forward
    prev_valid_x, prev_valid_y = cx, cy
    fwd_ref_x, fwd_ref_y = cx, cy
    fwd_vx, fwd_vy = 0.0, 0.0

    for j in range(center_frame + 1, min(n, center_frame + window_radius + 1)):
        pred_x = fwd_ref_x + fwd_vx
        pred_y = fwd_ref_y + fwd_vy

        cand, dist, picked_by = _pick_post_local_candidate(
            candidates_by_frame[j],
            pred_x,
            pred_y,
            radius=link_radius,
            allowed_side=chosen_side,
        )

        if cand is not None:
            curr_x, curr_y = float(cand.cx), float(cand.cy)
            fwd_vx = curr_x - prev_valid_x
            fwd_vy = curr_y - prev_valid_y
            prev_valid_x, prev_valid_y = curr_x, curr_y
            fwd_ref_x, fwd_ref_y = curr_x, curr_y

            points_dict[j] = WindowPoint(
                rel_idx=j - center_frame,
                frame_idx=j,
                x=curr_x,
                y=curr_y,
                score_blob=float(cand.score_blob),
                score_event=float(cand.score_event),
                track_score_1=float(cand.track_score_1),
                track_score_2=float(cand.track_score_2),
                support_5f=int(cand.support_5f),
                court_side=str(cand.court_side),
                distance_to_pred=float(dist),
                picked_by=picked_by,
                is_valid=1,
            )
        else:
            points_dict[j] = WindowPoint(
                rel_idx=j - center_frame,
                frame_idx=j,
                x=float(pred_x),
                y=float(pred_y),
                score_blob=0.0,
                score_event=0.0,
                track_score_1=0.0,
                track_score_2=0.0,
                support_5f=0,
                court_side=str(chosen_side),
                distance_to_pred=9999.0,
                picked_by="missing",
                is_valid=0,
            )
            fwd_ref_x, fwd_ref_y = pred_x, pred_y

    points = []
    for rel in range(-window_radius, window_radius + 1):
        fidx = center_frame + rel
        if 0 <= fidx < n:
            points.append(points_dict[fidx])

    return points


def score_postcheck_sequence(points: List[WindowPoint], chosen_side: str) -> Tuple[int, float, float, float, float, float]:
    if not points:
        return 0, 0.0, 0.0, 0.0, 1.0, 0.0

    valid_pts = [p for p in points if p.is_valid == 1]
    valid_count = len(valid_pts)
    if valid_count == 0:
        return 0, 0.0, 0.0, 0.0, 1.0, 0.0

    side_consistency = float(
        sum(1 for p in valid_pts if p.court_side == chosen_side) / max(1, valid_count)
    )

    qualities = []
    for p in valid_pts:
        q = (
            0.50 * p.score_event
            + 0.20 * p.score_blob
            + 0.15 * p.track_score_1
            + 0.10 * p.track_score_2
            + 0.05 * min(1.0, p.support_5f / 5.0)
        )
        qualities.append(q)
    quality_mean = float(np.mean(qualities)) if qualities else 0.0

    steps = []
    prev = None
    for p in points:
        if p.is_valid != 1:
            continue
        if prev is not None:
            steps.append(float(np.hypot(p.x - prev.x, p.y - prev.y)))
        prev = p

    if len(steps) >= 2:
        step_mean = float(np.mean(steps))
        step_std = float(np.std(steps))
        smoothness_score = float(np.clip(1.0 - step_std / max(step_mean, 1.0), 0.0, 1.0))
    elif len(steps) == 1:
        smoothness_score = 0.5
    else:
        smoothness_score = 0.0

    missing_runs = 0
    curr_run = 0
    for p in points:
        if p.is_valid == 0:
            curr_run += 1
        else:
            if curr_run > 0:
                missing_runs += curr_run
            curr_run = 0
    if curr_run > 0:
        missing_runs += curr_run

    gap_penalty = float(np.clip(missing_runs / 4.0, 0.0, 1.0))

    sequence_score = float(
        0.30 * min(1.0, valid_count / 7.0)
        + 0.22 * side_consistency
        + 0.28 * quality_mean
        + 0.20 * smoothness_score
        - 0.22 * gap_penalty
    )
    sequence_score = float(np.clip(sequence_score, 0.0, 1.0))

    return valid_count, side_consistency, quality_mean, smoothness_score, gap_penalty, sequence_score

def score_window_direction_change(
    points: List[WindowPoint],
    idx: int,
    min_side_points: int = 2,
) -> Tuple[float, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    用整段前半段 vs 後半段方向變化來算分。
    回傳:
      (direction_change_score, v_pre_unit, v_post_unit)
    """
    if idx <= 0 or idx >= len(points) - 1:
        return 0.0, None, None

    pre_pts = [p for p in points[:idx] if p.is_valid == 1]
    post_pts = [p for p in points[idx + 1:] if p.is_valid == 1]

    if len(pre_pts) < min_side_points or len(post_pts) < min_side_points:
        return 0.0, None, None

    # 用整段首尾差向量
    pre_start = pre_pts[0]
    pre_end = pre_pts[-1]
    post_start = post_pts[0]
    post_end = post_pts[-1]

    v_pre = np.array(
        [pre_end.x - pre_start.x, pre_end.y - pre_start.y],
        dtype=np.float32
    )
    v_post = np.array(
        [post_end.x - post_start.x, post_end.y - post_start.y],
        dtype=np.float32
    )

    n1 = float(np.linalg.norm(v_pre))
    n2 = float(np.linalg.norm(v_post))

    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0, None, None

    v_pre_unit = (float(v_pre[0] / n1), float(v_pre[1] / n1))
    v_post_unit = (float(v_post[0] / n2), float(v_post[1] / n2))

    cosang = float(np.dot(v_pre, v_post) / (n1 * n2 + 1e-6))
    score = float(np.clip((1.0 - cosang) * 0.5, 0.0, 1.0))

    return score, v_pre_unit, v_post_unit

def score_postcheck_bounce_moment(points: List[WindowPoint], idx: int) -> Tuple[float, float, float]:
    if idx <= 0 or idx >= len(points) - 1:
        return 0.0, float(points[idx].x), float(points[idx].y)

    p = points[idx]
    if p.is_valid != 1:
        return 0.0, float(p.x), float(p.y)

    prevs = [points[i] for i in range(0, idx) if points[i].is_valid == 1]
    nexts = [points[i] for i in range(idx + 1, len(points)) if points[i].is_valid == 1]

    if len(prevs) < 1 or len(nexts) < 1:
        return 0.0, float(p.x), float(p.y)

    # 保留最近鄰，給 speed dip 用
    p_prev = prevs[-1]
    p_next = nexts[0]

    prev_step = float(np.hypot(p.x - p_prev.x, p.y - p_prev.y))
    next_step = float(np.hypot(p_next.x - p.x, p_next.y - p.y))

    # 新版：方向改變看整段前後方向
    direction_change_score, _, _ = score_window_direction_change(points, idx, min_side_points=2)

    # 局部速度 dip
    all_steps = []
    for j in range(1, len(points)):
        if points[j - 1].is_valid == 1 and points[j].is_valid == 1:
            all_steps.append(float(np.hypot(points[j].x - points[j - 1].x, points[j].y - points[j - 1].y)))

    local_mean = float(np.mean(all_steps)) if all_steps else max(prev_step, next_step, 1.0)
    center_step = 0.5 * (prev_step + next_step)
    speed_dip_score = float(np.clip(1.0 - center_step / max(local_mean * 1.15, 1.0), 0.0, 1.0))

    local_quality = (
        0.42 * p.score_event
        + 0.18 * p.score_blob
        + 0.15 * p.track_score_1
        + 0.10 * p.track_score_2
        + 0.05 * min(1.0, p.support_5f / 5.0)
        + 0.10 * (1.0 if p.picked_by in ("direct", "pred") else 0.4)
    )

    support_score = min(1.0, 0.5 * min(len(prevs), 2) + 0.5 * min(len(nexts), 2)) / 2.0

    score = float(
        0.32 * direction_change_score
        + 0.26 * speed_dip_score
        + 0.26 * local_quality
        + 0.16 * support_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    return score, float(p.x), float(p.y)

def _pick_track_candidate(
    cands: List[BlobCandidate],
    ref_x: float,
    ref_y: float,
    radius: float,
    allowed_side: Optional[str] = None,
) -> Optional[BlobCandidate]:
    best = None
    best_score = -1e18
    radius = max(1.0, float(radius))

    for cand in cands:
        if allowed_side in ("near", "far") and cand.court_side != allowed_side:
            continue

        dist = float(np.hypot(cand.cx - ref_x, cand.cy - ref_y))
        if dist > radius:
            continue

        score = (
            0.56 * (1.0 - dist / radius)
            + 0.28 * float(cand.score_event)
            + 0.16 * float(cand.score_blob)
        )
        if score > best_score:
            best_score = score
            best = cand
    return best


def _collect_local_track(
    candidates_by_frame: List[List[BlobCandidate]],
    peak_frame: int,
    anchor_x: float,
    anchor_y: float,
    link_radius: float,
    window_radius: int = 3,
    allowed_side: Optional[str] = None,
) -> List[Tuple[int, float, float, float]]:
    start = max(0, peak_frame - window_radius)
    end = min(len(candidates_by_frame) - 1, peak_frame + window_radius)
    obs: List[Tuple[int, float, float, float]] = []

    center_cand = _pick_track_candidate(
        candidates_by_frame[peak_frame],
        anchor_x,
        anchor_y,
        link_radius * 1.35,
        allowed_side=allowed_side,
    )
    if center_cand is not None:
        obs.append((peak_frame, float(center_cand.cx), float(center_cand.cy), float(max(center_cand.score_event, center_cand.score_blob))))
        back_ref_x, back_ref_y = float(center_cand.cx), float(center_cand.cy)
        fwd_ref_x, fwd_ref_y = float(center_cand.cx), float(center_cand.cy)
    else:
        obs.append((peak_frame, float(anchor_x), float(anchor_y), 0.20))
        back_ref_x, back_ref_y = float(anchor_x), float(anchor_y)
        fwd_ref_x, fwd_ref_y = float(anchor_x), float(anchor_y)

    for j in range(peak_frame - 1, start - 1, -1):
        cand = _pick_track_candidate(
            candidates_by_frame[j],
            back_ref_x,
            back_ref_y,
            link_radius,
            allowed_side=allowed_side,
        )
        if cand is None:
            continue
        obs.append((j, float(cand.cx), float(cand.cy), float(max(cand.score_event, cand.score_blob))))
        back_ref_x, back_ref_y = float(cand.cx), float(cand.cy)

    for j in range(peak_frame + 1, end + 1):
        cand = _pick_track_candidate(
            candidates_by_frame[j],
            fwd_ref_x,
            fwd_ref_y,
            link_radius,
            allowed_side=allowed_side,
        )
        if cand is None:
            continue
        obs.append((j, float(cand.cx), float(cand.cy), float(max(cand.score_event, cand.score_blob))))
        fwd_ref_x, fwd_ref_y = float(cand.cx), float(cand.cy)

    obs.sort(key=lambda z: z[0])
    return obs


def _fit_line_xy(obs: List[Tuple[int, float, float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(obs) < 2:
        return None
    t = np.array([o[0] for o in obs], dtype=np.float32)
    if float(np.max(t) - np.min(t)) < 1e-6:
        return None
    x = np.array([o[1] for o in obs], dtype=np.float32)
    y = np.array([o[2] for o in obs], dtype=np.float32)
    w = np.array([max(1e-3, o[3]) for o in obs], dtype=np.float32)
    try:
        coef_x = np.polyfit(t, x, 1, w=w)
        coef_y = np.polyfit(t, y, 1, w=w)
    except Exception:
        coef_x = np.polyfit(t, x, 1)
        coef_y = np.polyfit(t, y, 1)
    return coef_x.astype(np.float32), coef_y.astype(np.float32)


def _predict_line_xy(fit: Tuple[np.ndarray, np.ndarray], t: float) -> Tuple[float, float]:
    coef_x, coef_y = fit
    x = float(coef_x[0] * t + coef_x[1])
    y = float(coef_y[0] * t + coef_y[1])
    return x, y


def _estimate_track_bounce_point(
    obs: List[Tuple[int, float, float, float]],
    peak_frame: int,
    coarse_x: float,
    coarse_y: float,
    start: int,
    end: int,
) -> Tuple[float, float, float, float, int, int]:
    pre_obs = [o for o in obs if o[0] < peak_frame]
    post_obs = [o for o in obs if o[0] > peak_frame]

    pre_fit = _fit_line_xy(pre_obs)
    post_fit = _fit_line_xy(post_obs)
    if pre_fit is None or post_fit is None:
        return float(coarse_x), float(coarse_y), float(peak_frame), 0.0, len(pre_obs), len(post_obs)

    t_lo = max(float(start), float(peak_frame) - 1.75)
    t_hi = min(float(end), float(peak_frame) + 1.75)
    if t_hi <= t_lo:
        t_lo, t_hi = float(start), float(end)
    t_grid = np.linspace(t_lo, t_hi, 31, dtype=np.float32)

    best_t = float(peak_frame)
    best_dist = 1e18
    best_xy = (float(coarse_x), float(coarse_y))
    for t in t_grid:
        px1, py1 = _predict_line_xy(pre_fit, float(t))
        px2, py2 = _predict_line_xy(post_fit, float(t))
        d = float(np.hypot(px1 - px2, py1 - py2))
        if d < best_dist:
            best_dist = d
            best_t = float(t)
            best_xy = (0.5 * (px1 + px2), 0.5 * (py1 + py2))

    track_score = float(np.clip(1.0 - best_dist / 12.0, 0.0, 1.0))
    return float(best_xy[0]), float(best_xy[1]), float(best_t), track_score, len(pre_obs), len(post_obs)


def _build_local_motion_heatmap(
    grays: List[np.ndarray],
    search_mask: np.ndarray,
    points: List[Tuple[float, float]],
    peak_frame: int,
    diff_th: int,
    window_radius: int,
    roi_radius: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float], float]:
    H, W = search_mask.shape[:2]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    margin = max(8, int(roi_radius))
    x1 = max(0, int(np.floor(min(xs) - margin)))
    y1 = max(0, int(np.floor(min(ys) - margin)))
    x2 = min(W, int(np.ceil(max(xs) + margin + 1)))
    y2 = min(H, int(np.ceil(max(ys) + margin + 1)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), np.float32), (0, 0, 1, 1), (float(points[0][0]), float(points[0][1])), 0.0

    heat = np.zeros((y2 - y1, x2 - x1), np.float32)
    start = max(0, peak_frame - window_radius)
    end = min(len(grays) - 1, peak_frame + window_radius)
    for j in range(start, end):
        diff = cv2.absdiff(grays[j + 1], grays[j])
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        diff_f = diff[y1:y2, x1:x2].astype(np.float32)
        diff_f[diff_f < float(diff_th)] = 0.0
        weight = 1.0 / (1.0 + abs((j + 0.5) - float(peak_frame)))
        heat += weight * diff_f

    if heat.size > 0:
        heat = cv2.GaussianBlur(heat, (5, 5), 0)
    mask_crop = (search_mask[y1:y2, x1:x2] > 0).astype(np.float32)
    heat *= mask_crop

    if heat.size == 0 or float(np.max(heat)) <= 1e-6:
        return heat, (x1, y1, x2, y2), (float(points[0][0]), float(points[0][1])), 0.0

    iy, ix = np.unravel_index(np.argmax(heat), heat.shape)
    peak_xy = (float(x1 + ix), float(y1 + iy))
    motion_score = float(np.max(heat))
    return heat, (x1, y1, x2, y2), peak_xy, motion_score


def refine_event_location_with_track_and_motion(
    event: BounceEvent,
    candidates_by_frame: List[List[BlobCandidate]],
    grays: List[np.ndarray],
    search_mask: np.ndarray,
    diff_th: int,
    near_neighbor_radius: float,
    far_neighbor_radius: float,
    net_y: Optional[float] = None,
    window_radius: int = 3,
) -> BounceEvent:
    side = event.court_side
    link_radius = (near_neighbor_radius + 7.0) if side == "near" else (far_neighbor_radius + 5.0)
    roi_radius = int(round((near_neighbor_radius * 1.8) if side == "near" else (far_neighbor_radius * 2.3)))
    roi_radius = max(10, roi_radius)

    obs = _collect_local_track(
        candidates_by_frame,
        peak_frame=event.peak_frame,
        anchor_x=event.coarse_cx,
        anchor_y=event.coarse_cy,
        link_radius=link_radius,
        window_radius=window_radius,
        allowed_side=event.court_side,
    )
    start = max(0, event.peak_frame - window_radius)
    end = min(len(candidates_by_frame) - 1, event.peak_frame + window_radius)
    track_x, track_y, track_t, track_score, n_pre, n_post = _estimate_track_bounce_point(
        obs,
        peak_frame=event.peak_frame,
        coarse_x=event.coarse_cx,
        coarse_y=event.coarse_cy,
        start=start,
        end=end,
    )

    heat, crop_box, motion_xy, motion_score_raw = _build_local_motion_heatmap(
        grays,
        search_mask,
        points=[(event.coarse_cx, event.coarse_cy), (track_x, track_y)],
        peak_frame=event.peak_frame,
        diff_th=diff_th,
        window_radius=window_radius,
        roi_radius=roi_radius,
    )
    x1, y1, x2, y2 = crop_box
    h, w = heat.shape[:2]
    refined_x = float(event.coarse_cx)
    refined_y = float(event.coarse_cy)
    refine_score = 0.0
    motion_norm_peak = 0.0

    if h > 0 and w > 0:
        heat_norm = heat / max(1e-6, float(np.max(heat))) if float(np.max(heat)) > 1e-6 else heat.copy()
        yy, xx = np.indices((h, w), dtype=np.float32)
        sigma = max(3.0, roi_radius * 0.28)
        track_prior = np.exp(-((xx - (track_x - x1)) ** 2 + (yy - (track_y - y1)) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        coarse_prior = np.exp(-((xx - (event.coarse_cx - x1)) ** 2 + (yy - (event.coarse_cy - y1)) ** 2) / (2.0 * (sigma * 1.25) * (sigma * 1.25))).astype(np.float32)
        combined = 0.58 * heat_norm + 0.30 * track_prior + 0.12 * coarse_prior
        combined *= (search_mask[y1:y2, x1:x2] > 0).astype(np.float32)

        # side clamp: 一旦 event side 已決定，refine 只能留在那一側
        if net_y is not None and event.court_side in ("near", "far"):
            yy_global = yy + float(y1)
            if event.court_side == "near":
                side_mask = (yy_global >= float(net_y)).astype(np.float32)
            else:
                side_mask = (yy_global < float(net_y)).astype(np.float32)
            combined *= side_mask
        if float(np.max(combined)) > 1e-6:
            iy, ix = np.unravel_index(np.argmax(combined), combined.shape)
            refined_x = float(x1 + ix)
            refined_y = float(y1 + iy)
            refine_score = float(np.max(combined))
            motion_norm_peak = float(heat_norm[iy, ix])
        else:
            refined_x, refined_y = float(track_x), float(track_y)
            refine_score = float(track_score)
    else:
        refined_x, refined_y = float(track_x), float(track_y)
        refine_score = float(track_score)

    event.track_fit_cx = float(track_x)
    event.track_fit_cy = float(track_y)
    event.track_fit_frame = float(track_t)
    event.motion_peak_cx = float(motion_xy[0])
    event.motion_peak_cy = float(motion_xy[1])
    event.motion_score = float(motion_norm_peak if motion_norm_peak > 0 else motion_score_raw)
    event.track_score_refine = float(track_score)
    event.local_track_count_pre = int(n_pre)
    event.local_track_count_post = int(n_post)
    event.refined_cx = float(refined_x)
    event.refined_cy = float(refined_y)
    new_frame = int(np.ceil(track_t)) if track_score > 0 else int(event.peak_frame)
    event.refined_frame = new_frame
    event.peak_frame = new_frame
    event.refine_score = float(refine_score)
    event.cx = float(refined_x)
    event.cy = float(refined_y)

    # 最終 side 與最終位置一致
    if net_y is not None:
        event.court_side = "near" if refined_y >= float(net_y) else "far"

    event.peak_score = float(0.82 * event.peak_score + 0.18 * refine_score)
    return event


def spatiotemporal_nms(events: List[BounceEvent], frame_gap: int = 8, dist_px: float = 28.0) -> List[BounceEvent]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e.peak_score, reverse=True)
    kept: List[BounceEvent] = []
    d2_lim = dist_px * dist_px

    for evt in events_sorted:
        suppress = False
        for k in kept:
            if abs(evt.peak_frame - k.peak_frame) <= frame_gap:
                dx = evt.cx - k.cx
                dy = evt.cy - k.cy
                if dx * dx + dy * dy <= d2_lim:
                    suppress = True
                    break
        if not suppress:
            kept.append(evt)

    kept.sort(key=lambda e: e.peak_frame)
    return kept


def stage2_validate_event_window(
    evt: BounceEvent,
    candidates_by_frame: List[List[BlobCandidate]],
    near_neighbor_radius: float,
    far_neighbor_radius: float,
    window_radius: int = 7,
) -> Stage2CheckResult:
    center_frame = int(evt.peak_frame)
    center_x = float(evt.cx)
    center_y = float(evt.cy)
    chosen_side = str(evt.court_side)

    points = build_postcheck_window_track(
        candidates_by_frame=candidates_by_frame,
        center_frame=center_frame,
        center_x=center_x,
        center_y=center_y,
        chosen_side=chosen_side,
        near_neighbor_radius=near_neighbor_radius,
        far_neighbor_radius=far_neighbor_radius,
        window_radius=window_radius,
    )

    valid_count, side_consistency, quality_mean, smoothness_score, gap_penalty, sequence_score = score_postcheck_sequence(
        points,
        chosen_side=chosen_side,
    )

    # 這裡只拿來評估整段像不像 bounce event，不改 frame
    best_bounce_score = -1.0
    direction_change_best = 0.0
    speed_dip_best = 0.0

    for i in range(1, len(points) - 1):
        s, x, y = score_postcheck_bounce_moment(points, i)
        if s > best_bounce_score:
            best_bounce_score = s

            direction_change_best, _, _ = score_window_direction_change(points, i, min_side_points=2)

            p = points[i]
            prevs = [points[j] for j in range(0, i) if points[j].is_valid == 1]
            nexts = [points[j] for j in range(i + 1, len(points)) if points[j].is_valid == 1]

            if len(prevs) >= 1 and len(nexts) >= 1:
                p_prev = prevs[-1]
                p_next = nexts[0]

                prev_step = float(np.hypot(p.x - p_prev.x, p.y - p_prev.y))
                next_step = float(np.hypot(p_next.x - p.x, p_next.y - p.y))

                all_steps = []
                for k in range(1, len(points)):
                    if points[k - 1].is_valid == 1 and points[k].is_valid == 1:
                        all_steps.append(float(np.hypot(points[k].x - points[k - 1].x, points[k].y - points[k - 1].y)))

                local_mean = float(np.mean(all_steps)) if all_steps else max(prev_step, next_step, 1.0)
                center_step = 0.5 * (prev_step + next_step)
                speed_dip_best = float(np.clip(1.0 - center_step / max(local_mean * 1.15, 1.0), 0.0, 1.0))

    if best_bounce_score < 0:
        best_bounce_score = 0.0

    event_likeness_score = float(
        0.34 * sequence_score
        + 0.26 * best_bounce_score
        + 0.18 * direction_change_best
        + 0.14 * speed_dip_best
        + 0.08 * quality_mean
    )
    event_likeness_score = float(np.clip(event_likeness_score, 0.0, 1.0))

    keep_event = 1
    if valid_count < 5:
        keep_event = 0
    if side_consistency < 0.60:
        keep_event = 0
    if event_likeness_score < 0.40:
        keep_event = 0

    return Stage2CheckResult(
        original_peak_frame=int(center_frame),
        refined_peak_frame=int(center_frame),   # 不改
        chosen_side=str(chosen_side),
        valid_count=int(valid_count),
        side_consistency=float(side_consistency),
        quality_mean=float(quality_mean),
        smoothness_score=float(smoothness_score),
        gap_penalty=float(gap_penalty),
        direction_change_score=float(direction_change_best),
        speed_dip_score=float(speed_dip_best),
        event_likeness_score=float(event_likeness_score),
        split_fit_score=0.0,
        final_score=float(event_likeness_score),
        refined_x=float(center_x),              # 不改
        refined_y=float(center_y),              # 不改
        keep_event=int(keep_event),
        points=points,
    )
    

def apply_stage2_result_to_event(evt: BounceEvent, s2: Stage2CheckResult) -> Optional[BounceEvent]:
    if s2.keep_event != 1:
        return None

    # 額外刪除缺乏真實前後軌跡支撐的假峰
    if evt.prev_step <= 1e-6 and evt.speed_dip_2 <= 1e-6 and evt.track_score_refine <= 1e-6:
        return None

    evt.peak_frame = int(s2.refined_peak_frame)
    evt.refined_frame = int(s2.refined_peak_frame)

    evt.cx = float(s2.refined_x)
    evt.cy = float(s2.refined_y)
    evt.refined_cx = float(s2.refined_x)
    evt.refined_cy = float(s2.refined_y)

    evt.court_side = str(s2.chosen_side)
    evt.support_5f = int(min(5, s2.valid_count))

    evt.peak_score = float(
        0.45 * evt.peak_score
        + 0.35 * s2.event_likeness_score
        + 0.20 * s2.split_fit_score
    )

    evt.motion_score = float(s2.event_likeness_score)
    evt.refine_score = float(s2.final_score)
    evt.track_score_refine = float(s2.smoothness_score)

    return evt

def draw_stage2_track_debug(
    out_path: Path,
    frame_path: Path,
    scale: float,
    points: List[WindowPoint],
    original_peak_frame: int,
    refined_peak_frame: int,
    net_y: Optional[float] = None,
    draw_lines: bool = True,
    show_invalid: bool = False,
):
    im = cv2.imread(str(frame_path))
    if im is None:
        return
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    vis = im.copy()

    # 畫 net line（可保留）
    if net_y is not None:
        y = int(round(net_y))
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (255, 0, 255), 1)

    valid_pts = [p for p in points if p.is_valid == 1]

    # 很細的軌跡線
    if draw_lines and len(valid_pts) >= 2:
        for i in range(1, len(valid_pts)):
            x1, y1 = int(round(valid_pts[i - 1].x)), int(round(valid_pts[i - 1].y))
            x2, y2 = int(round(valid_pts[i].x)), int(round(valid_pts[i].y))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 220), 1, cv2.LINE_AA)

    # 畫點：一般點小一點，peak 用不同顏色
    for p in points:
        x = int(round(p.x))
        y = int(round(p.y))

        if p.is_valid != 1:
            if show_invalid:
                cv2.circle(vis, (x, y), 2, (120, 120, 120), -1, lineType=cv2.LINE_AA)
            continue

        # refined peak：紅色
        if p.frame_idx == refined_peak_frame:
            color = (0, 0, 255)
            radius = 4
        # original peak：黃色（如果和 refined 不同）
        elif p.frame_idx == original_peak_frame:
            color = (0, 255, 255)
            radius = 3
        # 其他正常點：綠色
        else:
            color = (0, 255, 0)
            radius = 2

        cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

        # frame 編號，小字，放右上
        cv2.putText(
            vis,
            str(p.frame_idx),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 左上角只留很少資訊
    info = f"orig={original_peak_frame} ref={refined_peak_frame}"
    cv2.putText(
        vis,
        info,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), vis)
    
def draw_scan_debug(
    out_path: Path,
    gray: np.ndarray,
    search_mask: np.ndarray,
    player_mask: np.ndarray,
    fg_union: np.ndarray,
    fg_inter: np.ndarray,
    candidates: List[BlobCandidate],
    title: str,
    player_boxes: Optional[List[Tuple[int, int, int, int, float]]] = None,
    net_y: Optional[float] = None,
):
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    search_overlay = canvas.copy()
    search_overlay[search_mask > 0] = (0, 80, 0)
    canvas = cv2.addWeighted(canvas, 0.86, search_overlay, 0.14, 0)

    player_overlay = canvas.copy()
    player_overlay[player_mask > 0] = (0, 0, 180)
    canvas = cv2.addWeighted(canvas, 0.86, player_overlay, 0.25, 0)

    canvas[fg_union > 0] = (255, 180, 0)
    canvas[fg_inter > 0] = (0, 255, 255)

    if net_y is not None:
        y = int(round(net_y))
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (255, 0, 255), 1)
        cv2.putText(canvas, "net_y", (8, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if player_boxes:
        for x1, y1, x2, y2, conf in player_boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"person {conf:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for cand in candidates:
        cv2.rectangle(canvas, (cand.x, cand.y), (cand.x + cand.w, cand.y + cand.h), (255, 255, 255), 1)
        cv2.circle(canvas, (int(round(cand.cx)), int(round(cand.cy))), 2, (0, 0, 255), -1)
        side_tag = "N" if cand.court_side == "near" else ("F" if cand.court_side == "far" else "?")
        p1 = cand.prev_step if np.isfinite(cand.prev_step) and cand.prev_step < 9998 else -1.0
        n1 = cand.next_step if np.isfinite(cand.next_step) and cand.next_step < 9998 else -1.0
        txt = f"{side_tag} sc={cand.score_event:.2f} p={p1:.1f} n={n1:.1f} d1={cand.speed_dip_1:.2f}"
        cv2.putText(canvas, txt, (cand.x, max(12, cand.y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def draw_event_debug(out_path: Path, frame_path: Path, scale: float, event: BounceEvent, search_mask: np.ndarray, net_y: Optional[float] = None):
    im = cv2.imread(str(frame_path))
    if im is None:
        return
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    overlay = im.copy()
    overlay[search_mask > 0] = (0, 70, 0)
    vis = cv2.addWeighted(im, 0.86, overlay, 0.14, 0)

    if net_y is not None:
        y = int(round(net_y))
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (255, 0, 255), 1)

    coarse_p = (int(round(event.coarse_cx)), int(round(event.coarse_cy)))
    track_p = (int(round(event.track_fit_cx)), int(round(event.track_fit_cy)))
    motion_p = (int(round(event.motion_peak_cx)), int(round(event.motion_peak_cy)))
    final_p = (int(round(event.refined_cx)), int(round(event.refined_cy)))

    cv2.circle(vis, coarse_p, 8, (255, 180, 0), 2)
    cv2.circle(vis, track_p, 8, (255, 0, 255), 2)
    cv2.circle(vis, motion_p, 8, (0, 255, 255), 2)
    cv2.circle(vis, final_p, 11, (0, 0, 255), 2)
    cv2.circle(vis, final_p, 3, (255, 255, 255), -1)

    cv2.putText(vis, 'coarse', (coarse_p[0] + 6, coarse_p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, 'track', (track_p[0] + 6, track_p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, 'motion', (motion_p[0] + 6, motion_p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, 'final', (final_p[0] + 6, final_p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    txt1 = f"bounce frame={event.peak_frame} score={event.peak_score:.3f} side={event.court_side}"
    txt2 = f"support5={event.support_5f} prev={event.prev_step:.2f} next={event.next_step:.2f}"
    txt3 = f"track={event.track_score_refine:.2f} motion={event.motion_score:.2f} refine={event.refine_score:.2f}"
    cv2.putText(vis, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt3, (10, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)

def draw_hit_interval_y_debug(
    out_path: Path,
    ball_y_df: pd.DataFrame,
    hit_df: pd.DataFrame,
    net_y: Optional[float] = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        ball_y_df["frame_idx"],
        ball_y_df["y_filled"],
        alpha=0.35,
        label="y_filled"
    )

    ax.plot(
        ball_y_df["frame_idx"],
        ball_y_df["y_smooth"],
        linewidth=2,
        label="y_smooth"
    )

    # 沒有 candidate 的 frame 標出來，方便看遠端漏幀
    missing_df = ball_y_df[ball_y_df["has_candidate"] == 0]
    if len(missing_df) > 0:
        ax.scatter(
            missing_df["frame_idx"],
            missing_df["y_filled"],
            s=10,
            alpha=0.35,
            label="missing/interpolated"
        )

    # 只畫 near / far 的 hit 線，不畫區間背景
    if hit_df is not None and len(hit_df) > 0:
        for _, row in hit_df.iterrows():
            center = int(row["center_frame"])
            side = str(row["hit_side"])

            ax.axvline(center, linestyle="--", alpha=0.7)
            ax.text(
                center,
                float(row["center_y"]),
                side,
                fontsize=9,
                ha="center",
                va="bottom"
            )

    if net_y is not None:
        ax.axhline(
            float(net_y),
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="net_y"
        )

    ax.set_title("Detected hit lines by ball y")
    ax.set_xlabel("frame")
    ax.set_ylabel("ball y")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    
def run_bounce_detector(
    video_path: Path,
    scale: float = 1.0,
    diff_th: int = 10,
    min_area: int = 2,
    max_area: int = 90,
    max_wh: int = 22,
    near_neighbor_radius: float = 22.0,
    far_neighbor_radius: float = 8.0,
    peak_smooth: int = 1,
    peak_min_gap: int = 10,
    peak_min_score: float = 0.58,
    yolo_model_path: str = "yolov8n.pt",
    yolo_device: str = "auto",
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.25,
    yolo_every_n: int = 1,
    net_penalty_radius: float = 12.0,
    net_penalty_weight: float = 0.25,
    roi_expand_px: int = 8,
    far_roi_expand_px: int = 18,
    far_roi_expand_dx: int = 8,
    valid_mask_expand_px: int = 6,
    debug: bool = False,
):
    paths = set_video_path(video_path)
    frames_dir = paths["FRAMES_DIR"]
    mask_path = paths["MASK_PATH"]
    roi_json = paths["ROI_JSON"]
    out_dir = paths["OUT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, grays, target_hw = load_frames_gray(frames_dir, scale=scale)
    valid_mask = load_valid_mask(mask_path, target_hw)
    net_y = estimate_net_y(roi_json if roi_json.exists() else None, target_hw, scale=scale)
    search_mask_static = build_search_mask(
        valid_mask,
        roi_json if roi_json.exists() else None,
        scale=scale,
        erode_iter=1,
        roi_expand_px=roi_expand_px,
        valid_mask_expand_px=valid_mask_expand_px,
        far_expand_px=far_roi_expand_px,
        far_expand_dx=far_roi_expand_dx,
        net_y=net_y,
    )
    net_dist_map = build_net_distance_map(roi_json if roi_json.exists() else None, target_hw, scale=scale)

    if debug:
        cv2.imwrite(str(debug_dir / "valid_mask.png"), valid_mask)
        cv2.imwrite(str(debug_dir / "search_mask_static.png"), search_mask_static)
        if net_dist_map is not None:
            net_vis = np.clip(255.0 * np.minimum(net_dist_map / max(1.0, float(np.max(net_dist_map))), 1.0), 0, 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / "net_distance_map.png"), net_vis)
    
    n = len(frame_paths)
    candidates_by_frame: List[List[BlobCandidate]] = [[] for _ in range(n)]
    frame_rows = []
    # Original-frame YOLO boxes shared with vote_action.
    yolo_raw_rows = []
    tracked_player_rows = []

    cached_boxes: List[Tuple[int, int, int, int, float]] = []
    cached_tracked_players = []

    cached_core_mask = np.zeros_like(search_mask_static)
    cached_soft_mask = np.zeros_like(search_mask_static)
    
    roi_data = load_json(roi_json) if roi_json.exists() else {}
    net_line = roi_data.get("NET_LINE", [])
    
    # 2. 初始化 Detector 與 Tracker
    # Keep these parameters aligned with vote_action.py so shared cache uses the same YOLO settings.
    detector = PlayerDetector(
        model_path=yolo_model_path,
        device=yolo_device,
        imgsz=yolo_imgsz,
        conf=yolo_conf,
    )
    if not net_line:
        # 如果沒設網子，給個畫面中間的預設值
        h, w = grays[0].shape
        net_line = [[0, h//2], [w, h//2]]
    tracker = TwoPlayerTracker(NET_LINE=net_line)

    for i in range(1, n - 1):
        diff_prev, diff_next, fg_union, fg_inter = build_motion_triplet(
            grays[i - 1], grays[i], grays[i + 1],
            search_mask_static,
            diff_th=diff_th,
            blur_ksize=3,
        )
        if (i == 1) or (yolo_every_n <= 1) or (i % yolo_every_n == 0):
            # Important: run YOLO on the ORIGINAL frame.
            # vote_action uses original-frame geometry for crop; bounce masks may use scaled coordinates.
            frame_bgr_orig = cv2.imread(str(frame_paths[i]))
            if frame_bgr_orig is None:
                raise ValueError(f"Cannot read frame for YOLO: {frame_paths[i]}")

            orig_h, orig_w = frame_bgr_orig.shape[:2]
            raw_boxes_orig = detector.detect(frame_bgr_orig)
            append_yolo_raw_box_rows(
                yolo_raw_rows,
                frame_idx=i,
                boxes_orig=raw_boxes_orig,
                img_w=orig_w,
                img_h=orig_h,
                scale=scale,
            )

            # Convert raw boxes to bounce.py coordinate system for tracker/mask.
            raw_boxes_for_bounce = scale_yolo_boxes_for_bounce(
                raw_boxes_orig,
                scale=scale,
                target_hw=search_mask_static.shape[:2],
            )

            frame_bgr_for_mask = frame_bgr_orig
            if scale != 1.0:
                frame_bgr_for_mask = cv2.resize(
                    frame_bgr_orig,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            core_mask, soft_mask, cached_boxes, cached_tracked_players, _ = build_player_exclusion_masks_with_tracker(
                frame_bgr=frame_bgr_for_mask,
                search_mask=search_mask_static,
                detector=None,
                tracker=tracker,
                net_y=net_y,
                raw_boxes=raw_boxes_for_bounce,
                near_core_pad_ratio=0.08,
                near_soft_pad_ratio=0.28,
                far_core_pad_ratio=0.03,
                far_soft_pad_ratio=0.16,
            )
            cached_core_mask = core_mask.copy()
            cached_soft_mask = soft_mask.copy()
        else:
            core_mask = cached_core_mask.copy()
            soft_mask = cached_soft_mask.copy()
            
        # ------------------------------------------------------------
        # Save tracked player boxes for later pose / stroke classification
        # 注意：
        # - 如果 scale=1.0，x1/y1/x2/y2 就是原始 frame 座標
        # - 如果 scale != 1.0，這裡同時存 scaled 座標與 original 座標
        # ------------------------------------------------------------
        for player_name, box in cached_tracked_players:
            x1, y1, x2, y2, conf = box

            w_box = float(x2 - x1)
            h_box = float(y2 - y1)
            cx_box = float((x1 + x2) / 2.0)
            cy_box = float((y1 + y2) / 2.0)

            if scale != 0:
                x1_orig = float(x1) / float(scale)
                y1_orig = float(y1) / float(scale)
                x2_orig = float(x2) / float(scale)
                y2_orig = float(y2) / float(scale)
            else:
                x1_orig, y1_orig, x2_orig, y2_orig = float(x1), float(y1), float(x2), float(y2)

            tracked_player_rows.append({
                "frame_idx": int(i),
                "player": str(player_name),

                # 座標是 bounce.py 當下使用的座標；scale=1 時就是原圖座標
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf),
                "cx": cx_box,
                "cy": cy_box,
                "w": w_box,
                "h": h_box,

                # 保險起見，也存原始 frame 座標
                "x1_orig": x1_orig,
                "y1_orig": y1_orig,
                "x2_orig": x2_orig,
                "y2_orig": y2_orig,

                "scale": float(scale),
            })

        search_mask_dyn = cv2.bitwise_and(search_mask_static, cv2.bitwise_not(core_mask))
        fg_union = cv2.bitwise_and(fg_union, search_mask_dyn)
        fg_inter = cv2.bitwise_and(fg_inter, search_mask_dyn)
        
        human_dist_map = None
        if np.count_nonzero(soft_mask) > 0:
            inv_soft = cv2.bitwise_not(soft_mask)
            human_dist_map = cv2.distanceTransform(inv_soft, cv2.DIST_L2, 3).astype(np.float32)

        frame_cands = extract_blob_candidates(
            frame_idx=i,
            fg_union=fg_union,
            fg_inter=fg_inter,
            diff_prev=diff_prev,
            diff_next=diff_next,
            min_area=min_area,
            max_area=max_area,
            max_wh=max_wh,
            inter_weight=0.25,
            net_dist_map=net_dist_map,
            net_penalty_radius=net_penalty_radius,
            net_penalty_weight=net_penalty_weight,
            human_dist_map=human_dist_map,
            human_penalty_radius=18.0,
            human_penalty_weight=0.28,
        )
        candidates_by_frame[i] = frame_cands

        frame_rows.append({
            "frame_idx": i,
            "n_candidates": len(frame_cands),
            "motion_union_pixels": int(np.count_nonzero(fg_union)),
            "motion_inter_pixels": int(np.count_nonzero(fg_inter)),
            "player_mask_pixels": int(np.count_nonzero(cv2.bitwise_or(core_mask, soft_mask))),
            "n_player_boxes": int(len(cached_boxes)),
        })

        if debug and (len(frame_cands) > 0 or i % 30 == 0):
            player_mask_vis = cv2.bitwise_or(core_mask, soft_mask)

            draw_scan_debug(
                debug_dir / f"scan_{i:06d}.png",
                grays[i],
                search_mask_dyn,
                player_mask_vis,
                fg_union,
                fg_inter,
                frame_cands,
                title=f"frame={i} n={len(frame_cands)}",
                player_boxes=cached_boxes,
                net_y=net_y,
            )

    # 先補單幀缺口，再做正式 event scoring
    candidates_by_frame = interpolate_single_frame_gaps(
        candidates_by_frame=candidates_by_frame,
        search_mask=search_mask_static,
        net_y=net_y,
        near_neighbor_radius=near_neighbor_radius,
        far_neighbor_radius=far_neighbor_radius,
        max_interp_per_frame=1,
        debug=debug,
    )

    candidates_by_frame, frame_scores_raw = score_bounce_events(
        candidates_by_frame,
        net_y=net_y,
        near_neighbor_radius=near_neighbor_radius,
        far_neighbor_radius=far_neighbor_radius,
    )
    frame_scores_smooth = smooth_1d(frame_scores_raw, ksize=peak_smooth)
    
    # ========================================================
    # Hit interval detection by visible y high/low segments
    # --------------------------------------------------------
    # high segment -> near hit interval
    # low segment  -> far hit interval
    # consecutive same-side segments are cleaned; incomplete tail far is removed
    # ========================================================
    ball_y_df = build_ball_y_signal_from_candidates(
        candidates_by_frame=candidates_by_frame,
        max_interp_gap=12,
        smooth_win=5,
    )

    hit_interval_df = detect_hit_intervals_by_clear_high_low_segments(
        ball_y_df=ball_y_df,
        net_y=net_y,
        y_col="y_smooth",
        margin=18.0,
        min_segment_len=4,
        max_merge_gap=3,
        pad=2,
    )

    if debug:
        draw_hit_interval_y_debug(
            out_path=debug_dir / "hit_intervals_y_debug.png",
            ball_y_df=ball_y_df,
            hit_df=hit_interval_df,
            net_y=net_y,
        )

    vote_scores_raw, vote_anchors, vote_support = build_temporal_spatial_vote_signal(
        candidates_by_frame,
        near_neighbor_radius=near_neighbor_radius,
        far_neighbor_radius=far_neighbor_radius,
        vote_radius=3,
    )

    vote_scores_smooth = smooth_1d(vote_scores_raw, ksize=peak_smooth)

    # 先抓 vote peak，再對 raw==1 的 peak 做 relocation
    vote_peak_indices_base = find_peaks_1d(
        vote_scores_smooth,
        min_score=peak_min_score,
        min_gap=peak_min_gap
    )

    vote_peak_indices = []
    for idx in vote_peak_indices_base:
        new_idx = relocate_saturated_vote_peak(
            peak_idx=idx,
            vote_scores_raw=vote_scores_raw,
            vote_anchors=vote_anchors,
            candidates_by_frame=candidates_by_frame,
            search_window=5,
            one_th=0.999,
            debug=debug,
        )
        vote_peak_indices.append(new_idx)

    vote_peak_indices = sorted(set(vote_peak_indices))

    raw_peak_indices = find_peaks_1d(
        frame_scores_raw,
        min_score=max(0.45, peak_min_score - 0.08),
        min_gap=peak_min_gap
    )

    peak_indices = vote_peak_indices[:]

    if debug:
        print("[raw_peaks]", raw_peak_indices)
        print("[vote_base_peaks]", vote_peak_indices_base)
        print("[vote_relocated_peaks]", vote_peak_indices)
    
    raw_events: List[BounceEvent] = []
    for peak_idx in peak_indices:
        anchor = get_anchor_for_peak(
            peak_idx=peak_idx,
            vote_anchors=vote_anchors,
            candidates_by_frame=candidates_by_frame,
        )
        if anchor is None:
            continue

        evt = make_event_from_vote_peak(
            peak_idx,
            anchor,
            candidates_by_frame,
            near_refine_radius=near_neighbor_radius + 4.0,
            far_refine_radius=far_neighbor_radius + 4.0,
            window_radius=3,
            enable_side_arbitration=True,
            side_conflict_check_radius=1,
            side_look_radius=5,
        )
        if evt is not None:
            raw_events.append(evt)

    events = spatiotemporal_nms(
        raw_events,
        frame_gap=peak_min_gap,
        dist_px=max(near_neighbor_radius + 10.0, far_neighbor_radius + 14.0)
    )

    stage2_rows = []
    stage2_point_rows = []
    filtered_events = []

    for evt in events:
        s2 = stage2_validate_event_window(
            evt=evt,
            candidates_by_frame=candidates_by_frame,
            near_neighbor_radius=near_neighbor_radius,
            far_neighbor_radius=far_neighbor_radius,
            window_radius=7,
        )

        # if s2.keep_event == 1:
        #     s2 = refine_frame_by_piecewise_fit(s2)

        stage2_rows.append({
            "original_peak_frame": int(s2.original_peak_frame),
            "refined_peak_frame": int(s2.refined_peak_frame),
            "chosen_side": str(s2.chosen_side),
            "valid_count": int(s2.valid_count),
            "side_consistency": float(s2.side_consistency),
            "quality_mean": float(s2.quality_mean),
            "smoothness_score": float(s2.smoothness_score),
            "gap_penalty": float(s2.gap_penalty),
            "direction_change_score": float(s2.direction_change_score),
            "speed_dip_score": float(s2.speed_dip_score),
            "event_likeness_score": float(s2.event_likeness_score),
            "split_fit_score": float(s2.split_fit_score),
            "final_score": float(s2.final_score),
            "refined_x": float(s2.refined_x),
            "refined_y": float(s2.refined_y),
            "keep_event": int(s2.keep_event),
        })

        for p in s2.points:
            stage2_point_rows.append({
                "original_peak_frame": int(s2.original_peak_frame),
                "refined_peak_frame": int(s2.refined_peak_frame),
                "chosen_side": str(s2.chosen_side),
                "rel_idx": int(p.rel_idx),
                "frame_idx": int(p.frame_idx),
                "x": float(p.x),
                "y": float(p.y),
                "score_blob": float(p.score_blob),
                "score_event": float(p.score_event),
                "track_score_1": float(p.track_score_1),
                "track_score_2": float(p.track_score_2),
                "support_5f": int(p.support_5f),
                "court_side": str(p.court_side),
                "distance_to_pred": float(p.distance_to_pred),
                "picked_by": str(p.picked_by),
                "is_valid": int(p.is_valid),
            })

        new_evt = apply_stage2_result_to_event(evt, s2)
        if new_evt is not None:
            filtered_events.append(new_evt)
            
    events = filtered_events

    # 你原本的 refine 還是可以保留，放在 post-check 後面
    events = [
        refine_event_location_with_track_and_motion(
            evt,
            candidates_by_frame=candidates_by_frame,
            grays=grays,
            search_mask=search_mask_static,
            diff_th=diff_th,
            near_neighbor_radius=near_neighbor_radius,
            far_neighbor_radius=far_neighbor_radius,
            net_y=net_y,
            window_radius=3,
        )
        for evt in events
    ]

    events = spatiotemporal_nms(
        events,
        frame_gap=peak_min_gap,
        dist_px=max(near_neighbor_radius + 10.0, far_neighbor_radius + 14.0)
    )

    score_rows = []
    peak_set = set(peak_indices)
    final_peak_set = {e.peak_frame for e in events}
    print("[final_peaks]", sorted(final_peak_set))
    for i in range(n):
        anchor = vote_anchors[i] if i < len(vote_anchors) else None
        score_rows.append({
            "frame_idx": i,
            "candidate_max_score_raw": float(frame_scores_raw[i]) if i < len(frame_scores_raw) else 0.0,
            "candidate_max_score_smooth": float(frame_scores_smooth[i]) if i < len(frame_scores_smooth) else 0.0,
            "vote_score_raw": float(vote_scores_raw[i]) if i < len(vote_scores_raw) else 0.0,
            "vote_score_smooth": float(vote_scores_smooth[i]) if i < len(vote_scores_smooth) else 0.0,
            "vote_support_frames": float(vote_support[i]) if i < len(vote_support) else 0.0,
            "vote_anchor_cx": float(anchor.cx) if anchor is not None else np.nan,
            "vote_anchor_cy": float(anchor.cy) if anchor is not None else np.nan,
            "n_candidates": len(candidates_by_frame[i]),
            "is_peak_raw": int(i in peak_set),
            "is_peak_final": int(i in final_peak_set),
        })
    pd.DataFrame(score_rows).to_csv(out_dir / "frame_scores.csv", index=False, encoding="utf-8")

    cand_rows = []
    for frame_cands in candidates_by_frame:
        for cand in frame_cands:
            cand_rows.append(asdict(cand))
    pd.DataFrame(cand_rows).to_csv(out_dir / "bounce_candidates.csv", index=False, encoding="utf-8")

    pd.DataFrame([asdict(e) for e in events]).to_csv(out_dir / "bounce_events.csv", index=False, encoding="utf-8")
    pd.DataFrame(frame_rows).to_csv(out_dir / "scan_stats.csv", index=False, encoding="utf-8")
    pd.DataFrame(stage2_rows).to_csv(out_dir / "stage2_debug.csv", index=False, encoding="utf-8")
    pd.DataFrame(stage2_point_rows).to_csv(out_dir / "stage2_points_debug.csv", index=False, encoding="utf-8")

    # Shared YOLO cache for vote_action.
    yolo_raw_df = pd.DataFrame(yolo_raw_rows)
    yolo_raw_df.to_csv(out_dir / "yolo_person_boxes.csv", index=False, encoding="utf-8-sig")
    # Compatibility name used by vote_action_tracked_select_raw_crop.py style caches.
    yolo_raw_df.to_csv(out_dir / "yolo_person_boxes_vote_cache.csv", index=False, encoding="utf-8-sig")

    tracked_df = pd.DataFrame(tracked_player_rows)
    tracked_df.to_csv(
        out_dir / "tracked_players.csv",
        index=False,
        encoding="utf-8"
    )
    tracked_df.to_csv(
        out_dir / "tracked_players_vote_cache.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    ball_y_df.to_csv(
        out_dir / "ball_y_signal.csv",
        index=False,
        encoding="utf-8"
    )

    # 保留舊檔名，避免其他程式讀不到；內容已改為 high/low segment 結果。
    hit_interval_df.to_csv(
        out_dir / "hit_intervals_y_extrema.csv",
        index=False,
        encoding="utf-8"
    )
    hit_interval_df.to_csv(
        out_dir / "hit_intervals_y_segments.csv",
        index=False,
        encoding="utf-8"
    )

    if debug:
        final_track_dir = debug_dir / "final_peak_tracks"
        final_track_dir.mkdir(parents=True, exist_ok=True)

        for idx, evt in enumerate(events):
            draw_event_debug(
                debug_dir / f"event_{idx:02d}_frame_{evt.peak_frame:06d}_{evt.court_side}.png",
                frame_paths[evt.peak_frame],
                scale,
                evt,
                search_mask_static,
                net_y=net_y,
            )

            # 重新為 final event 建局部短軌跡
            final_points = build_postcheck_window_track(
                candidates_by_frame=candidates_by_frame,
                center_frame=int(evt.peak_frame),
                center_x=float(evt.cx),
                center_y=float(evt.cy),
                chosen_side=str(evt.court_side),
                near_neighbor_radius=near_neighbor_radius,
                far_neighbor_radius=far_neighbor_radius,
                window_radius=7,
            )

            draw_stage2_track_debug(
                out_path=final_track_dir / f"final_track_{idx:02d}_frame_{evt.peak_frame:06d}_{evt.court_side}.png",
                frame_path=frame_paths[int(evt.peak_frame)],
                scale=scale,
                points=final_points,
                original_peak_frame=int(evt.peak_frame),
                refined_peak_frame=int(evt.peak_frame),
                net_y=net_y,
                draw_lines=True,
                show_invalid=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="例如 raw_videos/xxx.mp4")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--diff-th", type=int, default=10)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=90)
    parser.add_argument("--max-wh", type=int, default=22)
    parser.add_argument("--near-neighbor-radius", type=float, default=22.0)
    parser.add_argument("--far-neighbor-radius", type=float, default=8.0)
    parser.add_argument("--peak-smooth", type=int, default=1)
    parser.add_argument("--peak-min-gap", type=int, default=10)
    parser.add_argument("--peak-min-score", type=float, default=0.58)
    parser.add_argument("--yolo-model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-device", type=str, default="auto")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-every-n", type=int, default=1)
    parser.add_argument("--net-penalty-radius", type=float, default=12.0)
    parser.add_argument("--net-penalty-weight", type=float, default=0.25)
    parser.add_argument("--roi-expand-px", type=int, default=8)
    parser.add_argument("--far-roi-expand-px", type=int, default=18)
    parser.add_argument("--far-roi-expand-dx", type=int, default=8)
    parser.add_argument("--valid-mask-expand-px", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_bounce_detector(
        video_path=Path(args.video_path),
        scale=args.scale,
        diff_th=args.diff_th,
        min_area=args.min_area,
        max_area=args.max_area,
        max_wh=args.max_wh,
        near_neighbor_radius=args.near_neighbor_radius,
        far_neighbor_radius=args.far_neighbor_radius,
        peak_smooth=args.peak_smooth,
        peak_min_gap=args.peak_min_gap,
        peak_min_score=args.peak_min_score,
        yolo_model_path=args.yolo_model_path,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_every_n=args.yolo_every_n,
        net_penalty_radius=args.net_penalty_radius,
        net_penalty_weight=args.net_penalty_weight,
        roi_expand_px=args.roi_expand_px,
        far_roi_expand_px=args.far_roi_expand_px,
        far_roi_expand_dx=args.far_roi_expand_dx,
        valid_mask_expand_px=args.valid_mask_expand_px,
        debug=args.debug,
    )

