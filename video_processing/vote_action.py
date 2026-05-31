"""
vote_action_tracked_select_raw_crop.py
------------------------------------------------------------
Event-level forehand/backhand voting for near-side tennis hits.

Core idea:
1. Run YOLO once over the full video frames in chronological order.
2. Feed the raw YOLO boxes into TwoPlayerTracker.
3. For each vote frame, use the tracker only to decide which person is the near/far player.
4. Match that tracked player back to the closest raw YOLO box.
5. Use the matched raw YOLO box with the original player_crop.py expand_box() policy.
6. If matching/cropping fails, skip that frame by default.

This keeps crop geometry close to the dataset-maker / original vote_action crop distribution,
while using tracking to reduce wrong-person selection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Reuse the exact model / transform definitions used during training.
from train_tennis_actions import build_model, build_transforms, get_device

# Reuse the exact crop policy from your photo-crop dataset maker.
from player_crop import (
    choose_nearest_largest_box,
    expand_box,
    save_debug_image,
)
from player_detector import PlayerDetector
from player_tracker import TwoPlayerTracker


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_name_from_path(video_path: Path) -> str:
    return video_path.stem


def default_paths(video_path: Path) -> Dict[str, Path]:
    video_name = video_name_from_path(video_path)
    base_dir = Path("dataset") / video_name
    out_dir = base_dir / "bounce_detector"
    return {
        "base_dir": base_dir,
        "frames_dir": base_dir / "frames",
        "bounce_out_dir": out_dir,
        "events_segments": out_dir / "hit_intervals_y_segments.csv",
        "events_extrema": out_dir / "hit_intervals_y_extrema.csv",
        "events_bounce": out_dir / "bounce_events.csv",
        "stroke_out_dir": out_dir / "stroke_vote_same_crop",
    }


def list_frame_paths(frames_dir: Path) -> List[Path]:
    frame_paths = []
    for ext in IMG_EXTS:
        frame_paths.extend(frames_dir.glob(f"*{ext}"))
        frame_paths.extend(frames_dir.glob(f"*{ext.upper()}"))
    frame_paths = sorted(set(frame_paths))
    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in: {frames_dir}")
    return frame_paths


def resolve_event_file(paths: Dict[str, Path], event_file: Optional[str]) -> Path:
    if event_file and event_file.lower() != "auto":
        p = Path(event_file)
        if not p.exists():
            raise FileNotFoundError(f"Event file not found: {p}")
        return p

    candidates = [
        paths["events_segments"],
        paths["events_extrema"],
        paths["events_bounce"],
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p, nrows=1)
                if len(df.columns) > 0:
                    return p
            except Exception:
                continue
    raise FileNotFoundError(
        "No non-empty event CSV found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def load_events(event_file: Path) -> pd.DataFrame:
    df = pd.read_csv(event_file)
    if len(df) == 0:
        raise ValueError(f"Event file is empty: {event_file}")

    if "center_frame" in df.columns:
        df["event_center_frame"] = df["center_frame"].astype(int)
    elif "peak_frame" in df.columns:
        df["event_center_frame"] = df["peak_frame"].astype(int)
    elif "refined_peak_frame" in df.columns:
        df["event_center_frame"] = df["refined_peak_frame"].astype(int)
    else:
        raise ValueError(
            "Cannot find event center column. Need one of: "
            "center_frame, peak_frame, refined_peak_frame"
        )

    if "hit_side" in df.columns:
        df["event_side"] = df["hit_side"].astype(str).str.lower()
    elif "court_side" in df.columns:
        df["event_side"] = df["court_side"].astype(str).str.lower()
    elif "chosen_side" in df.columns:
        df["event_side"] = df["chosen_side"].astype(str).str.lower()
    else:
        df["event_side"] = "unknown"

    if "center_x" in df.columns:
        df["event_x"] = pd.to_numeric(df["center_x"], errors="coerce")
    elif "cx" in df.columns:
        df["event_x"] = pd.to_numeric(df["cx"], errors="coerce")
    elif "refined_x" in df.columns:
        df["event_x"] = pd.to_numeric(df["refined_x"], errors="coerce")
    else:
        df["event_x"] = np.nan

    if "center_y" in df.columns:
        df["event_y"] = pd.to_numeric(df["center_y"], errors="coerce")
    elif "cy" in df.columns:
        df["event_y"] = pd.to_numeric(df["cy"], errors="coerce")
    elif "refined_y" in df.columns:
        df["event_y"] = pd.to_numeric(df["refined_y"], errors="coerce")
    else:
        df["event_y"] = np.nan

    df = df.reset_index(drop=True)
    df["event_id"] = np.arange(len(df), dtype=int)
    return df


def filter_events_by_side(events_df: pd.DataFrame, target_side: str) -> pd.DataFrame:
    target_side = str(target_side).lower().strip()
    if target_side == "all":
        out = events_df.copy().reset_index(drop=True)
    elif target_side in ("near", "far"):
        out = events_df[events_df["event_side"].astype(str).str.lower() == target_side].copy()
        out = out.reset_index(drop=True)
    else:
        raise ValueError("target_side must be one of: near, far, all")

    out["event_id"] = np.arange(len(out), dtype=int)
    return out


def cv2_bgr_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)



def load_roi_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_net_line(paths: Dict[str, Path], frame_paths: List[Path], roi_json_arg: Optional[str] = None) -> List[List[float]]:
    """
    TwoPlayerTracker needs NET_LINE. Prefer dataset/<video_name>/roi_config.json["NET_LINE"].
    If missing, fall back to a horizontal line at image mid-height.
    """
    roi_path = Path(roi_json_arg) if roi_json_arg else (paths["base_dir"] / "roi_config.json")
    data = load_roi_json(roi_path)

    net_line = data.get("NET_LINE", None)
    if isinstance(net_line, list) and len(net_line) >= 2:
        try:
            pts = [[float(p[0]), float(p[1])] for p in net_line]
            return pts
        except Exception:
            pass

    # Fallback: use first readable frame.
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is not None:
            h, w = img.shape[:2]
            return [[0.0, float(h // 2)], [float(w - 1), float(h // 2)]]

    return [[0.0, 360.0], [1280.0, 360.0]]


def box_center(box: Tuple[int, int, int, int, float]) -> Tuple[float, float]:
    x1, y1, x2, y2, _ = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def box_iou(a: Tuple[int, int, int, int, float], b: Tuple[int, int, int, int, float]) -> float:
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def center_distance(a: Tuple[int, int, int, int, float], b: Tuple[int, int, int, int, float]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return float(np.hypot(ax - bx, ay - by))


def clip_box_to_image(box: Tuple[int, int, int, int, float], img_w: int, img_h: int) -> Tuple[int, int, int, int, float]:
    x1, y1, x2, y2, conf = box
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))
    return (x1, y1, x2, y2, float(conf))


def tracked_box_for_side(
    tracked_players: List[Tuple[str, Tuple[int, int, int, int, float]]],
    side: str,
    net_y: float,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int, float]]]:
    """
    Tracker convention:
      Player1 = near
      Player2 = far
    Fallback uses box center relative to net_y.
    """
    side = str(side).lower().strip()

    preferred_names = []
    if side == "near":
        preferred_names = ["Player1", "near", "Near"]
    elif side == "far":
        preferred_names = ["Player2", "far", "Far"]

    for name, box in tracked_players:
        if str(name) in preferred_names:
            return str(name), box

    # Fallback by y position.
    candidates = []
    for name, box in tracked_players:
        _, cy = box_center(box)
        if side == "near" and cy >= net_y:
            candidates.append((cy, str(name), box))
        elif side == "far" and cy < net_y:
            candidates.append((-cy, str(name), box))

    if candidates:
        candidates.sort(reverse=True)
        _, name, box = candidates[0]
        return name, box

    return None, None


def match_raw_box_to_tracked_box(
    raw_boxes: List[Tuple[int, int, int, int, float]],
    tracked_box: Tuple[int, int, int, int, float],
    min_iou: float,
    max_center_dist: float,
) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int, float]], float, float]:
    """
    Tracker selects identity, but crop geometry should come from the matching raw YOLO box.
    A match is accepted if IoU is high enough OR center distance is small enough.
    """
    if not raw_boxes or tracked_box is None:
        return None, None, 0.0, 999999.0

    best_idx = None
    best_box = None
    best_score = -1e18
    best_iou = 0.0
    best_dist = 999999.0

    for idx, box in enumerate(raw_boxes):
        iou = box_iou(box, tracked_box)
        dist = center_distance(box, tracked_box)

        # Higher IoU, lower center distance, slightly higher confidence.
        dist_score = 1.0 / (1.0 + dist / 50.0)
        score = 2.2 * iou + 0.8 * dist_score + 0.08 * float(box[4])

        if score > best_score:
            best_score = score
            best_idx = idx
            best_box = box
            best_iou = float(iou)
            best_dist = float(dist)

    if best_box is None:
        return None, None, 0.0, 999999.0

    if best_iou >= float(min_iou) or best_dist <= float(max_center_dist):
        return int(best_idx), best_box, best_iou, best_dist

    return None, None, best_iou, best_dist


def save_debug_image_with_track(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    chosen_box: Optional[Tuple[int, int, int, int, float]],
    crop_box: Optional[Tuple[int, int, int, int]],
    tracked_box: Optional[Tuple[int, int, int, int, float]],
    out_path: Path,
) -> None:
    """
    Debug colors:
      blue    = all raw YOLO boxes
      magenta = tracked selected player
      green   = raw box used for crop
      red     = final crop box
    """
    debug = img.copy()

    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            debug,
            f"raw {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if tracked_box is not None:
        x1, y1, x2, y2, _ = tracked_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(
            debug,
            "tracked target",
            (x1, min(debug.shape[0] - 1, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if chosen_box is not None:
        x1, y1, x2, y2, conf = chosen_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            debug,
            f"crop raw {conf:.2f}",
            (x1, min(debug.shape[0] - 1, y2 + 38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            debug,
            "final crop",
            (x1, max(0, y1 - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)


def build_tracking_cache(
    frame_paths: List[Path],
    detector: PlayerDetector,
    net_line: List[List[float]],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[int, Dict[str, object]]:
    """
    Run YOLO exactly once per frame, update TwoPlayerTracker in chronological order,
    and keep both raw boxes and tracked player boxes for voting/debug.
    """
    tracker = TwoPlayerTracker(
        NET_LINE=net_line,
        max_miss=args.tracker_max_miss,
        line_judge_max_y=args.line_judge_max_y,
    )
    net_y = float(np.mean([p[1] for p in net_line]))

    cache: Dict[int, Dict[str, object]] = {}
    raw_rows: List[Dict[str, object]] = []
    tracked_rows: List[Dict[str, object]] = []

    print(f"[INFO] building YOLO + tracker cache for {len(frame_paths)} frames")
    for frame_idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            boxes: List[Tuple[int, int, int, int, float]] = []
            tracked_players = tracker.update([])
            cache[frame_idx] = {
                "boxes": boxes,
                "tracked_players": tracked_players,
                "img_w": 0,
                "img_h": 0,
                "read_ok": False,
                "net_y": net_y,
            }
            continue

        img_h, img_w = img.shape[:2]
        boxes = detector.detect(img)
        tracked_players = tracker.update(boxes)

        cache[frame_idx] = {
            "boxes": boxes,
            "tracked_players": tracked_players,
            "img_w": img_w,
            "img_h": img_h,
            "read_ok": True,
            "net_y": net_y,
        }

        for det_id, box in enumerate(boxes):
            x1, y1, x2, y2, conf = box
            raw_rows.append({
                "frame_idx": int(frame_idx),
                "det_id": int(det_id),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(conf),
                "img_w": int(img_w),
                "img_h": int(img_h),
            })

        for player_name, box in tracked_players:
            x1, y1, x2, y2, conf = box
            cx, cy = box_center(box)
            tracked_rows.append({
                "frame_idx": int(frame_idx),
                "player": str(player_name),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(conf),
                "cx": float(cx),
                "cy": float(cy),
                "side_by_net": "near" if cy >= net_y else "far",
                "net_y": float(net_y),
            })

        if args.cache_progress_every > 0 and (frame_idx + 1) % args.cache_progress_every == 0:
            print(f"[INFO] cached {frame_idx + 1}/{len(frame_paths)} frames")

    pd.DataFrame(raw_rows).to_csv(out_dir / "yolo_person_boxes_vote_cache.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tracked_rows).to_csv(out_dir / "tracked_players_vote_cache.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] raw YOLO boxes cache -> {out_dir / 'yolo_person_boxes_vote_cache.csv'}")
    print(f"[DONE] tracked players cache -> {out_dir / 'tracked_players_vote_cache.csv'}")

    return cache



def load_tracking_cache_from_csv(
    cache_dir: Path,
    frame_paths: List[Path],
    net_line: List[List[float]],
) -> Dict[int, Dict[str, object]]:
    """
    Load shared YOLO/tracker cache produced by bounce_shared_yolo_cache.py.

    Expected files under cache_dir:
      - yolo_person_boxes.csv or yolo_person_boxes_vote_cache.csv
      - tracked_players.csv or tracked_players_vote_cache.csv
    """
    raw_candidates = [
        cache_dir / "yolo_person_boxes.csv",
        cache_dir / "yolo_person_boxes_vote_cache.csv",
    ]
    tracked_candidates = [
        cache_dir / "tracked_players.csv",
        cache_dir / "tracked_players_vote_cache.csv",
    ]

    raw_file = next((p for p in raw_candidates if p.exists()), None)
    tracked_file = next((p for p in tracked_candidates if p.exists()), None)

    if raw_file is None:
        raise FileNotFoundError(
            "Cannot find shared YOLO raw box cache. Expected one of: "
            + ", ".join(str(p) for p in raw_candidates)
        )
    if tracked_file is None:
        raise FileNotFoundError(
            "Cannot find shared tracked player cache. Expected one of: "
            + ", ".join(str(p) for p in tracked_candidates)
        )

    raw_df = pd.read_csv(raw_file)
    tracked_df = pd.read_csv(tracked_file)
    net_y = float(np.mean([p[1] for p in net_line]))

    cache: Dict[int, Dict[str, object]] = {}
    for frame_idx, frame_path in enumerate(frame_paths):
        img_w = 0
        img_h = 0

        raw_part = raw_df[raw_df["frame_idx"].astype(int) == int(frame_idx)] if len(raw_df) else pd.DataFrame()
        boxes: List[Tuple[int, int, int, int, float]] = []
        if len(raw_part) > 0:
            for _, r in raw_part.iterrows():
                boxes.append((
                    int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]), float(r["conf"])
                ))
            if "img_w" in raw_part.columns and pd.notna(raw_part.iloc[0].get("img_w", np.nan)):
                img_w = int(raw_part.iloc[0]["img_w"])
            if "img_h" in raw_part.columns and pd.notna(raw_part.iloc[0].get("img_h", np.nan)):
                img_h = int(raw_part.iloc[0]["img_h"])

        tracked_part = tracked_df[tracked_df["frame_idx"].astype(int) == int(frame_idx)] if len(tracked_df) else pd.DataFrame()
        tracked_players: List[Tuple[str, Tuple[int, int, int, int, float]]] = []
        if len(tracked_part) > 0:
            for _, r in tracked_part.iterrows():
                # Prefer original coordinates if bounce.py was run with scale != 1.
                if all(c in tracked_part.columns for c in ["x1_orig", "y1_orig", "x2_orig", "y2_orig"]):
                    x1 = int(round(float(r["x1_orig"])))
                    y1 = int(round(float(r["y1_orig"])))
                    x2 = int(round(float(r["x2_orig"])))
                    y2 = int(round(float(r["y2_orig"])))
                else:
                    x1 = int(round(float(r["x1"])))
                    y1 = int(round(float(r["y1"])))
                    x2 = int(round(float(r["x2"])))
                    y2 = int(round(float(r["y2"])))
                tracked_players.append((
                    str(r.get("player", "unknown")),
                    (x1, y1, x2, y2, float(r.get("conf", 1.0))),
                ))

        # If no raw rows on this frame, still read image size when needed for crop/debug.
        if img_w <= 0 or img_h <= 0:
            img = cv2.imread(str(frame_path))
            if img is not None:
                img_h, img_w = img.shape[:2]

        cache[frame_idx] = {
            "boxes": boxes,
            "tracked_players": tracked_players,
            "img_w": int(img_w),
            "img_h": int(img_h),
            "read_ok": True,
            "net_y": net_y,
        }

    print("[INFO] shared YOLO/tracker cache loaded")
    return cache


def crop_frame_tracked_select_raw_geometry(
    frame_path: Path,
    frame_idx: int,
    event_side: str,
    tracking_cache: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[
    Optional[Image.Image],
    Dict[str, object],
    Optional[np.ndarray],
    List[Tuple[int, int, int, int, float]],
    Optional[Tuple[int, int, int, int, float]],
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int, float]],
]:
    """
    Return:
      crop_pil, info, original_img_bgr, raw_boxes, chosen_raw_box, crop_box, tracked_target_box

    crop_source modes:
      - tracked_match_raw: tracker selects identity, matched raw YOLO box decides crop geometry.
      - original_raw: original vote_action behavior.
      - tracked_box: tracker box decides identity and crop geometry; useful only for comparison.
    """
    info: Dict[str, object] = {
        "crop_mode_used": args.crop_source,
        "num_boxes": 0,
        "chosen_conf": np.nan,
        "chosen_det_id": np.nan,
        "matched_iou": np.nan,
        "matched_center_dist": np.nan,
        "tracked_player": "",
        "track_x1": np.nan,
        "track_y1": np.nan,
        "track_x2": np.nan,
        "track_y2": np.nan,
        "crop_x1": np.nan,
        "crop_y1": np.nan,
        "crop_x2": np.nan,
        "crop_y2": np.nan,
    }

    img = cv2.imread(str(frame_path))
    if img is None:
        info["skip_reason"] = "read_failed"
        return None, info, None, [], None, None, None

    img_h, img_w = img.shape[:2]
    cache_item = tracking_cache.get(frame_idx, {})
    raw_boxes = list(cache_item.get("boxes", []))
    tracked_players = list(cache_item.get("tracked_players", []))
    net_y = float(cache_item.get("net_y", img_h / 2.0))
    info["num_boxes"] = int(len(raw_boxes))

    crop_track_side = args.track_side
    if crop_track_side == "event":
        if str(event_side).lower() in ("near", "far"):
            crop_track_side = str(event_side).lower()
        elif str(args.target_side).lower() in ("near", "far"):
            crop_track_side = str(args.target_side).lower()
        else:
            crop_track_side = "near"

    tracked_name, tracked_box = tracked_box_for_side(tracked_players, crop_track_side, net_y)

    chosen_box = None
    chosen_det_id = None
    matched_iou = np.nan
    matched_dist = np.nan

    if args.crop_source == "original_raw":
        chosen_box = choose_nearest_largest_box(
            boxes=raw_boxes,
            img_w=img_w,
            img_h=img_h,
            min_area_ratio=args.min_area_ratio,
            area_weight=args.area_weight,
            y_weight=args.y_weight,
            conf_weight=args.conf_weight,
        )
        info["crop_mode_used"] = "original_raw_choose_nearest_largest"

    elif args.crop_source == "tracked_box":
        if tracked_box is not None:
            chosen_box = clip_box_to_image(tracked_box, img_w, img_h)
            chosen_det_id = -1
            info["crop_mode_used"] = "tracked_box_geometry"
        else:
            info["skip_reason"] = "no_tracked_player"
            return None, info, img, raw_boxes, None, None, None

    else:
        # Default: tracker selects identity; raw YOLO box determines crop geometry.
        if tracked_box is None:
            if args.track_fallback == "original_raw":
                chosen_box = choose_nearest_largest_box(
                    boxes=raw_boxes,
                    img_w=img_w,
                    img_h=img_h,
                    min_area_ratio=args.min_area_ratio,
                    area_weight=args.area_weight,
                    y_weight=args.y_weight,
                    conf_weight=args.conf_weight,
                )
                info["crop_mode_used"] = "fallback_original_raw_no_tracked_player"
            else:
                info["skip_reason"] = "no_tracked_player"
                return None, info, img, raw_boxes, None, None, None
        else:
            chosen_det_id, chosen_box, matched_iou, matched_dist = match_raw_box_to_tracked_box(
                raw_boxes=raw_boxes,
                tracked_box=tracked_box,
                min_iou=args.track_match_min_iou,
                max_center_dist=args.track_match_max_center_dist,
            )

            if chosen_box is None:
                if args.track_fallback == "original_raw":
                    chosen_box = choose_nearest_largest_box(
                        boxes=raw_boxes,
                        img_w=img_w,
                        img_h=img_h,
                        min_area_ratio=args.min_area_ratio,
                        area_weight=args.area_weight,
                        y_weight=args.y_weight,
                        conf_weight=args.conf_weight,
                    )
                    info["crop_mode_used"] = "fallback_original_raw_match_failed"
                elif args.track_fallback == "tracked_box":
                    chosen_box = clip_box_to_image(tracked_box, img_w, img_h)
                    chosen_det_id = -1
                    info["crop_mode_used"] = "fallback_tracked_box_match_failed"
                else:
                    info["skip_reason"] = "raw_match_failed"
                    info["matched_iou"] = float(matched_iou) if np.isfinite(matched_iou) else np.nan
                    info["matched_center_dist"] = float(matched_dist) if np.isfinite(matched_dist) else np.nan
                    return None, info, img, raw_boxes, None, None, tracked_box

    if tracked_box is not None:
        tx1, ty1, tx2, ty2, _ = tracked_box
        info.update({
            "tracked_player": str(tracked_name),
            "track_x1": int(tx1),
            "track_y1": int(ty1),
            "track_x2": int(tx2),
            "track_y2": int(ty2),
        })

    if chosen_box is None:
        info["skip_reason"] = "no_person_detected"
        return None, info, img, raw_boxes, None, None, tracked_box

    crop_box = expand_box(
        chosen_box,
        img_w=img_w,
        img_h=img_h,
        scale_w=args.scale_w,
        scale_h=args.scale_h,
        top_extra=args.top_extra,
        bottom_extra=args.bottom_extra,
    )

    x1, y1, x2, y2 = crop_box
    crop = img[y1:y2, x1:x2]

    info.update({
        "chosen_conf": float(chosen_box[4]),
        "chosen_det_id": int(chosen_det_id) if chosen_det_id is not None else np.nan,
        "matched_iou": float(matched_iou) if np.isfinite(matched_iou) else np.nan,
        "matched_center_dist": float(matched_dist) if np.isfinite(matched_dist) else np.nan,
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_x2": int(x2),
        "crop_y2": int(y2),
    })

    if crop.size == 0:
        info["skip_reason"] = "empty_crop"
        return None, info, img, raw_boxes, chosen_box, crop_box, tracked_box

    if args.crop_resize > 0:
        crop = cv2.resize(
            crop,
            (int(args.crop_resize), int(args.crop_resize)),
            interpolation=cv2.INTER_AREA,
        )

    info["skip_reason"] = ""
    return cv2_bgr_to_pil_rgb(crop), info, img, raw_boxes, chosen_box, crop_box, tracked_box

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    eval_tfms,
    image: Image.Image,
    device: torch.device,
    class_names: List[str],
) -> Tuple[str, Dict[str, float]]:
    x = eval_tfms(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    return pred_label, {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def decide_event_label(frame_rows: List[Dict[str, object]], class_names: List[str]) -> Dict[str, object]:
    valid = [r for r in frame_rows if int(r.get("valid_prediction", 0)) == 1]
    skipped = len(frame_rows) - len(valid)

    if not valid:
        out = {
            "event_pred_label": "unknown",
            "vote_count": 0,
            "skipped_frame_count": int(skipped),
            "winning_votes": 0,
            "vote_ratio": 0.0,
            "mean_winning_prob": 0.0,
            "tie_breaker": "none",
        }
        for c in class_names:
            out[f"votes_{c}"] = 0
            out[f"prob_sum_{c}"] = 0.0
            out[f"prob_mean_{c}"] = 0.0
        return out

    counts = {c: 0 for c in class_names}
    prob_sums = {c: 0.0 for c in class_names}

    for r in valid:
        pred = str(r["pred_label"])
        if pred in counts:
            counts[pred] += 1
        for c in class_names:
            prob_sums[c] += float(r.get(f"prob_{c}", 0.0))

    max_count = max(counts.values())
    winners = [c for c, v in counts.items() if v == max_count]

    if len(winners) == 1:
        final_label = winners[0]
        tie_breaker = "majority_vote"
    else:
        final_label = max(winners, key=lambda c: prob_sums[c])
        tie_breaker = "sum_probability"

    vote_count = len(valid)
    winning_votes = counts[final_label]
    vote_ratio = float(winning_votes / max(1, vote_count))
    mean_winning_prob = float(prob_sums[final_label] / max(1, vote_count))

    out = {
        "event_pred_label": final_label,
        "vote_count": int(vote_count),
        "skipped_frame_count": int(skipped),
        "winning_votes": int(winning_votes),
        "vote_ratio": vote_ratio,
        "mean_winning_prob": mean_winning_prob,
        "tie_breaker": tie_breaker,
    }
    for c in class_names:
        out[f"votes_{c}"] = int(counts[c])
        out[f"prob_sum_{c}"] = float(prob_sums[c])
        out[f"prob_mean_{c}"] = float(prob_sums[c] / max(1, vote_count))
    return out


def save_raw_crop(crop: Image.Image, out_path: Path) -> None:
    """Save the exact crop used for prediction. No text/labels are drawn on it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)



def run_vote(args: argparse.Namespace) -> None:
    video_path = Path(args.video_path)
    paths = default_paths(video_path)

    frames_dir = Path(args.frames_dir) if args.frames_dir else paths["frames_dir"]
    frame_paths = list_frame_paths(frames_dir)

    event_file = resolve_event_file(paths, args.event_file)
    events_df = load_events(event_file)

    before_filter = len(events_df)
    events_df = filter_events_by_side(events_df, args.target_side)
    after_filter = len(events_df)
    if after_filter == 0:
        print(f"[WARNING] No events left after --target-side {args.target_side}. "
              f"Original events: {before_filter}. Skipping stroke voting.")
        
        # Create empty output files to keep downstream tools happy
        out_dir = Path(args.out_dir) if args.out_dir else paths["stroke_out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stroke_vote_frame_predictions.csv").touch()
        (out_dir / "stroke_vote_events.csv").touch()
        return

    out_dir = Path(args.out_dir) if args.out_dir else paths["stroke_out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "prediction_crops"
    debug_dir = out_dir / "crop_debug"

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = list(ckpt["class_names"])
    img_size = int(ckpt.get("img_size", args.img_size))

    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_tfms = build_transforms(img_size=img_size, use_hflip=False)

    net_line = resolve_net_line(paths, frame_paths, roi_json_arg=args.roi_json)

    if args.yolo_cache_dir:
        tracking_cache = load_tracking_cache_from_csv(
            cache_dir=Path(args.yolo_cache_dir),
            frame_paths=frame_paths,
            net_line=net_line,
        )
    else:
        detector = PlayerDetector(
            model_path=args.model_path,
            device=args.yolo_device,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        tracking_cache = build_tracking_cache(
            frame_paths=frame_paths,
            detector=detector,
            net_line=net_line,
            args=args,
            out_dir=out_dir,
        )

    fallback_end_offset = int(args.fallback_end_offset) if args.fallback_end_offset is not None else int(args.start_offset) - 1

    print(
        f"[INFO] video={video_path.stem} | events={after_filter}/{before_filter} kept "
        f"({args.target_side}) | device={device} | crop={args.crop_source}"
    )
    print(
        f"[INFO] vote window: primary center{args.start_offset:+d}~center{args.end_offset:+d}; "
        f"fallback center{args.fallback_start_offset:+d}~center{fallback_end_offset:+d}"
    )

    if args.verbose:
        print(f"[DETAIL] frames_dir  = {frames_dir}")
        print(f"[DETAIL] event_file  = {event_file}")
        print(f"[DETAIL] checkpoint  = {args.checkpoint}")
        print(f"[DETAIL] classes     = {class_names}")
        print(f"[DETAIL] track_side  = {args.track_side}")
        print(f"[DETAIL] fallback    = {args.track_fallback}")
        print(f"[DETAIL] yolo        = model={args.model_path}, device={args.yolo_device}, imgsz={args.imgsz}, conf={args.conf}")

    crop_cache: Dict[int, Tuple[
        Optional[Image.Image],
        Dict[str, object],
        Optional[np.ndarray],
        List[Tuple[int, int, int, int, float]],
        Optional[Tuple[int, int, int, int, float]],
        Optional[Tuple[int, int, int, int]],
        Optional[Tuple[int, int, int, int, float]],
    ]] = {}

    all_frame_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    def has_valid_prediction(rows: List[Dict[str, object]]) -> bool:
        return any(int(r.get("valid_prediction", 0)) == 1 for r in rows)

    def run_offsets_for_event(
        event_id: int,
        center: int,
        event_side: str,
        offsets: range,
        phase: str,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []

        for offset in offsets:
            frame_idx = center + offset
            row: Dict[str, object] = {
                "event_id": event_id,
                "event_center_frame": center,
                "event_side": event_side,
                "vote_phase": phase,
                "offset": offset,
                "frame_idx": frame_idx,
                "frame_path": "",
                "valid_prediction": 0,
                "pred_label": "",
            }
            for c in class_names:
                row[f"prob_{c}"] = np.nan

            if frame_idx < 0 or frame_idx >= len(frame_paths):
                row["skip_reason"] = "frame_idx_out_of_range"
                all_frame_rows.append(row)
                rows.append(row)
                continue

            frame_path = frame_paths[frame_idx]
            row["frame_path"] = str(frame_path)

            # Cache final crop result per frame. For near-only voting, event_side does not change.
            # If you later use target_side=all and different sides share a frame, disable this cache
            # or include event_side in the cache key.
            cache_key = frame_idx if args.target_side != "all" else (frame_idx, str(event_side).lower())
            if cache_key not in crop_cache:
                crop_cache[cache_key] = crop_frame_tracked_select_raw_geometry(
                    frame_path=frame_path,
                    frame_idx=frame_idx,
                    event_side=event_side,
                    tracking_cache=tracking_cache,
                    args=args,
                )

            crop, crop_info, original_img, boxes, chosen_box, crop_box, tracked_box = crop_cache[cache_key]
            row.update(crop_info)

            if args.debug_crops and original_img is not None:
                save_debug_image_with_track(
                    original_img,
                    boxes,
                    chosen_box,
                    crop_box,
                    tracked_box,
                    debug_dir / f"{phase}_frame_{frame_idx:06d}.jpg",
                )

            if crop is None:
                row["valid_prediction"] = 0
                row["pred_label"] = ""
                row["pred_prob"] = np.nan
                all_frame_rows.append(row)
                rows.append(row)
                continue

            pred_label, probs = predict_image(model, eval_tfms, crop, device, class_names)
            row["valid_prediction"] = 1
            row["pred_label"] = pred_label
            row["pred_prob"] = float(probs[pred_label])
            row["skip_reason"] = ""
            for c in class_names:
                row[f"prob_{c}"] = float(probs[c])

            if args.save_crops:
                safe_pred = pred_label.replace("/", "_")
                save_raw_crop(
                    crop,
                    crops_dir / f"event_{event_id:03d}" / f"{phase}_frame_{frame_idx:06d}_off_{offset:+d}_{safe_pred}.jpg",
                )

            all_frame_rows.append(row)
            rows.append(row)

        return rows

    for _, event in events_df.iterrows():
        event_id = int(event["event_id"])
        center = int(event["event_center_frame"])
        event_side = str(event.get("event_side", "unknown"))

        primary_rows = run_offsets_for_event(
            event_id=event_id,
            center=center,
            event_side=event_side,
            offsets=range(int(args.start_offset), int(args.end_offset) + 1),
            phase="primary",
        )

        used_rows = primary_rows
        used_phase = "primary"
        fallback_rows: List[Dict[str, object]] = []

        if not has_valid_prediction(primary_rows):
            fallback_rows = run_offsets_for_event(
                event_id=event_id,
                center=center,
                event_side=event_side,
                offsets=range(int(args.fallback_start_offset), int(fallback_end_offset) + 1),
                phase="fallback_backward",
            )
            used_rows = fallback_rows
            used_phase = "fallback_backward"

        decision = decide_event_label(used_rows, class_names)
        decision["used_vote_phase"] = used_phase
        decision["primary_valid_count"] = int(sum(int(r.get("valid_prediction", 0)) == 1 for r in primary_rows))
        decision["fallback_valid_count"] = int(sum(int(r.get("valid_prediction", 0)) == 1 for r in fallback_rows))
        decision["primary_total_count"] = int(len(primary_rows))
        decision["fallback_total_count"] = int(len(fallback_rows))

        event_row = event.to_dict()
        event_row.update(decision)
        event_rows.append(event_row)

    frame_pred_df = pd.DataFrame(all_frame_rows)
    event_pred_df = pd.DataFrame(event_rows)

    frame_csv = out_dir / "stroke_vote_frame_predictions.csv"
    event_csv = out_dir / "stroke_vote_events.csv"
    frame_pred_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")
    event_pred_df.to_csv(event_csv, index=False, encoding="utf-8-sig")

    failed_count = int((frame_pred_df["valid_prediction"] != 1).sum()) if len(frame_pred_df) else 0
    valid_count = int((frame_pred_df["valid_prediction"] == 1).sum()) if len(frame_pred_df) else 0

    print("\n[DONE]")
    print(f"  frame CSV: {frame_csv}")
    print(f"  event CSV: {event_csv}")
    print(f"[SUMMARY] events={len(event_pred_df)} | valid_frame_votes={valid_count} | skipped_frames={failed_count}")

    compact_cols = [
        "event_id", "event_center_frame", "event_pred_label",
        "winning_votes", "vote_count", "skipped_frame_count",
        "vote_ratio", "mean_winning_prob", "used_vote_phase"
    ]
    compact_cols = [c for c in compact_cols if c in event_pred_df.columns]
    if compact_cols:
        compact_df = event_pred_df[compact_cols].copy()
        compact_df = compact_df.rename(columns={
            "event_id": "id",
            "event_center_frame": "frame",
            "event_pred_label": "pred",
            "winning_votes": "win",
            "vote_count": "votes",
            "skipped_frame_count": "skip",
            "vote_ratio": "ratio",
            "mean_winning_prob": "conf",
            "used_vote_phase": "phase",
        })

        for col in ["ratio", "conf"]:
            if col in compact_df.columns:
                compact_df[col] = compact_df[col].astype(float).round(3)

        print("\n[EVENTS]")
        print(compact_df.to_string(index=False))

    if args.verbose:
        cols = [
            "event_id", "event_center_frame", "event_side", "event_pred_label",
            "used_vote_phase", "winning_votes", "vote_count", "skipped_frame_count",
            "primary_valid_count", "fallback_valid_count", "vote_ratio",
            "mean_winning_prob", "tie_breaker"
        ]
        existing_cols = [c for c in cols if c in event_pred_df.columns]
        print("\n[VERBOSE EVENT SUMMARY]")
        print(event_pred_df[existing_cols].to_string(index=False))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="例如 raw_videos/xxx.mp4")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained best_model.pt")
    parser.add_argument("--event-file", type=str, default="auto", help="預設自動使用 hit_intervals_y_segments.csv")
    parser.add_argument("--frames-dir", type=str, default=None, help="預設 dataset/<video_name>/frames")
    parser.add_argument("--out-dir", type=str, default=None, help="預設 bounce_detector/stroke_vote_same_crop")
    parser.add_argument(
        "--yolo-cache-dir",
        type=str,
        default=None,
        help="讀取 bounce.py 產生的 shared YOLO cache 資料夾，例如 dataset/<video>/bounce_detector。若不填，會自行重跑 YOLO。",
    )
    parser.add_argument("--roi-json", type=str, default=None, help="預設 dataset/<video_name>/roi_config.json，用於讀 NET_LINE 初始化 tracker。")

    parser.add_argument(
        "--target-side",
        type=str,
        default="near",
        choices=["near", "far", "all"],
        help="只對哪一側的擊球 event 做正反拍投票；預設 near，只判斷近端球員。",
    )

    parser.add_argument("--start-offset", type=int, default=-5)
    parser.add_argument("--end-offset", type=int, default=2)
    parser.add_argument(
        "--fallback-start-offset",
        type=int,
        default=-10,
        help="如果 primary window 完全沒有成功 crop，往前補找的起始 offset；預設 center-10。",
    )
    parser.add_argument(
        "--fallback-end-offset",
        type=int,
        default=None,
        help="如果 primary window 完全沒有成功 crop，往前補找的結束 offset；預設為 start_offset-1，所以原設定下是 center-6。",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-crops", action="store_true", help="保存實際丟進模型的 crop；不會畫字，避免污染圖片。")
    parser.add_argument("--debug-crops", action="store_true", help="保存畫有 bbox/crop 框的 debug 圖。")

    # YOLO parameters.
    parser.add_argument("--model_path", type=str, default="yolov8n.pt", help="YOLO 模型路徑，預設 yolov8n.pt。")
    parser.add_argument("--yolo-device", type=str, default="auto", help="auto / cpu / 0 / 1。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size。")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO person detection confidence threshold。")

    # Same crop parameters as player_crop.py.
    parser.add_argument("--min_area_ratio", type=float, default=0.002, help="bbox area / image area 小於此值就丟掉。")
    parser.add_argument("--scale_w", type=float, default=1.7, help="crop bbox 橫向放大倍率。")
    parser.add_argument("--scale_h", type=float, default=1.7, help="crop bbox 縱向放大倍率。")
    parser.add_argument("--top_extra", type=float, default=0.05, help="額外往上補的比例，以原 bbox 高度為基準。")
    parser.add_argument("--bottom_extra", type=float, default=0.05, help="額外往下補的比例，以原 bbox 高度為基準。")
    parser.add_argument("--area_weight", type=float, default=1.0, help="選主要球員時，bbox 面積權重。")
    parser.add_argument("--y_weight", type=float, default=0.5, help="選主要球員時，bbox 越下面的權重。")
    parser.add_argument("--conf_weight", type=float, default=0.1, help="選主要球員時，confidence 權重。")
    parser.add_argument("--crop-resize", type=int, default=0, help="若 >0，裁切後先 resize 成 NxN。預設 0，交給模型 transform resize。")

    # New tracking-selection parameters.
    parser.add_argument(
        "--crop-source",
        type=str,
        default="tracked_match_raw",
        choices=["tracked_match_raw", "original_raw", "tracked_box"],
        help="tracked_match_raw=tracker 選人、raw YOLO box 決定 crop；original_raw=原 vote_action；tracked_box=直接用 tracked box crop。",
    )
    parser.add_argument(
        "--track-side",
        type=str,
        default="near",
        choices=["near", "far", "event"],
        help="tracker 要選哪一側球員；near 預設用 Player1。若設 event，依 event_side 決定。",
    )
    parser.add_argument(
        "--track-fallback",
        type=str,
        default="skip",
        choices=["skip", "original_raw", "tracked_box"],
        help="tracked_match_raw 找不到合理 raw match 時怎麼辦。建議先 skip，避免錯 crop 污染投票。",
    )
    parser.add_argument("--track-match-min-iou", type=float, default=0.10, help="tracked box 與 raw box 的最低 IoU 接受門檻。")
    parser.add_argument("--track-match-max-center-dist", type=float, default=80.0, help="若 IoU 不夠，中心距離小於此值仍接受。")
    parser.add_argument("--tracker-max-miss", type=int, default=12, help="TwoPlayerTracker max_miss。")
    parser.add_argument("--line-judge-max-y", type=int, default=20, help="TwoPlayerTracker 過濾極端靠上線審/雜訊的 y 門檻。")
    parser.add_argument("--cache-progress-every", type=int, default=100, help="建立 YOLO+tracker cache 時每幾幀印一次進度；0 表示不印。")
    parser.add_argument("--verbose", action="store_true", help="印出詳細執行資訊與完整 event summary。")

    return parser.parse_args()


if __name__ == "__main__":
    run_vote(parse_args())
