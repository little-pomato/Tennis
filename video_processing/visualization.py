import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# Debug overlay helpers — projects court model lines back onto
# the original video using H (court-model meters -> image px).
# ============================================================

# Court model line segments: list of ((X0,Y0),(X1,Y1)) in court meters
def _court_model_segments() -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    W, L = COURT_WIDTH, COURT_LENGTH
    SM = SINGLE_MARGIN
    SW = SINGLES_WIDTH
    NY = COURT_LENGTH / 2.0
    SF = SERVICE_FROM_NET
    segs = [
        # doubles outer boundary
        ((0, 0),   (W, 0)),
        ((W, 0),   (W, L)),
        ((W, L),   (0, L)),
        ((0, L),   (0, 0)),
        # singles sidelines
        ((SM,     0), (SM,     L)),
        ((SM+SW,  0), (SM+SW,  L)),
        # near/far service lines
        ((SM,    NY-SF), (SM+SW, NY-SF)),
        ((SM,    NY+SF), (SM+SW, NY+SF)),
        # center service line
        ((W/2,   NY-SF), (W/2,   NY+SF)),
        # net (drawn separately as dashed, but listed here for projection)
        ((0, NY), (W, NY)),
    ]
    return segs

# ============================================================
# Clean bounce landing visualization
# ------------------------------------------------------------
# Default sync source is dataset/<video>/frames, because bounce.py also
# detects bounce_events.csv from those extracted frames.
#
# This script intentionally does NOT use:
#   - tracked_path.csv
#   - PlayerDetector / TwoPlayerTracker
#   - ball/player overlays on the original video
#   - hit interval CSV
#
# It uses only:
#   - dataset/<video>/bounce_detector/bounce_events.csv
#   - dataset/<video>/roi_config.json  (only for H projection)
# ============================================================

COURT_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2.0
SINGLE_MARGIN = (COURT_WIDTH - SINGLES_WIDTH) / 2.0


def set_video_path(video_path: Path):
    global VIDEO_PATH, video_name, BASE, FRAMES_DIR, ROI_JSON, BOUNCE_DIR
    global BOUNCE_EVENTS_CSV, OUT_PATH_OVERLAY

    VIDEO_PATH = Path(video_path)
    video_name = VIDEO_PATH.stem
    BASE = Path("dataset") / video_name
    FRAMES_DIR = BASE / "frames"
    ROI_JSON = BASE / "roi_config.json"
    BOUNCE_DIR = BASE / "bounce_detector"
    BOUNCE_EVENTS_CSV = BOUNCE_DIR / "bounce_events.csv"
    OUT_PATH_OVERLAY = BOUNCE_DIR / "overlay_bounce_landings.mp4"


def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_bounce_events(path: Path, coord_scale: float = 1.0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bounce_events.csv not found: {path}")

    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["frame_idx", "x", "y", "court_side", "score"])
    if raw.empty:
        return pd.DataFrame(columns=["frame_idx", "x", "y", "court_side", "score"])

    frame_col = _pick_first_existing_column(raw, ["peak_frame", "refined_frame", "frame_idx"])
    x_col = _pick_first_existing_column(raw, ["cx", "refined_cx", "x"])
    y_col = _pick_first_existing_column(raw, ["cy", "refined_cy", "y"])
    side_col = _pick_first_existing_column(raw, ["court_side", "landing_side", "side"])
    score_col = _pick_first_existing_column(raw, ["peak_score", "refine_score", "score"])

    if frame_col is None or x_col is None or y_col is None:
        raise ValueError(
            "bounce_events.csv must contain frame/x/y columns, e.g. peak_frame, cx, cy."
        )

    out = pd.DataFrame()
    out["frame_idx"] = pd.to_numeric(raw[frame_col], errors="coerce")
    out["x"] = pd.to_numeric(raw[x_col], errors="coerce") * float(coord_scale)
    out["y"] = pd.to_numeric(raw[y_col], errors="coerce") * float(coord_scale)
    out["court_side"] = raw[side_col].astype(str).str.lower() if side_col else "unknown"
    out["score"] = pd.to_numeric(raw[score_col], errors="coerce") if score_col else np.nan

    out = out.dropna(subset=["frame_idx", "x", "y"]).copy()
    out["frame_idx"] = out["frame_idx"].round().astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)
    out = out.sort_values("frame_idx").reset_index(drop=True)
    out["landing_no"] = np.arange(1, len(out) + 1)
    return out


def load_homography_from_roi(roi_json: Path) -> np.ndarray:
    roi_json = Path(roi_json)
    if not roi_json.exists():
        raise FileNotFoundError(f"roi_config.json not found: {roi_json}")

    with open(roi_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "H" not in data:
        raise KeyError(
            f"{roi_json} does not contain H. The side-court projection needs the "
            "homography generated by court_detector.py."
        )

    H = np.asarray(data["H"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Invalid H shape in {roi_json}: {H.shape}")
    return H


def image_to_court_xy(x: float, y: float, H_court_to_image: np.ndarray) -> Optional[Tuple[float, float]]:
    """Project image pixel coordinate to court-meter coordinate.

    court_detector.py stores H as court-model coordinates -> image pixels,
    so image -> court uses inv(H).
    """
    try:
        H_inv = np.linalg.inv(np.asarray(H_court_to_image, dtype=np.float64))
        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, H_inv.astype(np.float64)).reshape(-1, 2)[0]
        X, Y = float(out[0]), float(out[1])
    except Exception:
        return None

    if not np.isfinite(X) or not np.isfinite(Y):
        return None
    return X, Y


def is_in_bounds(X: float, Y: float, singles: bool = True, tol: float = 0.03) -> bool:
    width = SINGLES_WIDTH if singles else COURT_WIDTH
    x_min = (COURT_WIDTH - width) / 2.0
    x_max = x_min + width
    return (
        x_min - tol <= float(X) <= x_max + tol
        and 0.0 - tol <= float(Y) <= COURT_LENGTH + tol
    )


def court_to_canvas_xy(X: float, Y: float, scale: float, margin_px: int) -> Tuple[int, int]:
    # Far baseline is top (Y=0), near baseline is bottom (Y=COURT_LENGTH)
    px = int(round(float(margin_px) + float(X) * float(scale)))
    py = int(round(float(margin_px) + float(Y) * float(scale)))
    return px, py


def draw_court(scale: float = 10.0, margin_ratio: float = 0.10) -> Tuple[np.ndarray, int]:
    margin_px = int(round(max(COURT_WIDTH, COURT_LENGTH) * float(scale) * float(margin_ratio)))
    W = int(round(COURT_WIDTH * float(scale) + 2 * margin_px))
    H_img = int(round(COURT_LENGTH * float(scale) + 2 * margin_px))
    img = np.full((H_img, W, 3), 255, dtype=np.uint8)

    green = (0, 210, 0)
    gray = (150, 150, 150)
    lw = max(1, int(round(float(scale) / 16)))

    def pt(x, y):
        return court_to_canvas_xy(x, y, scale, margin_px)

    cv2.rectangle(img, pt(0, 0), pt(COURT_WIDTH, COURT_LENGTH), green, lw)
    cv2.line(img, pt(SINGLE_MARGIN, 0), pt(SINGLE_MARGIN, COURT_LENGTH), green, lw)
    cv2.line(img, pt(SINGLE_MARGIN + SINGLES_WIDTH, 0), pt(SINGLE_MARGIN + SINGLES_WIDTH, COURT_LENGTH), green, lw)
    cv2.line(img, pt(0, NET_Y), pt(COURT_WIDTH, NET_Y), gray, lw)

    far_service = NET_Y - SERVICE_FROM_NET
    near_service = NET_Y + SERVICE_FROM_NET
    cv2.line(img, pt(SINGLE_MARGIN, far_service), pt(SINGLE_MARGIN + SINGLES_WIDTH, far_service), green, lw)
    cv2.line(img, pt(SINGLE_MARGIN, near_service), pt(SINGLE_MARGIN + SINGLES_WIDTH, near_service), green, lw)
    cv2.line(img, pt(COURT_WIDTH / 2.0, far_service), pt(COURT_WIDTH / 2.0, near_service), green, lw)

    return img, margin_px


def side_color(side: str, in_bound: bool = True) -> Tuple[int, int, int]:
    if not in_bound:
        return (130, 130, 130)
    side = str(side).lower()
    if side == "near":
        return (0, 80, 255)
    if side == "far":
        return (255, 170, 0)
    return (0, 0, 255)


def project_all_landings(bounce_df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, row in bounce_df.iterrows():
        projected = image_to_court_xy(row["x"], row["y"], H)
        if projected is None:
            continue
        X, Y = projected
        rows.append({
            "frame_idx": int(row["frame_idx"]),
            "landing_no": int(row["landing_no"]),
            "X": float(X),
            "Y": float(Y),
            "court_side": str(row.get("court_side", "unknown")),
            "in_bound": bool(is_in_bounds(X, Y, singles=True)),
            "score": float(row["score"]) if pd.notna(row.get("score", np.nan)) else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["frame_idx", "landing_no", "X", "Y", "court_side", "in_bound", "score"])
    return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)


def draw_landings_on_court(
    base_court: np.ndarray,
    landings: List[dict],
    scale: float,
    margin_px: int,
    current_frame_idx: int,
    show_index: bool = False,
) -> np.ndarray:
    court_img = base_court.copy()

    for hit in landings:
        px, py = court_to_canvas_xy(hit["X"], hit["Y"], scale, margin_px)
        in_bound = bool(hit.get("in_bound", True))
        color = side_color(hit.get("court_side", "unknown"), in_bound=in_bound)
        is_current = int(hit.get("frame_idx", -1)) == int(current_frame_idx)

        radius = 4 if is_current else 2
        cv2.circle(court_img, (px, py), radius, color, -1, cv2.LINE_AA)

        if is_current:
            cv2.circle(court_img, (px, py), radius + 6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(court_img, (px, py), radius + 8, (255, 255, 255), 1, cv2.LINE_AA)

        if show_index:
            cv2.putText(
                court_img,
                str(int(hit.get("landing_no", 0))),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

    return court_img


def get_video_fps(video_path: Path, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0 or not np.isfinite(fps):
        return float(fallback)
    return float(fps)


def get_frame_files(frames_dir: Path) -> List[Path]:
    frames_dir = Path(frames_dir)
    files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png")))
    return files


def _frame_iter_from_images(frame_files: List[Path]):
    for idx, fp in enumerate(frame_files):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        yield idx, frame


def _frame_iter_from_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        yield idx, frame
    cap.release()


def make_overlay_bounce_landings(
    video_path: Path,
    frames_dir: Path,
    bounce_csv: Path,
    roi_json: Path,
    out_path: Path,
    court_scale: float = 10.0,
    court_margin: float = 0.10,
    court_ratio: float = 0.35,
    bounce_coord_scale: float = 1.0,
    show_index: bool = False,
    source: str = "frames",
    fps: Optional[float] = None,
):
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)
    bounce_csv = Path(bounce_csv)
    roi_json = Path(roi_json)
    out_path = Path(out_path)

    bounce_df = load_bounce_events(bounce_csv, coord_scale=bounce_coord_scale)
    H = load_homography_from_roi(roi_json)
    landing_df = project_all_landings(bounce_df, H)

    print(f"[INFO] loaded bounce events: {len(bounce_df)}")
    print(f"[INFO] projected landings: {len(landing_df)}")

    landing_by_frame: Dict[int, pd.DataFrame] = {
        int(frame_idx): sub.copy() for frame_idx, sub in landing_df.groupby("frame_idx")
    }

    if fps is None or fps <= 0:
        fps = get_video_fps(video_path, fallback=30.0) if video_path.exists() else 30.0

    # Important: bounce.py uses dataset/<video>/frames frame_idx.
    # Therefore the default video source must be the extracted frame sequence,
    # not cv2.VideoCapture(raw_video), because VideoCapture can decode/drop frames
    # differently and create progressive drift.
    frame_files = get_frame_files(frames_dir)
    use_frames = source == "frames"
    if use_frames:
        if not frame_files:
            raise FileNotFoundError(f"No extracted frames found in {frames_dir}. Use --source video only for debugging.")
        first = cv2.imread(str(frame_files[0]))
        if first is None:
            raise ValueError(f"Cannot read first frame: {frame_files[0]}")
        H_img, W = first.shape[:2]
        frame_iter = _frame_iter_from_images(frame_files)
        total_frames = len(frame_files)
        print(f"[INFO] source=frames, frame_count={total_frames}, fps={fps:.3f}")
    else:
        if not video_path.exists():
            raise FileNotFoundError(f"video_path not found: {video_path}")
        cap_probe = cv2.VideoCapture(str(video_path))
        W = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_img = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()
        if W <= 0 or H_img <= 0:
            raise ValueError(f"Cannot read video dimensions: {video_path}")
        frame_iter = _frame_iter_from_video(video_path)
        print(f"[WARN] source=video. If sync drifts, use default source=frames.")
        print(f"[INFO] source=video, frame_count={total_frames}, fps={fps:.3f}")

    if not landing_df.empty:
        max_bounce_frame = int(landing_df["frame_idx"].max())
        if max_bounce_frame >= total_frames:
            print(
                f"[WARN] max bounce frame {max_bounce_frame} >= source frame count {total_frames}. "
                "Check whether bounce.py and visualization.py are using the same extracted frames."
            )

    court_w = int(round(W * float(court_ratio)))
    total_w = W + court_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), fourcc, float(fps), (total_w, H_img))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    base_court, margin_px = draw_court(scale=court_scale, margin_ratio=court_margin)
    accumulated_landings: List[dict] = []

    for frame_idx, frame in tqdm(frame_iter, total=total_frames):
        current = landing_by_frame.get(frame_idx)
        if current is not None and not current.empty:
            for _, row in current.iterrows():
                accumulated_landings.append(row.to_dict())

        court_img = draw_landings_on_court(
            base_court=base_court,
            landings=accumulated_landings,
            scale=court_scale,
            margin_px=margin_px,
            current_frame_idx=frame_idx,
            show_index=show_index,
        )
        court_resized = cv2.resize(court_img, (court_w, H_img), interpolation=cv2.INTER_AREA)

        combined = np.zeros((H_img, total_w, 3), dtype=np.uint8)
        combined[:, :W] = frame
        combined[:, W:] = court_resized
        out.write(combined)

    out.release()
    print(f"[DONE] overlay saved -> {out_path}")


def _court_pt_to_image(X: float, Y: float, H: np.ndarray) -> Optional[Tuple[int, int]]:
    """Project one court-model point (meters) to image pixels using H directly."""
    pts = np.array([[[float(X), float(Y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    if out is None:
        return None
    px, py = float(out[0, 0, 0]), float(out[0, 0, 1])
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return int(round(px)), int(round(py))


def draw_court_lines_on_frame(
    frame: np.ndarray,
    H: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 230, 0),
    net_color: Tuple[int, int, int] = (200, 200, 200),
    alpha: float = 0.55,
) -> np.ndarray:
    """Project court model lines onto the video frame using H.

    Draws semi-transparently so the underlying image shows through.
    This lets you verify that the projected court grid aligns with
    the real painted lines in the video.
    """
    overlay = frame.copy()
    H_arr = np.asarray(H, dtype=np.float64)
    H_img, W_img = frame.shape[:2]
    clip = (-W_img, -H_img, 2 * W_img, 2 * H_img)  # loose clip so lines near edges render

    def _in_clip(p):
        if p is None:
            return False
        return clip[0] <= p[0] <= clip[2] and clip[1] <= p[1] <= clip[3]

    for (x0, y0), (x1, y1) in _court_model_segments():
        p0 = _court_pt_to_image(x0, y0, H_arr)
        p1 = _court_pt_to_image(x1, y1, H_arr)
        if not (_in_clip(p0) and _in_clip(p1)):
            continue
        # Net drawn as a dashed line in a different colour
        is_net = (y0 == COURT_LENGTH / 2.0 and y1 == COURT_LENGTH / 2.0)
        color = net_color if is_net else line_color
        thickness = 2 if is_net else 1
        if is_net:
            # dashed
            total = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            n_dashes = max(1, total // 14)
            for k in range(n_dashes):
                t0 = k / n_dashes
                t1 = (k + 0.55) / n_dashes
                s = (int(round(p0[0] + t0 * (p1[0] - p0[0]))),
                     int(round(p0[1] + t0 * (p1[1] - p0[1]))))
                e = (int(round(p0[0] + t1 * (p1[0] - p0[0]))),
                     int(round(p0[1] + t1 * (p1[1] - p0[1]))))
                cv2.line(overlay, s, e, color, thickness, cv2.LINE_AA)
        else:
            cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def draw_bounces_on_frame(
    frame: np.ndarray,
    accumulated: List[dict],
    current_frame_idx: int,
    trail_length: int = 10,
) -> np.ndarray:
    """Draw bounce markers directly on the video frame.

    - Current frame's bounce: large crosshair + concentric rings + label.
    - Last `trail_length` previous bounces: fading dots.
    """
    out = frame.copy()
    H_img, W_img = frame.shape[:2]

    # Only keep the most recent trail_length bounces before current frame
    past = [b for b in accumulated if int(b["frame_idx"]) < current_frame_idx]
    past = past[-trail_length:]

    # Fading trail
    for i, b in enumerate(past):
        fade = (i + 1) / (len(past) + 1)   # 0→1 as we approach current
        bx, by = int(round(float(b["x"]))), int(round(float(b["y"])))
        if not (0 <= bx < W_img and 0 <= by < H_img):
            continue
        radius = max(3, int(round(4 * fade)))
        in_b = bool(b.get("in_bound", True))
        base_color = side_color(b.get("court_side", "unknown"), in_bound=in_b)
        # Dim the colour by fade factor
        color = tuple(int(c * fade) for c in base_color)
        cv2.circle(out, (bx, by), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), radius + 1, (30, 30, 30), 1, cv2.LINE_AA)

    # Current-frame bounce: full crosshair + rings + label
    cur_bounces = [b for b in accumulated if int(b["frame_idx"]) == current_frame_idx]
    for b in cur_bounces:
        bx, by = int(round(float(b["x"]))), int(round(float(b["y"])))
        if not (0 <= bx < W_img and 0 <= by < H_img):
            continue
        in_b = bool(b.get("in_bound", True))
        main_color = side_color(b.get("court_side", "unknown"), in_bound=in_b)
        out_label = "IN" if in_b else "OUT"
        out_color = (0, 200, 50) if in_b else (0, 60, 220)

        # Cross-hair arms
        arm = 18
        cv2.line(out, (bx - arm, by), (bx + arm, by), main_color, 2, cv2.LINE_AA)
        cv2.line(out, (bx, by - arm), (bx, by + arm), main_color, 2, cv2.LINE_AA)
        # Concentric rings
        cv2.circle(out, (bx, by), 6,  main_color,     -1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 10, (255, 255, 255),  1, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 16, (30,  30,  30),   1, cv2.LINE_AA)

        # Text label — two lines: "# score" and "SIDE IN/OUT"
        score_str = f"{float(b.get('score', 0)):.2f}" if pd.notna(b.get("score", float("nan"))) else "--"
        lbl_top = f"#{int(b.get('landing_no', 0))}  s={score_str}"
        lbl_bot = f"{str(b.get('court_side','?')).upper()}  {out_label}"

        tx, ty = bx + 18, by - 10
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.putText(out, lbl_top, (tx + dx, ty + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0),   2, cv2.LINE_AA)
            cv2.putText(out, lbl_bot, (tx + dx, ty + dy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl_top, (tx, ty),      cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, lbl_bot, (tx, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, out_color,       1, cv2.LINE_AA)

    return out


def make_debug_video_overlay(
    video_path: Path,
    frames_dir: Path,
    bounce_csv: Path,
    roi_json: Path,
    out_path: Path,
    bounce_coord_scale: float = 1.0,
    trail_length: int = 10,
    source: str = "frames",
    fps: Optional[float] = None,
    draw_court: bool = True,
    court_alpha: float = 0.55,
):
    """Write a debug video with bounce markers and projected court lines drawn
    directly on the original footage — no side-by-side 2D diagram.

    Use this to verify that the court boundary (projected from H) aligns with
    the real painted lines in the video.  If they don't align, the homography H
    needs to be recalibrated via court_detector.auto_pick_roi().
    """
    video_path  = Path(video_path)
    frames_dir  = Path(frames_dir)
    bounce_csv  = Path(bounce_csv)
    roi_json    = Path(roi_json)
    out_path    = Path(out_path)

    bounce_df = load_bounce_events(bounce_csv, coord_scale=bounce_coord_scale)
    H = load_homography_from_roi(roi_json)
    landing_df = project_all_landings(bounce_df, H)

    print(f"[INFO] loaded bounce events: {len(bounce_df)}")
    print(f"[INFO] projected landings:   {len(landing_df)}")

    # Build lookup: frame_idx -> list of dicts with pixel coords + projected court coords
    pixel_by_frame: Dict[int, List[dict]] = {}
    bounce_by_idx = {int(r["frame_idx"]): r for _, r in bounce_df.iterrows()}
    for _, row in landing_df.iterrows():
        fi = int(row["frame_idx"])
        brow = bounce_by_idx.get(fi, {})
        entry = {
            "frame_idx":  fi,
            "x":          float(brow.get("x", row.get("X", 0))),
            "y":          float(brow.get("y", row.get("Y", 0))),
            "court_side": str(row.get("court_side", "unknown")),
            "in_bound":   bool(row.get("in_bound", True)),
            "score":      row.get("score", float("nan")),
            "landing_no": int(row.get("landing_no", 0)),
        }
        pixel_by_frame.setdefault(fi, []).append(entry)

    if fps is None or fps <= 0:
        fps = get_video_fps(video_path, fallback=30.0) if video_path.exists() else 30.0

    frame_files = get_frame_files(frames_dir)
    use_frames = source == "frames"
    if use_frames:
        if not frame_files:
            raise FileNotFoundError(f"No frames in {frames_dir}. Use --source video.")
        first = cv2.imread(str(frame_files[0]))
        H_img, W = first.shape[:2]
        frame_iter = _frame_iter_from_images(frame_files)
        total_frames = len(frame_files)
    else:
        cap = cv2.VideoCapture(str(video_path))
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_iter = _frame_iter_from_video(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H_img))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    H_arr = np.asarray(H, dtype=np.float64)
    accumulated: List[dict] = []

    for frame_idx, frame in tqdm(frame_iter, total=total_frames, desc="debug overlay"):
        # 1. Project court lines onto the frame
        vis = draw_court_lines_on_frame(frame, H_arr, alpha=court_alpha) if draw_court else frame.copy()

        # 2. Accumulate new bounces
        new_hits = pixel_by_frame.get(frame_idx, [])
        accumulated.extend(new_hits)

        # 3. Draw bounce markers
        vis = draw_bounces_on_frame(vis, accumulated, frame_idx, trail_length=trail_length)

        # 4. Frame counter in corner
        cv2.putText(vis, f"f{frame_idx}", (8, H_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(vis, f"f{frame_idx}", (8, H_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(vis)

    writer.release()
    print(f"[DONE] debug overlay saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a bounce landing video.  Use --mode debug to overlay "
                    "bounce markers and court lines directly on the original footage."
    )
    parser.add_argument("video_path", type=str, help="例如 raw_videos/testVid.mp4")
    parser.add_argument(
        "--mode", choices=["overlay", "debug"], default="overlay",
        help=(
            "overlay (default): original video + 2D court diagram side-by-side. "
            "debug: bounce markers + projected court lines drawn ON the video — "
            "use this to check that the court lines align with the real painted lines."
        ),
    )
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--bounce-csv", type=str, default=None)
    parser.add_argument("--roi-json", type=str, default=None)
    parser.add_argument("--out-overlay", type=str, default=None, help="Output video path")
    parser.add_argument("--bounce-coord-scale", type=float, default=1.0,
                        help="Set to 2.0 when bounce.py was run with --scale 0.5")
    # overlay-mode options
    parser.add_argument("--court-scale", type=float, default=10.0)
    parser.add_argument("--court-margin", type=float, default=0.10)
    parser.add_argument("--court-ratio", type=float, default=0.35)
    parser.add_argument("--show-index", action="store_true")
    # debug-mode options
    parser.add_argument("--trail-length", type=int, default=10,
                        help="How many previous bounces to show as fading dots (debug mode)")
    parser.add_argument("--court-alpha", type=float, default=0.55,
                        help="Opacity of projected court lines 0–1 (debug mode)")
    parser.add_argument("--no-court-lines", action="store_true",
                        help="Skip projected court lines (debug mode)")
    parser.add_argument("--source", choices=["frames", "video"], default="frames")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    set_video_path(Path(args.video_path))

    if args.mode == "debug":
        default_out = BOUNCE_DIR / "debug_bounce_overlay.mp4"
        make_debug_video_overlay(
            video_path=Path(args.video_path),
            frames_dir=Path(args.frames_dir) if args.frames_dir else FRAMES_DIR,
            bounce_csv=Path(args.bounce_csv) if args.bounce_csv else BOUNCE_EVENTS_CSV,
            roi_json=Path(args.roi_json) if args.roi_json else ROI_JSON,
            out_path=Path(args.out_overlay) if args.out_overlay else default_out,
            bounce_coord_scale=args.bounce_coord_scale,
            trail_length=args.trail_length,
            source=args.source,
            fps=args.fps,
            draw_court=not args.no_court_lines,
            court_alpha=args.court_alpha,
        )
    else:
        make_overlay_bounce_landings(
            video_path=Path(args.video_path),
            frames_dir=Path(args.frames_dir) if args.frames_dir else FRAMES_DIR,
            bounce_csv=Path(args.bounce_csv) if args.bounce_csv else BOUNCE_EVENTS_CSV,
            roi_json=Path(args.roi_json) if args.roi_json else ROI_JSON,
            out_path=Path(args.out_overlay) if args.out_overlay else OUT_PATH_OVERLAY,
            court_scale=args.court_scale,
            court_margin=args.court_margin,
            court_ratio=args.court_ratio,
            bounce_coord_scale=args.bounce_coord_scale,
            show_index=args.show_index,
            source=args.source,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
