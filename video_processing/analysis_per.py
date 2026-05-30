import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ========================================================
# Court geometry from court_detector.py / roi_config.json
# --------------------------------------------------------
# court_detector.py stores H as: court model coordinates -> image pixels.
# Therefore image bounce coordinates must be projected with inv(H).
# The court model uses:
#   X: left -> right in meters
#   Y: far baseline (0) -> near baseline (COURT_LENGTH)
# ========================================================
COURT_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2.0
SINGLE_MARGIN = (COURT_WIDTH - SINGLES_WIDTH) / 2.0


def set_video_path(video_path: Path):
    global VIDEO_PATH, video_name, BASE, FRAMES_DIR, ROI_JSON
    global BOUNCE_DIR, BOUNCE_EVENTS_CSV, HIT_INTERVALS_CSV, HIT_SEGMENTS_CSV
    global STROKE_VOTE_EVENTS_CSV, STROKE_SUMMARY_CSV, CHARTS_DIR
    global EVENTS_CSV, REPORT_PDF

    VIDEO_PATH = Path(video_path)
    video_name = VIDEO_PATH.stem
    BASE = Path("dataset") / video_name
    FRAMES_DIR = BASE / "frames"
    ROI_JSON = BASE / "roi_config.json"
    BOUNCE_DIR = BASE / "bounce_detector"

    BOUNCE_EVENTS_CSV = BOUNCE_DIR / "bounce_events.csv"
    HIT_INTERVALS_CSV = BOUNCE_DIR / "hit_intervals_y_extrema.csv"
    HIT_SEGMENTS_CSV = BOUNCE_DIR / "hit_intervals_y_segments.csv"
    STROKE_VOTE_EVENTS_CSV = BOUNCE_DIR / "stroke_vote_same_crop" / "stroke_vote_events.csv"
    STROKE_SUMMARY_CSV = BOUNCE_DIR / "stroke_landing_summary.csv"
    CHARTS_DIR = BOUNCE_DIR / "analysis_charts"
    EVENTS_CSV = BOUNCE_DIR / "events_from_bounce_hit.csv"
    REPORT_PDF = BOUNCE_DIR / "analysis_report.pdf"


# ========================================================
# Homography / court helpers
# ========================================================
def load_court_homography(roi_json: Path) -> np.ndarray:
    roi_json = Path(roi_json)
    with open(roi_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "H" not in data:
        raise KeyError(
            f"{roi_json} does not contain H. Please regenerate roi_config.json "
            "with court_detector.auto_pick_roi()."
        )
    H = np.asarray(data["H"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Invalid H shape in {roi_json}: {H.shape}")
    return H


def project_to_court(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """Image pixel coordinate -> court-meter coordinate using inv(H)."""
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H_inv.astype(np.float64)).reshape(-1, 2)[0]
    return float(out[0]), float(out[1])


def is_in_bounds(X: float, Y: float, singles: bool = True, tol: float = 0.3) -> bool:
    """
    Checks if a point (X, Y) in court meters is inside the lines.
    Tennis rules state that if any part of the ball touches the line, it is IN.
    Standard line width is 5cm (0.05m).
    """
    width = SINGLES_WIDTH if singles else COURT_WIDTH
    x_min = (COURT_WIDTH - width) / 2.0
    x_max = x_min + width

    # The tolerance (tol) represents the line width + ball radius + error margin.
    return (
        x_min - tol <= float(X) <= x_max + tol
        and 0.0 - tol <= float(Y) <= COURT_LENGTH + tol
    )



def landing_side_from_y(Y: float) -> str:
    return "near" if float(Y) >= NET_Y else "far"


def court_to_canvas_xy(X: float, Y: float, scale: float, margin_px: float) -> Tuple[int, int]:
    """Court meters -> image pixels. Far side is top; near side is bottom."""
    return (
        int(round(float(margin_px) + float(X) * float(scale))),
        int(round(float(margin_px) + float(Y) * float(scale))),
    )


def draw_court(scale: float = 30, margin_ratio: float = 0.10) -> Tuple[np.ndarray, int, int]:
    margin_px = int(round(max(COURT_WIDTH, COURT_LENGTH) * float(scale) * float(margin_ratio)))
    W = int(round(COURT_WIDTH * float(scale) + 2 * margin_px))
    H_img = int(round(COURT_LENGTH * float(scale) + 2 * margin_px))
    img = np.full((H_img, W, 3), 255, dtype=np.uint8)
    color = (0, 220, 0)
    gray = (150, 150, 150)
    lw = max(1, int(round(float(scale) / 18)))

    def pt(x, y):
        return court_to_canvas_xy(x, y, scale, margin_px)

    cv2.rectangle(img, pt(0, 0), pt(COURT_WIDTH, COURT_LENGTH), color, lw)
    cv2.line(img, pt(SINGLE_MARGIN, 0), pt(SINGLE_MARGIN, COURT_LENGTH), color, lw)
    cv2.line(img, pt(SINGLE_MARGIN + SINGLES_WIDTH, 0), pt(SINGLE_MARGIN + SINGLES_WIDTH, COURT_LENGTH), color, lw)
    cv2.line(img, pt(0, NET_Y), pt(COURT_WIDTH, NET_Y), gray, lw)

    far_service = NET_Y - SERVICE_FROM_NET
    near_service = NET_Y + SERVICE_FROM_NET
    cv2.line(img, pt(SINGLE_MARGIN, far_service), pt(SINGLE_MARGIN + SINGLES_WIDTH, far_service), color, lw)
    cv2.line(img, pt(SINGLE_MARGIN, near_service), pt(SINGLE_MARGIN + SINGLES_WIDTH, near_service), color, lw)
    cv2.line(img, pt(COURT_WIDTH / 2.0, far_service), pt(COURT_WIDTH / 2.0, near_service), color, lw)
    return img, margin_px, margin_px


# ========================================================
# Load bounce.py outputs
# ========================================================
def _read_csv(path: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{name} not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if required:
            raise FileNotFoundError(f"{name} exists but is empty: {path}")
        return pd.DataFrame()


def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_bounce_events(path: Optional[Path] = None, coord_scale: float = 1.0) -> pd.DataFrame:
    if path is None:
        path = BOUNCE_EVENTS_CSV
    raw = _read_csv(path, "bounce_events.csv", required=True)

    frame_col = _first_col(raw, ["peak_frame", "refined_frame", "frame_idx"])
    x_col = _first_col(raw, ["cx", "refined_cx", "x"])
    y_col = _first_col(raw, ["cy", "refined_cy", "y"])
    side_col = _first_col(raw, ["court_side", "landing_side", "side"])
    score_col = _first_col(raw, ["peak_score", "refine_score", "score"])

    if frame_col is None or x_col is None or y_col is None:
        raise ValueError("bounce_events.csv must contain peak_frame/cx/cy or equivalent columns.")

    out = pd.DataFrame()
    out["frame_idx"] = pd.to_numeric(raw[frame_col], errors="coerce")
    out["x"] = pd.to_numeric(raw[x_col], errors="coerce") * float(coord_scale)
    out["y"] = pd.to_numeric(raw[y_col], errors="coerce") * float(coord_scale)
    out["landing_side_csv"] = raw[side_col].astype(str).str.lower() if side_col else "unknown"
    out["score"] = pd.to_numeric(raw[score_col], errors="coerce") if score_col else np.nan

    out = out.dropna(subset=["frame_idx", "x", "y"]).copy()
    out["frame_idx"] = out["frame_idx"].round().astype(int)
    out = out.sort_values("frame_idx").reset_index(drop=True)
    out["bounce_no"] = np.arange(1, len(out) + 1)
    return out


def load_hit_intervals(path: Optional[Path] = None, fallback_path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = HIT_INTERVALS_CSV
    if fallback_path is None:
        fallback_path = HIT_SEGMENTS_CSV
    if not Path(path).exists() and Path(fallback_path).exists():
        path = fallback_path

    raw = _read_csv(path, "hit_intervals.csv", required=False)
    if raw.empty:
        return pd.DataFrame(columns=["frame_idx", "start", "end", "hitter", "event_type", "hit_no"])

    start_col = _first_col(raw, ["interval_start_frame", "start_frame", "window_start"])
    end_col = _first_col(raw, ["interval_end_frame", "end_frame", "window_end"])
    center_col = _first_col(raw, ["center_frame", "peak_frame", "frame_idx"])
    side_col = _first_col(raw, ["hit_side", "court_side", "side"])

    if start_col is None or end_col is None or center_col is None or side_col is None:
        raise ValueError("hit interval csv must contain start/end/center/hit_side columns.")

    out = pd.DataFrame()
    out["frame_idx"] = pd.to_numeric(raw[center_col], errors="coerce")
    out["start"] = pd.to_numeric(raw[start_col], errors="coerce")
    out["end"] = pd.to_numeric(raw[end_col], errors="coerce")
    out["hitter"] = raw[side_col].astype(str).str.lower()
    out["event_type"] = "hit"
    out = out.dropna(subset=["frame_idx", "start", "end", "hitter"]).copy()
    out["frame_idx"] = out["frame_idx"].round().astype(int)
    out["start"] = out["start"].round().astype(int)
    out["end"] = out["end"].round().astype(int)
    out = out.sort_values("frame_idx").reset_index(drop=True)
    out["hit_no"] = np.arange(1, len(out) + 1)
    return out


def load_stroke_votes(path: Optional[Path] = None, required: bool = False) -> pd.DataFrame:
    """Load vote_action.py event-level forehand/backhand predictions.

    Expected default file:
      dataset/<video>/bounce_detector/stroke_vote_same_crop/stroke_vote_events.csv

    The important matching key is event_center_frame + event_side, because
    vote_action.py reads the same hit interval event file and then writes
    one row per voted hit event.
    """
    if path is None:
        path = STROKE_VOTE_EVENTS_CSV
    raw = _read_csv(Path(path), "stroke_vote_events.csv", required=required)
    if raw.empty:
        return pd.DataFrame(columns=[
            "event_center_frame", "event_side", "event_pred_label",
            "vote_count", "winning_votes", "vote_ratio", "mean_winning_prob",
            "used_vote_phase", "tie_breaker",
        ])

    frame_col = _first_col(raw, ["event_center_frame", "center_frame", "peak_frame", "frame_idx"])
    side_col = _first_col(raw, ["event_side", "hit_side", "court_side", "chosen_side"])
    label_col = _first_col(raw, ["event_pred_label", "pred_label", "stroke_label"])

    if frame_col is None or label_col is None:
        raise ValueError("stroke_vote_events.csv must contain event_center_frame and event_pred_label columns.")

    out = pd.DataFrame()
    out["event_center_frame"] = pd.to_numeric(raw[frame_col], errors="coerce")
    out["event_side"] = raw[side_col].astype(str).str.lower() if side_col else "unknown"
    out["event_pred_label"] = raw[label_col].astype(str).str.lower()

    for col in ["vote_count", "winning_votes", "vote_ratio", "mean_winning_prob"]:
        out[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan

    out["used_vote_phase"] = raw["used_vote_phase"].astype(str) if "used_vote_phase" in raw.columns else ""
    out["tie_breaker"] = raw["tie_breaker"].astype(str) if "tie_breaker" in raw.columns else ""

    out = out.dropna(subset=["event_center_frame"]).copy()
    out["event_center_frame"] = out["event_center_frame"].round().astype(int)
    out = out.sort_values(["event_center_frame", "event_side"]).reset_index(drop=True)
    return out


def attach_stroke_votes_to_hits(
    hits: pd.DataFrame,
    stroke_votes: pd.DataFrame,
    min_vote_count: int = 1,
    min_vote_ratio: float = 0.0,
    min_mean_prob: float = 0.0,
) -> pd.DataFrame:
    """Attach event-level stroke labels to hit intervals.

    Only hits with a reliable vote keep forehand/backhand. Others become unknown.
    This is intentional: far-side hits normally have no vote result when
    vote_action.py is run with --target-side near.
    """
    hits = hits.copy()
    default_cols = {
        "stroke_label": "unknown",
        "stroke_vote_count": np.nan,
        "stroke_winning_votes": np.nan,
        "stroke_vote_ratio": np.nan,
        "stroke_mean_prob": np.nan,
        "stroke_used_vote_phase": "",
        "stroke_tie_breaker": "",
        "stroke_vote_valid": 0,
    }
    for col, val in default_cols.items():
        hits[col] = val

    if hits.empty or stroke_votes is None or stroke_votes.empty:
        return hits

    sv = stroke_votes.copy()
    sv = sv.rename(columns={
        "event_center_frame": "frame_idx",
        "event_side": "hitter",
        "event_pred_label": "stroke_label",
        "vote_count": "stroke_vote_count",
        "winning_votes": "stroke_winning_votes",
        "vote_ratio": "stroke_vote_ratio",
        "mean_winning_prob": "stroke_mean_prob",
        "used_vote_phase": "stroke_used_vote_phase",
        "tie_breaker": "stroke_tie_breaker",
    })

    keep_cols = [
        "frame_idx", "hitter", "stroke_label", "stroke_vote_count",
        "stroke_winning_votes", "stroke_vote_ratio", "stroke_mean_prob",
        "stroke_used_vote_phase", "stroke_tie_breaker",
    ]
    sv = sv[[c for c in keep_cols if c in sv.columns]].copy()
    sv["frame_idx"] = pd.to_numeric(sv["frame_idx"], errors="coerce").round().astype("Int64")
    sv["hitter"] = sv["hitter"].astype(str).str.lower()
    sv = sv.dropna(subset=["frame_idx"]).copy()
    sv["frame_idx"] = sv["frame_idx"].astype(int)

    hits = hits.merge(sv, on=["frame_idx", "hitter"], how="left", suffixes=("", "_vote"))

    for col in ["stroke_label", "stroke_used_vote_phase", "stroke_tie_breaker"]:
        if f"{col}_vote" in hits.columns:
            hits[col] = hits[f"{col}_vote"].combine_first(hits[col])
            hits = hits.drop(columns=[f"{col}_vote"])

    for col in ["stroke_vote_count", "stroke_winning_votes", "stroke_vote_ratio", "stroke_mean_prob"]:
        if f"{col}_vote" in hits.columns:
            hits[col] = hits[f"{col}_vote"].combine_first(hits[col])
            hits = hits.drop(columns=[f"{col}_vote"])

    valid = (
        hits["stroke_label"].isin(["forehand", "backhand"]) &
        (pd.to_numeric(hits["stroke_vote_count"], errors="coerce").fillna(0) >= int(min_vote_count)) &
        (pd.to_numeric(hits["stroke_vote_ratio"], errors="coerce").fillna(0) >= float(min_vote_ratio)) &
        (pd.to_numeric(hits["stroke_mean_prob"], errors="coerce").fillna(0) >= float(min_mean_prob))
    )
    hits.loc[~valid, "stroke_label"] = "unknown"
    hits["stroke_vote_valid"] = valid.astype(int)
    return hits


def _opposite_side(side: str) -> Optional[str]:
    side = str(side).lower()
    if side == "near":
        return "far"
    if side == "far":
        return "near"
    return None


def assign_player_view_to_bounces(bounces: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Assign each bounce to the NEAR/FAR player-view map.

    This keeps the original user-facing report style:
      - NEAR Player Bounce Map shows landings from near player's viewing logic.
      - FAR Player Bounce Map shows landings from far player's viewing logic.

    Important first-bounce rule:
    If the first visible bounce occurs before the first hit center, it is the
    incoming ball to that first hitter. Therefore it belongs to the opposite
    player's view map. This prevents the first bounce from disappearing just
    because the clip starts mid-rally.
    """
    bounces = bounces.copy()
    bounces["player_view"] = "unknown"
    bounces["linked_hit_side"] = "unknown"
    bounces["linked_hit_frame"] = np.nan
    bounces["linked_hit_no"] = np.nan
    bounces["linked_stroke_label"] = "unknown"
    bounces["linked_stroke_vote_count"] = np.nan
    bounces["linked_stroke_vote_ratio"] = np.nan
    bounces["linked_stroke_mean_prob"] = np.nan
    bounces["assignment_note"] = ""

    if hits.empty:
        return bounces

    first_hit = hits.iloc[0]
    first_hit_frame = int(first_hit["frame_idx"])
    first_hit_side = str(first_hit["hitter"]).lower()

    for i, row in bounces.iterrows():
        f = int(row["frame_idx"])

        # Clip starts mid-rally: a bounce before the first hit center is the
        # ball coming from the opposite player. Keep it on the map.
        if f < first_hit_frame:
            opposite = _opposite_side(first_hit_side)
            if opposite is not None:
                bounces.at[i, "player_view"] = opposite
                bounces.at[i, "linked_hit_side"] = first_hit_side
                bounces.at[i, "linked_hit_frame"] = first_hit_frame
                bounces.at[i, "linked_hit_no"] = int(first_hit["hit_no"])
                bounces.at[i, "assignment_note"] = "pre_first_hit_opposite_side"
                continue

        previous_hits = hits[hits["frame_idx"] <= f]
        if previous_hits.empty:
            active_hits = hits[(hits["start"] <= f) & (hits["end"] >= f)]
            if active_hits.empty:
                continue
            h = active_hits.iloc[0]
            note = "active_interval"
        else:
            h = previous_hits.iloc[-1]
            note = "previous_hit"

        side = str(h["hitter"]).lower()
        if side not in ("near", "far"):
            continue

        bounces.at[i, "player_view"] = side
        bounces.at[i, "linked_hit_side"] = side
        bounces.at[i, "linked_hit_frame"] = int(h["frame_idx"])
        bounces.at[i, "linked_hit_no"] = int(h["hit_no"])
        bounces.at[i, "linked_stroke_label"] = str(h.get("stroke_label", "unknown"))
        bounces.at[i, "linked_stroke_vote_count"] = h.get("stroke_vote_count", np.nan)
        bounces.at[i, "linked_stroke_vote_ratio"] = h.get("stroke_vote_ratio", np.nan)
        bounces.at[i, "linked_stroke_mean_prob"] = h.get("stroke_mean_prob", np.nan)
        bounces.at[i, "assignment_note"] = note

    return bounces


def build_event_dataframe(bounces: pd.DataFrame, hits: pd.DataFrame, roi_json: Path) -> pd.DataFrame:
    H = load_court_homography(roi_json)
    bounces = assign_player_view_to_bounces(bounces, hits)

    bounce_rows = []
    for _, row in bounces.iterrows():
        X, Y = project_to_court(row["x"], row["y"], H)
        side_by_y = landing_side_from_y(Y)
        side_csv = str(row.get("landing_side_csv", "unknown")).lower()
        landing_side = side_csv if side_csv in ("near", "far") else side_by_y
        bounce_rows.append({
            "frame_idx": int(row["frame_idx"]),
            "event_type": "bounce",
            "is_hit": 0,
            "is_bounce": 1,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "court_x": float(X),
            "court_y": float(Y),
            "landing_side": landing_side,
            "landing_side_by_y": side_by_y,
            "player_view": str(row.get("player_view", "unknown")),
            "linked_hit_side": str(row.get("linked_hit_side", "unknown")),
            "linked_hit_frame": row.get("linked_hit_frame", np.nan),
            "linked_hit_no": row.get("linked_hit_no", np.nan),
            "linked_stroke_label": str(row.get("linked_stroke_label", "unknown")),
            "linked_stroke_vote_count": row.get("linked_stroke_vote_count", np.nan),
            "linked_stroke_vote_ratio": row.get("linked_stroke_vote_ratio", np.nan),
            "linked_stroke_mean_prob": row.get("linked_stroke_mean_prob", np.nan),
            "assignment_note": str(row.get("assignment_note", "")),
            "score": row.get("score", np.nan),
            "bounce_no": int(row["bounce_no"]),
            "in_singles": int(is_in_bounds(X, Y, singles=True)),
        })

    hit_rows = []
    for _, row in hits.iterrows():
        hit_rows.append({
            "frame_idx": int(row["frame_idx"]),
            "event_type": "hit",
            "is_hit": 1,
            "is_bounce": 0,
            "x": np.nan,
            "y": np.nan,
            "court_x": np.nan,
            "court_y": np.nan,
            "landing_side": None,
            "landing_side_by_y": None,
            "player_view": None,
            "linked_hit_side": None,
            "linked_hit_frame": np.nan,
            "linked_hit_no": np.nan,
            "assignment_note": "",
            "score": np.nan,
            "bounce_no": np.nan,
            "hit_side": str(row["hitter"]),
            "hit_no": int(row["hit_no"]),
            "hit_start": int(row["start"]),
            "hit_end": int(row["end"]),
            "stroke_label": str(row.get("stroke_label", "unknown")),
            "stroke_vote_count": row.get("stroke_vote_count", np.nan),
            "stroke_winning_votes": row.get("stroke_winning_votes", np.nan),
            "stroke_vote_ratio": row.get("stroke_vote_ratio", np.nan),
            "stroke_mean_prob": row.get("stroke_mean_prob", np.nan),
            "stroke_used_vote_phase": str(row.get("stroke_used_vote_phase", "")),
            "stroke_tie_breaker": str(row.get("stroke_tie_breaker", "")),
            "stroke_vote_valid": int(row.get("stroke_vote_valid", 0)),
            "in_singles": np.nan,
        })

    events = pd.DataFrame(hit_rows + bounce_rows)
    events = events.sort_values(["frame_idx", "is_hit"]).reset_index(drop=True)

    # Re-calculate in_singles for bounces with explicit tolerance to ensure line-hits are IN.
    # 0.3m (30cm) allows for line width + ball radius + projection drift.
    bounce_mask = events["is_bounce"] == 1
    events.loc[bounce_mask, "in_singles"] = events.loc[bounce_mask].apply(
        lambda r: int(is_in_bounds(r["court_x"], r["court_y"], singles=True, tol=0.3)), axis=1
    )
    return events


# ========================================================
# Landing statistics: original player-view report style
# ========================================================
def _bounce_rows(events: pd.DataFrame) -> pd.DataFrame:
    return events[(events["is_bounce"] == 1) & events["court_x"].notna() & events["court_y"].notna()].copy()


def _landing_half_from_court_y(Yc: float) -> str:
    return "near" if float(Yc) >= NET_Y else "far"


def _is_opponent_half_for_view(player_view: str, Yc: float) -> bool:
    """For the original player-view map, NEAR view targets far half, FAR view targets near half."""
    player_view = str(player_view).lower()
    landing_half = _landing_half_from_court_y(float(Yc))
    if player_view == "near":
        return landing_half == "far"
    if player_view == "far":
        return landing_half == "near"
    return False


def in_out_by_player_view(events: pd.DataFrame, only_opponent_half: bool = True) -> dict:
    bounces = _bounce_rows(events)
    stats = {"near": {"in": 0, "out": 0}, "far": {"in": 0, "out": 0}}
    for _, row in bounces.iterrows():
        player = str(row.get("player_view", "unknown")).lower()
        if player not in stats:
            continue
        if only_opponent_half and not _is_opponent_half_for_view(player, row["court_y"]):
            continue
        if bool(row["in_singles"]):
            stats[player]["in"] += 1
        else:
            stats[player]["out"] += 1
    return stats


def bounce_zone_stats(events: pd.DataFrame, view_side: str = "near", only_opponent_half: bool = True) -> Optional[dict]:
    """Stats for the original NEAR/FAR Player Bounce Map.

    view_side is the player-view label used in the report. Normally we show
    landings on the opponent half for that view. The first bounce before the
    first hit is not lost, because build_event_dataframe assigns it to the
    opposite player-view first.
    """
    view_side = str(view_side).lower()
    bounces = _bounce_rows(events)
    bounces = bounces[bounces["player_view"] == view_side].copy()
    if only_opponent_half:
        bounces = bounces[bounces["court_y"].apply(lambda y: _is_opponent_half_for_view(view_side, y))]
    if bounces.empty:
        return None

    mid_x = COURT_WIDTH / 2.0
    service_far = NET_Y - SERVICE_FROM_NET
    service_near = NET_Y + SERVICE_FROM_NET

    left_cnt = right_cnt = front_cnt = back_cnt = 0
    xy = []

    for _, row in bounces.iterrows():
        Xc = float(row["court_x"])
        Yc = float(row["court_y"])
        xy.append([Xc, Yc])

        # 左右：依球員視角。near 不反轉；far 視角左右反轉。
        if view_side == "near":
            if Xc < mid_x:
                left_cnt += 1
            else:
                right_cnt += 1
        else:
            if Xc < mid_x:
                right_cnt += 1
            else:
                left_cnt += 1

        # 前後：front = 對方半場靠近網子；back = 對方半場靠近底線。
        if view_side == "near":
            # near player view -> opponent half is far side: 0 ~ NET_Y
            if service_far <= Yc <= NET_Y:
                front_cnt += 1
            else:
                back_cnt += 1
        else:
            # far player view -> opponent half is near side: NET_Y ~ COURT_LENGTH
            if NET_Y <= Yc <= service_near:
                front_cnt += 1
            else:
                back_cnt += 1

    total = len(xy)
    return {
        "xy": np.asarray(xy, dtype=float),
        "count": int(total),
        "left": left_cnt / total * 100,
        "right": right_cnt / total * 100,
        "front": front_cnt / total * 100,
        "back": back_cnt / total * 100,
    }


def _stroke_zone_for_bounce(row: pd.Series, view_side: str) -> Tuple[str, str]:
    """Return left/right and front/back for a bounce from a player's view."""
    Xc = float(row["court_x"])
    Yc = float(row["court_y"])
    mid_x = COURT_WIDTH / 2.0
    service_far = NET_Y - SERVICE_FROM_NET
    service_near = NET_Y + SERVICE_FROM_NET

    if view_side == "near":
        lr = "left" if Xc < mid_x else "right"
        fb = "front" if service_far <= Yc <= NET_Y else "back"
    else:
        lr = "right" if Xc < mid_x else "left"
        fb = "front" if NET_Y <= Yc <= service_near else "back"
    return lr, fb


def stroke_landing_summary(
    events: pd.DataFrame,
    view_side: str = "near",
    only_opponent_half: bool = True,
) -> pd.DataFrame:
    """Summarize landing outcomes by forehand/backhand for one player view.

    This uses bounce rows, not hit rows, because the useful tactical outcome is
    where the ball landed after the labeled hit.
    """
    view_side = str(view_side).lower()
    bounces = _bounce_rows(events)
    bounces = bounces[bounces["player_view"].astype(str).str.lower() == view_side].copy()

    if only_opponent_half:
        bounces = bounces[bounces["court_y"].apply(lambda y: _is_opponent_half_for_view(view_side, y))]

    if bounces.empty or "linked_stroke_label" not in bounces.columns:
        return pd.DataFrame(columns=[
            "player_view", "stroke_label", "landings", "in_count", "out_count", "in_rate",
            "left_count", "right_count", "front_count", "back_count",
            "left_pct", "right_pct", "front_pct", "back_pct",
            "mean_vote_ratio", "mean_stroke_prob",
        ])

    bounces["linked_stroke_label"] = bounces["linked_stroke_label"].astype(str).str.lower()
    bounces = bounces[bounces["linked_stroke_label"].isin(["forehand", "backhand"])].copy()
    if bounces.empty:
        return pd.DataFrame(columns=[
            "player_view", "stroke_label", "landings", "in_count", "out_count", "in_rate",
            "left_count", "right_count", "front_count", "back_count",
            "left_pct", "right_pct", "front_pct", "back_pct",
            "mean_vote_ratio", "mean_stroke_prob",
        ])

    zones = bounces.apply(lambda r: _stroke_zone_for_bounce(r, view_side), axis=1)
    bounces["lr_zone"] = [z[0] for z in zones]
    bounces["fb_zone"] = [z[1] for z in zones]

    rows = []
    for stroke in ["forehand", "backhand"]:
        part = bounces[bounces["linked_stroke_label"] == stroke].copy()
        total = int(len(part))
        if total == 0:
            continue

        in_count = int(pd.to_numeric(part["in_singles"], errors="coerce").fillna(0).sum())
        out_count = int(total - in_count)
        left_count = int((part["lr_zone"] == "left").sum())
        right_count = int((part["lr_zone"] == "right").sum())
        front_count = int((part["fb_zone"] == "front").sum())
        back_count = int((part["fb_zone"] == "back").sum())

        rows.append({
            "player_view": view_side,
            "stroke_label": stroke,
            "landings": total,
            "in_count": in_count,
            "out_count": out_count,
            "in_rate": in_count / total if total else np.nan,
            "left_count": left_count,
            "right_count": right_count,
            "front_count": front_count,
            "back_count": back_count,
            "left_pct": left_count / total if total else np.nan,
            "right_pct": right_count / total if total else np.nan,
            "front_pct": front_count / total if total else np.nan,
            "back_pct": back_count / total if total else np.nan,
            "mean_vote_ratio": pd.to_numeric(part["linked_stroke_vote_ratio"], errors="coerce").mean(),
            "mean_stroke_prob": pd.to_numeric(part["linked_stroke_mean_prob"], errors="coerce").mean(),
        })

    return pd.DataFrame(rows)


# ========================================================
# Plot landing maps
# ========================================================
def _rect_from_court_coords(x1, y1, x2, y2, scale, margin_px):
    px1, py1 = court_to_canvas_xy(x1, y1, scale, margin_px)
    px2, py2 = court_to_canvas_xy(x2, y2, scale, margin_px)
    return (min(px1, px2), min(py1, py2)), abs(px2 - px1), abs(py2 - py1)


def plot_bounce_map_with_zones(stats: dict, view_side: str = "near", scale: float = 30):
    xy = stats["xy"]
    court_img, margin_px, _ = draw_court(scale=scale)
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.imshow(court_img)
    ax.axis("off")

    service_far = NET_Y - SERVICE_FROM_NET
    service_near = NET_Y + SERVICE_FROM_NET
    singles_left = SINGLE_MARGIN
    singles_right = SINGLE_MARGIN + SINGLES_WIDTH
    mid_x = COURT_WIDTH / 2.0

    if view_side == "near":
        # NEAR player-view: target/opponent half is the FAR side.
        half_top, half_bottom = 0.0, NET_Y
        front_top, front_bottom = service_far, NET_Y
        back_top, back_bottom = 0.0, service_far
        left_color, right_color = "blue", "red"
        left_label_x = SINGLE_MARGIN + SINGLES_WIDTH * 0.25
        right_label_x = SINGLE_MARGIN + SINGLES_WIDTH * 0.75
    else:
        # FAR player-view: target/opponent half is the NEAR side.
        half_top, half_bottom = NET_Y, COURT_LENGTH
        front_top, front_bottom = NET_Y, service_near
        back_top, back_bottom = service_near, COURT_LENGTH
        left_color, right_color = "red", "blue"  # far side perspective is reversed
        left_label_x = SINGLE_MARGIN + SINGLES_WIDTH * 0.75
        right_label_x = SINGLE_MARGIN + SINGLES_WIDTH * 0.25

    # left/right outlines on selected half
    (xy0, w, h) = _rect_from_court_coords(singles_left, half_top, mid_x, half_bottom, scale, margin_px)
    ax.add_patch(patches.Rectangle(xy0, w, h, facecolor="none", edgecolor=left_color, linewidth=1))
    (xy0, w, h) = _rect_from_court_coords(mid_x, half_top, singles_right, half_bottom, scale, margin_px)
    ax.add_patch(patches.Rectangle(xy0, w, h, facecolor="none", edgecolor=right_color, linewidth=1))

    # front / back shading
    (xy0, w, h) = _rect_from_court_coords(singles_left, front_top, singles_right, front_bottom, scale, margin_px)
    ax.add_patch(patches.Rectangle(xy0, w, h, facecolor="green", alpha=0.20))
    (xy0, w, h) = _rect_from_court_coords(singles_left, back_top, singles_right, back_bottom, scale, margin_px)
    ax.add_patch(patches.Rectangle(xy0, w, h, facecolor="#e76d0a", alpha=0.25))

    # labels
    _, label_y = court_to_canvas_xy(mid_x, (half_top + half_bottom) / 2.0, scale, margin_px)
    lx, _ = court_to_canvas_xy(left_label_x, 0, scale, margin_px)
    rx, _ = court_to_canvas_xy(right_label_x, 0, scale, margin_px)
    ax.text(lx, label_y, f"Left {stats['left']:.1f}%", color="blue", ha="center")
    ax.text(rx, label_y, f"Right {stats['right']:.1f}%", color="red", ha="center")

    fx, fy = court_to_canvas_xy(mid_x, (front_top + front_bottom) / 2.0, scale, margin_px)
    bx, by = court_to_canvas_xy(mid_x, (back_top + back_bottom) / 2.0, scale, margin_px)
    ax.text(fx + 4, fy, f"Front {stats['front']:.1f}%", color="green", ha="center", va="center")
    ax.text(bx + 4, by, f"Back {stats['back']:.1f}%", color="#e76d0a", ha="center", va="center")

    colors = ["black" if is_in_bounds(Xc, Yc, singles=True, tol=0.3) else "gray" for Xc, Yc in xy]
    xs = xy[:, 0] * scale + margin_px
    ys = xy[:, 1] * scale + margin_px
    ax.scatter(xs, ys, c=colors, s=20)

    ax.set_title(f"{view_side.upper()} Player Bounce Map ({stats['count']} landings)")
    return fig


# ========================================================
# Rally extraction: independent from landing maps
# ========================================================
def extract_rallies_true(events: pd.DataFrame) -> List[dict]:
    rows = events.sort_values(["frame_idx", "is_hit"]).reset_index(drop=True)
    rallies = []
    current_shots = 0
    in_rally = False

    for i, row in rows.iterrows():
        if int(row.get("is_hit", 0)) == 1:
            if not in_rally:
                in_rally = True
                current_shots = 1
            else:
                current_shots += 1
            continue

        if int(row.get("is_bounce", 0)) == 1 and in_rally:
            # OUT ends rally.
            if int(row.get("in_singles", 0)) == 0:
                rallies.append({"shots": current_shots, "reason": "OUT"})
                in_rally = False
                current_shots = 0
                continue

            # Double bounce: next event is also bounce before another hit.
            if i + 1 < len(rows) and int(rows.loc[i + 1, "is_bounce"]) == 1:
                rallies.append({"shots": current_shots, "reason": "double-bounce"})
                in_rally = False
                current_shots = 0
                continue

    if in_rally and current_shots > 0:
        rallies.append({"shots": current_shots, "reason": "video ended"})
    return rallies


# ========================================================
# PDF Summary / Visual analytics
# ========================================================
def _safe_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _safe_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"


def _save_pdf_page(pdf: PdfPages, fig, png_path: Optional[Path] = None):
    """Save one figure to both PDF and PNG without clipping page content."""
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _empty_chart(title: str, message: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=16)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    return fig


def plot_summary_dashboard(events: pd.DataFrame, rallies: List[dict], stroke_summary: pd.DataFrame):
    """A table-based summary page. This replaces long text lines that can overflow."""
    stats = in_out_by_player_view(events, only_opponent_half=True)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.axis("off")

    ax.text(0.04, 0.94, "Match Analysis Summary", fontsize=20, weight="bold", transform=ax.transAxes)
    ax.text(
        0.04,
        0.89,
        "Landing statistics are counted on the opponent half from each player's view.",
        fontsize=10,
        transform=ax.transAxes,
    )

    # Player-level in/out table.
    player_rows = []
    for player in ["near", "far"]:
        in_count = int(stats[player]["in"])
        out_count = int(stats[player]["out"])
        total = in_count + out_count
        player_rows.append([
            player.upper(),
            str(in_count),
            str(out_count),
            str(total),
            _safe_rate(in_count, total),
        ])

    ax.text(0.04, 0.82, "Player Landing Result", fontsize=13, weight="bold", transform=ax.transAxes)
    t1 = ax.table(
        cellText=player_rows,
        colLabels=["Player", "In", "Out", "Total", "In-rate"],
        cellLoc="center",
        colWidths=[0.16, 0.16, 0.16, 0.16, 0.16],
        loc="upper left",
        bbox=[0.04, 0.61, 0.42, 0.18],
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1, 1.3)

    # Stroke-level table.
    ax.text(0.52, 0.82, "NEAR Stroke Landing Summary", fontsize=13, weight="bold", transform=ax.transAxes)
    if stroke_summary is None or stroke_summary.empty:
        ax.text(0.52, 0.70, "No reliable near-player stroke labels found.", fontsize=11, transform=ax.transAxes)
    else:
        stroke_rows = []
        for _, r in stroke_summary.iterrows():
            stroke_rows.append([
                str(r["stroke_label"]).title(),
                str(int(r["landings"])),
                f"{int(r['in_count'])}/{int(r['out_count'])}",
                _safe_pct(r["in_rate"]),
                f"{_safe_pct(r['left_pct'])}\n/ {_safe_pct(r['right_pct'])}",
                f"{_safe_pct(r['front_pct'])}\n/ {_safe_pct(r['back_pct'])}",
            ])
        t2 = ax.table(
            cellText=stroke_rows,
            colLabels=["Stroke", "Landings", "In/Out", "In-rate", "L/R", "F/B"],
            cellLoc="center",
            colWidths=[0.15, 0.14, 0.14, 0.14, 0.16, 0.16],
            loc="upper left",
            bbox=[0.50, 0.55, 0.44, 0.26],
        )
        t2.auto_set_font_size(False)
        t2.set_fontsize(8.4)
        t2.scale(1, 1.45)
        for (_, _), cell in t2.get_celld().items():
            cell.set_text_props(va='center', ha='center')

    # Rally summary.
    rally_lengths = [int(r["shots"]) for r in rallies] if rallies else []
    reason_counts: Dict[str, int] = {}
    for r in rallies:
        reason = str(r.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    rally_rows = [
        ["Total rallies", str(len(rallies))],
        ["Average rally length", f"{np.mean(rally_lengths):.2f} shots" if rally_lengths else "N/A"],
        ["Longest rally", f"{max(rally_lengths)} shots" if rally_lengths else "N/A"],
        ["End reasons", ", ".join(f"{k}: {v}" for k, v in reason_counts.items()) if reason_counts else "N/A"],
    ]
    ax.text(0.04, 0.48, "Rally Summary", fontsize=13, weight="bold", transform=ax.transAxes)
    t3 = ax.table(
        cellText=rally_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colWidths=[0.28, 0.28],
        loc="upper left",
        bbox=[0.04, 0.26, 0.55, 0.18],
    )
    t3.auto_set_font_size(False)
    t3.set_fontsize(10)
    t3.scale(1, 1.25)
    
    return fig


def plot_stroke_landing_map(events: pd.DataFrame, view_side: str = "near", only_opponent_half: bool = True, scale: float = 30):
    """Court map with forehand/backhand landing points for the selected player view."""
    view_side = str(view_side).lower()
    bounces = _bounce_rows(events)
    bounces = bounces[bounces["player_view"].astype(str).str.lower() == view_side].copy()
    if only_opponent_half:
        bounces = bounces[bounces["court_y"].apply(lambda y: _is_opponent_half_for_view(view_side, y))]
    if "linked_stroke_label" in bounces.columns:
        bounces["linked_stroke_label"] = bounces["linked_stroke_label"].astype(str).str.lower()
    else:
        bounces["linked_stroke_label"] = "unknown"
    bounces = bounces[bounces["linked_stroke_label"].isin(["forehand", "backhand"])].copy()

    if bounces.empty:
        return _empty_chart(
            f"{view_side.upper()} Stroke Landing Map",
            "No reliable stroke-labeled landing points to plot.",
        )

    court_img, margin_px, _ = draw_court(scale=scale)
    fig, ax = plt.subplots(figsize=(5.8, 10.2))
    ax.imshow(court_img)
    ax.axis("off")

    markers = {"forehand": "o", "backhand": "^"}
    for stroke in ["forehand", "backhand"]:
        part = bounces[bounces["linked_stroke_label"] == stroke]
        if part.empty:
            continue
        in_part = part[pd.to_numeric(part["in_singles"], errors="coerce").fillna(0).astype(int) == 1]
        out_part = part[pd.to_numeric(part["in_singles"], errors="coerce").fillna(0).astype(int) == 0]

        if not in_part.empty:
            xs = in_part["court_x"].astype(float).to_numpy() * scale + margin_px
            ys = in_part["court_y"].astype(float).to_numpy() * scale + margin_px
            ax.scatter(xs, ys, s=58, marker=markers[stroke], alpha=0.85, label=f"{stroke.title()} IN ({len(in_part)})")

        if not out_part.empty:
            xs = out_part["court_x"].astype(float).to_numpy() * scale + margin_px
            ys = out_part["court_y"].astype(float).to_numpy() * scale + margin_px
            ax.scatter(xs, ys, s=72, marker="x", linewidths=2.0, label=f"{stroke.title()} OUT ({len(out_part)})")

    ax.set_title(f"{view_side.upper()} Player Stroke Landing Map", fontsize=14, pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)
    return fig


def plot_stroke_in_out_chart(stroke_summary: pd.DataFrame):
    """Stacked count chart: in/out landings by stroke."""
    if stroke_summary is None or stroke_summary.empty:
        return _empty_chart("Stroke In/Out Result", "No stroke summary data to plot.")

    df = stroke_summary.copy()
    df["stroke_label"] = df["stroke_label"].astype(str).str.title()
    x = np.arange(len(df))
    in_counts = pd.to_numeric(df["in_count"], errors="coerce").fillna(0).to_numpy()
    out_counts = pd.to_numeric(df["out_count"], errors="coerce").fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, in_counts, label="In")
    ax.bar(x, out_counts, bottom=in_counts, label="Out")

    for i, (_, r) in enumerate(df.iterrows()):
        total = int(r["landings"])
        text = _safe_pct(r["in_rate"])
        ax.text(i, total + 0.08, text, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(df["stroke_label"].tolist())
    ax.set_ylabel("Landing count")
    ax.set_title("NEAR Player: In/Out by Stroke")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ymax = max(1, int((in_counts + out_counts).max()))
    ax.set_ylim(0, ymax + 1)
    ax.set_yticks(np.arange(0, ymax + 2, 1))

    fig.tight_layout()
    return fig


def plot_stroke_zone_chart(stroke_summary: pd.DataFrame):
    """Two clear charts instead of one misleading 4-series chart.

    Important: Left/Right is one 100% split, and Front/Back is another 100% split.
    They should not be visually read as four categories summing to 100 together.
    """
    if stroke_summary is None or stroke_summary.empty:
        return _empty_chart("Stroke Zone Distribution", "No stroke summary data to plot.")

    df = stroke_summary.copy()
    df["stroke_label"] = df["stroke_label"].astype(str).str.title()
    labels = [f"{lbl}\n(n={int(n)})" for lbl, n in zip(df["stroke_label"], pd.to_numeric(df["landings"], errors="coerce").fillna(0))]
    x = np.arange(len(labels))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

    left_vals = pd.to_numeric(df["left_pct"], errors="coerce").fillna(0).to_numpy() * 100
    right_vals = pd.to_numeric(df["right_pct"], errors="coerce").fillna(0).to_numpy() * 100
    axes[0].bar(x - width / 2, left_vals, width, label="Left")
    axes[0].bar(x + width / 2, right_vals, width, label="Right")
    for xi, lv, rv in zip(x, left_vals, right_vals):
        axes[0].text(xi - width / 2, lv + 1, f"{lv:.0f}%", ha="center", va="bottom", fontsize=8)
        axes[0].text(xi + width / 2, rv + 1, f"{rv:.0f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Left / Right split")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percentage of landings")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper center")

    front_vals = pd.to_numeric(df["front_pct"], errors="coerce").fillna(0).to_numpy() * 100
    back_vals = pd.to_numeric(df["back_pct"], errors="coerce").fillna(0).to_numpy() * 100
    axes[1].bar(x - width / 2, front_vals, width, label="Front")
    axes[1].bar(x + width / 2, back_vals, width, label="Back")
    for xi, fv, bv in zip(x, front_vals, back_vals):
        axes[1].text(xi - width / 2, fv + 1, f"{fv:.0f}%", ha="center", va="bottom", fontsize=8)
        axes[1].text(xi + width / 2, bv + 1, f"{bv:.0f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Front / Back split")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper center")

    fig.suptitle(
        "NEAR Player: Landing Zone Distribution by Stroke\n"
        "Note: Left/Right and Front/Back are two separate 100% splits.",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_rally_length_chart(rallies: List[dict]):
    if not rallies:
        return _empty_chart("Rally Lengths", "No rally data to plot.")

    rally_ids = np.arange(1, len(rallies) + 1)
    lengths = np.asarray([int(r.get("shots", 0)) for r in rallies])
    reasons = [str(r.get("reason", "unknown")) for r in rallies]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(rally_ids, lengths)
    for x, y, reason in zip(rally_ids, lengths, reasons):
        ax.text(x, y + 0.1, reason, ha="center", va="bottom", fontsize=8, rotation=15)
    ax.set_xticks(rally_ids)
    ax.set_xlabel("Rally")
    ax.set_ylabel("Shots")
    ax.set_title("Rally Lengths")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(1, int(lengths.max())) + 2)
    fig.tight_layout()
    return fig


def generate_pdf_report(events: pd.DataFrame, rallies: List[dict], pdf_name: Optional[Path] = None):
    if pdf_name is None:
        pdf_name = REPORT_PDF
    pdf_name = Path(pdf_name)
    pdf_name.parent.mkdir(parents=True, exist_ok=True)

    charts_dir = CHARTS_DIR if "CHARTS_DIR" in globals() else pdf_name.parent / "analysis_charts"
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    near_stats = bounce_zone_stats(events, "near", only_opponent_half=True)
    far_stats = bounce_zone_stats(events, "far", only_opponent_half=True)
    near_stroke_summary = stroke_landing_summary(events, view_side="near", only_opponent_half=True)

    with PdfPages(str(pdf_name)) as pdf:
        # 1. Summary first, table-based to avoid clipped long lines.
        fig = plot_summary_dashboard(events, rallies, near_stroke_summary)
        _save_pdf_page(pdf, fig, charts_dir / "00_summary_dashboard.png")

        # 2. Original landing maps.
        if near_stats is not None:
            fig = plot_bounce_map_with_zones(near_stats, "near")
            _save_pdf_page(pdf, fig, charts_dir / "01_near_player_bounce_map.png")

        if far_stats is not None:
            fig = plot_bounce_map_with_zones(far_stats, "far")
            _save_pdf_page(pdf, fig, charts_dir / "02_far_player_bounce_map.png")

        # 3. Stroke-specific visuals.
        fig = plot_stroke_landing_map(events, view_side="near", only_opponent_half=True)
        _save_pdf_page(pdf, fig, charts_dir / "03_near_stroke_landing_map.png")

        fig = plot_stroke_in_out_chart(near_stroke_summary)
        _save_pdf_page(pdf, fig, charts_dir / "04_near_stroke_in_out.png")

        fig = plot_stroke_zone_chart(near_stroke_summary)
        _save_pdf_page(pdf, fig, charts_dir / "05_near_stroke_zone_distribution.png")

        # 4. Rally visual.
        fig = plot_rally_length_chart(rallies)
        _save_pdf_page(pdf, fig, charts_dir / "06_rally_lengths.png")

    print(f"PDF saved to {pdf_name}")
    print(f"Charts saved to {charts_dir}")


# ========================================================
# Main
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--bounce-csv", type=str, default=None)
    parser.add_argument("--hit-csv", type=str, default=None)
    parser.add_argument("--stroke-csv", type=str, default=None, help="vote_action.py 產生的 stroke_vote_events.csv；預設自動找 bounce_detector/stroke_vote_same_crop/stroke_vote_events.csv")
    parser.add_argument("--min-stroke-vote-count", type=int, default=1)
    parser.add_argument("--min-stroke-vote-ratio", type=float, default=0.0)
    parser.add_argument("--min-stroke-confidence", type=float, default=0.0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--bounce-coord-scale", type=float, default=1.0, help="若 bounce.py 用 --scale 0.5 跑，這裡填 2.0")
    args = parser.parse_args()

    set_video_path(Path(args.video_path))
    bounces = load_bounce_events(Path(args.bounce_csv) if args.bounce_csv else BOUNCE_EVENTS_CSV, coord_scale=args.bounce_coord_scale)
    hits = load_hit_intervals(Path(args.hit_csv) if args.hit_csv else HIT_INTERVALS_CSV, fallback_path=HIT_SEGMENTS_CSV)
    stroke_votes = load_stroke_votes(Path(args.stroke_csv) if args.stroke_csv else STROKE_VOTE_EVENTS_CSV, required=False)
    hits = attach_stroke_votes_to_hits(
        hits,
        stroke_votes,
        min_vote_count=args.min_stroke_vote_count,
        min_vote_ratio=args.min_stroke_vote_ratio,
        min_mean_prob=args.min_stroke_confidence,
    )
    events = build_event_dataframe(bounces, hits, ROI_JSON)

    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_CSV, index=False, encoding="utf-8")

    stroke_summary = stroke_landing_summary(events, view_side="near", only_opponent_half=True)
    stroke_summary.to_csv(STROKE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"Stroke landing summary saved to {STROKE_SUMMARY_CSV}")

    rallies = extract_rallies_true(events)
    generate_pdf_report(events, rallies, pdf_name=Path(args.out) if args.out else REPORT_PDF)


if __name__ == "__main__":
    main()
