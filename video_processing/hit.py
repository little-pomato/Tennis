from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _best_candidate_for_y_signal(cands: List[BlobCandidate]) -> Optional[BlobCandidate]:
    """
    每幀挑一顆最像球的 candidate，建立 y signal 用。
    這裡不要求一定是 bounce，只是要找整體球軌跡的 y 最大 / 最小。
    """
    if not cands:
        return None

    def score(c: BlobCandidate) -> float:
        return float(
            0.45 * c.score_event
            + 0.30 * c.score_blob
            + 0.15 * c.track_score_1
            + 0.10 * c.track_score_2
        )

    return max(cands, key=score)


def build_ball_y_signal_from_candidates(
    candidates_by_frame: List[List[BlobCandidate]],
    max_interp_gap: int = 12,
    smooth_win: int = 5,
) -> pd.DataFrame:
    """
    從 candidates_by_frame 建立每幀球的 y 軌跡。
    有 candidate 就用真實 candidate；
    沒 candidate 就先留 NaN，再用 interpolate 補短 gap。
    """
    rows = []

    for i, cands in enumerate(candidates_by_frame):
        best = _best_candidate_for_y_signal(cands)

        if best is None:
            rows.append({
                "frame_idx": i,
                "x": np.nan,
                "y": np.nan,
                "has_candidate": 0,
                "candidate_score": 0.0,
                "court_side": "missing",
                "n_candidates": len(cands),
            })
        else:
            candidate_score = float(
                0.45 * best.score_event
                + 0.30 * best.score_blob
                + 0.15 * best.track_score_1
                + 0.10 * best.track_score_2
            )

            rows.append({
                "frame_idx": i,
                "x": float(best.cx),
                "y": float(best.cy),
                "has_candidate": 1,
                "candidate_score": candidate_score,
                "court_side": str(best.court_side),
                "n_candidates": len(cands),
            })

    df = pd.DataFrame(rows)

    # 補短 gap：遠端漏 10 幀左右，所以先允許補到 12 幀
    df["x_filled"] = df["x"].interpolate(
        limit=max_interp_gap,
        limit_direction="both"
    )
    df["y_filled"] = df["y"].interpolate(
        limit=max_interp_gap,
        limit_direction="both"
    )

    # Smooth y — Savitzky-Golay preserves local extrema (ball apex/bounce minimum)
    # better than a rolling mean, which flattens and shifts peaks.
    filled = df["y_filled"].values.copy()
    n = len(filled)
    win = smooth_win if smooth_win % 2 == 1 else smooth_win + 1
    win = max(win, 3)
    if n >= win:
        df["y_smooth"] = savgol_filter(filled, window_length=win, polyorder=2, mode="nearest")
    else:
        df["y_smooth"] = (
            df["y_filled"]
            .rolling(window=smooth_win, center=True, min_periods=1)
            .mean()
        )

    return df


def _segments_from_bool_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    把 True/False mask 轉成連續區間。
    回傳 [(start_idx, end_idx), ...]
    """
    segments = []
    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i

        segments.append((start, end))
        i += 1

    return segments


def detect_hit_intervals_by_clear_high_low_segments(
    ball_y_df: pd.DataFrame,
    net_y: float,
    y_col: str = "y_smooth",
    margin: float = 18.0,
    min_segment_len: int = 4,
    max_merge_gap: int = 3,
    pad: int = 2,
    shape_window: int = 28,
    min_shape_change: float = 18.0,
) -> pd.DataFrame:
    """
    用「圖上看得到的高點區間 / 低點區間」找擊球區間。

    核心規則：
    1. y > net_y + margin 形成 near 高點區間，該區間取 y 最大。
    2. y < net_y - margin 形成 far 低點區間，該區間取 y 最小。
    3. 連續同側事件只保留一個：near 留最高、far 留最低。
       例如 near 58、near 70、far 89 -> 只保留 near 58、far 89。
    4. near 必須像完整山峰：高點後面要有明顯下降。
       far 必須像完整谷底：低點後面要有明顯回升。
       因此影片尾端只有往下掉、沒有回升的 far 會被刪掉。

    注意：這個函式輸出的 center_frame 是該區間的代表點；
    真正建議使用的是 interval_start_frame ~ interval_end_frame。
    """
    if net_y is None or not np.isfinite(net_y):
        return pd.DataFrame()

    df = ball_y_df.copy().reset_index(drop=True)
    if y_col not in df.columns:
        y_col = "y_smooth"

    y = df[y_col].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    valid = np.isfinite(y)
    near_mask = valid & (y > float(net_y) + float(margin))
    far_mask = valid & (y < float(net_y) - float(margin))

    def merge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        """把同一個高/低區間中間很短的 False gap 補起來。"""
        mask = mask.copy()
        m = len(mask)
        i = 0
        while i < m:
            if mask[i]:
                i += 1
                continue

            start = i
            while i < m and not mask[i]:
                i += 1
            end = i - 1

            gap_len = end - start + 1
            left_true = start - 1 >= 0 and mask[start - 1]
            right_true = end + 1 < m and mask[end + 1]

            if left_true and right_true and gap_len <= max_gap:
                mask[start:end + 1] = True

        return mask

    near_mask = merge_small_gaps(near_mask, int(max_merge_gap))
    far_mask = merge_small_gaps(far_mask, int(max_merge_gap))

    events = []

    def append_event(a: int, b: int, side: str):
        if b - a + 1 < int(min_segment_len):
            return

        seg_len = int(b - a + 1)

        interval_start_idx = max(0, a - int(pad))
        interval_end_idx = min(n - 1, b + int(pad))

        if "has_candidate" in df.columns:
            has_candidate_ratio = float(
                df.loc[interval_start_idx:interval_end_idx, "has_candidate"].mean()
            )
        else:
            has_candidate_ratio = 1.0

        # 先照原本邏輯找 extreme point
        # near: 最高點；far: 最低點
        if side == "near":
            extreme_idx = a + int(np.nanargmax(y[a:b + 1]))
            event_type = "clear_high_segment"
        else:
            extreme_idx = a + int(np.nanargmin(y[a:b + 1]))
            event_type = "clear_low_segment"

        def calc_swing(k: int) -> tuple[float, float]:
            """
            計算某個 center 左右兩側的形狀變化量。
            near: 看山峰左右是否有下降。
            far: 看谷底左右是否有回升。
            """
            left_start = max(0, k - int(shape_window))
            right_end = min(n - 1, k + int(shape_window))

            left = y[left_start:k]
            right = y[k + 1:right_end + 1]

            left_valid = left[np.isfinite(left)]
            right_valid = right[np.isfinite(right)]

            if not np.isfinite(y[k]):
                return np.nan, np.nan

            center_y_tmp = float(y[k])

            if side == "near":
                left_s = (
                    center_y_tmp - float(np.nanmin(left_valid))
                    if len(left_valid)
                    else np.nan
                )
                right_s = (
                    center_y_tmp - float(np.nanmin(right_valid))
                    if len(right_valid)
                    else np.nan
                )
            else:
                left_s = (
                    float(np.nanmax(left_valid)) - center_y_tmp
                    if len(left_valid)
                    else np.nan
                )
                right_s = (
                    float(np.nanmax(right_valid)) - center_y_tmp
                    if len(right_valid)
                    else np.nan
                )

            return left_s, right_s

        extreme_left_swing, extreme_right_swing = calc_swing(extreme_idx)

        center_idx = extreme_idx
        center_method = "extreme"

        # --------------------------------------------------------
        # 關鍵修正：
        # far segment 如果很長、candidate ratio 偏低，
        # 而且 extreme point 左側 swing 很小、右側 swing 很大，
        # 代表這個最低點很可能是漏幀 / 錯誤候選拖出來的假中心。
        #
        # 像 169~211：
        # segment_len = 39
        # has_candidate_ratio 約 0.63
        # left_swing 約 25.8
        # right_swing 約 224
        # 就會進入這個 fallback。
        # --------------------------------------------------------
        if side == "far":
            right_big = (
                np.isfinite(extreme_right_swing)
                and extreme_right_swing >= 80.0
            )

            left_too_small = (
                not np.isfinite(extreme_left_swing)
                or extreme_left_swing < max(45.0, 0.35 * extreme_right_swing)
            )

            long_sparse_segment = (
                seg_len >= 34
                and has_candidate_ratio < 0.75
            )

            if long_sparse_segment and right_big and left_too_small:
                # 用 segment 中點附近重新選 center
                # 不直接拿最低點，避免被漏幀後的錯誤 y_min 拖到太後面
                target_idx = int(round((a + b) / 2.0))

                # 在中點附近找有真實 candidate 的 frame
                # 如果 190 沒有 candidate，但 191 有 candidate，就會偏向 191
                search_radius = 3
                lo = max(a, target_idx - search_radius)
                hi = min(b, target_idx + search_radius)

                candidate_indices = []
                for k in range(lo, hi + 1):
                    if not np.isfinite(y[k]):
                        continue

                    if "has_candidate" in df.columns:
                        if int(df.loc[k, "has_candidate"]) != 1:
                            continue

                    candidate_indices.append(k)

                if candidate_indices:
                    def midpoint_score(k: int) -> float:
                        dist = abs(k - target_idx)
                        dist_score = 1.0 - dist / max(1.0, float(search_radius))

                        if "candidate_score" in df.columns and pd.notna(df.loc[k, "candidate_score"]):
                            cand_score = float(df.loc[k, "candidate_score"])
                            cand_score = float(np.clip(cand_score, 0.0, 1.0))
                        else:
                            cand_score = 0.0

                        return 0.75 * dist_score + 0.25 * cand_score

                    center_idx = max(candidate_indices, key=midpoint_score)
                else:
                    center_idx = target_idx

                center_method = "midpoint_fallback_sparse_far"

        center_y = float(y[center_idx])

        if side == "near":
            strength = float(center_y - float(net_y))
        else:
            strength = float(float(net_y) - center_y)

        left_swing, right_swing = calc_swing(center_idx)

        events.append({
            "interval_start_frame": int(df.loc[interval_start_idx, "frame_idx"]),
            "interval_end_frame": int(df.loc[interval_end_idx, "frame_idx"]),
            "center_frame": int(df.loc[center_idx, "frame_idx"]),
            "center_y": center_y,
            "hit_side": side,
            "event_type": event_type,
            "has_candidate_ratio": float(has_candidate_ratio),
            "segment_start_frame": int(df.loc[a, "frame_idx"]),
            "segment_end_frame": int(df.loc[b, "frame_idx"]),
            "segment_len": int(seg_len),
            "segment_strength": float(strength),
            "left_swing": float(left_swing) if np.isfinite(left_swing) else np.nan,
            "right_swing": float(right_swing) if np.isfinite(right_swing) else np.nan,

            # debug 欄位：之後你可以看這段到底有沒有啟動 fallback
            "extreme_frame": int(df.loc[extreme_idx, "frame_idx"]),
            "extreme_y": float(y[extreme_idx]),
            "extreme_left_swing": float(extreme_left_swing) if np.isfinite(extreme_left_swing) else np.nan,
            "extreme_right_swing": float(extreme_right_swing) if np.isfinite(extreme_right_swing) else np.nan,
            "center_method": center_method,
        })

    for a, b in _segments_from_bool_mask(near_mask):
        append_event(a, b, "near")

    for a, b in _segments_from_bool_mask(far_mask):
        append_event(a, b, "far")

    if not events:
        return pd.DataFrame()

    hit_df = pd.DataFrame(events).sort_values("center_frame").reset_index(drop=True)

    # 連續同側只留最極端的一個。
    cleaned = []
    group = []

    def flush_group(rows: List[dict]) -> Optional[dict]:
        if not rows:
            return None

        side = str(rows[0]["hit_side"])

        # 注意：
        # center_y 現在可能被 fallback 改成 midpoint 的 y，
        # 所以清理連續同側事件時，不能再用 center_y 判斷誰比較極端。
        # 要改用 extreme_y，這樣才不會破壞原本「同側只留最極端 segment」的邏輯。
        if side == "near":
            return max(rows, key=lambda r: float(r.get("extreme_y", r["center_y"])))

        return min(rows, key=lambda r: float(r.get("extreme_y", r["center_y"])))

    for _, row in hit_df.iterrows():
        r = row.to_dict()
        if not group:
            group.append(r)
            continue

        if str(r["hit_side"]) == str(group[-1]["hit_side"]):
            group.append(r)
        else:
            best = flush_group(group)
            if best is not None:
                cleaned.append(best)
            group = [r]

    best = flush_group(group)
    if best is not None:
        cleaned.append(best)

    hit_df = pd.DataFrame(cleaned).sort_values("center_frame").reset_index(drop=True)

    # 過濾不完整山峰 / 谷底。
    # 尾端 far 若沒有後續回升，right_swing 會太小或 NaN，因此會被刪掉。
    kept = []
    for idx, row in hit_df.iterrows():
        side = str(row["hit_side"])
        left_swing = float(row["left_swing"]) if pd.notna(row["left_swing"]) else np.nan
        right_swing = float(row["right_swing"]) if pd.notna(row["right_swing"]) else np.nan

        has_left_shape = np.isfinite(left_swing) and left_swing >= float(min_shape_change)
        has_right_shape = np.isfinite(right_swing) and right_swing >= float(min_shape_change)

        # 開頭第一個 near 可能沒有完整左側，但只要後面明顯掉下去，仍保留。
        is_first_event = (idx == 0)
        if side == "near":
            keep = has_right_shape and (has_left_shape or is_first_event)
        else:
            # far 一定要有右側回升；沒有回升就不是完整谷底。
            keep = has_left_shape and has_right_shape

        if keep:
            kept.append(row.to_dict())

    if not kept:
        return pd.DataFrame()

    return pd.DataFrame(kept).sort_values("center_frame").reset_index(drop=True)
