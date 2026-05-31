import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.homography import HomographyHandler
from src.core.court import TennisCourt
from src.detectors.stroke import StrokeDetector


class MatchAnalyzer(BaseDetector):
    def __init__(self, stroke_model_path: Optional[str] = None, device: str = 'cpu'):
        self.court_ref = TennisCourt()
        self.homography_handler = HomographyHandler(self.court_ref)
        self.stroke_detector = None
        if stroke_model_path:
            try:
                self.stroke_detector = StrokeDetector(stroke_model_path, device=device)
            except Exception as e:
                print(f"[MatchAnalyzer] Warning: stroke model failed to load ({e}). Falling back to geometric classification.")

    def process(self, context: VideoContext) -> VideoContext:
        print("[MatchAnalyzer] Starting analysis...")

        context.analytics_data["ball_speeds"] = []
        context.analytics_data["player_stats"] = {
            "top":    {"forehands": 0, "backhands": 0, "drops": []},
            "bottom": {"forehands": 0, "backhands": 0, "drops": []},
        }

        net_y = self.court_ref.config.net_y

        # --- 1. Project ball track to metric space ---
        metric_track = []
        for pos, matrix in zip(context.ball_track, context.homography_matrices):
            if pos[0] is not None and matrix is not None:
                metric_track.append(self.homography_handler.project_point(pos, matrix))
            else:
                metric_track.append(None)

        # --- 2. Build sorted anchor list (bounces with known 2D positions) ---
        anchors = sorted(
            [a for a in context.bounce_analysis if a.get("pos_2d") is not None],
            key=lambda x: x["frame"]
        )
        if not anchors:
            print("[MatchAnalyzer] No anchors with pos_2d — skipping analytics.")
            return context

        # --- 3. Bounce frames to skip when searching for hits ---
        bounce_frame_set = set(context.bounces)

        # --- 4. For each bounce, determine hitter from PHYSICS then find hit frame ---
        for idx, anchor in enumerate(anchors):
            bounce_y = anchor["pos_2d"][1]

            # ---------------------------------------------------------------
            # PHYSICS TRUTH: ball always crosses the net.
            # bounce on top-half  (y < net_y)  → hitter is the BOTTOM player
            # bounce on bot-half  (y > net_y)  → hitter is the TOP player
            # ---------------------------------------------------------------
            side = "bottom" if bounce_y < net_y else "top"

            # --- Find hit frame (for speed calc & stroke classification) ---
            prev_anchor_frame = anchors[idx - 1]["frame"] if idx > 0 else 0
            hit_frame = self._find_hit_frame(
                anchor["frame"], prev_anchor_frame, metric_track, bounce_frame_set
            )
            if hit_frame is None:
                # Fall back to midpoint between previous bounce and this one
                hit_frame = max(prev_anchor_frame + 2, anchor["frame"] - 20)

            # --- Speed over the hit→bounce segment ---
            speed_kmh = self._calc_segment_speed(hit_frame, anchor["frame"], metric_track, context.fps)

            # --- Stroke classification (forehand / backhand) ---
            stroke = self._classify_stroke(hit_frame, side, metric_track, context)

            # --- Record results ---
            context.analytics_data["ball_speeds"].append({
                "start": hit_frame,
                "end": anchor["frame"],
                "speed_kmh": speed_kmh,
                "side": side,
            })
            context.analytics_data["player_stats"][side][f"{stroke}s"] += 1
            context.analytics_data["player_stats"][side]["drops"].append(anchor["pos_2d"])

        total = sum(
            v for side in ("top", "bottom")
            for k, v in context.analytics_data["player_stats"][side].items()
            if k in ("forehands", "backhands")
        )
        print(f"[MatchAnalyzer] Done. {len(anchors)} bounces → {len(context.analytics_data['ball_speeds'])} events, {total} strokes.")
        return context

    # -----------------------------------------------------------------------
    # Hit frame search
    # -----------------------------------------------------------------------

    def _find_hit_frame(
        self,
        bounce_frame: int,
        prev_bounce_frame: int,
        metric_track: list,
        bounce_frame_set: set,
        max_lookback: int = 60,
    ) -> Optional[int]:
        """
        Scans backward from bounce_frame looking for a kinematic direction change.
        Skips frames that are themselves bounces to avoid confusing ground-bounces
        with player hits.
        Returns the frame index of the detected hit, or None.
        """
        start = max(prev_bounce_frame + 2, bounce_frame - max_lookback)

        for i in range(bounce_frame - 4, start, -1):
            if i in bounce_frame_set:
                continue

            p0 = metric_track[i - 1] if i - 1 >= 0 else None
            p1 = metric_track[i]
            p2 = metric_track[i + 1] if i + 1 < len(metric_track) else None

            if not (p0 and p1 and p2):
                continue

            v_in  = np.array([p1[0] - p0[0], p1[1] - p0[1]])
            v_out = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            mag_in  = float(np.linalg.norm(v_in))
            mag_out = float(np.linalg.norm(v_out))

            if mag_in < 0.02 or mag_out < 0.02:
                continue

            cos_t = float(np.clip(np.dot(v_in, v_out) / (mag_in * mag_out), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_t)))

            is_hit = angle > 30 or (np.dot(v_in, v_out) < 0 and abs(v_in[1]) > 0.05)
            if is_hit:
                return i

        return None

    # -----------------------------------------------------------------------
    # Speed calculation
    # -----------------------------------------------------------------------

    def _calc_segment_speed(
        self,
        start_frame: int,
        end_frame: int,
        metric_track: list,
        fps: float,
    ) -> float:
        length = end_frame - start_frame
        if length < 5:
            return 0.0

        trim = max(1, int(length * 0.1))
        segment = [p for p in metric_track[start_frame + trim : end_frame - trim] if p is not None]
        if len(segment) < 3:
            return 0.0

        speeds = []
        for k in range(1, len(segment)):
            dx = segment[k][0] - segment[k - 1][0]
            dy = segment[k][1] - segment[k - 1][1]
            speeds.append(np.sqrt(dx * dx + dy * dy) * fps * 3.6)

        med = float(np.median(speeds))
        return med if 20 < med < 280 else 0.0

    # -----------------------------------------------------------------------
    # Stroke classification
    # -----------------------------------------------------------------------

    def _classify_stroke(
        self,
        hit_frame: int,
        side: str,
        metric_track: list,
        context: VideoContext,
    ) -> str:
        # Try ML model first
        if self.stroke_detector and 0 <= hit_frame < len(context.players):
            bboxes = context.players[hit_frame].get(side, [])
            if bboxes:
                try:
                    frame_img = context.get_frame(hit_frame)
                    stroke = self.stroke_detector.classify_stroke(frame_img, bboxes[0])
                    if stroke in ("forehand", "backhand"):
                        # Bottom player faces opposite direction → swap labels
                        if side == "bottom":
                            stroke = "backhand" if stroke == "forehand" else "forehand"
                        return stroke
                except Exception:
                    pass

        # Fallback: geometric (ball position relative to player)
        ball_m = metric_track[hit_frame] if hit_frame < len(metric_track) else None
        if ball_m is None:
            return "forehand"

        player_m = self._get_player_pos_m(hit_frame, side, context)
        if player_m is None:
            return "forehand"

        is_right = ball_m[0] > player_m[0]
        if side == "top":
            stroke = "forehand" if not is_right else "backhand"
        else:
            # Bottom player faces upward, so left/right is mirrored
            stroke = "forehand" if is_right else "backhand"
        return stroke

    def _get_player_pos_m(self, frame_idx: int, side: str, context: VideoContext) -> Optional[tuple]:
        """Returns player foot position in metric space, searching ±5 frames."""
        for win in range(max(0, frame_idx - 5), min(len(context.players), frame_idx + 6)):
            bboxes = context.players[win].get(side, [])
            if not bboxes:
                continue
            matrix = context.homography_matrices[win]
            if matrix is None:
                continue
            bbox = bboxes[0]
            foot = ((bbox[0] + bbox[2]) / 2, bbox[3])
            pos_m = self.homography_handler.project_point(foot, matrix)
            if pos_m:
                return pos_m
        return None
