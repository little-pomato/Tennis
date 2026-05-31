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
            self.stroke_detector = StrokeDetector(stroke_model_path, device=device)

    def process(self, context: VideoContext) -> VideoContext:
        print("Analyzing match statistics (Robust Speed Engine)...")
        
        context.analytics_data["ball_speeds"] = []
        context.analytics_data["player_stats"] = {
            "top": {"forehands": 0, "backhands": 0, "drops": []},
            "bottom": {"forehands": 0, "backhands": 0, "drops": []}
        }

        # 1. Project full trajectory to metric space
        metric_track = []
        for i in range(len(context.ball_track)):
            pos = context.ball_track[i]
            matrix = context.homography_matrices[i]
            if pos[0] is not None and matrix is not None:
                metric_track.append(self.homography_handler.project_point(pos, matrix))
            else:
                metric_track.append(None)

        # 2. Identify all Hit Events
        anchors = sorted(context.bounce_analysis, key=lambda x: x["frame"])
        raw_hits = []
        for anchor in anchors:
            hit = self._find_preceding_hit(anchor, metric_track, context)
            if hit: raw_hits.append(hit)
        
        for i in range(5, len(metric_track) - 5):
            if any(abs(i - h["frame"]) < 12 for h in raw_hits): continue
            hit = self._scan_standalone_hit(i, metric_track, context)
            if hit: raw_hits.append(hit)
        
        raw_hits.sort(key=lambda x: x["frame"])
        
        # 3. Robust Speed Calculation: Average over the whole stroke
        for i, hit in enumerate(raw_hits):
            next_hit_frame = raw_hits[i+1]["frame"] if i + 1 < len(raw_hits) else len(metric_track)
            
            stroke_end = next_hit_frame
            next_anchor = None
            for anchor in anchors:
                if hit["frame"] < anchor["frame"] < next_hit_frame:
                    stroke_end = anchor["frame"]
                    next_anchor = anchor
                    break
            
            stroke_len = stroke_end - hit["frame"]
            if stroke_len > 10:
                trim = max(2, int(stroke_len * 0.15))
                s_idx = hit["frame"] + trim
                e_idx = stroke_end - trim
                
                segment = [p for p in metric_track[s_idx:e_idx] if p is not None]
                if len(segment) > 5:
                    speeds = []
                    for k in range(1, len(segment)):
                        d_m = np.sqrt((segment[k][0]-segment[k-1][0])**2 + (segment[k][1]-segment[k-1][1])**2)
                        speeds.append(d_m * context.fps * 3.6)
                    
                    avg_speed_kmh = float(np.median(speeds))
                    if 30 < avg_speed_kmh < 250:
                        hit["speed_kmh"] = avg_speed_kmh

            self._update_stats(hit, next_anchor, context)

        return context

    def _scan_standalone_hit(self, i, metric_track, context):
        """Checks frame i for a hit event using vector deviation and court boundaries."""
        p_curr = metric_track[i]
        p_prev = metric_track[i-1]
        p_next = metric_track[i+1]
        if not (p_curr and p_prev and p_next): return None
        
        court_w = self.court_ref.config.court_width
        court_l = self.court_ref.config.court_length
        if not (-1.5 < p_curr[0] < court_w + 1.5 and -2.0 < p_curr[1] < court_l + 2.0):
            return None

        v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        
        mag_in = np.linalg.norm(v_in)
        mag_out = np.linalg.norm(v_out)
        if mag_in < 0.02 or mag_out < 0.02: return None
        
        cos_theta = np.dot(v_in, v_out) / (mag_in * mag_out)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))
        
        is_kinematic_hit = (angle_deg > 30) or (np.dot(v_in, v_out) < 0 and abs(v_in[1]) > 0.05)
        
        if is_kinematic_hit:
            side, player_m = self._find_nearby_player(i, p_curr, context, window=3, max_dist=3.0)
            if side:
                speed_kmh = (mag_out * context.fps) * 3.6
                return {
                    "frame": i, "pos_m": p_curr, "player_m": player_m,
                    "side": side, "speed_kmh": speed_kmh, "confidence": angle_deg
                }
        return None

    def _find_nearby_player(self, frame_idx, ball_m, context, window=5, max_dist=4.0):
        """Finds a player near the ball within a temporal window."""
        for win_idx in range(max(0, frame_idx - window), min(len(context.players), frame_idx + window)):
            for side in ["top", "bottom"]:
                if context.players[win_idx][side]:
                    bbox = context.players[win_idx][side][0]
                    matrix = context.homography_matrices[win_idx]
                    p_foot = self.homography_handler.project_point(((bbox[0]+bbox[2])/2, bbox[3]), matrix)
                    if p_foot:
                        dist = np.sqrt((ball_m[0]-p_foot[0])**2 + (ball_m[1]-p_foot[1])**2)
                        if dist < max_dist:
                            return side, p_foot
        return None, None

    def _find_preceding_hit(self, anchor, metric_track, context, max_lookback=60):
        """Searches backward from a bounce/event using vector analysis."""
        end_f = anchor["frame"]
        start_f = max(2, end_f - max_lookback)
        
        for i in range(end_f - 4, start_f, -1):
            hit = self._scan_standalone_hit(i, metric_track, context)
            if hit:
                return hit
        return None

    def _update_stats(self, hit, anchor, context):
        side = hit["side"]
        frame_idx = hit["frame"]
        stroke = "unknown"
        
        if self.stroke_detector and context.players[frame_idx][side]:
            frame = context.get_frame(frame_idx)
            bbox = context.players[frame_idx][side][0]
            stroke = self.stroke_detector.classify_stroke(frame, bbox)
            if stroke not in ["forehand", "backhand"]:
                stroke = self._geometric_classification(hit)
        else:
            stroke = self._geometric_classification(hit)

        # Explicit Swap for Bottom Player (User Request)
        if side == "bottom":
            if stroke == "forehand": stroke = "backhand"
            elif stroke == "backhand": stroke = "forehand"
            
        context.analytics_data["player_stats"][side][f"{stroke}s"] += 1
        
        context.analytics_data["ball_speeds"].append({
            "start": hit["frame"],
            "end": anchor["frame"] if anchor else hit["frame"] + 30,
            "speed_kmh": hit.get("speed_kmh", 0),
            "side": side
        })
        
        if anchor and anchor.get("pos_2d"):
            context.analytics_data["player_stats"][side]["drops"].append(anchor["pos_2d"])

    def _geometric_classification(self, hit) -> str:
        side = hit["side"]
        is_right_of_player = hit["pos_m"][0] > hit["player_m"][0]
        if side == "bottom":
            return "forehand" if is_right_of_player else "backhand"
        else:
            return "forehand" if not is_right_of_player else "backhand"
