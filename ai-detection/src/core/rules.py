from typing import Dict, Any, Optional
from src.config import CourtConfig
from src.pipeline import VideoContext
from src.core.homography import HomographyHandler
from src.core.court import TennisCourt
from src.detectors.base import BaseDetector

class InoutRuleEngine(BaseDetector):
    def __init__(self):
        self.config = CourtConfig()
        self.court_ref = TennisCourt()
        self.homography_handler = HomographyHandler(self.court_ref)

    def process(self, context: VideoContext) -> VideoContext:
        analysis = []
        
        # 1. Detect Net Hits (Scan entire trajectory)
        net_hits = self._detect_net_hits(context)
        for hit in net_hits:
            analysis.append({
                "frame": hit["frame"],
                "pos_2d": hit["pos_2d"],
                "status": "Out - Hit Net"
            })

        # 2. Analyze Bounces
        for frame_idx in context.bounces:
            # Skip if we already marked this area as a net hit
            if any(abs(frame_idx - h["frame"]) < 5 for h in net_hits):
                continue

            ball_pos = context.ball_track[frame_idx]
            matrix = context.homography_matrices[frame_idx]
            
            if ball_pos[0] is None or matrix is None:
                analysis.append({
                    "frame": frame_idx,
                    "pos_2d": None,
                    "status": "Unknown (Missing Tracking/Court)"
                })
                continue

            pos_2d = self.homography_handler.project_point(ball_pos, matrix)
            status = self._classify_point(pos_2d)
            
            analysis.append({
                "frame": frame_idx,
                "pos_2d": pos_2d,
                "status": status
            })
            
        # Sort analysis by frame
        context.bounce_analysis = sorted(analysis, key=lambda x: x["frame"])
        return context

    def _detect_net_hits(self, context: VideoContext):
        """
        Detects net hits using multi-signal confidence:
        1. Direction reversal (Bounce back)
        2. Drastic velocity drop (Trickle down)
        3. Tracking termination in net zone (Lost in net)
        """
        net_hits = []
        net_y = self.config.net_y
        # Net zone is ~1.5 meters around the net
        zone_epsilon = 1.5 
        
        # Get projected metrics
        projected = []
        for i in range(len(context.ball_track)):
            pos = context.ball_track[i]
            matrix = context.homography_matrices[i]
            if pos[0] is not None and matrix is not None:
                projected.append(self.homography_handler.project_point(pos, matrix))
            else:
                projected.append(None)

        for i in range(2, len(projected) - 2):
            p_prev = projected[i-1]
            p_curr = projected[i]
            p_next = projected[i+1]
            
            if p_curr is None or p_prev is None:
                continue
            
            # Check proximity to net
            dist_to_net = abs(p_curr[1] - net_y)
            if dist_to_net < zone_epsilon:
                v_in = abs(p_curr[1] - p_prev[1])
                
                # Signal 1: Tracking ends abruptly in the net zone
                if p_next is None and v_in > 0.1: # Was moving, then disappeared
                    # Check if it stays dead for a few frames
                    if all(p is None for p in projected[i+1 : i+5]):
                        net_hits.append({"frame": i, "pos_2d": p_curr, "type": "termination"})
                        continue

                if p_next is not None:
                    v_out = abs(p_next[1] - p_curr[1])
                    
                    # Signal 2: Direction Reversal (Sign change in Y velocity)
                    # (p_curr[1]-p_prev[1]) and (p_next[1]-p_curr[1]) have different signs
                    if (p_curr[1]-p_prev[1]) * (p_next[1]-p_curr[1]) < 0:
                        if v_in > 0.05: # Ignore micro-jitters
                            net_hits.append({"frame": i, "pos_2d": p_curr, "type": "reversal"})
                            continue
                    
                    # Signal 3: Drastic Deceleration (The "Trickle")
                    if v_in > 0.15 and v_out < (v_in * 0.2):
                        net_hits.append({"frame": i, "pos_2d": p_curr, "type": "deceleration"})

        # Filter duplicates (cluster events within 15 frames)
        if not net_hits: return []
        net_hits.sort(key=lambda x: x["frame"])
        filtered = [net_hits[0]]
        for i in range(1, len(net_hits)):
            if net_hits[i]["frame"] - filtered[-1]["frame"] > 15:
                filtered.append(net_hits[i])
        return filtered

    def _classify_point(self, pos_2d) -> str:
        if pos_2d is None:
            return "Unknown"
        
        x, y = pos_2d
        
        # Check Long (y-axis)
        if y < self.config.y_min:
            return "Out - Long (Far)"
        if y > self.config.y_max:
            return "Out - Long (Near)"
            
        # Check Wide (x-axis)
        # Assuming doubles lines are the outer boundary
        if x < self.config.x_min or x > self.config.x_max:
            return "Out - Wide"
            
        # Optional: Check if it's within singles lines
        if x < self.config.singles_x_min or x > self.config.singles_x_max:
            return "In (Doubles Alley)"
            
        return "In"
