import cv2
import numpy as np
from typing import List
from tqdm import tqdm
from src.pipeline import VideoContext
from src.core.court import TennisCourt
from src.core.homography import HomographyHandler

class VideoExporter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.court_ref = TennisCourt()
        self.homography_handler = HomographyHandler(self.court_ref)
        
        # Minimap style settings
        self.m_scale = 15.0 # pixels per meter
        self.m_margin = 30 # pixels
        self.width_minimap = int(self.court_ref.config.court_width * self.m_scale + 2 * self.m_margin)
        self.height_minimap = int(self.court_ref.config.court_length * self.m_scale + 2 * self.m_margin)

    def export(self, context: VideoContext):
        if not context.frame_paths:
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, context.fps, (context.width, context.height))
        
        # Build base minimap
        base_minimap = self.court_ref.get_court_image(scale=self.m_scale, margin=self.m_margin)
        
        num_frames = len(context.frame_paths)
        for i in tqdm(range(num_frames), desc="Exporting video"):
            frame = context.get_frame(i)
            annotated = frame.copy()
            minimap = base_minimap.copy()
            
            # --- TRAJECTORY WITH HITTER COLORING ---
            for j in range(1, i + 1):
                p1 = context.ball_track[j-1]
                p2 = context.ball_track[j]
                if p1[0] is not None and p2[0] is not None:
                    # Determine color based on who hit this segment
                    color = (255, 255, 255) # Default: White
                    for speed_evt in context.analytics_data["ball_speeds"]:
                        if speed_evt["start"] <= j <= speed_evt["end"]:
                            if speed_evt.get("side") == "top":
                                color = (255, 0, 255) # Magenta
                            elif speed_evt.get("side") == "bottom":
                                color = (255, 255, 0) # Cyan
                            break
                    
                    alpha = max(0.1, 1.0 - (i - j) / 40.0)
                    cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2, cv2.LINE_AA)
            # --- END TRAJECTORY ---

            # 1. Draw Ball Current Pos
            pos = context.ball_track[i]
            if pos[0] is not None:
                cv2.circle(annotated, (int(pos[0]), int(pos[1])), 8, (0, 255, 255), -1, cv2.LINE_AA)
            
            # 2. Draw Court Keypoints
            if context.court_keypoints[i] is not None:
                for kp in context.court_keypoints[i]:
                    cv2.circle(annotated, (int(kp[0, 0]), int(kp[0, 1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
            
            # 3. Draw Players and update Minimap
            if i < len(context.players):
                player_data = context.players[i]
                matrix_img_to_court = context.homography_matrices[i]
                for side in ["top", "bottom"]:
                    color = (255, 0, 0) if side == "top" else (0, 0, 255)
                    for bbox in player_data[side]:
                        cv2.rectangle(annotated, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2, cv2.LINE_AA)
                        
                        # Project foot point to metric minimap
                        if matrix_img_to_court is not None:
                            foot_point = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                            pos_m = self.homography_handler.project_point(foot_point, matrix_img_to_court)
                            if pos_m:
                                px, py = self.court_ref.meter_to_px(pos_m[0], pos_m[1], self.m_scale, self.m_margin)
                                # Draw player dot
                                cv2.circle(minimap, (px, py), 6, color, -1, cv2.LINE_AA)
                                cv2.circle(minimap, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 4. Draw Hits and Bounces
            current_speed = None
            for speed_evt in context.analytics_data["ball_speeds"]:
                if speed_evt["start"] <= i <= speed_evt["end"]:
                    current_speed = speed_evt["speed_kmh"]
                    break
                    
                if speed_evt["start"] == i:
                    ball_p = context.ball_track[i]
                    if ball_p[0]:
                        cv2.circle(annotated, (int(ball_p[0]), int(ball_p[1])), 25, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(annotated, "HIT!", (int(ball_p[0]) - 20, int(ball_p[1]) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

            for bounce in context.bounce_analysis:
                if bounce["frame"] <= i:
                    if bounce["frame"] == i:
                        ball_pos = context.ball_track[i]
                        if ball_pos[0] is not None:
                            s_color = (0, 255, 255) if "In" in bounce["status"] else (0, 0, 255)
                            cv2.putText(annotated, bounce["status"], (int(ball_pos[0]), int(ball_pos[1]) - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2, cv2.LINE_AA)
                    
                    if bounce["pos_2d"]:
                        is_in = "In" in bounce["status"]
                        b_color = (0, 255, 255) if is_in else (0, 69, 255)
                        px, py = self.court_ref.meter_to_px(bounce["pos_2d"][0], bounce["pos_2d"][1], self.m_scale, self.m_margin)
                        
                        is_current = (bounce["frame"] == i)
                        radius = 5 if is_current else 3
                        cv2.circle(minimap, (px, py), radius, b_color, -1, cv2.LINE_AA)
                        if is_current:
                            cv2.circle(minimap, (px, py), radius + 4, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Overlay Minimap (Dynamic Scaling)
            m_h = int(context.height * 0.4) 
            m_w = int(m_h * (self.width_minimap / self.height_minimap))
            
            h_start, h_end = 30, 30 + m_h
            w_start, w_end = context.width - 30 - m_w, context.width - 30
            
            if h_end <= context.height and w_end <= context.width:
                minimap_resized = cv2.resize(minimap, (w_end - w_start, h_end - h_start))
                cv2.rectangle(minimap_resized, (0, 0), (minimap_resized.shape[1]-1, minimap_resized.shape[0]-1), (255, 255, 255), 1)
                annotated[h_start:h_end, w_start:w_end] = minimap_resized
            
            # 6. Draw Analytics Overlays
            if current_speed:
                speed_text = f"{int(current_speed)} km/h"
                cv2.putText(annotated, speed_text, (30, context.height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

            # Draw simple FH/BH scoreboards for each side
            for side, pos_y in [("top", 60), ("bottom", context.height - 100)]:
                stats = context.analytics_data["player_stats"][side]
                stat_text = f"{side.upper()}: FH {stats['forehands']} | BH {stats['backhands']}"
                cv2.putText(annotated, stat_text, (30, pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(annotated)
            
        out.release()
