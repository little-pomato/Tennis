from ultralytics import YOLO
import cv2
import torch
import numpy as np
from tqdm import tqdm
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.court import TennisCourt

class PlayerDetector(BaseDetector):
    def __init__(self, model_path: str = "yolov8n.pt", device: str = 'cpu'):
        self.device = device
        # Use YOLOv8 Nano for maximum speed
        self.model = YOLO(model_path)
        self.imgsz = 640
        self.conf = 0.3
        
        self.court_ref = TennisCourt()

    def process(self, context: VideoContext) -> VideoContext:
        players_per_frame = []
        num_frames = len(context.frame_paths)
        
        # Batch size for YOLO
        batch_size = 16 if ("cuda" in self.device or "mps" in self.device) else 4
        
        print(f"Tracking players (YOLO batch_size={batch_size}, device={self.device})...")
        
        for i in tqdm(range(0, num_frames, batch_size), desc="Tracking players"):
            end_idx = min(i + batch_size, num_frames)
            batch_frames = []
            
            # Load frames for the batch
            for idx in range(i, end_idx):
                batch_frames.append(context.get_frame(idx))
            
            # Run YOLO batch inference
            results = self.model.predict(
                batch_frames,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                verbose=False,
                classes=[0] # 0 is 'person' in COCO
            )
            
            # Post-process results
            for j, res in enumerate(results):
                frame_idx = i + j
                frame = batch_frames[j]
                
                if context.homography_matrices[frame_idx] is not None:
                    inv_matrix = context.homography_matrices[frame_idx]
                    top, bottom = self.filter_players(res, inv_matrix)
                    players_per_frame.append({"top": top, "bottom": bottom})
                else:
                    players_per_frame.append({"top": [], "bottom": []})
        
        context.players = players_per_frame
        return context

    def filter_players(self, yolo_result, inv_matrix):
        candidates_top, candidates_bottom = [], []
        net_y = self.court_ref.config.net_y
        court_w = self.court_ref.config.court_width
        court_l = self.court_ref.config.court_length
        
        for box in yolo_result.boxes.xyxy:
            bbox = box.cpu().numpy()
            # 1. Metric coordinates of feet
            px = (bbox[0] + bbox[2]) / 2
            py = bbox[3]
            pt = np.array([px, py], dtype=np.float32).reshape(1, 1, 2)
            projected = cv2.perspectiveTransform(pt, inv_matrix)
            x_m, y_m = projected[0, 0, 0], projected[0, 0, 1]
            
            # 2. Area of bbox (Players are usually larger than background referees)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # 3. Scoring Heuristics
            # Distance to center line
            dist_x = abs(x_m - (court_w / 2))
            
            # Depth penalty: Referees are often far behind the baseline
            # Optimal Y is near the baseline (0 or 23.77)
            if y_m < net_y:
                dist_y_baseline = abs(y_m - 0)
            else:
                dist_y_baseline = abs(y_m - court_l)
            
            # Calculate a "Player Score" (Higher is more likely a player)
            # - Reward larger area
            # - Reward proximity to center-line (X)
            # - Penalize being too far out-of-bounds (Y)
            score = area / 1000.0 # Base on size
            score -= dist_x * 2.0 # Penalty for being far from center line
            score -= dist_y_baseline * 1.5 # Penalty for being far behind baseline
            
            if y_m < net_y:
                candidates_top.append({"bbox": bbox, "score": score})
            else:
                candidates_bottom.append({"bbox": bbox, "score": score})
        
        # Pick the highest scoring candidate on each side
        top_players = [sorted(candidates_top, key=lambda x: x["score"], reverse=True)[0]["bbox"]] if candidates_top else []
        bottom_players = [sorted(candidates_bottom, key=lambda x: x["score"], reverse=True)[0]["bbox"]] if candidates_bottom else []
                
        return top_players, bottom_players
