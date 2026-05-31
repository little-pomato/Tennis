import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import distance
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.models.tracknet import BallTrackerNet

class BallDetector(BaseDetector):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Optimization: Use FP16 if supported by device
        self.use_fp16 = ("cuda" in device or "mps" in device)
        if self.use_fp16:
            self.model = self.model.half()

        self.input_width = 640
        self.input_height = 360

    def process(self, context: VideoContext) -> VideoContext:
        scale_w = context.width / self.input_width
        scale_h = context.height / self.input_height
        
        num_frames = len(context.frame_paths)
        ball_track = [(None, None)] * num_frames
        prev_pred = (None, None)
        
        # Batch size for GPU acceleration
        batch_size = 16 if self.use_fp16 else 1
        print(f"Tracking ball (batch_size={batch_size}, device={self.device}, fp16={self.use_fp16})...")
        
        # Cache for resized frames
        resized_cache = {}

        for i in tqdm(range(2, num_frames, batch_size), desc="Tracking ball"):
            end_idx = min(i + batch_size, num_frames)
            current_batch_size = end_idx - i
            
            batch_inps = []
            for idx in range(i, end_idx):
                for target_idx in [idx, idx-1, idx-2]:
                    if target_idx not in resized_cache:
                        full_frame = context.get_frame(target_idx)
                        resized_cache[target_idx] = cv2.resize(full_frame, (self.input_width, self.input_height))
                
                img = resized_cache[idx]
                img_prev = resized_cache[idx-1]
                img_preprev = resized_cache[idx-2]
                
                imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
                # (H, W, C) -> (C, H, W)
                imgs = np.transpose(imgs, (2, 0, 1))
                batch_inps.append(imgs)
            
            # Memory cleanup
            min_needed = i - 2
            keys_to_remove = [k for k in list(resized_cache.keys()) if k < min_needed]
            for k in keys_to_remove:
                del resized_cache[k]

            # Transfer and convert to FP16 if needed
            inp_tensor = torch.from_numpy(np.array(batch_inps)).float().div(255.0).to(self.device)
            if self.use_fp16:
                inp_tensor = inp_tensor.half()

            with torch.no_grad():
                out = self.model(inp_tensor)
            
            if len(out.shape) == 3:
                out = out.view(current_batch_size, self.model.out_channels, self.input_height, self.input_width)
            
            # Get argmax on device
            outputs = out.argmax(dim=1).detach().cpu().numpy()
            
            for j in range(current_batch_size):
                frame_idx = i + j
                x_pred, y_pred = self.postprocess(outputs[j], prev_pred, scale_w, scale_h)
                prev_pred = (x_pred, y_pred)
                ball_track[frame_idx] = (x_pred, y_pred)
            
        context.ball_track = ball_track
        return context

    def postprocess(self, feature_map, prev_pred, scale_w, scale_h, max_dist=80):
        # Optimization: Threshold and use FindContours instead of HoughCircles
        feature_map = (feature_map * 255).astype(np.uint8)
        _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = (M["m10"] / M["m00"]) * scale_w
                cy = (M["m01"] / M["m00"]) * scale_h
                candidates.append((cx, cy))
        
        x, y = None, None
        if candidates:
            if prev_pred[0] is not None:
                # Find nearest candidate to previous prediction
                best_dist = max_dist
                for cx, cy in candidates:
                    dist = distance.euclidean((cx, cy), prev_pred)
                    if dist < best_dist:
                        best_dist = dist
                        x, y = cx, cy
            else:
                # Use largest/first candidate
                x, y = candidates[0]
                
        return x, y
