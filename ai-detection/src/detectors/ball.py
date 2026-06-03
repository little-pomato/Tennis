import cv2
import os
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.models.tracknet import BallTrackerNet

class BallDetector(BaseDetector):
    def __init__(self, model_path: str, device: str = 'cpu', batch_size: int = None):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        self.device_type = torch.device(device).type
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.use_fp16 = self._resolve_fp16()
        if self.use_fp16:
            self.model = self.model.half()
        if self.device_type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.input_width = 640
        self.input_height = 360
        self.batch_size = self._resolve_batch_size(batch_size)

    def process(self, context: VideoContext) -> VideoContext:
        scale_w = context.width / self.input_width
        scale_h = context.height / self.input_height
        
        num_frames = len(context.frame_paths)
        ball_track = [(None, None)] * num_frames
        prev_pred = (None, None)

        batch_size = self.batch_size
        print(f"Tracking ball (batch_size={batch_size}, device={self.device}, fp16={self.use_fp16})...")

        preprocess_time = 0.0
        inference_time = 0.0
        postprocess_time = 0.0

        for i in tqdm(range(2, num_frames, batch_size), desc="Tracking ball"):
            end_idx = min(i + batch_size, num_frames)
            current_batch_size = end_idx - i

            started = perf_counter()
            batch_inps = np.empty(
                (current_batch_size, 9, self.input_height, self.input_width),
                dtype=np.float32,
            )
            for idx in range(i, end_idx):
                batch_offset = idx - i
                img = context.get_resized_frame(idx, self.input_width, self.input_height)
                img_prev = context.get_resized_frame(idx - 1, self.input_width, self.input_height)
                img_preprev = context.get_resized_frame(idx - 2, self.input_width, self.input_height)

                batch_inps[batch_offset, 0:3] = np.transpose(img, (2, 0, 1))
                batch_inps[batch_offset, 3:6] = np.transpose(img_prev, (2, 0, 1))
                batch_inps[batch_offset, 6:9] = np.transpose(img_preprev, (2, 0, 1))
            preprocess_time += perf_counter() - started

            inp_tensor = torch.from_numpy(batch_inps).div_(255.0).to(self.device, non_blocking=True)
            if self.use_fp16:
                inp_tensor = inp_tensor.half()

            started = perf_counter()
            with torch.inference_mode():
                out = self.model(inp_tensor)
                out_shape = out.shape
                class_maps = out.argmax(dim=1).to(torch.uint8).detach().cpu().numpy()
            inference_time += perf_counter() - started

            del inp_tensor, out

            if len(out_shape) == 3:
                class_maps = class_maps.reshape(current_batch_size, self.input_height, self.input_width)

            started = perf_counter()
            for j in range(current_batch_size):
                frame_idx = i + j
                x_pred, y_pred = self.postprocess(class_maps[j], prev_pred, scale_w, scale_h)
                prev_pred = (x_pred, y_pred)
                ball_track[frame_idx] = (x_pred, y_pred)
            postprocess_time += perf_counter() - started

        context.ball_track = ball_track
        print(
            "[Timing] BallDetector detail: "
            f"preprocess={preprocess_time:.2f}s, "
            f"inference={inference_time:.2f}s, "
            f"postprocess={postprocess_time:.2f}s"
        )
        return context

    def postprocess(self, feature_map, prev_pred, scale_w, scale_h, max_dist=80):
        heatmap = (feature_map * 255).astype(np.uint8, copy=False)
        _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
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
                    dist = ((cx - prev_pred[0]) ** 2 + (cy - prev_pred[1]) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        x, y = cx, cy
            else:
                # Use largest/first candidate
                x, y = candidates[0]
                
        return x, y

    def _resolve_batch_size(self, batch_size):
        if batch_size is not None:
            return max(1, int(batch_size))

        env_value = os.getenv("TRACKNET_BATCH_SIZE")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                print(f"Warning: invalid TRACKNET_BATCH_SIZE={env_value!r}; using default.")

        if self.device_type == "cuda":
            return 8
        if self.device_type == "mps":
            return 4
        return 1

    def _resolve_fp16(self):
        env_value = os.getenv("TRACKNET_FP16", "auto").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return self.device_type in {"cuda", "mps"}
        if env_value in {"0", "false", "no", "off"}:
            return False
        return self.device_type == "cuda"
