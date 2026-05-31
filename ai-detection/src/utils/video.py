import cv2
import numpy as np
import os
from typing import List, Tuple
from scenedetect import SceneManager, open_video, ContentDetector

def read_video_to_disk(path: str, temp_dir: str = "temp_frames", max_long_side: int = 640, target_fps: int = 20) -> Tuple[List[str], int, int, int]:
    """Extracts, resizes, and potentially skips frames to disk. Returns (paths, fps, width, height)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {path}")
        
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0: source_fps = 30.0
    
    # Calculate skip rate if target_fps is provided
    skip_rate = 1
    if target_fps > 0 and source_fps > target_fps:
        skip_rate = round(source_fps / target_fps)
        actual_fps = source_fps / skip_rate
    else:
        actual_fps = source_fps
        
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    
    idx = 0
    source_idx = 0
    
    print(f"Extracting frames: source_fps={source_fps:.2f}, target_fps={actual_fps:.2f}, skip_rate={skip_rate}, max_long_side={max_long_side}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if source_idx % skip_rate == 0:
            # Resize
            h, w = frame.shape[:2]
            if max_long_side > 0:
                scale = max_long_side / max(h, w)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if idx == 0:
                final_h, final_w = frame.shape[:2]
                
            frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_paths.append(frame_path)
            idx += 1
            
        source_idx += 1
        
    cap.release()
    return frame_paths, actual_fps, final_w, final_h

def scene_detect(path_video: str) -> List[List[int]]:
    """Split video into disjoint fragments."""
    video = open_video(path_video)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    if not scene_list:
        # Fallback to single scene of the entire video
        cap = cv2.VideoCapture(path_video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [[0, num_frames]]
        
    return [[s[0].get_frames(), s[1].get_frames()] for s in scene_list]
