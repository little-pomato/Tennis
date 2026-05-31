from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import cv2

@dataclass
class VideoContext:
    frame_paths: List[str]
    fps: int
    width: int
    height: int
    
    # Detection results
    ball_track: List[Tuple[Optional[float], Optional[float]]] = field(default_factory=list)
    bounces: List[int] = field(default_factory=list) # Frame indices
    homography_matrices: List[Optional[np.ndarray]] = field(default_factory=list)
    court_keypoints: List[Optional[np.ndarray]] = field(default_factory=list)
    players: List[Dict[str, Any]] = field(default_factory=list) # [{frame_idx: {top: [], bottom: []}}]
    
    # Analysis results
    bounce_analysis: List[Dict[str, Any]] = field(default_factory=list) # [{frame: idx, pos_2d: (x,y), status: "In/Out"}]
    analytics_data: Dict[str, Any] = field(default_factory=lambda: {
        "ball_speeds": [], # [{frame: i, speed_kmh: x}]
        "player_stats": {
            "top": {"forehands": 0, "backhands": 0, "drops": []},
            "bottom": {"forehands": 0, "backhands": 0, "drops": []}
        }
    })

    def __post_init__(self):
        num_frames = len(self.frame_paths)
        if not self.ball_track:
            self.ball_track = [(None, None)] * num_frames
        if not self.homography_matrices:
            self.homography_matrices = [None] * num_frames
        if not self.court_keypoints:
            self.court_keypoints = [None] * num_frames

    def get_frame(self, idx: int) -> np.ndarray:
        return cv2.imread(self.frame_paths[idx])

class Pipeline:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        return self

    def run(self, context: VideoContext) -> VideoContext:
        for node in self.nodes:
            context = node.process(context)
        return context
