from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import cv2

@dataclass
class VideoContext:
    frame_paths: List[str]
    fps: int
    width: int
    height: int
    frame_cache_size: int = 96
    resized_cache_size: int = 128
    
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
    timings: Dict[str, float] = field(default_factory=dict)
    _frame_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _resized_cache: OrderedDict[Tuple[int, int, int], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self):
        num_frames = len(self.frame_paths)
        if not self.ball_track:
            self.ball_track = [(None, None)] * num_frames
        if not self.homography_matrices:
            self.homography_matrices = [None] * num_frames
        if not self.court_keypoints:
            self.court_keypoints = [None] * num_frames

    def get_frame(self, idx: int) -> np.ndarray:
        if idx in self._frame_cache:
            frame = self._frame_cache.pop(idx)
            self._frame_cache[idx] = frame
            return frame

        frame = cv2.imread(self.frame_paths[idx])
        if frame is None:
            raise IOError(f"Could not read frame: {self.frame_paths[idx]}")

        if self.frame_cache_size > 0:
            self._frame_cache[idx] = frame
            while len(self._frame_cache) > self.frame_cache_size:
                self._frame_cache.popitem(last=False)

        return frame

    def get_resized_frame(self, idx: int, width: int, height: int) -> np.ndarray:
        key = (idx, width, height)
        if key in self._resized_cache:
            frame = self._resized_cache.pop(key)
            self._resized_cache[key] = frame
            return frame

        frame = self.get_frame(idx)
        if frame.shape[1] == width and frame.shape[0] == height:
            resized = frame
        else:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        if self.resized_cache_size > 0:
            self._resized_cache[key] = resized
            while len(self._resized_cache) > self.resized_cache_size:
                self._resized_cache.popitem(last=False)

        return resized

    def clear_frame_cache(self):
        self._frame_cache.clear()
        self._resized_cache.clear()

class Pipeline:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        return self

    def run(self, context: VideoContext) -> VideoContext:
        for node in self.nodes:
            name = node.__class__.__name__
            started = perf_counter()
            context = node.process(context)
            elapsed = perf_counter() - started
            context.timings[name] = context.timings.get(name, 0.0) + elapsed
            print(f"[Timing] {name}: {elapsed:.2f}s")
        return context
