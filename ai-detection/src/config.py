from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

@dataclass
class CourtConfig:
    # Standard dimensions in METERS (Reference coordinate system)
    # Origin (0,0) is far-left corner of the outer boundary (doubles)
    court_width: float = 10.97
    court_length: float = 23.77
    singles_width: float = 8.23
    service_line_from_net: float = 6.40
    net_y: float = 23.77 / 2.0
    single_margin: float = (10.97 - 8.23) / 2.0
    
    # Boundary definitions in METERS
    x_min: float = 0.0
    x_max: float = 10.97
    y_min: float = 0.0
    y_max: float = 23.77
    
    # Singles lines
    singles_x_min: float = (10.97 - 8.23) / 2.0
    singles_x_max: float = 10.97 - ((10.97 - 8.23) / 2.0)
    
    # Service lines
    service_y_far: float = (23.77 / 2.0) - 6.40
    service_y_near: float = (23.77 / 2.0) + 6.40

    @property
    def total_width(self) -> float:
        return self.court_width

    @property
    def total_height(self) -> float:
        return self.court_length

@dataclass
class ModelConfig:
    ball_model_path: str = "models/tracknet.pt"
    court_model_path: str = "models/model_tennis_court_det.pt"
    bounce_model_path: str = "models/ctb_regr_bounce.cbm"
    stroke_model_path: str = "models/best_model.pt"
