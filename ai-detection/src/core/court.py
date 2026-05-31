import numpy as np
import cv2
from src.config import CourtConfig

class TennisCourt:
    def __init__(self):
        self.config = CourtConfig()
        
        # Standard Key points in METERS
        # Using standard court layout: (x, y)
        W = self.config.court_width
        L = self.config.court_length
        SM = self.config.single_margin
        SW = self.config.singles_width
        NY = self.config.net_y
        SF = self.config.service_line_from_net
        
        self.baseline_top = ((0, 0), (W, 0))
        self.baseline_bottom = ((0, L), (W, L))
        self.net = ((0, NY), (W, NY))
        self.left_court_line = ((0, 0), (0, L))
        self.right_court_line = ((W, 0), (W, L))
        self.left_inner_line = ((SM, 0), (SM, L))
        self.right_inner_line = ((SM+SW, 0), (SM+SW, L))
        self.middle_line = ((W/2, NY-SF), (W/2, NY+SF))
        self.top_inner_line = ((SM, NY-SF), (SM+SW, NY-SF))
        self.bottom_inner_line = ((SM, NY+SF), (SM+SW, NY+SF))
        
        # Mapping for detection network (which likely uses different pixel-based training)
        # We must keep these original pixel-based values for the DETECTOR only.
        # But for 2D visualization, we use the METRIC values above.
        self.detector_keypoints = [
            (286, 561), (1379, 561),   # Baseline top
            (286, 2935), (1379, 2935), # Baseline bottom
            (423, 561), (423, 2935),   # Left inner
            (1242, 561), (1242, 2935), # Right inner
            (423, 1110), (1242, 1110), # Top inner
            (423, 2386), (1242, 2386), # Bottom inner
            (832, 1110), (832, 2386)   # Middle
        ]
        
        # Reference coordinates in METERS for the same points
        self.metric_keypoints = [
            (0, 0), (W, 0),            # Baseline top
            (0, L), (W, L),            # Baseline bottom
            (SM, 0), (SM, L),          # Left inner
            (SM+SW, 0), (SM+SW, L),    # Right inner
            (SM, NY-SF), (SM+SW, NY-SF), # Top inner
            (SM, NY+SF), (SM+SW, NY+SF), # Bottom inner
            (W/2, NY-SF), (W/2, NY+SF)   # Middle
        ]

        self.court_configurations = {
            1: [self.metric_keypoints[0], self.metric_keypoints[1], self.metric_keypoints[2], self.metric_keypoints[3]],
            2: [self.metric_keypoints[4], self.metric_keypoints[6], self.metric_keypoints[5], self.metric_keypoints[7]],
            3: [self.metric_keypoints[4], (W, 0), self.metric_keypoints[5], (W, L)],
            4: [(0, 0), self.metric_keypoints[6], (0, L), self.metric_keypoints[7]],
            5: [self.metric_keypoints[8], self.metric_keypoints[9], self.metric_keypoints[10], self.metric_keypoints[11]],
            # ... can add more mappings if needed
        }

    def get_reference_keypoints(self) -> np.ndarray:
        # The detector expects points in its training coordinate space
        return np.array(self.detector_keypoints, dtype=np.float32).reshape((-1, 1, 2))
    
    def get_metric_keypoints(self) -> np.ndarray:
        # For mapping image to metric space
        return np.array(self.metric_keypoints, dtype=np.float32).reshape((-1, 1, 2))

    def get_court_image(self, scale: float = 20.0, margin: int = 40) -> np.ndarray:
        """Builds a high-quality BGR court image using metric scale."""
        W = int(self.config.court_width * scale)
        L = int(self.config.court_length * scale)
        
        canvas_w = W + 2 * margin
        canvas_h = L + 2 * margin
        
        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        # Professional Blue court color (often used in modern tournaments)
        img[:] = (150, 100, 50) 
        # Out-of-bounds area (lighter blue or green)
        cv2.rectangle(img, (0, 0), (canvas_w, canvas_h), (120, 80, 40), -1)
        cv2.rectangle(img, (margin, margin), (margin + W, margin + L), (150, 100, 50), -1)

        white = (255, 255, 255)
        thick = max(1, int(scale / 5))

        def t(x_m, y_m):
            return (int(x_m * scale + margin), int(y_m * scale + margin))

        lines = [
            (self.baseline_top[0], self.baseline_top[1]),
            (self.baseline_bottom[0], self.baseline_bottom[1]),
            ((0, self.config.net_y), (self.config.court_width, self.config.net_y)), # Net
            (self.top_inner_line[0], self.top_inner_line[1]),
            (self.bottom_inner_line[0], self.bottom_inner_line[1]),
            (self.left_court_line[0], self.left_court_line[1]),
            (self.right_court_line[0], self.right_court_line[1]),
            (self.left_inner_line[0], self.left_inner_line[1]),
            (self.right_inner_line[0], self.right_inner_line[1]),
            (self.middle_line[0], self.middle_line[1])
        ]

        for p1, p2 in lines:
            cv2.line(img, t(*p1), t(*p2), white, thick, cv2.LINE_AA)
            
        return img

    def meter_to_px(self, x_m, y_m, scale, margin):
        return int(x_m * scale + margin), int(y_m * scale + margin)
