import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.models.tracknet import BallTrackerNet
from src.core.utils.kps import refine_kps
from src.core.homography import HomographyHandler
from src.core.court import TennisCourt

class CourtDetector(BaseDetector):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = BallTrackerNet(out_channels=15)
        self.device = device
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.court_ref = TennisCourt()
        self.homography_handler = HomographyHandler(self.court_ref)

    def process(self, context: VideoContext) -> VideoContext:
        output_width = 640
        output_height = 360
        scale_w = context.width / output_width
        scale_h = context.height / output_height
        
        # Optimization: Detect once for static camera
        print("Detecting court (static camera optimization)...")
        first_frame = context.get_frame(0)
        img = context.get_resized_frame(0, output_width, output_height)
        inp = (img.astype(np.float32) / 255.)
        inp = torch.from_numpy(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.model(inp.float())[0]
        pred = torch.sigmoid(out).detach().cpu().numpy()

        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                       param1=50, param2=2, minRadius=10, maxRadius=25)
            
            if circles is not None:
                x_pred = circles[0][0][0] * scale_w
                y_pred = circles[0][0][1] * scale_h
                if kps_num not in [8, 12, 9]:
                    x_pred, y_pred = refine_kps(first_frame, int(y_pred), int(x_pred))
                points.append((x_pred, y_pred))                
            else:
                points.append(None)

        matrix_trans = self.homography_handler.get_trans_matrix(points)
        
        if matrix_trans is not None:
            # Important: Get the inverse matrix to map Image -> Metric
            # self.homography_handler.get_trans_matrix returns Metric -> Image
            matrix_img_to_metric = cv2.invert(matrix_trans)[1]
            
            # Use detector_keypoints for visual refinement on the image
            ref_kps = self.court_ref.get_reference_keypoints()
            court_kps_img = cv2.perspectiveTransform(ref_kps, matrix_trans)
            
            # Reuse the same detection for all frames
            num_frames = len(context.frame_paths)
            for i in range(num_frames):
                context.court_keypoints[i] = court_kps_img
                context.homography_matrices[i] = matrix_img_to_metric
        else:
            print("Warning: Court not detected in the first frame.")
            num_frames = len(context.frame_paths)
            for i in range(num_frames):
                context.court_keypoints[i] = None
                context.homography_matrices[i] = None
                
        return context
