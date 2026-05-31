import numpy as np
import cv2
from scipy.spatial import distance
from src.core.court import TennisCourt

class HomographyHandler:
    def __init__(self, court: TennisCourt):
        self.court = court
        self.reference_kps = court.get_reference_keypoints()
        self.config_indices = self._build_config_indices()

    def _build_config_indices(self):
        config_indices = {}
        for i, conf in self.court.court_configurations.items():
            inds = []
            for point in conf:
                # Find index in metric_keypoints list
                for k, mk in enumerate(self.court.metric_keypoints):
                    if mk[0] == point[0] and mk[1] == point[1]:
                        inds.append(k)
                        break
            config_indices[i] = inds
        return config_indices

    def get_trans_matrix(self, detected_points):
        """Determine the best homography matrix from detected court points."""
        matrix_trans = None
        min_error = np.inf
        
        # We want matrix that projects Metric -> Image
        metric_kps = self.court.get_metric_keypoints()
        
        for conf_id, inds in self.config_indices.items():
            conf_points_m = self.court.court_configurations[conf_id]
            detected_subset = [detected_points[i] for i in inds]
            
            if None not in detected_subset:
                matrix, _ = cv2.findHomography(np.float32(conf_points_m), np.float32(detected_subset), method=0)
                if matrix is None:
                    continue
                    
                projected_kps = cv2.perspectiveTransform(metric_kps, matrix).squeeze(1)
                
                errors = []
                for i in range(len(detected_points)):
                    if i not in inds and detected_points[i] is not None:
                        errors.append(distance.euclidean(detected_points[i], projected_kps[i]))
                
                error = np.mean(errors) if errors else 0
                if error < min_error:
                    matrix_trans = matrix
                    min_error = error
                    
        return matrix_trans

    @staticmethod
    def project_point(point, matrix):
        """Project a point (x, y) using the given homography matrix."""
        if point[0] is None or matrix is None:
            return None
        pt = np.array([point[0], point[1]], dtype=np.float32).reshape(1, 1, 2)
        projected = cv2.perspectiveTransform(pt, matrix)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])
