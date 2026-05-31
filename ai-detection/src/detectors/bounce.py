import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext

class BounceDetector(BaseDetector):
    def __init__(self, model_path: str):
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(model_path)
        self.threshold = 0.45

    def process(self, context: VideoContext) -> VideoContext:
        x_ball = [pt[0] for pt in context.ball_track]
        y_ball = [pt[1] for pt in context.ball_track]
        
        # Smooth and extrapolate missing detections
        x_smooth, y_smooth = self.smooth_predictions(list(x_ball), list(y_ball))
        
        features, frame_indices = self.prepare_features(x_smooth, y_smooth)
        if features.empty:
            context.bounces = []
            return context
            
        preds = self.model.predict(features)
        
        # Adaptive Thresholding: 
        # Far-court (small Y) has a weaker signal, so we lower the threshold.
        # Near-court (large Y) has a stronger signal, so we keep a higher threshold.
        # Assuming Y typically ranges from ~0 to ~height (e.g., 720 or 1080)
        # Note: In most tennis videos, smaller Y is further away.
        y_values = np.array([y_smooth[i] for i in frame_indices])
        
        # Threshold scales from 0.3 (far) to 0.5 (near)
        # We'll use the 10th and 90th percentiles of Y as bounds for the scaling
        y_min, y_max = np.percentile(y_values, [10, 90])
        adaptive_thresholds = np.interp(y_values, [y_min, y_max], [0.3, 0.5])
        
        ind_bounce = np.where(preds > adaptive_thresholds)[0]
        
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess_indices(ind_bounce, preds)
            context.bounces = [frame_indices[i] for i in ind_bounce]
        else:
            context.bounces = []
            
        return context

    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels[f'x_lag_{i}'] = labels['x-coordinate'].shift(i)
            labels[f'x_lag_inv_{i}'] = labels['x-coordinate'].shift(-i)
            labels[f'y_lag_{i}'] = labels['y-coordinate'].shift(i)
            labels[f'y_lag_inv_{i}'] = labels['y-coordinate'].shift(-i) 
            labels[f'x_diff_{i}'] = abs(labels[f'x_lag_{i}'] - labels['x-coordinate'])
            labels[f'y_diff_{i}'] = labels[f'y_lag_{i}'] - labels['y-coordinate']
            labels[f'x_diff_inv_{i}'] = abs(labels[f'x_lag_inv_{i}'] - labels['x-coordinate'])
            labels[f'y_diff_inv_{i}'] = labels[f'y_lag_inv_{i}'] - labels['y-coordinate']
            labels[f'x_div_{i}'] = abs(labels[f'x_diff_{i}']/(labels[f'x_diff_inv_{i}'] + eps))
            labels[f'y_div_{i}'] = labels[f'y_diff_{i}']/(labels[f'y_diff_inv_{i}'] + eps)

        for i in range(1, num):
            labels = labels[labels[f'x_lag_{i}'].notna()]
            labels = labels[labels[f'x_lag_inv_{i}'].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        colnames = [f'x_diff_{i}' for i in range(1, num)] + \
                   [f'x_diff_inv_{i}' for i in range(1, num)] + \
                   [f'x_div_{i}' for i in range(1, num)] + \
                   [f'y_diff_{i}' for i in range(1, num)] + \
                   [f'y_diff_inv_{i}' for i in range(1, num)] + \
                   [f'y_div_{i}' for i in range(1, num)]

        return labels[colnames], list(labels['frame'])

    def smooth_predictions(self, x_ball, y_ball):
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball)-1):
            if not x_ball[num] and sum(is_none[num-interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num-interp:num], y_ball[num-interp:num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num+1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num+1], y_ball[num+1]))
                    if dist > 80:
                        x_ball[num+1], y_ball[num+1], is_none[num+1] = None, None, 1
                counter += 1
            else:
                counter = 0  
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        return float(func_x(len(x_coords))), float(func_y(len(x_coords)))    

    def postprocess_indices(self, ind_bounce, preds):
        if len(ind_bounce) == 0: return []
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                ind_bounce_filtered.append(ind_bounce[i])
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered
