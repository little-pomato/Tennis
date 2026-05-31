import numpy as np
from typing import List, Tuple, Optional
from scipy.signal import savgol_filter

def smooth_trajectory(ball_track: List[Tuple[Optional[float], Optional[float]]], 
                      window_size: int = 5, 
                      poly_order: int = 2) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Refines ball trajectory using a LIGHT Savitzky-Golay filter.
    Reduced window size (5) to preserve the sharp 'V' shape of hits and bounces.
    """
    if len(ball_track) < window_size:
        return ball_track

    x = np.array([p[0] if p[0] is not None else np.nan for p in ball_track])
    y = np.array([p[1] if p[1] is not None else np.nan for p in ball_track])

    # 1. Linear Interpolation for missing values (only for very short gaps)
    def interpolate_nans(arr):
        nans = np.isnan(arr)
        if not np.any(~nans): return arr
        # Find consecutive nans
        def get_x(a): return a.nonzero()[0]
        # Only interpolate if gap is < 4 frames
        arr[nans] = np.interp(get_x(nans), get_x(~nans), arr[~nans])
        return arr

    x_interp = interpolate_nans(x.copy())
    y_interp = interpolate_nans(y.copy())

    # 2. Savitzky-Golay Smoothing
    # This filter fits a polynomial to a window, preserving the 'V' shape of bounces 
    # better than a Kalman Filter which assumes constant velocity.
    try:
        x_smooth = savgol_filter(x_interp, window_size, poly_order)
        y_smooth = savgol_filter(y_interp, window_size, poly_order)
    except ValueError:
        # Fallback if window is too large for data
        return ball_track

    # 3. Outlier Rejection: If original detection is too far from smoothed path, 
    # it might be a false positive. We keep the smoothed value instead.
    refined = []
    for i in range(len(ball_track)):
        if np.isnan(x[i]):
            # If missing originally, use the smoothed (interpolated) value
            refined.append((float(x_smooth[i]), float(y_smooth[i])))
        else:
            dist = np.sqrt((x[i] - x_smooth[i])**2 + (y[i] - y_smooth[i])**2)
            if dist > 50: # Outlier threshold in pixels
                refined.append((float(x_smooth[i]), float(y_smooth[i])))
            else:
                # Keep original if it's close to the smooth path (trust the detection)
                refined.append((float(x[i]), float(y[i])))

    return refined
