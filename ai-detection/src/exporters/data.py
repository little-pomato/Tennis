import json
import numpy as np
from src.pipeline import VideoContext

class DataExporter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def export(self, context: VideoContext):
        data = {
            "metadata": {
                "fps": context.fps,
                "width": context.width,
                "height": context.height,
                "total_frames": len(context.frame_paths)
            },
            "results": {
                "ball_track": [
                    {"frame": i, "x": pt[0], "y": pt[1]} 
                    for i, pt in enumerate(context.ball_track)
                ],
                "bounces": context.bounce_analysis,
                "players": context.players,
                # Convert matrices to lists for JSON serialization
                "homography_matrices": [
                    m.tolist() if m is not None else None 
                    for m in context.homography_matrices
                ],
                "court_keypoints": [
                    kp.tolist() if kp is not None else None 
                    for kp in context.court_keypoints
                ]
            }
        }
        
        # Helper to handle numpy types
        def default_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=4, default=default_serializer)

    @staticmethod
    def load_to_context(path: str, context: VideoContext):
        """Loads results from JSON back into a VideoContext."""
        with open(path, 'r') as f:
            data = json.load(f)["results"]
        
        context.ball_track = [(p["x"], p["y"]) for p in data["ball_track"]]
        context.bounce_analysis = data["bounces"]
        context.players = data["players"]
        
        context.homography_matrices = [
            np.array(m) if m is not None else None 
            for m in data["homography_matrices"]
        ]
        context.court_keypoints = [
            np.array(kp) if kp is not None else None 
            for kp in data["court_keypoints"]
        ]
        return context
