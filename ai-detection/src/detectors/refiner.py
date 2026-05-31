from src.detectors.base import BaseDetector
from src.pipeline import VideoContext
from src.core.utils.trajectory import smooth_trajectory

class TrajectoryRefiner(BaseDetector):
    """Pipeline node that refines the ball trajectory before bounce detection."""
    def process(self, context: VideoContext) -> VideoContext:
        print("Refining ball trajectory...")
        context.ball_track = smooth_trajectory(context.ball_track)
        return context
