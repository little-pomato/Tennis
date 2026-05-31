from abc import ABC, abstractmethod
from src.pipeline import VideoContext

class BaseDetector(ABC):
    @abstractmethod
    def process(self, context: VideoContext) -> VideoContext:
        """Process the video context and update it with detection results."""
        pass
