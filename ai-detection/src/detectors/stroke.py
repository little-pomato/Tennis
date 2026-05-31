import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from src.detectors.base import BaseDetector
from src.pipeline import VideoContext

class StrokeDetector(BaseDetector):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.class_names = ["backhand", "forehand"]
        
        # Build model architecture (MobileNetV3-Small)
        self.model = models.mobilenet_v3_small()
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Define transforms (Matching training/video_processing version)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def process(self, context: VideoContext) -> VideoContext:
        # Stroke detection is typically handled during MatchAnalyzer hit detection
        # This node can be used to pre-calculate or provide classification utility
        return context

    def classify_stroke(self, frame: np.ndarray, bbox: np.ndarray) -> str:
        """Crops a player from the frame and classifies their stroke."""
        # 1. Expand bbox (Matching player_crop.py policy)
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        
        # Expansion scale: 1.7x (from player_crop.py)
        new_w, new_h = bw * 1.7, bh * 1.7
        
        nx1 = max(0, int(cx - new_w / 2))
        ny1 = max(0, int(cy - new_h / 2 - bh * 0.05)) # top_extra 0.05
        nx2 = min(frame.shape[1], int(cx + new_w / 2))
        ny2 = min(frame.shape[0], int(cy + new_h / 2 + bh * 0.05)) # bottom_extra 0.05
        
        if nx2 <= nx1 or ny2 <= ny1:
            return "unknown"

        crop = frame[ny1:ny2, nx1:nx2]
        
        # 2. Transform and Predict
        # Convert BGR (OpenCV) to RGB (PIL/Torch)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        inp = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inp)
            _, preds = torch.max(outputs, 1)
            
        return self.class_names[preds[0].item()]
