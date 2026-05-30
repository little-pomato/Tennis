from ultralytics import YOLO
import torch

ULTRA_MODEL = r"yolov8n.pt"
PLAYER_CLASSES = {"person"}       # 要視作球員的類別


class PlayerDetector:
    def __init__(self, model_path=ULTRA_MODEL, device="auto", imgsz=640, conf=0.25):
        """
        device:
            "auto" -> 有 GPU 就用 GPU，沒有就用 CPU
            "cpu"  -> 強制 CPU
            0      -> 使用第 0 張 GPU
            1      -> 使用第 1 張 GPU
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf

        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else "cpu"
        elif isinstance(device, str) and device.isdigit():
            self.device = int(device)
        else:
            self.device = device

        self.ultra = YOLO(self.model_path)

        if self.device == "cpu":
            print(f"[INFO] YOLOv8 loaded on CPU: {self.model_path}")
        else:
            gpu_name = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "unknown GPU"
            print(f"[INFO] YOLOv8 loaded on GPU {self.device}: {gpu_name}")

    def detect(self, img):
        """回傳 [(x1, y1, x2, y2, conf), ...]"""
        h, w = img.shape[:2]
        boxes = []

        # 執行 YOLO 偵測
        # 重點是 device=self.device
        res = self.ultra.predict(
            img,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device
        )[0]

        name_map = getattr(self.ultra.model, "names", None) or getattr(res, "names", None) or {}

        # 取出偵測結果
        for b in res.boxes:
            try:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            except Exception:
                continue

            cls_name = name_map.get(cls_id, "person")

            if cls_name in PLAYER_CLASSES and conf > self.conf:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                boxes.append((x1, y1, x2, y2, conf))

        return boxes