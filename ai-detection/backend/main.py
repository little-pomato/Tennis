import os
import sys
import uuid
import shutil
import json
import torch
import logging
import numpy as np
from pathlib import Path
from time import perf_counter
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tennis-api")

# Add project root to sys.path to allow importing from src
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

try:
    from src.utils.video import read_video_to_disk
    from src.pipeline import VideoContext, Pipeline
    from src.config import ModelConfig
    from src.detectors.court import CourtDetector
    from src.detectors.ball import BallDetector
    from src.detectors.bounce import BounceDetector
    from src.detectors.player import PlayerDetector
    from src.detectors.refiner import TrajectoryRefiner
    from src.core.rules import InoutRuleEngine
    from src.core.analytics import MatchAnalyzer
    from src.exporters.data import DataExporter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

app = FastAPI(title="Tennis Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEMP_DIR = root_path / "temp"
TEMP_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_cfg = ModelConfig()

def get_env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %r", name, value, default)
        return default
    return parsed if parsed > 0 else default

target_fps = get_env_int("VIDEO_TARGET_FPS", 20)
max_long_side = get_env_int("VIDEO_MAX_LONG_SIDE", 640)
ball_batch_size = get_env_int("TRACKNET_BATCH_SIZE", None)
frame_cache_size = get_env_int("VIDEO_FRAME_CACHE_SIZE", 96)
resized_cache_size = get_env_int("VIDEO_RESIZED_CACHE_SIZE", 128)

# In-memory progress storage (In production, use Redis or a database)
processing_status = {}

def run_detection_pipeline(video_path: str, request_id: str):
    """Runs the full tennis detection pipeline and updates progress."""
    request_temp_dir = TEMP_DIR / request_id
    request_temp_dir.mkdir(exist_ok=True)
    
    output_json_path = request_temp_dir / "output.json"
    
    try:
        processing_status[request_id] = {"status": "processing", "progress": 10, "message": "Extracting frames..."}
        
        # 1. Initialize Context
        frames_dir = request_temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        started = perf_counter()
        frame_paths, fps, width, height = read_video_to_disk(
            video_path, 
            temp_dir=str(frames_dir), 
            max_long_side=max_long_side,
            target_fps=target_fps
        )
        context = VideoContext(
            frame_paths=frame_paths,
            fps=fps,
            width=width,
            height=height,
            frame_cache_size=frame_cache_size,
            resized_cache_size=resized_cache_size,
        )
        context.timings["FrameExtraction"] = perf_counter() - started

        processing_status[request_id] = {"status": "processing", "progress": 30, "message": "Initializing models..."}
        
        # 2. Build and Run Pipeline
        pipeline = Pipeline()
        pipeline.add_node(CourtDetector(model_cfg.court_model_path, device=device))
        pipeline.add_node(BallDetector(model_cfg.ball_model_path, device=device, batch_size=ball_batch_size))
        pipeline.add_node(TrajectoryRefiner())
        pipeline.add_node(BounceDetector(model_cfg.bounce_model_path))
        pipeline.add_node(PlayerDetector(device=device))
        pipeline.add_node(InoutRuleEngine())
        pipeline.add_node(MatchAnalyzer(model_cfg.stroke_model_path, device=device))

        processing_status[request_id] = {"status": "processing", "progress": 50, "message": "Running AI inference..."}
        
        # Run pipeline
        context = pipeline.run(context)
        
        processing_status[request_id] = {"status": "processing", "progress": 90, "message": "Exporting results..."}
        
        # PRE-CALCULATE PLAYER METRIC PATHS
        player_metric_tracks = {"top": [], "bottom": []}
        from src.core.homography import HomographyHandler
        from src.core.court import TennisCourt
        handler = HomographyHandler(TennisCourt())
        
        for i in range(len(context.players)):
            matrix = context.homography_matrices[i]
            p_data = context.players[i]
            
            for side in ["top", "bottom"]:
                if p_data[side] and matrix is not None:
                    bbox = p_data[side][0]
                    # Feet position (bottom center of bbox)
                    foot = ((bbox[0] + bbox[2]) / 2, bbox[3])
                    pos_m = handler.project_point(foot, matrix)
                    player_metric_tracks[side].append({"frame": i, "x": float(pos_m[0]), "y": float(pos_m[1])})
                else:
                    player_metric_tracks[side].append({"frame": i, "x": None, "y": None})

        # 3. Export Results
        data = {
            "metadata": {
                "fps": float(context.fps),
                "width": context.width,
                "height": context.height,
                "total_frames": len(context.frame_paths),
                "timings": context.timings
            },
            "results": {
                "ball_track": [{"frame": i, "x": pt[0], "y": pt[1]} for i, pt in enumerate(context.ball_track)],
                "bounces": context.bounce_analysis,
                "players": context.players,
                "player_metric_tracks": player_metric_tracks, # NEW: Running paths
                "analytics": context.analytics_data,
                "homography_matrices": [m.tolist() if m is not None else None for m in context.homography_matrices]
            }
        }
        
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.floating)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj

        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4, default=default_serializer)
            
        with open(output_json_path, 'r') as f:
            results = json.load(f)
            
        processing_status[request_id] = {"status": "completed", "progress": 100, "result": results}
        logger.info(f"[{request_id}] Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        processing_status[request_id] = {"status": "failed", "progress": 0, "error": str(e)}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if request_temp_dir.exists():
            shutil.rmtree(request_temp_dir, ignore_errors=True)

@app.post("/upload")
async def upload_and_detect(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    request_id = str(uuid.uuid4())
    input_video_path = TEMP_DIR / f"{request_id}_{file.filename}"
    try:
        with input_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    processing_status[request_id] = {"status": "queued", "progress": 0, "message": "Waiting in queue..."}
    background_tasks.add_task(run_detection_pipeline, str(input_video_path), request_id)
    return {"request_id": request_id}

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Request not found")
    return processing_status[request_id]

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": device,
        "target_fps": target_fps,
        "max_long_side": max_long_side,
        "tracknet_batch_size": ball_batch_size,
        "frame_cache_size": frame_cache_size,
        "resized_cache_size": resized_cache_size,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
