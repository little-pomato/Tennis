# Tennis Detection Backend API

This is a FastAPI-based backend for the Tennis Detection Pipeline.

## Features
- **Video Upload**: Upload a tennis video (MP4, AVI, MOV) and get detection results.
- **JSON Results**: Returns a comprehensive JSON containing ball tracks, player positions, court keypoints, and match analysis.
- **Cross-Platform**: Automatically detects and uses CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

## Installation

1. Ensure you have the main project dependencies installed (Ultralytics, OpenCV, torch, etc.).
2. Install backend-specific dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

## Running the API

From the project root:
```bash
python backend/main.py
```
The API will be available at `http://0.0.0.0:8000`.

## API Endpoints

### POST `/upload`
Upload a video file to run the detection pipeline.

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/upload" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/video.mp4"
```

### GET `/health`
Check if the API is running and see which device (CPU/CUDA/MPS) is being used.

## Best Practices Implemented
- **Unique Request Scoping**: Each request is processed in a unique temporary directory to prevent race conditions.
- **Automatic Cleanup**: Temporary frames and uploaded videos are deleted after processing.
- **CORS Enabled**: Ready for frontend integration.
- **Modular Integration**: Leverages the existing `src` pipeline logic directly.
