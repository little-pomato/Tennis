# Tennis Video Analysis Project

This project is an end-to-end tennis video analysis system. It processes a match video, extracts synchronized frames, detects the court, detects ball bounce events, estimates hit intervals, classifies near-player forehand/backhand strokes, generates tactical landing statistics, and exposes the pipeline through a backend API for a web interface.

The project has three main parts:

1. **Video analysis pipeline** — runs the full tennis video processing workflow.
2. **Stroke classifier training** — trains the forehand/backhand classification model used by `vote_action.py`.
3. **Backend API** — wraps the pipeline so the final web interface can trigger video analysis and retrieve results.

## Repository Structure

Recommended repository layout:

```text
project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── video_processing_new/
│   ├── run_pipeline.py              # Runs the full video analysis pipeline
│   ├── preprocessing.py             # Extracts frames, creates ROI config, builds valid mask
│   ├── frame_extractor.py           # Extracts resized synchronized frames from the input video
│   ├── pick_roi.py                  # Auto/manual ROI selection tool
│   ├── court_detector.py            # Detects court lines and generates roi_config.json
│   ├── bounce.py                    # Detects bounce events, hit intervals, player tracks, YOLO cache
│   ├── vote_action.py               # Classifies near-player strokes using event-level voting
│   ├── analysis_per.py              # Creates event table, stroke summary, charts, and PDF report
│   ├── visualization.py             # Creates overlay video with bounce landings
│   ├── player_detector.py           # YOLOv8-based player detector
│   ├── player_tracker.py            # Two-player tracker
│   ├── hit.py                       # Hit interval detection helpers
│   └── mask.py                      # Court/search-mask helpers
│
├── training/
│   ├── train_tennis_action.py       # Trains the forehand/backhand stroke classifier
│   └── README.md                    # Optional: training data format and training command
│
├── models/
│   ├── README.md                    # Explains how to obtain/place model checkpoints
│   └── best_model.pt                # Stroke classifier checkpoint, preferably via Git LFS or Release
│
├── backend/
│   ├── main.py                      # Backend API for the web interface
│   └── README.md                    # Optional: backend run instructions and API notes
│
├── raw_videos/                      # Local input videos; usually not committed to Git
└── dataset/                         # Generated outputs; usually not committed to Git
```

If `run_pipeline.py` is placed inside `video_processing_new/`, it treats the parent folder as the project root. In that setup, generated outputs are written to:

```text
project/dataset/
```

instead of:

```text
project/video_processing_new/dataset/
```

## Main Workflow

The full video analysis workflow is:

```text
preprocessing.py
bounce.py
vote_action.py
analysis_per.py
visualization.py
```

### Stage Overview

| Stage | Script | Main responsibility |
|---|---|---|
| Preprocessing | `preprocessing.py` | Extract frames, detect or select court ROI, create `roi_config.json` and `valid_mask.png` |
| Bounce detection | `bounce.py` | Detect ball bounces, estimate hit intervals, track players, save YOLO/person cache |
| Stroke voting | `vote_action.py` | Use the trained stroke classifier to classify near-player hit events as forehand/backhand |
| Analysis | `analysis_per.py` | Combine bounce, hit, and stroke-vote results into CSV summaries, charts, and a PDF report |
| Visualization | `visualization.py` | Create a synchronized overlay video showing bounce landings |

## Features

- Extracts synchronized frames from tennis videos at a fixed sampling rate.
- Stores both timestamps and source-frame indices in `frame_map.csv`.
- Detects tennis court geometry and computes homography for court-coordinate projection.
- Supports manual ROI selection if automatic court detection fails.
- Detects ball bounce events using frame differences, candidate scoring, temporal support, spatial voting, and second-stage validation.
- Uses YOLOv8 person detection and a two-player tracker to reduce false bounce detections near players.
- Saves YOLO/person detection cache files so `vote_action.py` can reuse the same player detections.
- Estimates near/far hit intervals from the ball's vertical movement pattern.
- Classifies near-player hit events as forehand/backhand using a trained PyTorch checkpoint.
- Generates landing maps, in/out statistics, stroke-specific landing charts, and rally summaries.
- Generates a synchronized overlay video for presentation or demo.
- Provides a backend API entry point for integrating the pipeline into a web interface.

## Requirements

Recommended Python version:

```text
Python 3.9+
```

Required packages:

```bash
pip install opencv-python numpy pandas matplotlib tqdm ultralytics torch
```

If the backend uses FastAPI, also install:

```bash
pip install fastapi uvicorn python-multipart
```

The player detector uses `yolov8n.pt` through the `ultralytics` package. If the model is not available locally, Ultralytics may attempt to download it when the detector is initialized.

## Model Checkpoint

`vote_action.py` requires a trained forehand/backhand stroke classifier checkpoint.

Recommended location:

```text
models/best_model.pt
```

If your current `run_pipeline.py` still uses another default checkpoint path, you can either:

1. Pass the model path explicitly:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt
```

2. Or change the default checkpoint path in `run_pipeline.py` to:

```python
DEFAULT_STROKE_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pt"
```

### Should `best_model.pt` be committed to GitHub?

The checkpoint must be available for the project to run stroke voting, but it should not be committed as a normal Git file if it is large.

Recommended options:

| Situation | Recommended approach |
|---|---|
| Small checkpoint | Commit directly or use Git LFS |
| Medium/large checkpoint | Use Git LFS |
| Very large checkpoint | Upload to GitHub Releases or cloud storage, then document the download location |
| Private or unstable model | Do not commit it; provide instructions for placing it under `models/` |

If using Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/best_model.pt
git commit -m "Add stroke classifier checkpoint"
```

If the checkpoint is not included in the repository, keep `models/README.md` and explain where users should place the file:

```text
models/best_model.pt
```

## Training the Stroke Classifier

The training code should be stored separately from the inference pipeline:

```text
training/train_tennis_action.py
```

This script is responsible for training the forehand/backhand classifier used by `vote_action.py`.

Example command:

```bash
python training/train_tennis_action.py
```

The exact training command may depend on your dataset path, model settings, and output directory. A recommended `training/README.md` should document:

- training data folder structure
- class names, such as `forehand/` and `backhand/`
- input image/crop format
- training command
- output checkpoint path
- how the resulting `best_model.pt` should be copied into `models/`

Recommended training data layout:

```text
dataset/tennis_actions/
├── train/
│   ├── forehand/
│   └── backhand/
├── valid/
│   ├── forehand/
│   └── backhand/
└── test/
    ├── forehand/
    └── backhand/
```

After training, place the selected checkpoint at:

```text
models/best_model.pt
```

or pass its path to `run_pipeline.py` using `--checkpoint`.

## Quick Start

Place a raw tennis video under:

```text
raw_videos/testVid.mp4
```

Run the full pipeline:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt
```

If you want to skip forehand/backhand voting:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 --skip-vote
```

Useful pipeline options:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 --debug-bounce
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 --skip-analysis
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 --skip-visualization
```

The pipeline will create:

```text
dataset/<video_name>/
```

and store all intermediate and final outputs there.

## Running Each Step Manually

For debugging, each stage can be run separately.

### 1. Preprocessing

```bash
python video_processing_new/preprocessing.py raw_videos/testVid.mp4
```

This creates:

```text
dataset/<video_name>/frames/
dataset/<video_name>/frame_map.csv
dataset/<video_name>/roi_config.json
dataset/<video_name>/valid_mask.png
```

`frame_map.csv` contains:

| Column | Description |
|---|---|
| `index` | Extracted-frame index used by later scripts |
| `filename` | Extracted frame filename |
| `timestamp_ms` | Timestamp in the source video |
| `source_frame_idx` | Original video frame index used to generate the extracted frame |

### 2. Bounce Detection

```bash
python video_processing_new/bounce.py raw_videos/testVid.mp4
```

Useful arguments:

```bash
python video_processing_new/bounce.py raw_videos/testVid.mp4 --debug
python video_processing_new/bounce.py raw_videos/testVid.mp4 --scale 0.5
python video_processing_new/bounce.py raw_videos/testVid.mp4 --peak-min-score 0.58
python video_processing_new/bounce.py raw_videos/testVid.mp4 --near-neighbor-radius 22 --far-neighbor-radius 8
python video_processing_new/bounce.py raw_videos/testVid.mp4 --yolo-every-n 3
python video_processing_new/bounce.py raw_videos/testVid.mp4 --yolo-device auto --yolo-imgsz 640 --yolo-conf 0.25
```

This outputs results to:

```text
dataset/<video_name>/bounce_detector/
```

### 3. Stroke Vote

```bash
python video_processing_new/vote_action.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt \
  --yolo-cache-dir dataset/testVid/bounce_detector
```

This step should be run after `bounce.py`, because it reuses the YOLO/person cache generated during bounce detection.

Expected output:

```text
dataset/<video_name>/bounce_detector/stroke_vote_same_crop/stroke_vote_events.csv
```

### 4. Analysis Report

```bash
python video_processing_new/analysis_per.py raw_videos/testVid.mp4
```

Useful arguments:

```bash
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 --stroke-csv path/to/stroke_vote_events.csv
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 --min-stroke-vote-count 1
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 --min-stroke-vote-ratio 0.5
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 --min-stroke-confidence 0.5
```

If `bounce.py` was run with `--scale 0.5`, use:

```bash
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 --bounce-coord-scale 2.0
```

### 5. Visualization

```bash
python video_processing_new/visualization.py raw_videos/testVid.mp4
```

Useful arguments:

```bash
python video_processing_new/visualization.py raw_videos/testVid.mp4 --show-index
python video_processing_new/visualization.py raw_videos/testVid.mp4 --out-overlay dataset/testVid/bounce_detector/my_overlay.mp4
python video_processing_new/visualization.py raw_videos/testVid.mp4 --court-ratio 0.35
```

If `bounce.py` was run with `--scale 0.5`, use:

```bash
python video_processing_new/visualization.py raw_videos/testVid.mp4 --bounce-coord-scale 2.0
```

## Backend API

The backend code is stored in:

```text
backend/main.py
```

This backend is intended to connect the web interface with the tennis analysis pipeline. It should handle tasks such as:

- receiving uploaded videos
- triggering the analysis pipeline
- saving outputs under `dataset/<video_name>/`
- returning analysis status and result paths to the frontend
- serving or linking generated reports, charts, and overlay videos

If the backend uses FastAPI, run it from the project root with:

```bash
uvicorn backend.main:app --reload
```

When calling the pipeline from the backend, use project-root-relative paths instead of hard-coded absolute paths. For example:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PROJECT_ROOT / "video_processing_new" / "run_pipeline.py"
```

## Output Structure

After running the pipeline, the output is stored under:

```text
dataset/<video_name>/
├── frames/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── frame_map.csv
├── roi_config.json
├── valid_mask.png
├── court_detector_debug/
└── bounce_detector/
    ├── frame_scores.csv
    ├── bounce_candidates.csv
    ├── bounce_events.csv
    ├── scan_stats.csv
    ├── stage2_debug.csv
    ├── stage2_points_debug.csv
    ├── yolo_person_boxes.csv
    ├── yolo_person_boxes_vote_cache.csv
    ├── tracked_players.csv
    ├── tracked_players_vote_cache.csv
    ├── ball_y_signal.csv
    ├── hit_intervals_y_extrema.csv
    ├── hit_intervals_y_segments.csv
    ├── events_from_bounce_hit.csv
    ├── stroke_landing_summary.csv
    ├── analysis_report.pdf
    ├── overlay_bounce_landings.mp4
    ├── stroke_vote_same_crop/
    │   └── stroke_vote_events.csv
    ├── analysis_charts/
    │   ├── 00_summary_dashboard.png
    │   ├── 01_near_player_bounce_map.png
    │   ├── 02_far_player_bounce_map.png
    │   ├── 03_near_stroke_landing_map.png
    │   ├── 04_near_stroke_in_out.png
    │   ├── 05_near_stroke_zone_distribution.png
    │   └── 06_rally_lengths.png
    └── debug/
        ├── valid_mask.png
        ├── search_mask_static.png
        ├── hit_intervals_y_debug.png
        ├── scan_*.png
        ├── event_*.png
        └── final_peak_tracks/
```

The old output folder name `bounce_event_v3/` has been replaced by:

```text
bounce_detector/
```

## What Should Be Committed to GitHub?

Recommended to commit:

```text
README.md
requirements.txt
.gitignore
video_processing_new/*.py
training/train_tennis_action.py
backend/main.py
models/README.md
training/README.md
backend/README.md
```

Usually not recommended to commit:

```text
raw_videos/*
dataset/*
*.mp4
*.avi
*.mov
*.mkv
*.csv
*.pdf
__pycache__/
.venv/
```

Model checkpoints such as `best_model.pt` should be handled carefully:

- use Git LFS if the file should be included in the repo;
- use GitHub Releases or external storage if it is too large;
- document the expected placement path in `models/README.md`.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/

# Local videos
raw_videos/*
!raw_videos/.gitkeep
!raw_videos/README.md

# Generated outputs
dataset/*
!dataset/.gitkeep
!dataset/README.md

# Generated media and reports
*.mp4
*.avi
*.mov
*.mkv
*.pdf

# Optional: generated CSV files
*.csv

# Model checkpoints
# If using Git LFS for .pt files, remove this rule.
*.pt
*.pth
*.onnx

# Editor / OS
.DS_Store
.vscode/
.idea/
```

## Common Problems

### 1. `ModuleNotFoundError`

Install the required packages:

```bash
pip install opencv-python numpy pandas matplotlib tqdm ultralytics torch
```

If using the backend:

```bash
pip install fastapi uvicorn python-multipart
```

### 2. Stroke checkpoint not found

Check whether the checkpoint exists:

```text
models/best_model.pt
```

Then run:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt
```

### 3. No stroke labels in the PDF report

Check whether this file exists:

```text
dataset/<video_name>/bounce_detector/stroke_vote_same_crop/stroke_vote_events.csv
```

If it does not exist, run `vote_action.py` or run the full pipeline without `--skip-vote`.

If it exists but no labels appear, try relaxing thresholds:

```bash
python video_processing_new/analysis_per.py raw_videos/testVid.mp4 \
  --min-stroke-vote-count 1 \
  --min-stroke-vote-ratio 0.0 \
  --min-stroke-confidence 0.0
```

### 4. Output folder not found

The current scripts use:

```text
dataset/<video_name>/bounce_detector/
```

not:

```text
dataset/<video_name>/bounce_event_v3/
```

### 5. Overlay video and bounce events are not synchronized

Use extracted frames instead of decoding the raw video directly. The pipeline is based on extracted-frame indices, so direct raw-video decoding may cause frame drift.

### 6. Automatic court detection fails

Run preprocessing again and use manual ROI selection when prompted.

### 7. YOLO runs too slowly

Try reducing YOLO frequency:

```bash
python video_processing_new/bounce.py raw_videos/testVid.mp4 --yolo-every-n 3
```

This may improve speed, but it can also reduce player-exclusion and stroke-vote cache accuracy.

## Recommended Development Workflow

For a new video, debug step by step:

```bash
python video_processing_new/preprocessing.py raw_videos/testVid.mp4
python video_processing_new/bounce.py raw_videos/testVid.mp4 --debug
python video_processing_new/vote_action.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt \
  --yolo-cache-dir dataset/testVid/bounce_detector
python video_processing_new/analysis_per.py raw_videos/testVid.mp4
python video_processing_new/visualization.py raw_videos/testVid.mp4 --show-index
```

After the outputs look correct, use the full pipeline:

```bash
python video_processing_new/run_pipeline.py raw_videos/testVid.mp4 \
  --checkpoint models/best_model.pt
```

For frontend/backend integration, start the backend:

```bash
uvicorn backend.main:app --reload
```

Then upload a video through the web interface and confirm that the backend creates outputs under:

```text
dataset/<video_name>/
```

## Output Summary

| Stage | Main output |
|---|---|
| `preprocessing.py` | `frames/`, `frame_map.csv`, `roi_config.json`, `valid_mask.png` |
| `bounce.py` | `bounce_events.csv`, hit intervals, YOLO/person caches, debug CSVs, tracked players |
| `vote_action.py` | `stroke_vote_same_crop/stroke_vote_events.csv` |
| `analysis_per.py` | `events_from_bounce_hit.csv`, `stroke_landing_summary.csv`, `analysis_report.pdf`, `analysis_charts/` |
| `visualization.py` | `overlay_bounce_landings.mp4` |
| `training/train_tennis_action.py` | trained stroke classifier checkpoint |
| `backend/main.py` | API service for web integration |

## License

No license has been specified yet.
