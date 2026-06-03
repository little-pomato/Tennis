import argparse
import torch
import os
import shutil
from pathlib import Path
from time import perf_counter
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
from src.exporters.video import VideoExporter
from src.exporters.data import DataExporter

def main():
    parser = argparse.ArgumentParser(description="Modular Tennis Detection Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("--output_json", type=str, default="output.json", help="Path to output JSON data")
    parser.add_argument("--load_results", type=str, help="Path to existing JSON results to skip inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), help="Device to run on")
    parser.add_argument("--target_fps", type=int, default=20, help="Frame rate to sample for analysis")
    parser.add_argument("--max_long_side", type=int, default=640, help="Resize input video so the long side is at most this size")
    parser.add_argument("--ball_batch_size", type=int, help="Override TrackNet batch size")
    parser.add_argument("--frame_cache_size", type=int, default=96, help="Number of decoded frames to keep in memory")
    parser.add_argument("--resized_cache_size", type=int, default=128, help="Number of resized frames to keep in memory")
    parser.add_argument("--keep_frames", action="store_true", help="Keep extracted frames after processing")
    args = parser.parse_args()

    # 1. Initialize Context
    print(f"Loading video: {args.input}")
    frames_dir = Path("temp_frames")
    started = perf_counter()
    frame_paths, fps, width, height = read_video_to_disk(
        args.input,
        temp_dir=str(frames_dir),
        max_long_side=args.max_long_side,
        target_fps=args.target_fps,
    )
    context = VideoContext(
        frame_paths=frame_paths,
        fps=fps,
        width=width,
        height=height,
        frame_cache_size=args.frame_cache_size,
        resized_cache_size=args.resized_cache_size,
    )
    context.timings["FrameExtraction"] = perf_counter() - started

    # 2. Build and Run Pipeline
    if args.load_results and os.path.exists(args.load_results):
        print(f"Skipping inference. Loading results from {args.load_results}...")
        context = DataExporter.load_to_context(args.load_results, context)
    else:
        model_cfg = ModelConfig()
        pipeline = Pipeline()
        
        print("Initializing pipeline nodes...")
        pipeline.add_node(CourtDetector(model_cfg.court_model_path, device=args.device))
        pipeline.add_node(BallDetector(model_cfg.ball_model_path, device=args.device, batch_size=args.ball_batch_size))
        pipeline.add_node(TrajectoryRefiner())
        pipeline.add_node(BounceDetector(model_cfg.bounce_model_path))
        pipeline.add_node(PlayerDetector(device=args.device))
        pipeline.add_node(InoutRuleEngine())
        pipeline.add_node(MatchAnalyzer(model_cfg.stroke_model_path, device=args.device))

        print("Running pipeline...")
        context = pipeline.run(context)
        
        print(f"Saving results to {args.output_json}...")
        DataExporter(args.output_json).export(context)

    # 3. Export Visuals
    print(f"Exporting video to {args.output_video}...")
    VideoExporter(args.output_video).export(context)

    if not args.keep_frames:
        context.clear_frame_cache()
        shutil.rmtree(frames_dir, ignore_errors=True)
    
    print("Done!")

if __name__ == "__main__":
    main()
