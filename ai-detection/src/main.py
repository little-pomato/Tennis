import argparse
import torch
import os
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
    args = parser.parse_args()

    # 1. Initialize Context
    print(f"Loading video: {args.input}")
    frame_paths, fps, width, height = read_video_to_disk(args.input, max_long_side=640, target_fps=20)
    context = VideoContext(frame_paths=frame_paths, fps=fps, width=width, height=height)

    # 2. Build and Run Pipeline
    if args.load_results and os.path.exists(args.load_results):
        print(f"Skipping inference. Loading results from {args.load_results}...")
        context = DataExporter.load_to_context(args.load_results, context)
    else:
        model_cfg = ModelConfig()
        pipeline = Pipeline()
        
        print("Initializing pipeline nodes...")
        pipeline.add_node(CourtDetector(model_cfg.court_model_path, device=args.device))
        pipeline.add_node(BallDetector(model_cfg.ball_model_path, device=args.device))
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
    
    print("Done!")

if __name__ == "__main__":
    main()
