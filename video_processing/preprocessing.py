from pathlib import Path
import argparse
from frame_extractor import extract_frames
from pick_roi import run_pick_roi
import cv2, json
import numpy as np

INPUT_PATH = None
PRE_OUTPUT_DIR = Path("dataset")

def set_video_path(video_path: Path):
    global INPUT_PATH
    INPUT_PATH = video_path

def load_roi(roi_json_path):
    with open(roi_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ROI_POLY  = np.array(cfg["ROI_POLY"], dtype=np.int32)
    BAN_BOXES = [tuple(b) for b in cfg.get("BAN_BOXES", [])]
    FAR_WARP_SRC_PTS = np.array(cfg.get("FAR_WARP_SRC_PTS", []), dtype=np.float32)
    FAR_WARP_DST_SIZE= tuple(cfg.get("FAR_WARP_DST_SIZE", [640,360]))
    FAR_Y_RATIO_EDGE = float(cfg.get("FAR_Y_RATIO_EDGE", 0.5))
    MID_LINE = [tuple(p) for p in cfg.get("MID_LINE", [])]
    SINGLES_LINES = [tuple(p) for p in cfg.get("SINGLES_LINES", [])]
    return ROI_POLY, BAN_BOXES, FAR_WARP_SRC_PTS, FAR_WARP_DST_SIZE, FAR_Y_RATIO_EDGE, MID_LINE, SINGLES_LINES

def build_masks(h, w, ROI_POLY, BAN_BOXES):
    roi_mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(roi_mask, [ROI_POLY], 255)
    ban_mask = np.zeros((h,w), np.uint8)
    for (x,y,ww,hh) in BAN_BOXES:
        cv2.rectangle(ban_mask, (x,y), (x+ww, y+hh), 255, -1)
    valid = cv2.bitwise_and(roi_mask, cv2.bitwise_not(ban_mask))
    return valid

def preprocess(video_path, root_dir):
    video_path = Path(video_path)
    video_name = video_path.stem        # testVid

    # dataset/testVid/
    out_root = Path(root_dir) / video_name
    out_root.mkdir(exist_ok=True, parents=True)

    # dataset/testVid/frames/
    frames_dir = out_root / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"\n[1] Extracting frames to {frames_dir}")
    extract_frames(video_path, frames_dir)

    # pick ROI on first frame
    first_frame = sorted(frames_dir.glob("*.jpg"))[0]
    roi_json = out_root / "roi_config.json"

    print(f"\n[2] Running ROI picker on {first_frame}")
    run_pick_roi(first_frame, roi_json, auto=True)

    print(f"\n[3] Loading ROI + building mask")
    ROI_POLY, BAN_BOXES, *_ = load_roi(roi_json)
    sample = cv2.imread(str(first_frame))
    h, w = sample.shape[:2]
    valid_mask = build_masks(h, w, ROI_POLY, BAN_BOXES)

    mask_path = out_root / "valid_mask.png"
    cv2.imwrite(str(mask_path), valid_mask)
    print(f"[OK] Mask saved -> {mask_path}")

    print("\n[Preprocessing Completed]")
    return frames_dir, roi_json, mask_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    args = parser.parse_args()

    set_video_path(Path(args.video_path))
    
    preprocess(INPUT_PATH, PRE_OUTPUT_DIR)