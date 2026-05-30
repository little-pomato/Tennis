from pathlib import Path
import csv, cv2, sys
import math
from tqdm import tqdm

FPS         = 20.0          # 固定抽幀的頻率（例：10 fps）
LONG_SIDE   = 640           # 縮圖後的長邊尺寸（例：640）
IMG_EXT     = "jpg"         # "jpg" 或 "png"
JPG_QUALITY = 90            # JPG 品質 1~100（建議 80~90）
OVERWRITE   = False         # 已處理過是否覆蓋重做
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

lastLineLen = 0

def make_dir(path:Path):
    path.mkdir(parents=True, exist_ok=True)
    
def resize_frame(frame, longSide:int):
    height, width = frame.shape[:2]
    curLongSide = max(height, width)
    if curLongSide <= longSide:
        return frame
    scale = longSide / curLongSide
    newHeight, newWidth = int(height * scale), int(width * scale)
    return cv2.resize(frame, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

def write_frame(outPath:Path, frame):
    if IMG_EXT.lower() == "jpg" or IMG_EXT.lower() == "jpeg":
        cv2.imwrite(str(outPath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    else:
        cv2.imwrite(str(outPath), frame)

def format_eta(sec: float) -> str:
    if sec is None or math.isinf(sec) or sec < 0:
        return "ETA --:--"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"ETA {h:d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"

def print_progress_inline(msg: str, end=False):
    global lastLineLen
    # 補空白把舊字蓋掉
    padded = msg if len(msg) >= lastLineLen else msg + " " * (lastLineLen - len(msg))
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()
    lastLineLen = len(msg)
    if end:
        sys.stdout.write("\n")
        sys.stdout.flush()
        lastLineLen = 0

# def extract_frames(videoPath: Path, outRoot: Path) -> Path:
#     # 直接使用 outRoot，不再自動附加 videoPath.stem
#     outDir = outRoot
#     make_dir(outDir)

#     mapCsv = outDir / "frame_map.csv"

#     if mapCsv.exists() and not OVERWRITE:
#         print(f"[Skip] {videoPath} already processed.")
#         return outDir

#     cap = cv2.VideoCapture(str(videoPath))
#     if not cap.isOpened():
#         print(f"[Error] Cannot open: {videoPath}")
#         return outDir

#     dtMs = 1000.0 / FPS
#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps_meta    = cap.get(cv2.CAP_PROP_FPS)

#     if frame_count > 0 and fps_meta and fps_meta > 1e-6:
#         duration_ms = (frame_count / fps_meta) * 1000.0
#         expected_total = int(max(1, math.floor(duration_ms / dtMs) + 1))
#     else:
#         expected_total = None

#     print(f"[INFO] Extracting frames from: {videoPath}")

#     tMs = 0.0
#     idx = 0

#     pbar = tqdm(total=expected_total, desc="Extracting Frames", unit="frame") if expected_total \
#            else tqdm(desc="Extracting Frames", unit="frame")

#     with open(mapCsv, "w", newline="", encoding="utf-8") as csvFile:
#         csvWriter = csv.writer(csvFile)
#         csvWriter.writerow(["index", "filename", "timestamp_ms"])

#         while True:
#             cap.set(cv2.CAP_PROP_POS_MSEC, tMs)
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frameSmall = resize_frame(frame, LONG_SIDE)

#             fileName = f"{idx:06d}.{IMG_EXT}"
#             outPath = outDir / fileName
#             write_frame(outPath, frameSmall)

#             csvWriter.writerow([idx, fileName, int(tMs)])

#             idx += 1
#             tMs += dtMs
#             pbar.update(1)

#     pbar.close()
#     cap.release()

#     print(f"[Done] {videoPath}: {idx} frames extracted.")
#     return outDir

def extract_frames(videoPath: Path, outRoot: Path) -> Path:
    outDir = outRoot
    make_dir(outDir)

    mapCsv = outDir / "frame_map.csv"

    if mapCsv.exists() and not OVERWRITE:
        print(f"[Skip] {videoPath} already processed.")
        return outDir

    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        print(f"[Error] Cannot open: {videoPath}")
        return outDir

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_meta = cap.get(cv2.CAP_PROP_FPS)

    if not fps_meta or fps_meta <= 1e-6:
        fps_meta = FPS

    dtMs = 1000.0 / FPS

    if frame_count > 0 and fps_meta > 1e-6:
        duration_ms = (frame_count / fps_meta) * 1000.0
        expected_total = int(max(1, math.floor(duration_ms / dtMs) + 1))
    else:
        expected_total = None

    print(f"[INFO] Extracting frames from: {videoPath}")
    print(f"[INFO] source fps = {fps_meta:.3f}, target fps = {FPS:.3f}")

    idx = 0
    source_frame_idx = 0
    next_tMs = 0.0

    pbar = tqdm(total=expected_total, desc="Extracting Frames", unit="frame") if expected_total \
           else tqdm(desc="Extracting Frames", unit="frame")

    with open(mapCsv, "w", newline="", encoding="utf-8") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["index", "filename", "timestamp_ms", "source_frame_idx"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_tMs = source_frame_idx * 1000.0 / fps_meta

            # 如果目前這一幀已經到達下一個抽樣時間點，就存下來
            if current_tMs + 1e-6 >= next_tMs:
                frameSmall = resize_frame(frame, LONG_SIDE)

                fileName = f"{idx:06d}.{IMG_EXT}"
                outPath = outDir / fileName
                write_frame(outPath, frameSmall)

                csvWriter.writerow([
                    idx,
                    fileName,
                    int(round(current_tMs)),
                    source_frame_idx
                ])

                idx += 1
                next_tMs += dtMs
                pbar.update(1)

            source_frame_idx += 1

    pbar.close()
    cap.release()

    print(f"[Done] {videoPath}: {idx} frames extracted.")
    return outDir