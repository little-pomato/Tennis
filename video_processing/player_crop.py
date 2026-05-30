"""
make_player_crop_dataset.py

專門處理「照片資料集」的 player crop 程式。

功能：
1. 對資料夾內所有圖片做 YOLO person detection
2. 每張照片獨立處理，不使用 tracker
3. 從所有 person bbox 中選出「最大、最靠近鏡頭」的球員
4. 將 bbox 放大，盡量保留手臂與球拍
5. 保留原本資料夾結構輸出到新的 cropped dataset
6. 產生 detection_log.csv 與 failed_detection.txt，方便人工檢查

使用前提：
- 此檔案請和 player_detector.py 放在同一個資料夾
- player_detector.py 內已有 PlayerDetector class
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from player_detector import PlayerDetector


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def choose_nearest_largest_box(
    boxes,
    img_w,
    img_h,
    min_area_ratio=0.002,
    area_weight=1.0,
    y_weight=0.5,
    conf_weight=0.1,
):
    """
    從 YOLO 偵測到的所有 person boxes 中，選出最可能是「近端主要球員」的 bbox。

    判斷邏輯：
    - bbox 面積越大，通常代表越靠近鏡頭
    - bbox 中心點 y 越下面，通常代表越靠近鏡頭
    - confidence 越高稍微加分
    - 過濾掉太小的人，例如遠方觀眾、線審、背景人物

    boxes format:
        [(x1, y1, x2, y2, conf), ...]
    """
    if not boxes:
        return None

    img_area = img_w * img_h
    candidates = []

    for box in boxes:
        x1, y1, x2, y2, conf = box

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh

        if area <= 0:
            continue

        area_ratio = area / img_area

        # 太小的 person 多半不是主要球員
        if area_ratio < min_area_ratio:
            continue

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # normalize 到 0~1
        y_score = cy / img_h
        conf_score = conf

        score = (
            area_weight * area_ratio
            + y_weight * y_score
            + conf_weight * conf_score
        )

        candidates.append((score, box, area_ratio, y_score))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def expand_box(
    box,
    img_w,
    img_h,
    scale_w=1.7,
    scale_h=1.7,
    top_extra=0.05,
    bottom_extra=0.05,
):
    """
    放大 bbox，避免只裁到身體而裁掉球拍、手臂、擊球動作。

    top_extra / bottom_extra 是以原 bbox 高度為基準，額外往上/下補一點。
    """
    x1, y1, x2, y2, conf = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1

    new_w = bw * scale_w
    new_h = bh * scale_h

    nx1 = int(cx - new_w / 2)
    nx2 = int(cx + new_w / 2)

    ny1 = int(cy - new_h / 2 - bh * top_extra)
    ny2 = int(cy + new_h / 2 + bh * bottom_extra)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w - 1, nx2)
    ny2 = min(img_h - 1, ny2)

    return nx1, ny1, nx2, ny2


def save_debug_image(img, boxes, chosen_box, crop_box, out_path):
    """
    輸出 debug 圖：
    - 所有偵測到的人：藍框
    - 被選中的人：綠框
    - 最終 crop 範圍：紅框
    """
    debug = img.copy()

    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            debug,
            f"person {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if chosen_box is not None:
        x1, y1, x2, y2, conf = chosen_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            debug,
            "chosen",
            (x1, min(debug.shape[0] - 1, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            debug,
            "crop",
            (x1, max(0, y1 - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)


def process_photo_dataset(args):
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root 不存在：{input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    debug_root = output_root / "_debug"
    log_csv_path = output_root / "detection_log.csv"
    failed_txt_path = output_root / "failed_detection.txt"

    detector = PlayerDetector(
        model_path=args.model_path,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    image_paths = sorted([
        p for p in input_root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])

    print(f"[INFO] input_root  = {input_root}")
    print(f"[INFO] output_root = {output_root}")
    print(f"[INFO] found images = {len(image_paths)}")

    failed = []
    log_rows = []

    for img_path in tqdm(image_paths, desc="cropping photos"):
        rel_path = img_path.relative_to(input_root)
        out_path = output_root / rel_path

        # 避免 output_root 如果放在 input_root 裡面，重複處理輸出圖片
        if output_root in img_path.parents:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            failed.append(str(img_path))
            log_rows.append({
                "path": str(img_path),
                "status": "read_failed",
                "num_boxes": 0,
                "chosen_conf": "",
                "crop_x1": "",
                "crop_y1": "",
                "crop_x2": "",
                "crop_y2": "",
                "output_path": "",
            })
            continue

        img_h, img_w = img.shape[:2]

        boxes = detector.detect(img)

        chosen_box = choose_nearest_largest_box(
            boxes=boxes,
            img_w=img_w,
            img_h=img_h,
            min_area_ratio=args.min_area_ratio,
            area_weight=args.area_weight,
            y_weight=args.y_weight,
            conf_weight=args.conf_weight,
        )

        crop_box = None
        status = "ok"

        if chosen_box is None:
            failed.append(str(img_path))
            status = "no_person_detected"

            if args.on_fail == "copy":
                cv2.imwrite(str(out_path), img)
                output_saved = str(out_path)
            elif args.on_fail == "black":
                black = img.copy()
                black[:] = 0
                cv2.imwrite(str(out_path), black)
                output_saved = str(out_path)
            else:
                output_saved = ""

            log_rows.append({
                "path": str(img_path),
                "status": status,
                "num_boxes": len(boxes),
                "chosen_conf": "",
                "crop_x1": "",
                "crop_y1": "",
                "crop_x2": "",
                "crop_y2": "",
                "output_path": output_saved,
            })

            if args.debug:
                debug_path = debug_root / rel_path
                save_debug_image(img, boxes, chosen_box, crop_box, debug_path)

            continue

        crop_box = expand_box(
            chosen_box,
            img_w=img_w,
            img_h=img_h,
            scale_w=args.scale_w,
            scale_h=args.scale_h,
            top_extra=args.top_extra,
            bottom_extra=args.bottom_extra,
        )

        x1, y1, x2, y2 = crop_box
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            failed.append(str(img_path))
            status = "empty_crop"

            if args.on_fail == "copy":
                cv2.imwrite(str(out_path), img)
                output_saved = str(out_path)
            elif args.on_fail == "black":
                black = img.copy()
                black[:] = 0
                cv2.imwrite(str(out_path), black)
                output_saved = str(out_path)
            else:
                output_saved = ""
        else:
            if args.resize > 0:
                crop = cv2.resize(
                    crop,
                    (args.resize, args.resize),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imwrite(str(out_path), crop)
            output_saved = str(out_path)

        if args.debug:
            debug_path = debug_root / rel_path
            save_debug_image(img, boxes, chosen_box, crop_box, debug_path)

        log_rows.append({
            "path": str(img_path),
            "status": status,
            "num_boxes": len(boxes),
            "chosen_conf": chosen_box[4],
            "crop_x1": crop_box[0],
            "crop_y1": crop_box[1],
            "crop_x2": crop_box[2],
            "crop_y2": crop_box[3],
            "output_path": output_saved,
        })

    with open(log_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "status",
                "num_boxes",
                "chosen_conf",
                "crop_x1",
                "crop_y1",
                "crop_x2",
                "crop_y2",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    failed_txt_path.write_text("\n".join(failed), encoding="utf-8")

    print("[DONE]")
    print(f"[INFO] saved cropped dataset to: {output_root}")
    print(f"[INFO] log csv: {log_csv_path}")
    print(f"[INFO] failed txt: {failed_txt_path}")
    print(f"[INFO] failed count: {len(failed)}")

    if args.debug:
        print(f"[INFO] debug images: {debug_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Photo-only player crop dataset maker"
    )

    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="原始照片資料夾。可以是 dataset/images，也可以是包含 class folders 的根目錄。",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="輸出的 cropped dataset 資料夾。",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="yolov8n.pt",
        help="YOLO 模型路徑，預設使用 yolov8n.pt。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='auto / cpu / 0 / 1。和 player_detector.py 的 device 設定一致。',
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size。",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO person detection confidence threshold。",
    )

    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=0.002,
        help="過濾太小 person bbox 的門檻。bbox area / image area 小於此值就丟掉。",
    )

    parser.add_argument(
        "--scale_w",
        type=float,
        default=1.7,
        help="crop bbox 橫向放大倍率。建議 1.5~1.9。",
    )

    parser.add_argument(
        "--scale_h",
        type=float,
        default=1.7,
        help="crop bbox 縱向放大倍率。建議 1.5~1.9。",
    )

    parser.add_argument(
        "--top_extra",
        type=float,
        default=0.05,
        help="額外往上補的比例，以原 bbox 高度為基準。",
    )

    parser.add_argument(
        "--bottom_extra",
        type=float,
        default=0.05,
        help="額外往下補的比例，以原 bbox 高度為基準。",
    )

    parser.add_argument(
        "--area_weight",
        type=float,
        default=1.0,
        help="選主要球員時，bbox 面積分數權重。",
    )

    parser.add_argument(
        "--y_weight",
        type=float,
        default=0.5,
        help="選主要球員時，bbox 越下面的加分權重。",
    )

    parser.add_argument(
        "--conf_weight",
        type=float,
        default=0.1,
        help="選主要球員時，YOLO confidence 加分權重。",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="若 > 0，將 crop 後圖片 resize 成 resize x resize。建議先設 0，讓 train transform 自己 resize。",
    )

    parser.add_argument(
        "--on_fail",
        type=str,
        default="copy",
        choices=["copy", "skip", "black"],
        help="偵測不到人時怎麼處理：copy=保留原圖；skip=不輸出；black=輸出黑圖。",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="輸出 debug 圖，會畫出所有 bbox、選中的 bbox、最終 crop 範圍。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_photo_dataset(args)
