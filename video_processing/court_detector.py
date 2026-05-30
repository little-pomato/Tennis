import cv2
import numpy as np
import os
import sys
import math
import json
from pathlib import Path

# =========================================================
# utils
# =========================================================
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_detector_paths(input_path, output_dir=None):
    """
    類似 court_mapper / bounce.py 的讀檔方式：
    raw_videos/testVid.mp4 ->
        dataset/testVid/
        dataset/testVid/frames/
        dataset/testVid/roi_config.json
        dataset/testVid/court_detector_debug/
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in VIDEO_EXTS:
        video_name = input_path.stem
        base = Path("dataset") / video_name

        return {
            "INPUT_PATH": input_path,
            "VIDEO_NAME": video_name,
            "BASE": base,
            "FRAMES_DIR": base / "frames",
            "ROI_JSON": base / "roi_config.json",
            "OUT_DIR": Path(output_dir) if output_dir is not None else base / "court_detector_debug",
            "IS_VIDEO": True,
        }

    else:
        out_dir = Path(output_dir) if output_dir is not None else Path("court_detector_debug")

        return {
            "INPUT_PATH": input_path,
            "VIDEO_NAME": input_path.stem,
            "BASE": out_dir,
            "FRAMES_DIR": None,
            "ROI_JSON": out_dir / "roi_config.json",
            "OUT_DIR": out_dir,
            "IS_VIDEO": False,
        }


def find_first_frame(frames_dir: Path):
    """
    從 dataset/<video_name>/frames 裡找第一張 frame。
    """
    if frames_dir is None or not frames_dir.exists():
        return None

    frame_paths = sorted(
        list(frames_dir.glob("*.jpg")) +
        list(frames_dir.glob("*.jpeg")) +
        list(frames_dir.glob("*.png"))
    )

    if not frame_paths:
        return None

    return frame_paths[0]


def read_image_or_first_frame(input_path, frames_dir=None):
    """
    若 input 是圖片：直接讀圖片。
    若 input 是影片：
      1. 優先讀 dataset/<video_name>/frames 的第一張圖
      2. 如果沒有 frames，就直接從 mp4 抓第一幀
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        img = cv2.imread(str(input_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")
        return img, input_path

    if suffix in VIDEO_EXTS:
        first_frame_path = find_first_frame(frames_dir)

        if first_frame_path is not None:
            img = cv2.imread(str(first_frame_path))
            if img is not None:
                return img, first_frame_path

        cap = cv2.VideoCapture(str(input_path))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise FileNotFoundError(f"Cannot read video or first frame: {input_path}")

        return frame, input_path

    raise ValueError(f"Unsupported input type: {input_path}")

def normalize_angle_deg(angle):
    while angle < 0:
        angle += 180
    while angle >= 180:
        angle -= 180
    return angle


def angle_diff_deg(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def seg_length(seg):
    return math.hypot(seg["x2"] - seg["x1"], seg["y2"] - seg["y1"])


def seg_mid(seg):
    return ((seg["x1"] + seg["x2"]) / 2.0, (seg["y1"] + seg["y2"]) / 2.0)


def line_coeff(seg):
    x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    n = math.hypot(a, b)
    if n < 1e-8:
        return None
    return a / n, b / n, c / n


def point_line_dist(px, py, seg):
    coeff = line_coeff(seg)
    if coeff is None:
        return 1e9
    a, b, c = coeff
    return abs(a * px + b * py + c)


def line_intersection(seg1, seg2):
    x1, y1, x2, y2 = seg1["x1"], seg1["y1"], seg1["x2"], seg1["y2"]
    x3, y3, x4, y4 = seg2["x1"], seg2["y1"], seg2["x2"], seg2["y2"]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def clip_pt(pt, w, h):
    x, y = pt
    return (int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1))))


def draw_seg(img, seg, color, thickness=2):
    cv2.line(
        img,
        (int(seg["x1"]), int(seg["y1"])),
        (int(seg["x2"]), int(seg["y2"])),
        color,
        thickness,
    )


def draw_segs(img, segs, color, thickness=2):
    out = img.copy()
    for s in segs:
        draw_seg(out, s, color, thickness)
    return out


def fit_line_through_points(points, out_w, out_h):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return None

    if abs(vx) >= abs(vy):
        x_left = 0
        y_left = y0 + (x_left - x0) * (vy / max(vx, 1e-8))
        x_right = out_w - 1
        y_right = y0 + (x_right - x0) * (vy / max(vx, 1e-8))
        p1 = (x_left, y_left)
        p2 = (x_right, y_right)
    else:
        y_top = 0
        x_top = x0 + (y_top - y0) * (vx / max(vy, 1e-8))
        y_bot = out_h - 1
        x_bot = x0 + (y_bot - y0) * (vx / max(vy, 1e-8))
        p1 = (x_top, y_top)
        p2 = (x_bot, y_bot)

    angle = normalize_angle_deg(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    return {
        "x1": int(round(p1[0])), "y1": int(round(p1[1])),
        "x2": int(round(p2[0])), "y2": int(round(p2[1])),
        "angle_deg": angle,
        "length": float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])),
    }


# =========================================================
# preprocess
# =========================================================
def build_static_ignore_mask(img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Broadcast scoreboards usually live in the upper corners. Keep the top
    # center available because the far baseline often crosses that area.
    corner_w = int(0.22 * w)
    corner_h = int(0.18 * h)
    cv2.rectangle(mask, (0, 0), (corner_w, corner_h), 255, -1)
    cv2.rectangle(mask, (w - corner_w, 0), (w - 1, corner_h), 255, -1)
    return mask


def preprocess(img_bgr, ignore_mask=None):
    h, w = img_bgr.shape[:2]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    eq = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [int(0.30 * w), int(0.08 * h)],
        [int(0.70 * w), int(0.08 * h)],
        [int(0.92 * w), int(0.92 * h)],
        [int(0.08 * w), int(0.92 * h)],
    ], dtype=np.int32)
    cv2.fillConvexPoly(roi, pts, 255)
    roi[int(0.94 * h):, :] = 0
    if ignore_mask is not None:
        roi = cv2.bitwise_and(roi, cv2.bitwise_not(ignore_mask))

    edges_roi = cv2.bitwise_and(edges, roi)

    hsv = cv2.cvtColor(eq, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (180, 110, 255))
    white = cv2.bitwise_and(white, roi)

    return eq, gray, blur, edges, roi, edges_roi, white


# =========================================================
# line detection / merge
# =========================================================
def detect_segments(edges_roi, img_shape):
    h, w = img_shape[:2]
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(25, int(0.04 * w)),
        maxLineGap=max(12, int(0.02 * w)),
    )

    if lines is None:
        return []

    segs = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, l)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(20, 0.03 * w):
            continue
        angle = normalize_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        segs.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "angle_deg": angle,
            "length": float(length),
        })
    return segs


def merge_two(base, seg):
    pts = np.array([
        [base["x1"], base["y1"]], [base["x2"], base["y2"]],
        [seg["x1"], seg["y1"]], [seg["x2"], seg["y2"]],
    ], dtype=np.float32)

    ref = base if base["length"] >= seg["length"] else seg
    dx = ref["x2"] - ref["x1"]
    dy = ref["y2"] - ref["y1"]
    n = math.hypot(dx, dy)
    if n < 1e-8:
        return base
    ux, uy = dx / n, dy / n
    proj = pts[:, 0] * ux + pts[:, 1] * uy
    p1 = pts[np.argmin(proj)]
    p2 = pts[np.argmax(proj)]

    return {
        "x1": int(round(p1[0])), "y1": int(round(p1[1])),
        "x2": int(round(p2[0])), "y2": int(round(p2[1])),
        "angle_deg": ref["angle_deg"],
        "length": float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])),
    }


def merge_segments(segs, angle_thresh=4.0, dist_thresh=14, gap_thresh=30):
    if not segs:
        return []

    work = sorted(segs, key=lambda s: s["length"], reverse=True)
    merged = []

    for seg in work:
        attached = False
        for i, base in enumerate(merged):
            if angle_diff_deg(seg["angle_deg"], base["angle_deg"]) > angle_thresh:
                continue

            mx, my = seg_mid(seg)
            d = point_line_dist(mx, my, base)
            if d > dist_thresh:
                continue

            endpoints1 = [(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])]
            endpoints2 = [(base["x1"], base["y1"]), (base["x2"], base["y2"])]
            min_gap = min(math.hypot(x1 - x2, y1 - y2) for (x1, y1) in endpoints1 for (x2, y2) in endpoints2)
            if min_gap > gap_thresh and d > dist_thresh / 2.0:
                continue

            merged[i] = merge_two(base, seg)
            attached = True
            break

        if not attached:
            merged.append(seg)

    return merged


# =========================================================
# orientation split
# =========================================================
def split_horizontal_vertical(segs, h_tol=20, v_tol=100):
    horizontal = []
    vertical = []
    other = []

    for s in segs:
        a = s["angle_deg"] % 180
        dh = min(abs(a - 0), abs(a - 180))
        dv = abs(a - 90)

        if dh <= h_tol:
            horizontal.append(s)
        elif dv <= v_tol:
            vertical.append(s)
        else:
            other.append(s)

    return horizontal, vertical, other


# =========================================================
# candidate banding / fitting
# =========================================================
def cluster_by_coordinate(segs, axis="x", tol=18):
    if not segs:
        return []

    values = []
    for s in segs:
        mx, my = seg_mid(s)
        values.append(mx if axis == "x" else my)

    order = np.argsort(values)
    clusters = []
    current = [segs[order[0]]]
    current_vals = [values[order[0]]]

    for idx in order[1:]:
        v = values[idx]
        mean_v = float(np.mean(current_vals))
        if abs(v - mean_v) <= tol:
            current.append(segs[idx])
            current_vals.append(v)
        else:
            clusters.append(current)
            current = [segs[idx]]
            current_vals = [v]
    clusters.append(current)
    return clusters


def fit_cluster_to_line(cluster, img_w, img_h):
    pts = []
    for s in cluster:
        pts.append([s["x1"], s["y1"]])
        pts.append([s["x2"], s["y2"]])
    return fit_line_through_points(pts, img_w, img_h)


# =========================================================
# choose 5 main lines
# =========================================================
def choose_vertical_five(vertical_group, img_w, img_h):
    if len(vertical_group) < 4:
        return None

    clusters = cluster_by_coordinate(vertical_group, axis="x", tol=max(14, int(0.02 * img_w)))
    candidates = []

    for cl in clusters:
        line = fit_cluster_to_line(cl, img_w, img_h)
        if line is None:
            continue
        mx, my = seg_mid(line)
        total_len = sum(s["length"] for s in cl)
        angle_penalty = abs(angle_diff_deg(line["angle_deg"], 90))
        score = total_len - 8.0 * angle_penalty
        candidates.append({
            "line": line,
            "mx": mx,
            "my": my,
            "total_len": total_len,
            "score": score,
            "cluster": cl,
        })

    if len(candidates) < 4:
        return None

    candidates.sort(key=lambda c: c["mx"])

    n = len(candidates)
    best = None
    best_score = -1e18

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    chosen = [candidates[i], candidates[j], candidates[k], candidates[m]]
                    xs = [c["mx"] for c in chosen]
                    x1, x2, x3, x4 = xs
                    d12 = x2 - x1
                    d23 = x3 - x2
                    d34 = x4 - x3
                    if min(d12, d23, d34) < 0.03 * img_w:
                        continue

                    symmetry_penalty = abs(d12 - d34)
                    ratio_penalty = abs((d12 + d34) - 0.66 * d23)
                    total_len = sum(c["total_len"] for c in chosen)

                    boundary_penalty = 0.0
                    boundary_penalty += max(0, 0.08 * img_w - x1) * 2.0
                    boundary_penalty += max(0, x4 - 0.92 * img_w) * 2.0

                    score = total_len - 2.0 * symmetry_penalty - 1.4 * ratio_penalty - boundary_penalty
                    if score > best_score:
                        best_score = score
                        best = chosen

    if best is None:
        return None

    return {
        "left_doubles": best[0]["line"],
        "left_singles": best[1]["line"],
        "right_singles": best[2]["line"],
        "right_doubles": best[3]["line"],
    }

def point_in_reasonable_image_range(pt, img_w, img_h, margin=0.25):
    if pt is None:
        return False
    x, y = pt
    return (-margin * img_w <= x <= (1.0 + margin) * img_w and
            -margin * img_h <= y <= (1.0 + margin) * img_h)


def safe_intersection(line1, line2, img_w, img_h, margin=0.25):
    pt = line_intersection(line1, line2)
    if not point_in_reasonable_image_range(pt, img_w, img_h, margin=margin):
        return None
    return pt


def horizontal_angle_penalty(seg):
    a = seg["angle_deg"] % 180
    return min(abs(a - 0), abs(a - 180))


def top_corner_penalty(seg, img_w, img_h):
    mx, my = seg_mid(seg)
    in_top = my < 0.18 * img_h
    in_left = mx < 0.22 * img_w
    in_right = mx > 0.78 * img_w
    if in_top and (in_left or in_right):
        return 120.0
    return 0.0

def point_projection_ratio_on_seg(pt, seg):
    x1, y1 = seg["x1"], seg["y1"]
    x2, y2 = seg["x2"], seg["y2"]
    px, py = pt

    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom < 1e-8:
        return None

    t = ((px - x1) * vx + (py - y1) * vy) / denom
    return float(t)


def line_through_point_with_angle(pt, angle_deg, img_w, img_h):
    px, py = pt
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None

    candidates = []
    if abs(dx) > 1e-8:
        for x in (0.0, float(img_w - 1)):
            t = (x - px) / dx
            y = py + t * dy
            if -img_h <= y <= 2 * img_h:
                candidates.append((x, y))
    if abs(dy) > 1e-8:
        for y in (0.0, float(img_h - 1)):
            t = (y - py) / dy
            x = px + t * dx
            if -img_w <= x <= 2 * img_w:
                candidates.append((x, y))

    if len(candidates) < 2:
        scale = max(img_w, img_h)
        candidates = [(px - dx * scale, py - dy * scale), (px + dx * scale, py + dy * scale)]

    p1 = candidates[0]
    p2 = max(candidates[1:], key=lambda p: math.hypot(p[0] - p1[0], p[1] - p1[1]))
    angle = normalize_angle_deg(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    return {
        "x1": int(round(p1[0])), "y1": int(round(p1[1])),
        "x2": int(round(p2[0])), "y2": int(round(p2[1])),
        "angle_deg": angle,
        "length": float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])),
    }


def make_segment_from_points(p1, p2):
    angle = normalize_angle_deg(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
    return {
        "x1": float(p1[0]), "y1": float(p1[1]),
        "x2": float(p2[0]), "y2": float(p2[1]),
        "angle_deg": angle,
        "length": float(math.hypot(p2[0] - p1[0], p2[1] - p1[1])),
    }


def clipped_segment_between_lines(base_line, left_line, right_line, img_w, img_h, margin=0.35):
    left_pt = safe_intersection(base_line, left_line, img_w, img_h, margin=margin)
    right_pt = safe_intersection(base_line, right_line, img_w, img_h, margin=margin)
    if left_pt is None or right_pt is None:
        return None, left_pt, right_pt
    if left_pt[0] > right_pt[0]:
        left_pt, right_pt = right_pt, left_pt
    return make_segment_from_points(left_pt, right_pt), left_pt, right_pt


def segment_support_metrics(seg, edges_roi, white_roi, thickness=7):
    band = line_band_mask(edges_roi.shape, seg, thickness=thickness)
    band_pixels = max(1, int(np.count_nonzero(band)))
    edge_hits = int(np.count_nonzero(cv2.bitwise_and(band, edges_roi)))
    white_hits = int(np.count_nonzero(cv2.bitwise_and(band, white_roi)))
    return {
        "band_pixels": band_pixels,
        "edge_hits": edge_hits,
        "white_hits": white_hits,
        "edge_ratio": edge_hits / band_pixels,
        "white_ratio": white_hits / band_pixels,
        "support_ratio": (edge_hits + 1.4 * white_hits) / band_pixels,
    }


def outside_clipped_support_ratio(line, clipped_seg, edges_roi, white_roi):
    full_band = line_band_mask(edges_roi.shape, line, thickness=7)
    clipped_band = line_band_mask(edges_roi.shape, clipped_seg, thickness=9)
    outside = cv2.bitwise_and(full_band, cv2.bitwise_not(clipped_band))
    outside_pixels = max(1, int(np.count_nonzero(outside)))
    edge_hits = int(np.count_nonzero(cv2.bitwise_and(outside, edges_roi)))
    white_hits = int(np.count_nonzero(cv2.bitwise_and(outside, white_roi)))
    return (edge_hits + 1.4 * white_hits) / outside_pixels

def score_service_line_candidate(line, cluster, vertical_five, img_w, img_h):
    ld = vertical_five["left_doubles"]
    ls = vertical_five["left_singles"]
    cs = vertical_five["center_service"]
    rs = vertical_five["right_singles"]
    rd = vertical_five["right_doubles"]

    mx, my = seg_mid(line)
    total_len = sum(s["length"] for s in cluster)

    # --- 1) 先算最重要的 T 字與單打線交點 ---
    inter_ls = safe_intersection(line, ls, img_w, img_h)
    inter_cs = safe_intersection(line, cs, img_w, img_h)
    inter_rs = safe_intersection(line, rs, img_w, img_h)

    if None in (inter_ls, inter_cs, inter_rs):
        return None

    # 交點的左右順序必須合理：ls < cs < rs
    if not (inter_ls[0] < inter_cs[0] < inter_rs[0]):
        return None

    left_half = inter_cs[0] - inter_ls[0]
    right_half = inter_rs[0] - inter_cs[0]
    inner_width = inter_rs[0] - inter_ls[0]

    if left_half < 0.04 * img_w or right_half < 0.04 * img_w:
        return None

    if inner_width < 0.10 * img_w:
        return None

    # T 字對稱性：中線到左右單打線距離應差不多
    t_sym_penalty = abs(left_half - right_half)

    # 候選水平線必須夠水平
    horiz_pen = horizontal_angle_penalty(line)

    # --- 2) 再看 doubles 外框，當次要約束 ---
    inter_ld = safe_intersection(line, ld, img_w, img_h)
    inter_rd = safe_intersection(line, rd, img_w, img_h)

    outer_width = None
    outer_sym_penalty = 0.0
    if inter_ld is not None and inter_rd is not None and inter_ld[0] < inter_rd[0]:
        outer_width = inter_rd[0] - inter_ld[0]
        if outer_width < 0.15 * img_w:
            return None

        # doubles 相對中線也希望左右大致對稱
        left_outer = inter_cs[0] - inter_ld[0]
        right_outer = inter_rd[0] - inter_cs[0]
        outer_sym_penalty = abs(left_outer - right_outer)

    # --- 3) 依照影像中的真正上端 / 下端 來判斷 far / near ---
    t_cs = point_projection_ratio_on_seg(inter_cs, cs)
    if t_cs is None:
        return None

    # 先判斷 cs 的哪一端是影像上端、哪一端是影像下端
    # top_t 對應影像較小 y 的那一端
    # bot_t 對應影像較大 y 的那一端
    if cs["y1"] <= cs["y2"]:
        top_t = 0.0
        bot_t = 1.0
    else:
        top_t = 1.0
        bot_t = 0.0

    far_err = abs(t_cs - top_t)   # far 應靠近影像上端
    near_err = abs(t_cs - bot_t)  # near 應靠近影像下端

    endpoint_tol = 0.22
    if min(far_err, near_err) > endpoint_tol:
        return None

    if far_err < near_err:
        service_side = "far"
        endpoint_penalty = far_err
    else:
        service_side = "near"
        endpoint_penalty = near_err

    # --- 4) 避免太靠近畫面極端 ---
    if my < 0.05 * img_h or my > 0.95 * img_h:
        return None

    # --- 5) 綜合分數 ---
    score = (
        1.2 * total_len
        - 10.0 * horiz_pen
        - 1.6 * t_sym_penalty
        - 0.7 * outer_sym_penalty
        + 0.4 * inner_width
        - 180.0 * endpoint_penalty
        - top_corner_penalty(line, img_w, img_h)
    )
    
    # print(
    #     f"service cand | y={my:.1f} t={t_cs:.3f} "
    #     f"top_t={top_t:.1f} bot_t={bot_t:.1f} side={service_side}"
    # )

    return {
        "line": line,
        "cluster": cluster,
        "score": score,
        "service_side": service_side,   # "far" or "near"
        "mx": mx,
        "my": my,
        "t_point": inter_cs,            # 和 center_service 的交點
        "intersections": {
            "ld": inter_ld,
            "ls": inter_ls,
            "cs": inter_cs,
            "rs": inter_rs,
            "rd": inter_rd,
        },
        "metrics": {
            "total_len": total_len,
            "horiz_pen": horiz_pen,
            "t_sym_penalty": t_sym_penalty,
            "outer_sym_penalty": outer_sym_penalty,
            "inner_width": inner_width,
            "outer_width": outer_width,
        }
    }
    
def choose_service_lines(horizontal_group, vertical_five, img_w, img_h, debug=False):
    if not horizontal_group or vertical_five is None:
        return None

    cs = vertical_five["center_service"]
    if cs["y1"] <= cs["y2"]:
        top_t = 0.0
        bot_t = 1.0
    else:
        top_t = 1.0
        bot_t = 0.0

    clusters = cluster_by_coordinate(horizontal_group, axis="y", tol=max(12, int(0.018 * img_h)))
    candidates = []

    for idx, cl in enumerate(clusters):
        line = fit_cluster_to_line(cl, img_w, img_h)
        if line is None:
            continue

        cand = score_service_line_candidate(line, cl, vertical_five, img_w, img_h)
        if cand is None:
            continue

        cand["cluster_id"] = idx
        candidates.append(cand)

    if not candidates:
        return None

    far_candidates = [c for c in candidates if c["service_side"] == "far"]
    near_candidates = [c for c in candidates if c["service_side"] == "near"]

    if not far_candidates or not near_candidates:
        if near_candidates:
            best_near = max(near_candidates, key=lambda c: c["score"])
            top_pt, _ = seg_endpoints_top_bottom(cs)
            inferred_far = line_through_point_with_angle(top_pt, best_near["line"]["angle_deg"], img_w, img_h)
            if inferred_far is not None:
                return {
                    "far_service": inferred_far,
                    "near_service": best_near["line"],
                    "far_candidate": {
                        "line": inferred_far,
                        "cluster": [],
                        "score": best_near["score"] * 0.45,
                        "service_side": "far",
                        "source": "projected",
                        "mx": seg_mid(inferred_far)[0],
                        "my": seg_mid(inferred_far)[1],
                    },
                    "near_candidate": best_near,
                    "all_candidates": candidates,
                    "line_sources": {
                        "far_service": "projected",
                        "near_service": "detected",
                    },
                    "warnings": ["far_service_projected_from_center_service_endpoint"],
                }
        return None

    best_pair = None
    best_pair_score = -1e18

    for f in far_candidates:
        for n in near_candidates:
            # far 必須在 near 上方
            if not (f["my"] < n["my"]):
                continue

            vertical_gap = n["my"] - f["my"]
            if vertical_gap < 0.08 * img_h:
                continue

            t_far = point_projection_ratio_on_seg(f["t_point"], cs)
            t_near = point_projection_ratio_on_seg(n["t_point"], cs)
            if t_far is None or t_near is None:
                continue

            far_err = abs(t_far - top_t)
            near_err = abs(t_near - bot_t)

            # 一定要靠近真正的上/下端
            if far_err > 0.18:
                continue
            if near_err > 0.18:
                continue

            width_consistency = abs(
                f["metrics"]["inner_width"] - n["metrics"]["inner_width"]
            )

            pair_score = (
                f["score"] + n["score"]
                - 1.5 * width_consistency
                - 120.0 * far_err
                - 120.0 * near_err
            )

            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (f, n)
        
    if best_pair is None:
        return None

    best_far, best_near = best_pair
    return {
        "far_service": best_far["line"],
        "near_service": best_near["line"],
        "far_candidate": best_far,
        "near_candidate": best_near,
        "all_candidates": candidates,
        "line_sources": {
            "far_service": "detected",
            "near_service": "detected",
        },
        "warnings": [],
    }
    
def save_service_lines_debug(img, vertical_five, service_lines, out_path):
    out = img.copy()

    draw_named_seg(out, vertical_five["left_doubles"],   (0, 0, 255),   "left_doubles")
    draw_named_seg(out, vertical_five["left_singles"],   (0, 255, 0),   "left_singles")
    draw_named_seg(out, vertical_five["center_service"], (255, 0, 255), "center_service")
    draw_named_seg(out, vertical_five["right_singles"],  (255, 0, 0),   "right_singles")
    draw_named_seg(out, vertical_five["right_doubles"],  (0, 255, 255), "right_doubles")

    draw_named_seg(out, service_lines["far_service"],    (255, 255, 255), "far_service", 2)
    draw_named_seg(out, service_lines["near_service"],   (180, 255, 180), "near_service", 2)

    cv2.imwrite(out_path, out)

def score_baseline_candidate(line, cluster, side, vertical_five, service_lines, H_hint, model, img_w, img_h, edges_roi, white_roi):
    ld = vertical_five["left_doubles"]
    ls = vertical_five["left_singles"]
    rs = vertical_five["right_singles"]
    rd = vertical_five["right_doubles"]
    far_service = service_lines["far_service"]
    near_service = service_lines["near_service"]

    inter_ld = safe_intersection(line, ld, img_w, img_h)
    inter_ls = safe_intersection(line, ls, img_w, img_h)
    inter_rs = safe_intersection(line, rs, img_w, img_h)
    inter_rd = safe_intersection(line, rd, img_w, img_h)
    if None in (inter_ld, inter_ls, inter_rs, inter_rd):
        return None

    if not (inter_ld[0] < inter_ls[0] < inter_rs[0] < inter_rd[0]):
        return None

    inner_width = inter_rs[0] - inter_ls[0]
    outer_width = inter_rd[0] - inter_ld[0]
    if inner_width < 0.10 * img_w or outer_width < 0.15 * img_w:
        return None

    clipped_outer, _, _ = clipped_segment_between_lines(line, ld, rd, img_w, img_h)
    clipped_inner, _, _ = clipped_segment_between_lines(line, ls, rs, img_w, img_h)
    if clipped_outer is None or clipped_inner is None:
        return None

    alley_l = inter_ls[0] - inter_ld[0]
    alley_r = inter_rd[0] - inter_rs[0]
    alley_penalty = abs(alley_l - alley_r)
    horiz_pen = horizontal_angle_penalty(line)
    total_len = sum(s["length"] for s in cluster)
    mx, my = seg_mid(line)

    model_name = f"{side}_baseline"
    p1, p2 = model["lines_dict"][model_name]
    pred = project_points(H_hint, [p1, p2])
    pred_line = {
        "x1": float(pred[0][0]), "y1": float(pred[0][1]),
        "x2": float(pred[1][0]), "y2": float(pred[1][1]),
        "angle_deg": normalize_angle_deg(math.degrees(math.atan2(pred[1][1]-pred[0][1], pred[1][0]-pred[0][0]))),
        "length": float(math.hypot(pred[1][0]-pred[0][0], pred[1][1]-pred[0][1])),
    }
    pred_mx, pred_my = seg_mid(pred_line)

    y_penalty = abs(my - pred_my)
    x_penalty = abs(mx - pred_mx)
    angle_match_penalty = angle_diff_deg(line["angle_deg"], pred_line["angle_deg"])

    far_service_y = seg_mid(far_service)[1]
    near_service_y = seg_mid(near_service)[1]
    if side == "far":
        if my >= far_service_y:
            return None
        service_gap = far_service_y - my
        min_gap = 0.065 * img_h
        max_gap = 0.28 * img_h
        if service_gap < min_gap or service_gap > max_gap:
            return None
    else:
        if my <= near_service_y:
            return None
        service_gap = my - near_service_y
        min_gap = 0.075 * img_h
        max_gap = 0.35 * img_h
        if service_gap < min_gap or service_gap > max_gap:
            return None

    support = segment_support_metrics(clipped_outer, edges_roi, white_roi, thickness=7)
    outside_support = outside_clipped_support_ratio(line, clipped_outer, edges_roi, white_roi)
    outside_penalty = max(0.0, outside_support - support["support_ratio"]) * 260.0

    inter_pred_ld = safe_intersection(pred_line, ld, img_w, img_h)
    inter_pred_rd = safe_intersection(pred_line, rd, img_w, img_h)
    width_penalty = 0.0
    if inter_pred_ld is not None and inter_pred_rd is not None and inter_pred_ld[0] < inter_pred_rd[0]:
        pred_outer = inter_pred_rd[0] - inter_pred_ld[0]
        width_penalty = abs(outer_width - pred_outer)

    score = (
        0.18 * min(total_len, clipped_outer["length"] * 1.35)
        + 360.0 * support["support_ratio"]
        + 0.35 * clipped_outer["length"]
        - 11.0 * horiz_pen
        - 1.2 * alley_penalty
        - 2.6 * y_penalty
        - 0.25 * x_penalty
        - 12.0 * angle_match_penalty
        - 1.15 * width_penalty
        - outside_penalty
        - top_corner_penalty(line, img_w, img_h)
    )

    return {
        "line": line,
        "cluster": cluster,
        "score": score,
        "side": side,
        "mx": mx,
        "my": my,
        "intersections": {
            "ld": inter_ld,
            "ls": inter_ls,
            "rs": inter_rs,
            "rd": inter_rd,
        },
        "clipped_outer": clipped_outer,
        "clipped_inner": clipped_inner,
        "pred_line": pred_line,
        "metrics": {
            "total_len": total_len,
            "inner_width": inner_width,
            "outer_width": outer_width,
            "clipped_outer_length": clipped_outer["length"],
            "alley_penalty": alley_penalty,
            "horiz_pen": horiz_pen,
            "y_penalty": y_penalty,
            "x_penalty": x_penalty,
            "angle_match_penalty": angle_match_penalty,
            "width_penalty": width_penalty,
            "service_gap": service_gap,
            "edge_ratio": support["edge_ratio"],
            "white_ratio": support["white_ratio"],
            "support_ratio": support["support_ratio"],
            "outside_support_ratio": outside_support,
            "outside_penalty": outside_penalty,
        }
    }


def choose_baselines(horizontal_group, vertical_five, service_lines, H_hint, model, img_w, img_h, edges_roi, white_roi):
    if not horizontal_group or vertical_five is None or service_lines is None or H_hint is None:
        return None

    far_service = service_lines["far_service"]
    near_service = service_lines["near_service"]
    far_service_y = seg_mid(far_service)[1]
    near_service_y = seg_mid(near_service)[1]

    pred_far = project_points(H_hint, list(model["lines_dict"]["far_baseline"]))
    pred_near = project_points(H_hint, list(model["lines_dict"]["near_baseline"]))
    pred_far_y = float((pred_far[0][1] + pred_far[1][1]) / 2.0)
    pred_near_y = float((pred_near[0][1] + pred_near[1][1]) / 2.0)

    clusters = cluster_by_coordinate(horizontal_group, axis="y", tol=max(12, int(0.018 * img_h)))
    far_candidates = []
    near_candidates = []

    for idx, cl in enumerate(clusters):
        line = fit_cluster_to_line(cl, img_w, img_h)
        if line is None:
            continue

        my = seg_mid(line)[1]
        if my >= near_service_y - 0.03 * img_h:
            cand = score_baseline_candidate(line, cl, "near", vertical_five, service_lines, H_hint, model, img_w, img_h, edges_roi, white_roi)
            if cand is not None:
                cand["cluster_id"] = idx
                cand["source"] = "detected"
                near_candidates.append(cand)

        if my <= far_service_y + 0.03 * img_h:
            cand = score_baseline_candidate(line, cl, "far", vertical_five, service_lines, H_hint, model, img_w, img_h, edges_roi, white_roi)
            if cand is not None:
                cand["cluster_id"] = idx
                cand["source"] = "detected"
                far_candidates.append(cand)

    if not far_candidates or not near_candidates:
        def projected_candidate(side):
            line = project_model_line(H_hint, model, f"{side}_baseline")
            ld = vertical_five["left_doubles"]
            rd = vertical_five["right_doubles"]
            inter_ld = safe_intersection(line, ld, img_w, img_h, margin=0.45)
            inter_rd = safe_intersection(line, rd, img_w, img_h, margin=0.45)
            if inter_ld is None or inter_rd is None:
                return None
            outer_width = abs(inter_rd[0] - inter_ld[0])
            mx, my = seg_mid(line)
            clipped_outer, _, _ = clipped_segment_between_lines(line, vertical_five["left_doubles"], vertical_five["right_doubles"], img_w, img_h)
            return {
                "line": line,
                "cluster": [],
                "score": -250.0,
                "side": side,
                "source": "projected",
                "mx": mx,
                "my": my,
                "clipped_outer": clipped_outer if clipped_outer is not None else line,
                "intersections": {"ld": inter_ld, "rd": inter_rd},
                "pred_line": line,
                "metrics": {
                    "total_len": 0.0,
                    "inner_width": outer_width * 0.75,
                    "outer_width": outer_width,
                    "alley_penalty": 0.0,
                    "horiz_pen": horizontal_angle_penalty(line),
                    "y_penalty": 0.0,
                    "x_penalty": 0.0,
                    "angle_match_penalty": 0.0,
                    "width_penalty": 0.0,
                    "service_gap": 0.0,
                    "edge_ratio": 0.0,
                    "white_ratio": 0.0,
                    "support_ratio": 0.0,
                    "outside_support_ratio": 0.0,
                    "outside_penalty": 0.0,
                },
            }

        if not far_candidates:
            cand = projected_candidate("far")
            if cand is not None:
                far_candidates.append(cand)
        if not near_candidates:
            cand = projected_candidate("near")
            if cand is not None:
                near_candidates.append(cand)
        if not far_candidates or not near_candidates:
            return None

    best_pair = None
    best_pair_score = -1e18
    for f in far_candidates:
        for n in near_candidates:
            if not (f["my"] < far_service_y < near_service_y < n["my"]):
                continue
            if n["my"] - f["my"] < 0.25 * img_h:
                continue

            order_penalty = abs(f["my"] - pred_far_y) + abs(n["my"] - pred_near_y)
            width_consistency = abs(f["metrics"]["outer_width"] - n["metrics"]["outer_width"])
            pair_score = f["score"] + n["score"] - 1.2 * width_consistency - 1.5 * order_penalty
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (f, n)

    if best_pair is None:
        return None

    best_far, best_near = best_pair
    return {
        "far_baseline": best_far["line"],
        "near_baseline": best_near["line"],
        "far_candidate": best_far,
        "near_candidate": best_near,
        "all_far_candidates": far_candidates,
        "all_near_candidates": near_candidates,
        "line_sources": {
            "far_baseline": best_far.get("source", "detected"),
            "near_baseline": best_near.get("source", "detected"),
        },
        "warnings": [
            f"{side}_baseline_projected_from_service_homography"
            for side, cand in (("far", best_far), ("near", best_near))
            if cand.get("source") == "projected"
        ],
    }


def save_baselines_debug(img, vertical_five, service_lines, baselines, out_path):
    out = img.copy()
    draw_named_seg(out, vertical_five["left_doubles"],   (0, 0, 255),   "left_doubles")
    draw_named_seg(out, vertical_five["left_singles"],   (0, 255, 0),   "left_singles")
    draw_named_seg(out, vertical_five["center_service"], (255, 0, 255), "center_service")
    draw_named_seg(out, vertical_five["right_singles"],  (255, 0, 0),   "right_singles")
    draw_named_seg(out, vertical_five["right_doubles"],  (0, 255, 255), "right_doubles")
    draw_named_seg(out, service_lines["far_service"],    (255, 255, 255), "far_service", 2)
    draw_named_seg(out, service_lines["near_service"],   (180, 255, 180), "near_service", 2)
    draw_named_seg(out, baselines["far_baseline"],       (255, 200, 0), "far_baseline", 2)
    draw_named_seg(out, baselines["near_baseline"],      (0, 200, 255), "near_baseline", 2)
    cv2.imwrite(out_path, out)


def save_baseline_candidates_debug(img, baselines, out_path):
    out = img.copy()
    candidates = []
    for side, color in (("far", (255, 200, 0)), ("near", (0, 200, 255))):
        key = f"all_{side}_candidates"
        for cand in baselines.get(key, []):
            candidates.append((side, cand, color))

    for side, cand, color in sorted(candidates, key=lambda item: item[1].get("score", 0.0), reverse=True):
        line = cand["line"]
        clipped = cand.get("clipped_outer", line)
        faint = tuple(int(c * 0.45) for c in color)
        draw_seg(out, line, faint, 1)
        draw_seg(out, clipped, color, 3)
        mx, my = seg_mid(clipped)
        metrics = cand.get("metrics", {})
        label = (
            f"{side} {cand.get('score', 0):.0f} "
            f"s={metrics.get('support_ratio', 0):.2f} "
            f"out={metrics.get('outside_support_ratio', 0):.2f}"
        )
        cv2.putText(
            out,
            label,
            (int(mx) + 4, int(my) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, out)

def choose_vertical_five_with_center(vertical_group, img_w, img_h):
    cand = []
    for s in vertical_group:
        mx = (s["x1"] + s["x2"]) / 2
        a = s["angle_deg"] % 180
        cand.append({
            "seg": s,
            "mx": mx,
            "angle": a,
            "len": s["length"],
        })

    if len(cand) < 5:
        return None

    cand = sorted(cand, key=lambda z: z["mx"])

    best = None
    best_score = -1e18
    n = len(cand)

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    for e in range(d + 1, n):
                        chosen = [cand[a], cand[b], cand[c], cand[d], cand[e]]
                        xs = [z["mx"] for z in chosen]
                        x1, x2, x3, x4, x5 = xs

                        d12 = x2 - x1
                        d23 = x3 - x2
                        d34 = x4 - x3
                        d45 = x5 - x4

                        # 基本條件：間距都要正且不能太小
                        if min(d12, d23, d34, d45) < 0.02 * img_w:
                            continue

                        # 左右對稱
                        symmetry_penalty = abs(d12 - d45) + abs(d23 - d34)

                        # 中線居中
                        center_penalty = abs(x3 - 0.5 * img_w)

                        # 單打到中線應比雙打到單打寬
                        spacing_penalty = 0.0
                        if d23 <= d12:
                            spacing_penalty += (d12 - d23) * 3.0
                        if d34 <= d45:
                            spacing_penalty += (d45 - d34) * 3.0

                        # 整體左右展開要夠大
                        spread = x5 - x1
                        if spread < 0.35 * img_w:
                            continue

                        # 角度仍要像直向線，但只當輕微輔助
                        angle_penalty = sum(abs(z["angle"] - 90) for z in chosen)

                        score = (
                            3.0 * spread
                            - 2.5 * symmetry_penalty
                            - 1.5 * center_penalty
                            - 2.0 * spacing_penalty
                            - 0.3 * angle_penalty
                        )

                        if score > best_score:
                            best_score = score
                            best = chosen

    if best is None:
        return None

    return {
        "left_doubles": best[0]["seg"],
        "left_singles": best[1]["seg"],
        "center_service": best[2]["seg"],
        "right_singles": best[3]["seg"],
        "right_doubles": best[4]["seg"],
    }

def clip_point(x, y, w, h, margin=2000):
    x = max(-margin, min(w + margin, int(x)))
    y = max(-margin, min(h + margin, int(y)))
    return x, y

def draw_named_seg(img, seg, color, name=None, thickness=3):
    h, w = img.shape[:2]
    x1, y1 = clip_point(seg["x1"], seg["y1"], w, h)
    x2, y2 = clip_point(seg["x2"], seg["y2"], w, h)

    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if name is not None:
        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)
        cv2.putText(
            img,
            name,
            (mx + 5, my - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        
def save_vertical_candidates_debug(img, vertical_group, img_w, img_h, debug_dir="court_detector_debug"):
    clusters = cluster_by_coordinate(vertical_group, axis="x", tol=max(14, int(0.02 * img_w)))
    candidates = []

    for idx, cl in enumerate(clusters):
        line = fit_cluster_to_line(cl, img_w, img_h)
        if line is None:
            continue
        mx, my = seg_mid(line)
        total_len = sum(s["length"] for s in cl)
        candidates.append({
            "id": idx,
            "line": line,
            "mx": mx,
            "my": my,
            "total_len": total_len,
            "cluster": cl,
        })

    out = img.copy()
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (180, 100, 255), (100, 255, 180)
    ]

    for i, c in enumerate(sorted(candidates, key=lambda z: z["mx"])):
        color = colors[i % len(colors)]
        seg = c["line"]
        draw_named_seg(
            out,
            seg,
            color,
            f"id={c['id']} mx={c['mx']:.1f} a={seg['angle_deg']:.1f}",
            3
        )
    return candidates

def save_segments_debug(img, segs, out_path, color=(0, 255, 255), thickness=2, put_idx=True):
    out = img.copy()
    for i, s in enumerate(segs):
        x1, y1 = int(s["x1"]), int(s["y1"])
        x2, y2 = int(s["x2"]), int(s["y2"])
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
        if put_idx:
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            txt = f"{i}:{s['angle_deg']:.1f}"
            cv2.putText(out, txt, (mx + 3, my - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)
    
def visualize_split_groups(img, horizontal, vertical, other):
    out = img.copy()
    for s in horizontal:
        draw_seg(out, s, (255, 0, 255), 2)   # 粉
    for s in vertical:
        draw_seg(out, s, (0, 255, 255), 2)   # 黃
    for s in other:
        draw_seg(out, s, (128, 128, 128), 2) # 灰
    return out
    
# =========================================================
# model / homography
# =========================================================
def build_tennis_court_model():
    Wd = 10.97
    Ws = 8.23
    L = 23.77
    half_L = L / 2.0
    service_from_net = 6.40
    center_x = Wd / 2.0
    single_margin = (Wd - Ws) / 2.0

    lines = {
        "near_baseline": ((0.0, L), (Wd, L)),
        "far_baseline": ((0.0, 0.0), (Wd, 0.0)),
        "left_doubles": ((0.0, 0.0), (0.0, L)),
        "right_doubles": ((Wd, 0.0), (Wd, L)),
        "left_singles": ((single_margin, 0.0), (single_margin, L)),
        "right_singles": ((single_margin + Ws, 0.0), (single_margin + Ws, L)),
        "near_service": ((single_margin, half_L + service_from_net), (single_margin + Ws, half_L + service_from_net)),
        "far_service": ((single_margin, half_L - service_from_net), (single_margin + Ws, half_L - service_from_net)),
        "center_service": ((center_x, half_L - service_from_net), (center_x, half_L + service_from_net)),
        "net": ((0.0, half_L), (Wd, half_L)),
    }

    key_points = {
        "far_left_doubles": np.array([0.0, 0.0], dtype=np.float32),
        "far_left_singles": np.array([single_margin, 0.0], dtype=np.float32),
        "far_right_singles": np.array([single_margin + Ws, 0.0], dtype=np.float32),
        "far_right_doubles": np.array([Wd, 0.0], dtype=np.float32),
        "near_left_doubles": np.array([0.0, L], dtype=np.float32),
        "near_left_singles": np.array([single_margin, L], dtype=np.float32),
        "near_right_singles": np.array([single_margin + Ws, L], dtype=np.float32),
        "near_right_doubles": np.array([Wd, L], dtype=np.float32),
        "far_service_left": np.array([single_margin, half_L - service_from_net], dtype=np.float32),
        "far_service_right": np.array([single_margin + Ws, half_L - service_from_net], dtype=np.float32),
        "near_service_left": np.array([single_margin, half_L + service_from_net], dtype=np.float32),
        "near_service_right": np.array([single_margin + Ws, half_L + service_from_net], dtype=np.float32),
    }

    return {"Wd": Wd, "Ws": Ws, "L": L, "lines_dict": lines, "lines": list(lines.values()), "key_points": key_points}


def project_points(H, pts):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def compute_homography_from_services(vertical_five, service_lines, model):
    ld = vertical_five["left_doubles"]
    ls = vertical_five["left_singles"]
    rs = vertical_five["right_singles"]
    rd = vertical_five["right_doubles"]
    far_service = service_lines["far_service"]
    near_service = service_lines["near_service"]

    pts_img = {
        "far_left_singles": line_intersection(ls, far_service),
        "far_right_singles": line_intersection(rs, far_service),
        "near_left_singles": line_intersection(ls, near_service),
        "near_right_singles": line_intersection(rs, near_service),
        "far_left_doubles": line_intersection(ld, far_service),
        "far_right_doubles": line_intersection(rd, far_service),
        "near_left_doubles": line_intersection(ld, near_service),
        "near_right_doubles": line_intersection(rd, near_service),
    }

    pts_model = [
        np.array(model["lines_dict"]["far_service"][0], dtype=np.float32),
        np.array(model["lines_dict"]["far_service"][1], dtype=np.float32),
        np.array(model["lines_dict"]["near_service"][0], dtype=np.float32),
        np.array(model["lines_dict"]["near_service"][1], dtype=np.float32),
        model["key_points"]["far_left_doubles"],
        model["key_points"]["far_right_doubles"],
        model["key_points"]["near_left_doubles"],
        model["key_points"]["near_right_doubles"],
    ]
    pts_dst = [
        pts_img["far_left_singles"],
        pts_img["far_right_singles"],
        pts_img["near_left_singles"],
        pts_img["near_right_singles"],
        pts_img["far_left_doubles"],
        pts_img["far_right_doubles"],
        pts_img["near_left_doubles"],
        pts_img["near_right_doubles"],
    ]

    if any(p is None for p in pts_dst):
        return None

    H, _ = cv2.findHomography(
        np.array(pts_model, dtype=np.float32),
        np.array(pts_dst, dtype=np.float32),
        method=0,
    )
    return H

def compute_homography_full_court(vertical_five, service_lines, baselines, model):
    ld = vertical_five["left_doubles"]
    ls = vertical_five["left_singles"]
    rs = vertical_five["right_singles"]
    rd = vertical_five["right_doubles"]
    fs = service_lines["far_service"]
    ns = service_lines["near_service"]
    fb = baselines["far_baseline"]
    nb = baselines["near_baseline"]

    pairs = [
        (line_intersection(ls, fs), model["key_points"]["far_service_left"]),
        (line_intersection(rs, fs), model["key_points"]["far_service_right"]),
        (line_intersection(ls, ns), model["key_points"]["near_service_left"]),
        (line_intersection(rs, ns), model["key_points"]["near_service_right"]),
        (line_intersection(ld, fb), model["key_points"]["far_left_doubles"]),
        (line_intersection(ls, fb), model["key_points"]["far_left_singles"]),
        (line_intersection(rs, fb), model["key_points"]["far_right_singles"]),
        (line_intersection(rd, fb), model["key_points"]["far_right_doubles"]),
        (line_intersection(ld, nb), model["key_points"]["near_left_doubles"]),
        (line_intersection(ls, nb), model["key_points"]["near_left_singles"]),
        (line_intersection(rs, nb), model["key_points"]["near_right_singles"]),
        (line_intersection(rd, nb), model["key_points"]["near_right_doubles"]),
    ]

    pts_img = []
    pts_model = []
    for img_pt, model_pt in pairs:
        if img_pt is None:
            continue
        pts_img.append(img_pt)
        pts_model.append(model_pt)

    if len(pts_img) < 8:
        return None

    H, _ = cv2.findHomography(
        np.array(pts_model, dtype=np.float32),
        np.array(pts_img, dtype=np.float32),
        method=0,
    )
    return H

def draw_projected_model(img, H, model, color=(0, 255, 255), thickness=2, net_top_line=None):
    out = img.copy()

    # 先畫 ground-plane court lines
    for p1, p2 in model["lines"]:
        pr = project_points(H, [p1, p2])
        cv2.line(
            out,
            tuple(np.int32(np.round(pr[0]))),
            tuple(np.int32(np.round(pr[1]))),
            color,
            thickness
        )

    # 再額外畫網子上緣
    if net_top_line is not None:
        cv2.line(
            out,
            (int(net_top_line["x1"]), int(net_top_line["y1"])),
            (int(net_top_line["x2"]), int(net_top_line["y2"])),
            (0, 255, 255),
            thickness
        )

    return out

def rasterize_model(mask_shape, H, model, thickness=2):
    h, w = mask_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for p1, p2 in model["lines"]:
        pr = project_points(H, [p1, p2])
        cv2.line(canvas, clip_pt(pr[0], w, h), clip_pt(pr[1], w, h), 255, thickness)
    return canvas


def line_band_mask(mask_shape, seg, thickness=5):
    h, w = mask_shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.line(m, clip_pt((seg["x1"], seg["y1"]), w, h), clip_pt((seg["x2"], seg["y2"]), w, h), 255, thickness)
    return m


# =========================================================
# ROI from 5 lines
# =========================================================
def make_main_roi(img_shape, vertical_five, near_service, H, model):
    h, w = img_shape[:2]
    roi = np.zeros((h, w), dtype=np.uint8)

    court_corners_model = np.array([
        [0.0, model["L"]],
        [model["Wd"], model["L"]],
        [model["Wd"], 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    corners_img = project_points(H, court_corners_model)
    poly = np.array([clip_pt(p, w, h) for p in corners_img], dtype=np.int32)
    cv2.fillConvexPoly(roi, poly, 255)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    roi = cv2.dilate(roi, kern, iterations=1)
    return roi

def seg_endpoints_top_bottom(seg):
    p1 = (float(seg["x1"]), float(seg["y1"]))
    p2 = (float(seg["x2"]), float(seg["y2"]))
    if p1[1] <= p2[1]:
        return p1, p2
    return p2, p1


def project_model_line(H, model, line_name):
    p1, p2 = model["lines_dict"][line_name]
    pr = project_points(H, [p1, p2])
    return {
        "x1": float(pr[0][0]), "y1": float(pr[0][1]),
        "x2": float(pr[1][0]), "y2": float(pr[1][1]),
        "angle_deg": normalize_angle_deg(math.degrees(math.atan2(pr[1][1] - pr[0][1], pr[1][0] - pr[0][0]))),
        "length": float(math.hypot(pr[1][0] - pr[0][0], pr[1][1] - pr[0][1])),
    }

def seg_x_range(seg):
    return min(seg["x1"], seg["x2"]), max(seg["x1"], seg["x2"])


def overlap_1d(a1, a2, b1, b2):
    left = max(a1, b1)
    right = min(a2, b2)
    return max(0.0, right - left)


def choose_net_top_candidate(horizontal_lines, net_bottom_line, img_w, img_h):
    """
    從 horizontal 線段中找最像 '網子上緣' 的那一條。
    條件：
    - 必須在 net_bottom_line 上方
    - y 距離不能太遠
    - x 範圍要和 net_bottom 有足夠重疊
    - 越水平越好、越長越好
    """
    nb_mx, nb_my = seg_mid(net_bottom_line)
    nb_x1, nb_x2 = seg_x_range(net_bottom_line)
    nb_width = nb_x2 - nb_x1

    best = None
    best_score = -1e18

    for s in horizontal_lines:
        sx1, sx2 = seg_x_range(s)
        smx, smy = seg_mid(s)

        # 1) 一定要在 net bottom 上方
        dy = nb_my - smy
        if dy <= 0:
            continue

        # 2) 不要離太遠
        if dy > 0.12 * img_h:
            continue

        # 3) x 方向要跟 net 有夠多重疊
        ov = overlap_1d(sx1, sx2, nb_x1, nb_x2)
        overlap_ratio = ov / max(1.0, min(sx2 - sx1, nb_width))
        if overlap_ratio < 0.45:
            continue

        # 4) 中心 x 不要差太多
        x_center_penalty = abs(smx - nb_mx)
        if x_center_penalty > 0.12 * img_w:
            continue

        # 5) 越水平越好
        horiz_pen = horizontal_angle_penalty(s)

        # 6) 越長越好
        length_bonus = s["length"]

        # 7) 網上緣通常比 net bottom 稍微短一點或差不多
        width_penalty = abs((sx2 - sx1) - nb_width)

        score = (
            1.5 * length_bonus
            - 10.0 * horiz_pen
            - 2.0 * x_center_penalty
            - 1.2 * width_penalty
            - 25.0 * abs(dy - 0.035 * img_h)   # 偏好在上方一小段距離
        )

        if score > best_score:
            best_score = score
            best = s

    return best

def detect_net_top_line(result, merged_lines, img_shape):
    h, w = img_shape[:2]

    # 先拿 homography 投影出的 net bottom
    net_bottom_line = project_model_line(result["H"], result["model"], "net")

    horizontal, _, _ = split_horizontal_vertical(merged_lines, h_tol=20, v_tol=18)
    net_top_line = choose_net_top_candidate(horizontal, net_bottom_line, w, h)

    return {
        "net_bottom_line": net_bottom_line,
        "net_top_line": net_top_line,
    }

def build_roi_config_from_result(image_path, img_shape, result):
    h, w = img_shape[:2]

    H = result["H"]
    model = result["model"]
    vf = result["vertical_five"]
    baselines = result["baselines"]
    far_service = result["far_service"]
    near_service = result["near_service"]
    far_baseline = baselines["far_baseline"]
    near_baseline = baselines["near_baseline"]

    # ground-plane 的 net 線（也就是 net bottom / net location）
    net_line = project_model_line(H, model, "net")

    # 若前面有另外偵測到網子上緣，就一起拿進來
    net_top_line = None
    if result.get("net_lines") is not None:
        net_top_line = result["net_lines"].get("net_top_line")

    # =====================================================
    # 1) ROI_POLY：整個球場四角
    # =====================================================
    court_corners_model = np.array([
        [0.0, model["L"]],          # near left doubles
        [model["Wd"], model["L"]],  # near right doubles
        [model["Wd"], 0.0],         # far right doubles
        [0.0, 0.0],                 # far left doubles
    ], dtype=np.float32)

    corners_img = project_points(H, court_corners_model)

    near_left  = tuple(map(float, corners_img[0]))
    near_right = tuple(map(float, corners_img[1]))
    far_right  = tuple(map(float, corners_img[2]))
    far_left   = tuple(map(float, corners_img[3]))

    ROI_POLY = [
        clip_pt(far_left, w, h),    # TL
        clip_pt(far_right, w, h),   # TR
        clip_pt(near_right, w, h),  # BR
        clip_pt(near_left, w, h),   # BL
    ]

    # =====================================================
    # 2) FAR_WARP_SRC_PTS：遠端半場（網子 -> 對面底線，左右雙打線）
    # =====================================================
    far_tl = line_intersection(vf["left_doubles"], far_baseline)
    far_tr = line_intersection(vf["right_doubles"], far_baseline)
    far_bl = line_intersection(vf["left_doubles"], net_line)
    far_br = line_intersection(vf["right_doubles"], net_line)

    if None in (far_tl, far_tr, far_br, far_bl):
        raise RuntimeError("Failed to build FAR_WARP_SRC_PTS from detected court.")

    FAR_WARP_SRC_PTS = [
        clip_pt(far_tl, w, h),   # TL
        clip_pt(far_tr, w, h),   # TR
        clip_pt(far_br, w, h),   # BR
        clip_pt(far_bl, w, h),   # BL
    ]

    FAR_COURT_POLY = [
        clip_pt(far_tl, w, h),
        clip_pt(far_tr, w, h),
        clip_pt(far_br, w, h),
        clip_pt(far_bl, w, h),
    ]

    br_y = max(FAR_WARP_SRC_PTS[2][1], FAR_WARP_SRC_PTS[3][1])
    FAR_Y_RATIO_EDGE = float(br_y) / float(h)
    FAR_WARP_DST_SIZE = [640, 360]

    # =====================================================
    # 3) MID_LINE：中線（service center line）top / bottom
    # =====================================================
    mid_top, mid_bottom = seg_endpoints_top_bottom(vf["center_service"])
    MID_LINE = [
        clip_pt(mid_top, w, h),
        clip_pt(mid_bottom, w, h),
    ]

    # =====================================================
    # 4) NET lines
    # =====================================================
    net_left = line_intersection(vf["left_doubles"], net_line)
    net_right = line_intersection(vf["right_doubles"], net_line)
    if None in (net_left, net_right):
        raise RuntimeError("Failed to build NET_LINE from detected court.")

    NET_LINE = [
        clip_pt(net_left, w, h),
        clip_pt(net_right, w, h),
    ]

    NET_BOTTOM_LINE = NET_LINE

    NET_TOP_LINE = None
    if net_top_line is not None:
        NET_TOP_LINE = [
            clip_pt((net_top_line["x1"], net_top_line["y1"]), w, h),
            clip_pt((net_top_line["x2"], net_top_line["y2"]), w, h),
        ]

    # =====================================================
    # 5) SERVICE_LINES：far / near 發球線（只取單打線之間那段）
    # =====================================================
    far_service_left = line_intersection(vf["left_singles"], far_service)
    far_service_right = line_intersection(vf["right_singles"], far_service)
    near_service_left = line_intersection(vf["left_singles"], near_service)
    near_service_right = line_intersection(vf["right_singles"], near_service)

    if None in (far_service_left, far_service_right, near_service_left, near_service_right):
        raise RuntimeError("Failed to build SERVICE_LINES from detected court.")

    FAR_SERVICE_LINE = [
        clip_pt(far_service_left, w, h),
        clip_pt(far_service_right, w, h),
    ]
    NEAR_SERVICE_LINE = [
        clip_pt(near_service_left, w, h),
        clip_pt(near_service_right, w, h),
    ]

    # =====================================================
    # 6) BASELINES：far / near baseline（取雙打線之間那段）
    # =====================================================
    far_baseline_left = line_intersection(vf["left_doubles"], far_baseline)
    far_baseline_right = line_intersection(vf["right_doubles"], far_baseline)
    near_baseline_left = line_intersection(vf["left_doubles"], near_baseline)
    near_baseline_right = line_intersection(vf["right_doubles"], near_baseline)

    if None in (far_baseline_left, far_baseline_right, near_baseline_left, near_baseline_right):
        raise RuntimeError("Failed to build BASELINES from detected court.")

    FAR_BASELINE = [
        clip_pt(far_baseline_left, w, h),
        clip_pt(far_baseline_right, w, h),
    ]
    NEAR_BASELINE = [
        clip_pt(near_baseline_left, w, h),
        clip_pt(near_baseline_right, w, h),
    ]

    # =====================================================
    # 7) SINGLES_LINES：TL, BL, TR, BR
    # Save true intersections with the baselines so the exported vertical
    # lines always meet the court rectangle.
    # =====================================================
    left_top = line_intersection(vf["left_singles"], far_baseline)
    left_bottom = line_intersection(vf["left_singles"], near_baseline)
    right_top = line_intersection(vf["right_singles"], far_baseline)
    right_bottom = line_intersection(vf["right_singles"], near_baseline)
    if None in (left_top, left_bottom, right_top, right_bottom):
        left_top, left_bottom = seg_endpoints_top_bottom(vf["left_singles"])
        right_top, right_bottom = seg_endpoints_top_bottom(vf["right_singles"])

    SINGLES_LINES = [
        clip_pt(left_top, w, h),
        clip_pt(left_bottom, w, h),
        clip_pt(right_top, w, h),
        clip_pt(right_bottom, w, h),
    ]

    # =====================================================
    # 8) DOUBLES_LINES：TL, BL, TR, BR
    # =====================================================
    ld_top = line_intersection(vf["left_doubles"], far_baseline)
    ld_bottom = line_intersection(vf["left_doubles"], near_baseline)
    rd_top = line_intersection(vf["right_doubles"], far_baseline)
    rd_bottom = line_intersection(vf["right_doubles"], near_baseline)
    if None in (ld_top, ld_bottom, rd_top, rd_bottom):
        ld_top, ld_bottom = seg_endpoints_top_bottom(vf["left_doubles"])
        rd_top, rd_bottom = seg_endpoints_top_bottom(vf["right_doubles"])

    DOUBLES_LINES = [
        clip_pt(ld_top, w, h),
        clip_pt(ld_bottom, w, h),
        clip_pt(rd_top, w, h),
        clip_pt(rd_bottom, w, h),
    ]

    out = {
        "image_path": str(image_path),
        "image_size": [w, h],

        # optional but very useful for bounce.py
        "H": np.asarray(H, dtype=float).tolist(),

        "ROI_POLY": ROI_POLY,
        "BAN_BOXES": [],

        "FAR_WARP_SRC_PTS": FAR_WARP_SRC_PTS,
        "FAR_COURT_POLY": FAR_COURT_POLY,
        "FAR_Y_RATIO_EDGE": FAR_Y_RATIO_EDGE,
        "FAR_WARP_DST_SIZE": FAR_WARP_DST_SIZE,

        "MID_LINE": MID_LINE,

        "NET_LINE": NET_LINE,
        "NET_BOTTOM_LINE": NET_BOTTOM_LINE,
        "NET_TOP_LINE": NET_TOP_LINE,

        "FAR_SERVICE_LINE": FAR_SERVICE_LINE,
        "NEAR_SERVICE_LINE": NEAR_SERVICE_LINE,

        "FAR_BASELINE": FAR_BASELINE,
        "NEAR_BASELINE": NEAR_BASELINE,

        "SINGLES_LINES": SINGLES_LINES,
        "DOUBLES_LINES": DOUBLES_LINES,
        "detector": {
            "type": "hybrid_line_v1",
            "frames_sampled": int(result.get("frames_sampled", 1)),
            "selected_frame": str(result.get("selected_frame", image_path)),
            "score": float(result.get("score", 0.0)),
            "line_sources": result.get("line_sources", {}),
            "warnings": result.get("warnings", []),
        },
    }
    return out

# =========================================================
# verify / score with other lines
# =========================================================
def score_hypothesis(H, model, edges_roi, white_roi, merged_lines):
    model_mask = rasterize_model(edges_roi.shape, H, model, thickness=2)
    dil = cv2.dilate(model_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    overlap_edges = cv2.bitwise_and(dil, edges_roi)
    overlap_white = cv2.bitwise_and(dil, white_roi)

    edge_hits = int(np.count_nonzero(overlap_edges))
    white_hits = int(np.count_nonzero(overlap_white))
    model_pixels = max(1, int(np.count_nonzero(dil)))

    coverage = edge_hits / model_pixels
    white_ratio = white_hits / model_pixels

    extras = ["center_service", "net", "far_service", "far_baseline"]
    extra_score = 0.0
    for name in extras:
        p1, p2 = model["lines_dict"][name]
        pr = project_points(H, [p1, p2])
        seg = {
            "x1": pr[0][0], "y1": pr[0][1], "x2": pr[1][0], "y2": pr[1][1],
            "angle_deg": normalize_angle_deg(math.degrees(math.atan2(pr[1][1] - pr[0][1], pr[1][0] - pr[0][0]))),
            "length": float(math.hypot(pr[1][0] - pr[0][0], pr[1][1] - pr[0][1])),
        }
        band = line_band_mask(edges_roi.shape, seg, thickness=7)
        extra_score += 0.004 * np.count_nonzero(cv2.bitwise_and(band, edges_roi))
        extra_score += 0.002 * np.count_nonzero(cv2.bitwise_and(band, white_roi))

    align_bonus = 0.0
    for s in merged_lines:
        mx, my = seg_mid(s)
        if dil[int(np.clip(my, 0, dil.shape[0] - 1)), int(np.clip(mx, 0, dil.shape[1] - 1))] > 0:
            align_bonus += min(2.0, 0.01 * s["length"])

    score = 1500.0 * coverage + 220.0 * white_ratio + extra_score + align_bonus
    return score, model_mask, overlap_edges, overlap_white


# =========================================================
# main pipeline
# =========================================================
def reconstruct_from_five_line_method(img, edges_roi, white_roi, merged_lines, debug_dir=None):
    debug_dir = Path(debug_dir) if debug_dir is not None else None
    if debug_dir is not None:
        ensure_dir(debug_dir)
    verbose = debug_dir is not None
    h, w = img.shape[:2]
    horizontal, vertical, other = split_horizontal_vertical(merged_lines)
   
    if debug_dir is not None:
        save_segments_debug(img, vertical, str(debug_dir / "09a_vertical_raw.jpg"), color=(0,255,255))
        save_segments_debug(img, horizontal, str(debug_dir / "09b_horizontal_raw.jpg"), color=(255,0,255))
        save_vertical_candidates_debug(img, vertical, w, h, debug_dir=debug_dir)
    
    vertical_five = choose_vertical_five_with_center(vertical, w, h)
    
    if vertical_five is not None and debug_dir is not None:
        dbg = img.copy()

        draw_named_seg(dbg, vertical_five["left_doubles"],  (0, 0, 255),   "left_doubles")
        draw_named_seg(dbg, vertical_five["left_singles"],  (0, 255, 0),   "left_singles")
        draw_named_seg(dbg, vertical_five["right_singles"], (255, 0, 0),   "right_singles")
        draw_named_seg(dbg, vertical_five["right_doubles"], (0, 255, 255), "right_doubles")
        draw_named_seg(dbg, vertical_five["center_service"], (255, 0, 255), "center_service")
        
        cv2.imwrite(str(debug_dir / "11_five_lines_debug.jpg"), dbg)

    if vertical_five is None:
        if verbose:
            print("FAIL: choose_vertical_five")
        return None

    service_lines = choose_service_lines(horizontal, vertical_five, w, h, debug=True)
    if service_lines is None:
        if verbose:
            print("FAIL: choose_service_lines")
        return None

    near_service = service_lines["near_service"]
    far_service = service_lines["far_service"]
    if near_service is None:
        return None
    if far_service is None:
        return None
    
    if debug_dir is not None:
        save_service_lines_debug(
            img,
            vertical_five,
            service_lines,
            str(debug_dir / "12_service_lines_debug.jpg")
        )
    
    model = build_tennis_court_model()
    
    H0 = compute_homography_from_services(vertical_five, service_lines, model)
    if H0 is None:
        if verbose:
            print("FAIL: compute_homography_from_services")
        return None

    baselines = choose_baselines(horizontal, vertical_five, service_lines, H0, model, w, h, edges_roi, white_roi)
    if baselines is None:
        if verbose:
            print("FAIL: choose_baselines")
        return None

    if debug_dir is not None:
        save_baselines_debug(
            img,
            vertical_five,
            service_lines,
            baselines,
            str(debug_dir / "13_baselines_debug.jpg")
        )
        save_baseline_candidates_debug(
            img,
            baselines,
            str(debug_dir / "17_baseline_candidates_scored.jpg")
        )

    H1 = compute_homography_full_court(vertical_five, service_lines, baselines, model)
    if H1 is None:
        if verbose:
            print("WARN: compute_homography_full_court failed, fallback to H0")
        H1 = H0

    main_roi = make_main_roi(img.shape, vertical_five, baselines["near_baseline"], H1, model)
    edges_main = cv2.bitwise_and(edges_roi, main_roi)
    white_main = cv2.bitwise_and(white_roi, main_roi)

    score, model_mask, overlap_edges, overlap_white = score_hypothesis(H1, model, edges_main, white_main, merged_lines)
    
    net_lines = detect_net_top_line(
        result={
            "H": H1,
            "model": model,
        },
        merged_lines=merged_lines,
        img_shape=img.shape,
    )

    line_sources = {
        "left_doubles": "detected",
        "left_singles": "detected",
        "center_service": "detected",
        "right_singles": "detected",
        "right_doubles": "detected",
    }
    line_sources.update(service_lines.get("line_sources", {}))
    line_sources.update(baselines.get("line_sources", {}))
    warnings = []
    warnings.extend(service_lines.get("warnings", []))
    warnings.extend(baselines.get("warnings", []))

    return {
        "H": H1,
        "H0": H0,
        "score": score,
        "model": model,
        "vertical_five": vertical_five,
        "near_service": near_service,
        "far_service": far_service,
        "baselines": baselines,
        "main_roi": main_roi,
        "model_mask": model_mask,
        "overlap_edges": overlap_edges,
        "overlap_white": overlap_white,
        "groups": {"horizontal": horizontal, "vertical": vertical},
        "net_lines": net_lines,
        "line_sources": line_sources,
        "warnings": warnings,
    }


# =========================================================
# visualizations
# =========================================================
def visualize_groups(img, result):
    out = img.copy()
    for s in result["groups"]["horizontal"]:
        draw_seg(out, s, (0, 255, 0), 2)
    for s in result["groups"]["vertical"]:
        draw_seg(out, s, (0, 0, 255), 2)
    return out


def visualize_main_five(img, result):
    out = img.copy()
    vf = result["vertical_five"]
    draw_seg(out, vf["left_doubles"], (255, 0, 255), 3)
    draw_seg(out, vf["left_singles"], (255, 128, 0), 3)
    draw_seg(out, vf["right_singles"], (255, 128, 0), 3)
    draw_seg(out, vf["right_doubles"], (255, 0, 255), 3)
    draw_seg(out, result["near_service"], (0, 255, 255), 3)
    draw_seg(out, result["far_service"], (255, 255, 255), 3)
    draw_seg(out, result["baselines"]["near_baseline"], (0, 200, 255), 3)
    draw_seg(out, result["baselines"]["far_baseline"], (255, 200, 0), 3)
    return out


def polygon_area(poly):
    pts = np.asarray(poly, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def validate_roi_config(cfg, img_shape):
    h, w = img_shape[:2]
    area = polygon_area(cfg.get("ROI_POLY", []))
    frame_area = float(max(1, w * h))
    area_ratio = area / frame_area
    if area_ratio < 0.05 or area_ratio > 0.95:
        return False, f"roi_area_ratio_out_of_range:{area_ratio:.3f}"

    try:
        far_y = np.mean([p[1] for p in cfg["FAR_BASELINE"]])
        near_y = np.mean([p[1] for p in cfg["NEAR_BASELINE"]])
        net_y = np.mean([p[1] for p in cfg["NET_LINE"]])
        far_service_y = np.mean([p[1] for p in cfg["FAR_SERVICE_LINE"]])
        near_service_y = np.mean([p[1] for p in cfg["NEAR_SERVICE_LINE"]])
    except Exception:
        return False, "missing_required_court_lines"

    if not (far_y < far_service_y < net_y < near_service_y < near_y):
        return False, "court_y_order_invalid"
    if far_y > 0.235 * h:
        return False, "far_baseline_too_low"
    if far_service_y - far_y < 0.065 * h:
        return False, "far_service_too_close_to_far_baseline"

    singles = cfg.get("SINGLES_LINES", [])
    doubles = cfg.get("DOUBLES_LINES", [])
    if len(singles) != 4 or len(doubles) != 4:
        return False, "vertical_line_points_missing"
    if not (doubles[0][0] <= singles[0][0] <= singles[2][0] <= doubles[2][0]):
        return False, "far_vertical_order_invalid"
    if not (doubles[1][0] <= singles[1][0] <= singles[3][0] <= doubles[3][0]):
        return False, "near_vertical_order_invalid"

    return True, "ok"


def collect_candidate_image_sources(input_path, frames_dir=None, max_frames=15, search_first=90):
    input_path = Path(input_path)
    frame_dir = Path(frames_dir) if frames_dir is not None else None

    if frame_dir is None and input_path.suffix.lower() in IMAGE_EXTS and input_path.parent.name == "frames":
        frame_dir = input_path.parent

    if frame_dir is not None and frame_dir.exists():
        frame_paths = sorted(
            list(frame_dir.glob("*.jpg")) +
            list(frame_dir.glob("*.jpeg")) +
            list(frame_dir.glob("*.png"))
        )
        if frame_paths:
            frame_paths = frame_paths[:max(1, min(search_first, len(frame_paths)))]
            if len(frame_paths) <= max_frames:
                return frame_paths
            idxs = np.linspace(0, len(frame_paths) - 1, max_frames)
            return [frame_paths[int(round(i))] for i in idxs]

    return [input_path]


def run_line_detector_on_image(image_path, debug_dir=None, write_debug_inputs=False):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    ignore_mask = build_static_ignore_mask(img.shape)
    eq, gray, blur, edges, roi, edges_roi, white = preprocess(img, ignore_mask=ignore_mask)
    raw = detect_segments(edges_roi, img.shape)
    merged = merge_segments(raw, angle_thresh=4.0, dist_thresh=14, gap_thresh=28)

    if debug_dir is not None and write_debug_inputs:
        debug_dir = Path(debug_dir)
        ensure_dir(debug_dir)
        cv2.imwrite(str(debug_dir / "01_original.jpg"), img)
        cv2.imwrite(str(debug_dir / "02_equalized.jpg"), eq)
        cv2.imwrite(str(debug_dir / "03_gray.jpg"), gray)
        cv2.imwrite(str(debug_dir / "04_edges.jpg"), edges)
        cv2.imwrite(str(debug_dir / "05_roi.jpg"), roi)
        cv2.imwrite(str(debug_dir / "06_edges_roi.jpg"), edges_roi)
        cv2.imwrite(str(debug_dir / "07_white.jpg"), white)
        cv2.imwrite(str(debug_dir / "08_raw_segments.jpg"), draw_segs(img, raw, (0, 255, 255), 2))
        cv2.imwrite(str(debug_dir / "09_merged_segments.jpg"), draw_segs(img, merged, (255, 255, 0), 2))

    result = reconstruct_from_five_line_method(img, edges_roi, white, merged, debug_dir=debug_dir)
    if result is None:
        return {
            "image_path": Path(image_path),
            "img": img,
            "raw_count": len(raw),
            "merged_count": len(merged),
            "result": None,
            "cfg": None,
            "valid": False,
            "reason": "reconstruction_failed",
        }

    try:
        cfg = build_roi_config_from_result(image_path, img.shape, result)
        valid, reason = validate_roi_config(cfg, img.shape)
    except Exception as exc:
        cfg = None
        valid = False
        reason = f"config_failed:{exc}"

    return {
        "image_path": Path(image_path),
        "img": img,
        "raw_count": len(raw),
        "merged_count": len(merged),
        "result": result,
        "cfg": cfg,
        "valid": valid,
        "reason": reason,
    }


def roi_corner_distance(cfg_a, cfg_b):
    a = np.asarray(cfg_a["ROI_POLY"], dtype=np.float32)
    b = np.asarray(cfg_b["ROI_POLY"], dtype=np.float32)
    if a.shape != b.shape:
        return 1e9
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def select_best_candidate(candidates):
    valid = [c for c in candidates if c.get("valid") and c.get("cfg") is not None and c.get("result") is not None]
    if not valid:
        return None, [{
            "frame": str(c.get("image_path", "")),
            "valid": False,
            "reason": c.get("reason", "unknown"),
            "raw_segments": int(c.get("raw_count", 0)),
            "merged_segments": int(c.get("merged_count", 0)),
        } for c in candidates]

    summary = []
    for c in valid:
        distances = [roi_corner_distance(c["cfg"], other["cfg"]) for other in valid if other is not c]
        median_dist = float(np.median(distances)) if distances else 0.0
        support = int(sum(d <= 24.0 for d in distances)) + 1
        warning_count = len(c["result"].get("warnings", []))
        adjusted_score = float(c["result"].get("score", 0.0)) - 0.75 * median_dist + 28.0 * support - 55.0 * warning_count
        c["adjusted_score"] = adjusted_score
        c["support"] = support
        c["median_corner_distance"] = median_dist
        summary.append({
            "frame": str(c["image_path"]),
            "valid": True,
            "reason": c["reason"],
            "score": float(c["result"].get("score", 0.0)),
            "adjusted_score": adjusted_score,
            "support": support,
            "median_corner_distance": median_dist,
            "raw_segments": int(c["raw_count"]),
            "merged_segments": int(c["merged_count"]),
            "line_sources": c["result"].get("line_sources", {}),
            "warnings": c["result"].get("warnings", []),
        })

    valid_ids = {id(c) for c in valid}
    invalid_summary = [{
        "frame": str(c["image_path"]),
        "valid": False,
        "reason": c.get("reason", "unknown"),
        "raw_segments": int(c.get("raw_count", 0)),
        "merged_segments": int(c.get("merged_count", 0)),
    } for c in candidates if id(c) not in valid_ids]

    best = max(valid, key=lambda c: c["adjusted_score"])
    return best, summary + invalid_summary


def save_selected_debug_images(debug_dir, selected):
    debug_dir = Path(debug_dir)
    ensure_dir(debug_dir)
    img = selected["img"]
    result = selected["result"]
    group_vis = visualize_groups(img, result)
    five_vis = visualize_main_five(img, result)

    net_top_line = None
    if result.get("net_lines") is not None:
        net_top_line = result["net_lines"].get("net_top_line")

    court_vis = draw_projected_model(
        img,
        result["H"],
        result["model"],
        (0, 255, 255),
        2,
        net_top_line=net_top_line
    )

    cv2.imwrite(str(debug_dir / "10_groups.jpg"), group_vis)
    cv2.imwrite(str(debug_dir / "11_main_five_lines.jpg"), five_vis)
    cv2.imwrite(str(debug_dir / "12_main_roi.jpg"), result["main_roi"])
    cv2.imwrite(str(debug_dir / "13_projected_court.jpg"), court_vis)
    cv2.imwrite(str(debug_dir / "14_projected_model_debug.jpg"), court_vis)
    cv2.imwrite(str(debug_dir / "14_model_mask.jpg"), result["model_mask"])
    cv2.imwrite(str(debug_dir / "15_overlap_edges.jpg"), result["overlap_edges"])
    cv2.imwrite(str(debug_dir / "16_overlap_white.jpg"), result["overlap_white"])


def detect_best_court(input_path, out_json=None, output_dir=None):
    paths = setup_detector_paths(input_path, output_dir=output_dir)

    debug_dir = paths["OUT_DIR"]
    if output_dir is None and out_json is not None:
        debug_dir = Path(out_json).parent / "court_detector_debug"
    debug_dir = Path(debug_dir)
    ensure_dir(debug_dir)

    candidates_paths = collect_candidate_image_sources(paths["INPUT_PATH"], paths["FRAMES_DIR"])
    candidates = []
    for image_path in candidates_paths:
        candidate = run_line_detector_on_image(image_path, debug_dir=None, write_debug_inputs=False)
        if candidate is not None:
            candidates.append(candidate)

    selected, summary = select_best_candidate(candidates)
    with open(debug_dir / "15_multiframe_candidates.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if selected is None:
        return None, None, debug_dir

    selected = run_line_detector_on_image(selected["image_path"], debug_dir=debug_dir, write_debug_inputs=True)
    if selected is None or selected["result"] is None:
        return None, None, debug_dir

    result = selected["result"]
    result["frames_sampled"] = len(candidates)
    result["selected_frame"] = str(selected["image_path"])
    result["score"] = float(selected["result"].get("score", 0.0))

    cfg = build_roi_config_from_result(selected["image_path"], selected["img"].shape, result)
    valid, reason = validate_roi_config(cfg, selected["img"].shape)
    if not valid:
        result.setdefault("warnings", []).append(reason)
        cfg = build_roi_config_from_result(selected["image_path"], selected["img"].shape, result)

    save_selected_debug_images(debug_dir, selected)
    with open(debug_dir / "16_selected_config_preview.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return cfg, selected, debug_dir


def auto_pick_roi(input_path, out_json=None, output_dir=None):
    cfg, selected, debug_dir = detect_best_court(input_path, out_json=out_json, output_dir=output_dir)
    if cfg is None:
        return None

    if out_json is None:
        paths = setup_detector_paths(input_path, output_dir=output_dir)
        out_json = paths["ROI_JSON"]

    out_json = Path(out_json)
    ensure_dir(out_json.parent)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return cfg

# =========================================================
# entry
# =========================================================
def main(input_path, output_dir=None):
    paths = setup_detector_paths(input_path, output_dir=output_dir)
    roi_json = paths["ROI_JSON"]

    ensure_dir(roi_json.parent)

    cfg, selected, out_dir = detect_best_court(input_path, output_dir=output_dir)
    if cfg is None or selected is None:
        print("Could not reconstruct court with five-line method.")
        return

    with open(roi_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Done. debug saved to {out_dir}")
    print(f"roi_config saved to {roi_json}")
    print(f"image source = {selected['image_path']}")
    print(f"raw segments = {selected['raw_count']}")
    print(f"merged segments = {selected['merged_count']}")
    print(f"score = {selected['result']['score']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python court_detector.py <video_path_or_image_path> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else None

    main(input_path, output_dir)
