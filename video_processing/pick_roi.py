from matplotlib.path import Path
import cv2, json, numpy as np
from court_detector import auto_pick_roi

def run_pick_roi(image_path, out_json, max_show=1200, auto=False, ):
    """
    完整互動式 pick ROI 工具，可被 preprocessing 呼叫。
    保留原本所有提示文字與 UI 操作方式。
    """
    if auto:
        try:
            out = auto_pick_roi(image_path, out_json=out_json)
            if out is not None:
                print("\n==== Auto ROI Config Saved ====")
                print(f"-> {out_json}\n")
                return out
            else:
                print("[WARN] auto ROI failed, fallback to manual mode.")
        except Exception as e:
            print(f"[WARN] auto ROI exception: {e}")
            print("[WARN] fallback to manual mode.")  
        
    # -----------------------------------------------------------
    # Load image
    # -----------------------------------------------------------
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    scale = min(1.0, max_show / max(H, W))
    disp = cv2.resize(img, (int(W * scale), int(H * scale))) if scale < 1.0 else img.copy()

    def to_orig(pt):
        x, y = pt
        return (int(round(x / scale)), int(round(y / scale)))

    # -----------------------------------------------------------
    # STEP 1 — ROI_POLY
    # -----------------------------------------------------------
    pts_roi = []
    done_roi = False

    def on_mouse_roi(event, x, y, flags, param):
        nonlocal pts_roi, done_roi
        if done_roi:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_roi.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts_roi:
            pts_roi.pop()

    win1 = "STEP1 ROI_POLY  (L-click add, R-click undo, ENTER finish)"
    cv2.namedWindow(win1)
    cv2.setMouseCallback(win1, on_mouse_roi)

    while True:
        canvas = disp.copy()
        # draw points
        for i, (x, y) in enumerate(pts_roi):
            cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
            if i > 0:
                cv2.line(canvas, pts_roi[i - 1], (x, y), (0, 255, 255), 2)

        # draw polyline
        if len(pts_roi) >= 3:
            cv2.polylines(canvas, [np.array(pts_roi, np.int32)], False, (0, 255, 0), 2)

        tip = "STEP1 ROI: L-click add, R-click undo, ENTER finish"
        cv2.putText(canvas, tip, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 240, 20), 2)

        cv2.imshow(win1, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts_roi) >= 3:  # ENTER
            done_roi = True
            break

    cv2.destroyWindow(win1)
    ROI_POLY = [to_orig(p) for p in pts_roi]

    # -----------------------------------------------------------
    # STEP 2 — BAN_BOXES
    # -----------------------------------------------------------
    BAN_BOXES = []
    disp_step2 = disp.copy()

    while True:
        msg = "STEP2 BAN: drag rectangles (scoreboard etc.). ENTER on empty box to finish."
        cv2.putText(disp_step2, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 240, 20), 2)

        r = cv2.selectROI("STEP2 BAN_BOXES (ENTER when done)", disp_step2, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("STEP2 BAN_BOXES (ENTER when done)")

        x, y, w, h = map(int, r)
        if w == 0 or h == 0:
            break  # finish

        x0, y0 = to_orig((x, y))
        w0 = int(round(w / scale))
        h0 = int(round(h / scale))
        BAN_BOXES.append((x0, y0, w0, h0))

    # -----------------------------------------------------------
    # STEP 3 — FAR_WARP_SRC_PTS
    # -----------------------------------------------------------
    pts_warp = []
    disp_step3 = disp.copy()

    def on_mouse_warp(event, x, y, flags, param):
        nonlocal pts_warp
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_warp) < 4:
            pts_warp.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts_warp:
            pts_warp.pop()

    win3 = "STEP3 FAR_WARP PTS  (order: TL, TR, BR, BL)"
    cv2.namedWindow(win3)
    cv2.setMouseCallback(win3, on_mouse_warp)

    while True:
        canvas = disp_step3.copy()

        hint = "STEP3 FAR_WARP: click TL, TR, BR, BL (R-undo), ENTER when 4 pts selected."
        cv2.putText(canvas, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        for p in pts_warp:
            cv2.circle(canvas, p, 5, (255, 0, 255), -1)

        if len(pts_warp) >= 2:
            cv2.polylines(canvas, [np.array(pts_warp, np.int32)], False, (255, 0, 255), 2)

        cv2.imshow(win3, canvas)

        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts_warp) == 4:
            break

    cv2.destroyWindow(win3)
    FAR_WARP_SRC_PTS = [to_orig(p) for p in pts_warp]

    br_y = max(FAR_WARP_SRC_PTS[2][1], FAR_WARP_SRC_PTS[3][1])
    FAR_Y_RATIO_EDGE = float(br_y) / float(H)
    FAR_WARP_DST_SIZE = [640, 360]

    # -----------------------------------------------------------
    # STEP 4 — MID_LINE
    # -----------------------------------------------------------
    mid_pts = []
    disp_step4 = disp.copy()

    def on_mouse_mid(event, x, y, flags, param):
        nonlocal mid_pts
        if event == cv2.EVENT_LBUTTONDOWN and len(mid_pts) < 2:
            mid_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and mid_pts:
            mid_pts.pop()

    win4 = "STEP4 MID_LINE (click top & bottom)"
    cv2.namedWindow(win4)
    cv2.setMouseCallback(win4, on_mouse_mid)

    while True:
        canvas = disp_step4.copy()
        tip = "STEP4: click top & bottom of midline (R-undo, ENTER confirm)"
        cv2.putText(canvas, tip, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for p in mid_pts:
            cv2.circle(canvas, p, 5, (0, 0, 255), -1)

        if len(mid_pts) == 2:
            cv2.line(canvas, mid_pts[0], mid_pts[1], (0, 0, 255), 2)

        cv2.imshow(win4, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(mid_pts) == 2:
            break

    cv2.destroyWindow(win4)
    MID_LINE = [to_orig(p) for p in mid_pts]

    # -----------------------------------------------------------
    # STEP 5 — SINGLES_LINES
    # -----------------------------------------------------------
    singles_pts = []
    disp_step5 = disp.copy()

    def on_mouse_single(event, x, y, flags, param):
        nonlocal singles_pts
        if event == cv2.EVENT_LBUTTONDOWN and len(singles_pts) < 4:
            singles_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and singles_pts:
            singles_pts.pop()

    win5 = "STEP5 SINGLES_LINES (click TL, BL, TR, BR)"
    cv2.namedWindow(win5)
    cv2.setMouseCallback(win5, on_mouse_single)

    while True:
        canvas = disp_step5.copy()

        tip = "STEP5: click TL, BL, TR, BR (R-undo, ENTER confirm)"
        cv2.putText(canvas, tip, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for p in singles_pts:
            cv2.circle(canvas, p, 5, (0, 255, 255), -1)

        if len(singles_pts) == 4:
            cv2.polylines(canvas, [np.array(singles_pts, np.int32)], True, (0, 255, 255), 2)

        cv2.imshow(win5, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(singles_pts) == 4:
            break

    cv2.destroyWindow(win5)
    SINGLES_LINES = [to_orig(p) for p in singles_pts]

    # -----------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------
    out = {
        "image_path": str(image_path),
        "image_size": [W, H],
        "ROI_POLY": ROI_POLY,
        "BAN_BOXES": BAN_BOXES,
        "FAR_WARP_SRC_PTS": FAR_WARP_SRC_PTS,
        "FAR_Y_RATIO_EDGE": FAR_Y_RATIO_EDGE,
        "FAR_WARP_DST_SIZE": FAR_WARP_DST_SIZE,
        "MID_LINE": MID_LINE,
        "SINGLES_LINES": SINGLES_LINES
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n==== Saved ROI Config ====")
    print(f"-> {out_json}\n")

    return out