import numpy as np

# =============================================================
#                          Kalman2D
# =============================================================
class Kalman2D:
    def __init__(self):
        self.x = np.zeros((4,1))      # [x, y, vx, vy]
        self.F = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.P = np.eye(4) * 400
        self.Q = np.eye(4) * 1.2
        self.R = np.eye(2) * 25
        self.initialized = False

    def init_state(self, cx, cy):
        self.x = np.array([[cx],[cy],[0],[0]], float)
        self.P = np.eye(4)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].ravel()

    def update(self, cx, cy):
        if not self.initialized:
            self.init_state(cx, cy)
            return self.x[:2].ravel()

        z = np.array([[cx],[cy]], float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].ravel()


# =============================================================
#                  TwoPlayerTracker (Ultimate Version)
# =============================================================
class TwoPlayerTracker:
    def __init__(self, NET_LINE, max_miss=12, line_judge_max_y=20):

        # ----- 網子中線：取全部 y 平均 -----
        ys = [p[1] for p in NET_LINE]
        self.net_y = sum(ys) / len(ys)

        # ----- 卡爾曼 -----
        self.kf1 = Kalman2D()     # Player1 = near (cy > mid_y)
        self.kf2 = Kalman2D()     # Player2 = far  (cy < mid_y)

        # ----- box 大小 smoothing -----
        self.near_w = None
        self.near_h = None
        self.far_w  = None
        self.far_h  = None

        # ----- miss -----
        self.miss1 = 0
        self.miss2 = 0
        self.max_miss = max_miss

        # ----- 線審區域（只濾掉極端靠上的噪音）-----
        self.LINE_JUDGE_Y_MAX = line_judge_max_y
        
        self.last_pos1 = None
        self.last_pos2 = None

    # =============================================================
    #                       Utility
    # =============================================================
    def _make_box(self, cx, cy, w, h):
        return (
            int(cx - w/2), int(cy - h/2),
            int(cx + w/2), int(cy + h/2),
            1.0
        )

    def _center(self, det):
        x1,y1,x2,y2,conf = det
        return ((x1+x2)/2, (y1+y2)/2)


    # =============================================================
    #                          UPDATE()
    # =============================================================
    def update(self, detections):
        # 取得所有偵測框的中心與 w,h
        det_boxes = []
        for det in detections:
            x1,y1,x2,y2,conf = det
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            w = x2 - x1
            h = y2 - y1
            det_boxes.append(((cx,cy),(w,h)))

        # --------------------------
        # step 1: YOLO 轉成中心點 + 過濾線審
        # --------------------------
        det_centers = [self._center(d) for d in detections]
        filtered = [(cx,cy) for (cx,cy) in det_centers if cy > self.LINE_JUDGE_Y_MAX]

        # Kalman predict
        p1 = self.kf1.predict()
        p2 = self.kf2.predict()

        # --------------------------
        # case 0: 沒人偵測
        # --------------------------
        if len(filtered) == 0:
            self.miss1 += 1
            self.miss2 += 1
            return self._output()


        # --------------------------
        # case 1: 一人偵測 —— 用 mid_y 判斷是誰 + 更新框大小
        # --------------------------
        if len(filtered) == 1:
            cx, cy = filtered[0]
            is_near = cy > self.net_y

            # 找出 YOLO 對應框的 w,h
            # 從 det_boxes 找出中心最接近 (cx,cy) 的那個
            matched_w = None
            matched_h = None
            for (c,(w,h)) in det_boxes:
                if abs(c[0]-cx) < 5 and abs(c[1]-cy) < 5:
                    matched_w = w
                    matched_h = h
                    break

            if is_near:
                # ---- 更新 Player1 ----
                self.kf1.update(cx, cy)
                self.miss1 = 0

                # ---- 框框大小 smoothing ----
                if matched_w is not None:
                    if self.near_w is None:
                        self.near_w, self.near_h = matched_w, matched_h
                    else:
                        self.near_w = 0.7*self.near_w + 0.3*matched_w
                        self.near_h = 0.7*self.near_h + 0.3*matched_h

                # ---- Player2 predict only ----
                self.miss2 += 1
                if self.last_pos2 is not None:
                    self.kf2.x[0,0], self.kf2.x[1,0] = self.last_pos2

            else:
                # ---- 更新 Player2 ----
                self.kf2.update(cx, cy)
                self.miss2 = 0

                # ---- 框框大小 smoothing ----
                if matched_w is not None:
                    if self.far_w is None:
                        self.far_w, self.far_h = matched_w, matched_h
                    else:
                        self.far_w = 0.7*self.far_w + 0.3*matched_w
                        self.far_h = 0.7*self.far_h + 0.3*matched_h

                # ---- Player1 predict only ----
                self.miss1 += 1
                if self.last_pos1 is not None:
                    self.kf1.x[0,0], self.kf1.x[1,0] = self.last_pos1

            return self._output()

        # --------------------------
        # case 2: 雙人偵測
        # --------------------------
        sorted_pts = sorted(filtered, key=lambda c: c[1])
        far = sorted_pts[0]
        near = sorted_pts[-1]

        # ----- Player2 = far -----
        if far[1] < self.net_y:  # 遠端區域
            if not self.kf2.initialized:
                self.kf2.init_state(far[0], far[1])
            else:
                self.kf2.update(far[0], far[1])

            # --- 框框大小（EMA smoothing）---
            for (c,(w,h)) in det_boxes:
                if abs(c[0]-far[0])<5 and abs(c[1]-far[1])<5:
                    if self.far_w is None:
                        self.far_w, self.far_h = w, h
                    else:
                        self.far_w = 0.7*self.far_w + 0.3*w
                        self.far_h = 0.7*self.far_h + 0.3*h

            self.miss2 = 0
        else:
            self.miss2 += 1


        # ----- Player1 = near -----
        if near[1] > self.net_y:  # 近端區域
            if not self.kf1.initialized:
                self.kf1.init_state(near[0], near[1])
            else:
                self.kf1.update(near[0], near[1])

            # --- 框框大小（EMA smoothing）---
            for (c,(w,h)) in det_boxes:
                if abs(c[0]-near[0])<5 and abs(c[1]-near[1])<5:
                    if self.near_w is None:
                        self.near_w, self.near_h = w, h
                    else:
                        self.near_w = 0.7*self.near_w + 0.3*w
                        self.near_h = 0.7*self.near_h + 0.3*h

            self.miss1 = 0
        else:
            self.miss1 += 1

        return self._output()

    # =============================================================
    #                         OUTPUT()
    # =============================================================
    def _output(self):
        outs = []

        # Player1 (near)
        if self.kf1.initialized and self.miss1 <= self.max_miss:
            cx, cy = self.kf1.x[:2].ravel()
            w = self.near_w if self.near_w else 50
            h = self.near_h if self.near_h else 100
            outs.append(("Player1", self._make_box(cx, cy, w, h)))

        # Player2 (far)
        if self.kf2.initialized and self.miss2 <= self.max_miss:
            cx, cy = self.kf2.x[:2].ravel()
            w = self.far_w if self.far_w else 40
            h = self.far_h if self.far_h else 80
            outs.append(("Player2", self._make_box(cx, cy, w, h)))
            
        # Update last stable positions
        if self.kf1.initialized and self.miss1 == 0:
            self.last_pos1 = self.kf1.x[:2].ravel()

        if self.kf2.initialized and self.miss2 == 0:
            self.last_pos2 = self.kf2.x[:2].ravel()

        return outs