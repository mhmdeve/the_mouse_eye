#!/usr/bin/env python3
"""
the_mouse_eye - control your mouse with your eyes using a laptop webcam.

  * Gaze (iris + head position) moves the cursor (after a 9-point calibration).
  * Blink BOTH eyes (hold ~0.5 s)  -> left click
  * Wink LEFT eye   (hold ~0.3 s)  -> left click
  * Wink RIGHT eye  (hold ~0.3 s)  -> right click
  * Scroll: look at the TOP strip of the screen to scroll up,
            look at the BOTTOM strip of the screen to scroll down.

Keys (in the preview window):
  c = recalibrate    p = pause / resume    q or ESC = quit

Requires: opencv-python, mediapipe, numpy, pyautogui
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import pyautogui

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe is not installed. Run: pip install -r requirements.txt")

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

# ----------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh, refine_landmarks=True)
# The frame is mirrored before processing, so index group "33..." is the
# user's LEFT eye as seen in the mirror view, and "362..." is the RIGHT eye.
# Use --swap-eyes if your left/right clicks come out reversed.
# ----------------------------------------------------------------------------
EYE_A = {"p": [33, 160, 158, 133, 153, 144], "iris": [468, 469, 470, 471, 472],
         "outer": 33, "inner": 133}
EYE_B = {"p": [362, 385, 387, 263, 373, 380], "iris": [473, 474, 475, 476, 477],
         "outer": 263, "inner": 362}
NOSE_TIP = 1


def ear(pts):
    """Eye Aspect Ratio from 6 points (p1..p6)."""
    p1, p2, p3, p4, p5, p6 = pts
    v = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    h = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return v / h


class OneEuro:
    """One Euro filter: smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class ClickDetector:
    """
    Turns eye-closed states into clicks. A click fires once while the eye(s)
    stay closed longer than the hold time, so natural blinks (<0.25 s) are
    ignored.
    """

    def __init__(self, wink_hold, blink_hold, cooldown):
        self.wink_hold, self.blink_hold, self.cooldown = wink_hold, blink_hold, cooldown
        self.t_left = None
        self.t_right = None
        self.fired = False
        self.last_click = 0.0

    def update(self, left_closed, right_closed, now):
        """Returns 'left', 'right' or None."""
        self.t_left = (self.t_left or now) if left_closed else None
        self.t_right = (self.t_right or now) if right_closed else None

        if not left_closed and not right_closed:
            self.fired = False
            return None
        if self.fired or now - self.last_click < self.cooldown:
            return None

        dl = now - self.t_left if self.t_left else 0
        dr = now - self.t_right if self.t_right else 0

        action = None
        if left_closed and right_closed:
            if min(dl, dr) >= self.blink_hold:
                action = "left"           # both eyes -> left click
        elif left_closed and dl >= self.wink_hold and not right_closed:
            # make sure the other eye has been open for the whole wink
            action = "left"
        elif right_closed and dr >= self.wink_hold and not left_closed:
            action = "right"

        if action:
            self.fired = True
            self.last_click = now
        return action


class EyeMouse:
    def __init__(self, args):
        self.args = args
        self.screen_w, self.screen_h = pyautogui.size()
        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open camera {args.camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6)

        self.fx = OneEuro(min_cutoff=args.smooth, beta=0.05)
        self.fy = OneEuro(min_cutoff=args.smooth, beta=0.05)
        self.clicks = ClickDetector(args.wink_time, args.blink_time, 0.6)
        self.model = None          # 2 x 5 matrix mapping features -> screen (0..1)
        self.open_ear = 0.30       # updated during calibration
        self.paused = False
        self.last_scroll = 0.0
        self.eye_a, self.eye_b = (EYE_B, EYE_A) if args.swap_eyes else (EYE_A, EYE_B)

    # ------------------------------------------------------------------ vision
    def read_features(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return frame, None
        lm = res.multi_face_landmarks[0].landmark
        pt = lambda i: np.array([lm[i].x * w, lm[i].y * h])

        def eye_info(eye):
            pts = [pt(i) for i in eye["p"]]
            iris = np.mean([pt(i) for i in eye["iris"]], axis=0)
            a, b = pt(eye["outer"]), pt(eye["inner"])
            axis = b - a
            width = np.linalg.norm(axis) + 1e-6
            u = axis / width
            n = np.array([-u[1], u[0]])
            centre = (a + b) / 2
            gx = np.dot(iris - centre, u) / width       # along the eye
            gy = np.dot(iris - centre, n) / width       # across the eye
            return ear(pts), gx, gy

        ear_a, gxa, gya = eye_info(self.eye_a)
        ear_b, gxb, gyb = eye_info(self.eye_b)
        nose = pt(NOSE_TIP)
        feats = np.array([
            (gxa + gxb) / 2 * 10, (gya + gyb) / 2 * 10,
            nose[0] / w * 3, nose[1] / h * 3, 1.0])
        info = {"ear_l": ear_a, "ear_r": ear_b, "feats": feats}
        # draw a little feedback
        for i in self.eye_a["p"] + self.eye_b["p"]:
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 1, (0, 255, 0), -1)
        return frame, info

    # ------------------------------------------------------------- calibration
    def calibrate(self):
        grid = [(0.08, 0.08), (0.5, 0.08), (0.92, 0.08),
                (0.08, 0.5), (0.5, 0.5), (0.92, 0.5),
                (0.08, 0.92), (0.5, 0.92), (0.92, 0.92)]
        win = "calibration"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        X, Y, ears = [], [], []
        for k, (tx, ty) in enumerate(grid):
            t0 = time.time()
            samples = []
            while time.time() - t0 < 2.2:
                canvas = np.zeros((self.screen_h, self.screen_w, 3), np.uint8)
                cx, cy = int(tx * self.screen_w), int(ty * self.screen_h)
                r = int(18 + 10 * np.sin((time.time() - t0) * 6))
                cv2.circle(canvas, (cx, cy), r, (0, 200, 255), -1)
                cv2.circle(canvas, (cx, cy), 4, (0, 0, 0), -1)
                cv2.putText(canvas, f"Look at the dot ({k + 1}/9) - keep your head natural",
                            (self.screen_w // 2 - 330, self.screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyWindow(win)
                    return False
                _, info = self.read_features()
                if info and time.time() - t0 > 0.9:       # let the eyes settle
                    samples.append(info["feats"])
                    ears.append((info["ear_l"] + info["ear_r"]) / 2)
            if len(samples) < 5:
                print("Face not detected during calibration, try again.")
                cv2.destroyWindow(win)
                return False
            X.append(np.median(samples, axis=0))
            Y.append([tx, ty])
        cv2.destroyWindow(win)

        X, Y = np.array(X), np.array(Y)
        # ridge-regularised least squares
        lam = 1e-3
        A = X.T @ X + lam * np.eye(X.shape[1])
        self.model = np.linalg.solve(A, X.T @ Y).T       # 2 x 5
        self.open_ear = float(np.percentile(ears, 70))
        with open(CALIB_FILE, "w") as f:
            json.dump({"model": self.model.tolist(), "open_ear": self.open_ear,
                       "screen": [self.screen_w, self.screen_h]}, f)
        print(f"Calibration saved. open EAR = {self.open_ear:.3f}")
        return True

    def load_calibration(self):
        try:
            with open(CALIB_FILE) as f:
                d = json.load(f)
            if d.get("screen") != [self.screen_w, self.screen_h]:
                return False
            self.model = np.array(d["model"])
            self.open_ear = d["open_ear"]
            return True
        except (OSError, ValueError, KeyError):
            return False

    # -------------------------------------------------------------------- main
    def run(self):
        if self.args.recalibrate or not self.load_calibration():
            print("Starting calibration: follow the dot with your eyes (ESC to abort).")
            if not self.calibrate():
                sys.exit("Calibration failed.")
        closed_thr = self.open_ear * self.args.closed_ratio
        print("Running. Keys in preview window: c=recalibrate, p=pause, q/ESC=quit")

        while True:
            frame, info = self.read_features()
            now = time.time()
            if frame is None:
                break
            status = "no face"
            if info and not self.paused:
                # --- eye closed? (EAR gets unreliable for gaze while closed)
                l_closed = info["ear_l"] < closed_thr
                r_closed = info["ear_r"] < closed_thr

                if not (l_closed or r_closed):
                    sx, sy = np.clip(self.model @ info["feats"], 0, 1)
                    x = self.fx(sx * self.screen_w, now)
                    y = self.fy(sy * self.screen_h, now)
                    x = int(np.clip(x, 2, self.screen_w - 3))
                    y = int(np.clip(y, 2, self.screen_h - 3))
                    pyautogui.moveTo(x, y, _pause=False)
                    self._scroll(y, now)
                    status = f"gaze {x},{y}"

                action = self.clicks.update(l_closed, r_closed, now)
                if action == "left":
                    pyautogui.click(button="left")
                    status = "LEFT CLICK"
                elif action == "right":
                    pyautogui.click(button="right")
                    status = "RIGHT CLICK"
            elif self.paused:
                status = "paused"

            if frame is not None:
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.imshow("the_mouse_eye (q to quit)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("p"):
                self.paused = not self.paused
            if k == ord("c"):
                self.calibrate()
                closed_thr = self.open_ear * self.args.closed_ratio

        self.cap.release()
        cv2.destroyAllWindows()

    def _scroll(self, y, now):
        """Scroll when gaze is in the top / bottom strip of the screen."""
        if now - self.last_scroll < 0.05:
            return
        zone = self.args.scroll_zone * self.screen_h
        if y < zone:
            depth = (zone - y) / zone
            pyautogui.scroll(int(self.args.scroll_speed * (0.3 + depth)))
            self.last_scroll = now
        elif y > self.screen_h - zone:
            depth = (y - (self.screen_h - zone)) / zone
            pyautogui.scroll(-int(self.args.scroll_speed * (0.3 + depth)))
            self.last_scroll = now


def parse_args():
    p = argparse.ArgumentParser(description="Control the mouse with your eyes.")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--recalibrate", action="store_true", help="force calibration")
    p.add_argument("--swap-eyes", action="store_true", help="swap left/right wink buttons")
    p.add_argument("--smooth", type=float, default=0.8,
                   help="cursor smoothing, lower = smoother (default 0.8)")
    p.add_argument("--wink-time", type=float, default=0.30, help="seconds to hold a wink")
    p.add_argument("--blink-time", type=float, default=0.50,
                   help="seconds to hold a both-eyes blink")
    p.add_argument("--closed-ratio", type=float, default=0.62,
                   help="eye counts as closed below this fraction of open EAR")
    p.add_argument("--scroll-zone", type=float, default=0.07,
                   help="top/bottom screen fraction that triggers scroll")
    p.add_argument("--scroll-speed", type=int, default=60, help="scroll amount per tick")
    return p.parse_args()


if __name__ == "__main__":
    EyeMouse(parse_args()).run()
