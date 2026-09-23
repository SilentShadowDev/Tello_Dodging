"""
Ball-tracking distance-trend obstacle reactor.

Simulates the acoustic-shield drone's evasion logic using a webcam + a
colored ball standing in for an ultrasonic/ToF sensor reading. Reaction
is driven by *how the distance is changing* (closing vs. opening), using
a time-to-contact estimate, rather than static "is it inside this band"
thresholds.
"""

import time
from collections import deque

import cv2
import numpy as np

# =====================================================================
# CONFIG — single entry point for all tunables
# =====================================================================
CONFIG = {
    # --- Color detection (HSV) for the tracked ball ---
    "lower_yellow": np.array([20, 100, 100]),
    "upper_yellow": np.array([35, 255, 255]),
    "min_contour_area": 300,
    "min_radius": 2,

    # --- Calibration ---
    "reference_distance_cm": 100,     # hold the ball this far away, press 'c'

    # --- Distance-trend estimation ---
    "distance_window_s": 0.6,         # seconds of history used for the rate fit
    "min_samples_for_rate": 4,        # don't trust a rate until we have this many points
    "closing_rate_deadband_cm_s": 8,  # ignore jitter below this speed (either direction)

    # --- Reaction thresholds (time-to-contact, not raw distance bands) ---
    "critical_ttc_s": 1.0,            # closing this fast relative to distance -> emergency
    "warning_ttc_s": 2.5,             # closing this fast -> reposition
    "hard_floor_cm": 30,              # fail-safe: always react under this, even if not closing

    # --- Lateral collision-course prediction (same idea as before, now time-based) ---
    "lookahead_s": 0.5,
    "grid_size": 5,
    "center_cell": (2, 2),            # (row, col)

    # --- Cooldowns ---
    "reposition_cooldown_s": 1.0,
    "dodge_cooldown_s": 0.5,
}


# =====================================================================
# Vision helpers
# =====================================================================
def get_mask(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["lower_yellow"], cfg["upper_yellow"])
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask


def find_ball(mask, cfg):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < cfg["min_contour_area"]:
        return None

    (x, y), radius = cv2.minEnclosingCircle(c)
    if radius < cfg["min_radius"]:
        return None

    return int(x), int(y), int(radius)


# =====================================================================
# Trend / time-to-contact estimation
# =====================================================================
def linear_rate(samples):
    """
    Least-squares slope of value vs. time over a list of (t, value) samples.
    Robust to uneven webcam frame timing and per-frame measurement jitter.
    Returns units-per-second, or None if there isn't enough data.
    """
    if len(samples) < 2:
        return None
    t = np.array([s[0] for s in samples], dtype=np.float64)
    v = np.array([s[1] for s in samples], dtype=np.float64)
    t = t - t[0]
    if t[-1] == 0:
        return None
    a, _ = np.polyfit(t, v, 1)  # v = a*t + b -> a is the rate
    return a


def get_cell(x, y, frame_width, frame_height, cfg):
    grid_size = cfg["grid_size"]
    col = min(int(x / (frame_width / grid_size)), grid_size - 1)
    row = min(int(y / (frame_height / grid_size)), grid_size - 1)
    return row, col


def decide_direction(x, y, frame_width, frame_height):
    h = "right" if x < frame_width // 2 else "left"
    v = "up" if y > frame_height // 2 else "down"
    return h, v


def draw_grid(frame, frame_width, frame_height, cfg):
    grid_size = cfg["grid_size"]
    cell_w = frame_width // grid_size
    cell_h = frame_height // grid_size

    for i in range(1, grid_size):
        cv2.line(frame, (i * cell_w, 0), (i * cell_w, frame_height), (100, 100, 100), 1)
        cv2.line(frame, (0, i * cell_h), (frame_width, i * cell_h), (100, 100, 100), 1)

    row, col = cfg["center_cell"]
    cx1, cy1 = col * cell_w, row * cell_h
    cv2.rectangle(frame, (cx1, cy1), (cx1 + cell_w, cy1 + cell_h), (255, 255, 0), 2)


# =====================================================================
# Tracker — owns all running state and the trend-based decision logic
# =====================================================================
class ObstacleTracker:
    """
    Holds calibration, distance history, lateral position history, and
    cooldown timers (previously scattered module-level globals). This is
    what makes the closing-rate / time-to-contact calculation possible,
    and it drops in cleanly wherever the ball is swapped for a real
    ultrasonic/ToF sensor reading later.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reference_radius = None
        self.distance_history = deque()   # (timestamp, distance_cm)
        self.position_history = deque()   # (timestamp, x, y)
        self.last_dodge_time = 0.0
        self.last_reposition_time = 0.0

    # -- calibration --
    def calibrate(self, radius):
        self.reference_radius = radius
        print(f"Calibrated! Reference radius = {radius}px at "
              f"{self.cfg['reference_distance_cm']}cm")

    def is_calibrated(self):
        return self.reference_radius is not None

    def estimate_distance(self, radius):
        if not self.is_calibrated() or radius == 0:
            return None
        return (self.reference_radius * self.cfg["reference_distance_cm"]) / radius

    # -- history --
    def _prune(self, history, now):
        window = self.cfg["distance_window_s"]
        while history and now - history[0][0] > window:
            history.popleft()

    def update(self, x, y, distance_cm, now):
        self.distance_history.append((now, distance_cm))
        self.position_history.append((now, x, y))
        self._prune(self.distance_history, now)
        self._prune(self.position_history, now)

    def clear(self):
        self.distance_history.clear()
        self.position_history.clear()

    # -- trend --
    def closing_rate_cm_s(self):
        """Positive = distance shrinking (approaching). Negative = receding."""
        if len(self.distance_history) < self.cfg["min_samples_for_rate"]:
            return None
        rate = linear_rate(list(self.distance_history))
        if rate is None:
            return None
        return -rate  # distance decreasing over time -> positive "closing" speed

    def time_to_contact(self, distance_cm, closing_rate):
        if closing_rate is None or closing_rate <= self.cfg["closing_rate_deadband_cm_s"]:
            return None  # not meaningfully closing -> treat as "infinite" TTC
        return distance_cm / closing_rate

    def lateral_velocity(self):
        hist = list(self.position_history)
        if len(hist) < 2:
            return 0.0, 0.0
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt

    def on_collision_course(self, x, y, frame_w, frame_h):
        vx, vy = self.lateral_velocity()
        future_x = x + vx * self.cfg["lookahead_s"]
        future_y = y + vy * self.cfg["lookahead_s"]
        return get_cell(future_x, future_y, frame_w, frame_h, self.cfg) == self.cfg["center_cell"]

    # -- decision --
    def evaluate(self, x, y, distance_cm, frame_w, frame_h):
        """
        Returns (state, closing_rate, ttc, on_course).

        Driven primarily by whether the object is closing in, and how fast
        relative to its current distance (time-to-contact) — not by which
        static distance band it happens to sit in right now. A hard floor
        is kept as a fail-safe: if it's already this close, react even if
        the trend can't be measured yet (e.g. right after re-detection).
        Remove that check if you want pure trend-only behavior.
        """
        if distance_cm is None:
            return "SAFE", None, None, False

        on_course = self.on_collision_course(x, y, frame_w, frame_h)
        closing_rate = self.closing_rate_cm_s()
        ttc = self.time_to_contact(distance_cm, closing_rate)

        if distance_cm < self.cfg["hard_floor_cm"]:
            return "RED", closing_rate, ttc, on_course

        if ttc is not None and ttc < self.cfg["critical_ttc_s"]:
            return "RED", closing_rate, ttc, on_course

        if on_course and ttc is not None and ttc < self.cfg["warning_ttc_s"]:
            return "ORANGE", closing_rate, ttc, on_course

        return "SAFE", closing_rate, ttc, on_course


# =====================================================================
# Main loop
# =====================================================================
def main():
    cfg = CONFIG
    tracker = ObstacleTracker(cfg)
    cap = cv2.VideoCapture(1)

    state_colors = {"SAFE": (0, 255, 0), "ORANGE": (0, 165, 255), "RED": (0, 0, 255)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        now = time.time()

        mask = get_mask(frame, cfg)
        draw_grid(frame, frame_width, frame_height, cfg)

        ball_info = find_ball(mask, cfg)

        if ball_info:
            x, y, radius = ball_info

            if not tracker.is_calibrated():
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
                cv2.putText(frame, f"Hold ball at {cfg['reference_distance_cm']}cm, press 'c'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"current radius: {radius}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                distance_cm = tracker.estimate_distance(radius)
                tracker.update(x, y, distance_cm, now)

                state, closing_rate, ttc, on_course = tracker.evaluate(
                    x, y, distance_cm, frame_width, frame_height
                )
                color = state_colors[state]

                rate_txt = f"{closing_rate:+.0f}cm/s" if closing_rate is not None else "n/a"
                ttc_txt = f"{ttc:.1f}s" if ttc is not None else "inf"
                cv2.circle(frame, (x, y), radius, color, 2)
                cv2.putText(
                    frame,
                    f"{state}  dist={int(distance_cm)}cm  rate={rate_txt}  ttc={ttc_txt}  course={on_course}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

                if state == "RED" and (now - tracker.last_dodge_time) > cfg["dodge_cooldown_s"]:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f">>> EMERGENCY DODGE {h.upper()} + {v.upper()}  (ttc={ttc_txt})")
                    tracker.last_dodge_time = now

                elif state == "ORANGE" and (now - tracker.last_reposition_time) > cfg["reposition_cooldown_s"]:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f"repositioning {h} and {v} to get off its path (ttc={ttc_txt})")
                    tracker.last_reposition_time = now
        else:
            tracker.clear()

        cv2.imshow("Simulated Drone Feed", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and ball_info:
            tracker.calibrate(ball_info[2])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()