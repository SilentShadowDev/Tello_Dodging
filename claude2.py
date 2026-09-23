"""
Ball-tracking obstacle reactor.

Simulates the acoustic-shield drone's evasion logic using a webcam + a
colored ball standing in for an ultrasonic/ToF sensor reading.

- SAFE / ORANGE / RED are just a distance-band *status label* — shown on
  screen, not used to decide anything.
- The drone only reacts when the ball's distance is actually decreasing
  (it's closing in). Zone doesn't matter for that decision at all: a ball
  sitting still deep in the RED zone gets no reaction; a ball closing in
  fast from the SAFE zone gets one immediately.
- If the distance isn't decreasing (steady or moving away), the drone
  stays put.
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

    # --- Zone thresholds (display / status only — do NOT gate reaction) ---
    "red_zone_cm": 40,
    "orange_zone_cm": 80,

    # --- Approach detection (this is what actually triggers a reaction) ---
    "distance_window_s": 0.6,         # seconds of history used for the rate fit
    "min_samples_for_rate": 4,        # don't trust a rate until we have this many points
    "closing_rate_deadband_cm_s": 8,  # must be closing faster than this to count as "approaching"

    # --- Cooldown so one approach doesn't spam repeated move commands ---
    "reaction_cooldown_s": 0.5,

    # --- Visual reference grid only (no longer used for any decision) ---
    "grid_size": 5,
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


# =====================================================================
# Tracker — owns calibration, distance history, and the reaction decision
# =====================================================================
class ObstacleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reference_radius = None
        self.distance_history = deque()  # (timestamp, distance_cm)
        self.last_reaction_time = 0.0

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
    def update(self, distance_cm, now):
        self.distance_history.append((now, distance_cm))
        window = self.cfg["distance_window_s"]
        while self.distance_history and now - self.distance_history[0][0] > window:
            self.distance_history.popleft()

    def clear(self):
        self.distance_history.clear()

    # -- status label only, never used to decide anything --
    def zone_for_distance(self, distance_cm):
        if distance_cm is None:
            return "SAFE"
        if distance_cm < self.cfg["red_zone_cm"]:
            return "RED"
        if distance_cm < self.cfg["orange_zone_cm"]:
            return "ORANGE"
        return "SAFE"

    # -- the actual decision driver --
    def closing_rate_cm_s(self):
        """Positive = distance shrinking (approaching). Negative = receding."""
        if len(self.distance_history) < self.cfg["min_samples_for_rate"]:
            return None
        rate = linear_rate(list(self.distance_history))
        if rate is None:
            return None
        return -rate  # distance decreasing over time -> positive "closing" speed

    def is_approaching(self, closing_rate):
        return closing_rate is not None and closing_rate > self.cfg["closing_rate_deadband_cm_s"]


# =====================================================================
# Main loop
# =====================================================================
def main():
    cfg = CONFIG
    tracker = ObstacleTracker(cfg)
    cap = cv2.VideoCapture(1)

    zone_colors = {"SAFE": (0, 255, 0), "ORANGE": (0, 165, 255), "RED": (0, 0, 255)}

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
                tracker.update(distance_cm, now)

                zone = tracker.zone_for_distance(distance_cm)          # label only
                closing_rate = tracker.closing_rate_cm_s()
                approaching = tracker.is_approaching(closing_rate)      # decision driver

                color = zone_colors[zone]
                rate_txt = f"{closing_rate:+.0f}cm/s" if closing_rate is not None else "n/a"

                cv2.circle(frame, (x, y), radius, color, 2)
                cv2.putText(
                    frame,
                    f"zone={zone}  dist={int(distance_cm)}cm  rate={rate_txt}  approaching={approaching}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

                if approaching and (now - tracker.last_reaction_time) > cfg["reaction_cooldown_s"]:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f">>> BALL CLOSING IN — MOVE {h.upper()} + {v.upper()}  "
                          f"(zone={zone}, rate={rate_txt})")
                    tracker.last_reaction_time = now
                # else: distance isn't decreasing -> no command issued, drone stays put.
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