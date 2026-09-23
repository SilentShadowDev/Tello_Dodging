"""
Ball-tracking obstacle reactor — now flying a real Tello.

Uses the Tello's own camera feed + a colored ball standing in for an
ultrasonic/ToF sensor reading, and reacts by actually moving the drone
via djitellopy — non-blocking rc-control bursts, the same pattern
Final1.0 uses — instead of just printing what it would do.

- SAFE / ORANGE / RED are a distance-band *status label* only — shown on
  screen, never used to decide anything.
- The only thing that triggers a reaction is the ball's distance actually
  decreasing (closing in). Zone doesn't gate that at all.
- Detection is a "detect, then track" loop: once the ball is found, later
  frames only search a small region around its last known position (fast).
  Full-frame search only runs on the first sighting or after the ball is
  lost for a few frames — which is also what keeps it able to pick up a
  ball that first appears small and far away, instead of only ever
  scanning a limited area.
- Movement is fire-and-forget, like Final1.0's start_move/update_move_state:
  send_rc_control kicks off a burst and a timestamp (not a blocking
  sleep()) tells the loop when to zero the sticks out again, so detection
  keeps running at full frame rate through a dodge instead of stalling.
"""

import time
from collections import deque
from djitellopy import Tello
import cv2
import numpy as np

# =====================================================================
# CONFIG — single entry point for all tunables
# =====================================================================
CONFIG = {
    # --- Color detection (HSV) for the tracked ball ---
    "lower_yellow": np.array([20, 100, 100]),
    "upper_yellow": np.array([35, 255, 255]),

    # Lowered vs. before: at ~5m a ball can be a genuinely small blob.
    # Shape (circularity), not just area, is what keeps this from picking
    # up noise once the area floor is this low.
    "min_contour_area": 8,
    "min_radius": 1.5,
    "min_circularity": 0.55,   # 4*pi*area/perimeter^2; 1.0 = perfect circle

    # --- Calibration ---
    "reference_distance_cm": 100,     # hold the ball this far away, press 'c'

    # --- Capture resolution ---
    # The Tello's own stream comes in at 960x720; kept here as config
    # (rather than a magic number) so the rest of the code can reference it.
    "capture_width": 960,
    "capture_height": 720,

    # --- Detect-then-track (this is what keeps frametime down) ---
    "roi_radius_multiplier": 4,   # search box half-size = radius * this + margin
    "roi_margin_px": 40,
    "max_consecutive_misses_before_full_scan": 5,

    # --- Zone thresholds (display / status only — do NOT gate reaction) ---
    "red_zone_cm": 40,
    "orange_zone_cm": 80,

    # --- Approach detection (this is what actually triggers a reaction) ---
    "distance_window_s": 0.6,
    "min_samples_for_rate": 4,
    "closing_rate_deadband_cm_s": 8,

    "reaction_cooldown_s": 0.5,

    # --- Movement (fire-and-forget send_rc_control bursts, no blocking sleep) ---
    "move_speed": 60,        # rc magnitude sent on left_right / up_down, 0-100
    "move_duration_s": 0.4,  # how long a burst runs before auto-zeroing out

    "grid_size": 5,
    "debug_overlay": True,   # set False to skip all drawing/text for max speed
}

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# =====================================================================
# Vision helpers
# =====================================================================
def get_mask(bgr_region, cfg):
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg["lower_yellow"], cfg["upper_yellow"])
    # A single MORPH_CLOSE (dilate then erode) fills small speckle holes
    # without the risk that MORPH_OPEN (erode first) has of erasing a
    # genuinely tiny, far-away blob before it ever gets a chance to grow
    # back. Also one call instead of two separate erode/dilate calls.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return mask


def find_ball(mask, cfg, offset=(0, 0)):
    """Returns (x, y, radius) in full-frame coordinates, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < cfg["min_contour_area"]:
        return None

    perimeter = cv2.arcLength(c, True)
    if perimeter <= 0:
        return None
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < cfg["min_circularity"]:
        return None  # shape-based reject: keeps the low area floor from picking up noise

    (x, y), radius = cv2.minEnclosingCircle(c)
    if radius < cfg["min_radius"]:
        return None

    ox, oy = offset
    return int(x + ox), int(y + oy), int(radius)


def compute_roi(last_x, last_y, last_r, frame_w, frame_h, cfg):
    half = last_r * cfg["roi_radius_multiplier"] + cfg["roi_margin_px"]
    x0 = max(0, int(last_x - half))
    y0 = max(0, int(last_y - half))
    x1 = min(frame_w, int(last_x + half))
    y1 = min(frame_h, int(last_y + half))
    return x0, y0, x1, y1


def linear_rate(samples):
    """Least-squares slope of value vs. time over (t, value) samples, per second."""
    if len(samples) < 2:
        return None
    t = np.array([s[0] for s in samples], dtype=np.float64)
    v = np.array([s[1] for s in samples], dtype=np.float64)
    t = t - t[0]
    if t[-1] == 0:
        return None
    a, _ = np.polyfit(t, v, 1)
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
# Tracker — owns calibration, distance history, the reaction decision,
# and now the drone's non-blocking movement bursts.
# =====================================================================
class ObstacleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reference_radius = None
        self.distance_history = deque()  # (timestamp, distance_cm)
        self.last_reaction_time = 0.0

        # detect-then-track state
        self.last_detection = None   # (x, y, radius) in full-frame coords
        self.consecutive_misses = 0

        # non-blocking movement state
        self.active_move_until = 0.0
        self.move_in_progress = False

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

    # -- detect-then-track --
    def find(self, frame, cfg):
        h, w = frame.shape[:2]
        use_roi = (
            self.last_detection is not None
            and self.consecutive_misses < cfg["max_consecutive_misses_before_full_scan"]
        )

        if use_roi:
            lx, ly, lr = self.last_detection
            x0, y0, x1, y1 = compute_roi(lx, ly, lr, w, h, cfg)
            region = frame[y0:y1, x0:x1]
            mask = get_mask(region, cfg)
            result = find_ball(mask, cfg, offset=(x0, y0))
            if result is None:
                # Ball may have moved faster than the ROI, or we're not
                # tracking well — fall back to a full scan next frame(s).
                self.consecutive_misses += 1
            return result, mask, (x0, y0)

        mask = get_mask(frame, cfg)
        result = find_ball(mask, cfg, offset=(0, 0))
        return result, mask, (0, 0)

    def on_detection(self, ball_info):
        if ball_info is None:
            if self.consecutive_misses >= self.cfg["max_consecutive_misses_before_full_scan"]:
                self.last_detection = None
            return
        self.consecutive_misses = 0
        self.last_detection = ball_info

    # -- distance history --
    def update_distance(self, distance_cm, now):
        self.distance_history.append((now, distance_cm))
        window = self.cfg["distance_window_s"]
        while self.distance_history and now - self.distance_history[0][0] > window:
            self.distance_history.popleft()

    def clear(self):
        self.distance_history.clear()
        self.last_detection = None
        self.consecutive_misses = 0

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
        if len(self.distance_history) < self.cfg["min_samples_for_rate"]:
            return None
        rate = linear_rate(list(self.distance_history))
        if rate is None:
            return None
        return -rate

    def is_approaching(self, closing_rate):
        return closing_rate is not None and closing_rate > self.cfg["closing_rate_deadband_cm_s"]

    # -- movement: fire-and-forget rc bursts (mirrors Final1.0's
    # start_move / update_move_state, just owned by the tracker instead
    # of module-level globals) --
    def start_move(self, tello, h_dir, v_dir, speed):
        left_right = speed if h_dir == "right" else -speed
        up_down = speed if v_dir == "up" else -speed
        tello.send_rc_control(left_right, 0, up_down, 0)
        self.active_move_until = time.time() + self.cfg["move_duration_s"]
        self.move_in_progress = True 

    def update_move_state(self, tello):
        """Call every loop iteration — zeroes the sticks once the burst's
        duration has elapsed, without ever blocking on sleep()."""
        if self.move_in_progress and time.time() >= self.active_move_until:
            tello.send_rc_control(0, 0, 0, 0)
            self.move_in_progress = False


# =====================================================================
# Main loop
# =====================================================================
def main():
    cfg = CONFIG
    tracker = ObstacleTracker(cfg)

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()
    frame_reader = tello.get_frame_read()

    # Create both windows up front, resizable. Combined with the fixed-size
    # mask canvas below, this stops the window from jumping around every
    # frame — which was also swallowing keyboard focus so 'q'/'c' stopped
    # registering.
    cv2.namedWindow("Tello Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    zone_colors = {"SAFE": (0, 255, 0), "ORANGE": (0, 165, 255), "RED": (0, 0, 255)}

    prev_t = time.time()
    fps = 0.0
    airborne = False

    print("Controls: 'c' calibrate on the visible ball, 't' takeoff, "
          "'l' land, 'q' quit (auto-lands if still airborne).")

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue
            # Tello's stream comes in RGB; the rest of the pipeline (and
            # cv2.imshow) expects BGR. No horizontal flip here — unlike a
            # selfie-style webcam, the drone's forward camera shouldn't be
            # mirrored, or left/right dodge decisions would be backwards.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_height, frame_width = frame.shape[:2]
            now = time.time()

            # smoothed FPS for the overlay
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            if cfg["debug_overlay"]:
                draw_grid(frame, frame_width, frame_height, cfg)

            # Auto-stop any movement burst whose duration has elapsed.
            # Checked every iteration, same as detection, so a dodge never
            # stalls the vision pipeline the way a blocking sleep() would.
            tracker.update_move_state(tello)

            ball_info, mask, roi_origin = tracker.find(frame, cfg)
            tracker.on_detection(ball_info)

            if ball_info:
                x, y, radius = ball_info

                if not tracker.is_calibrated():
                    if cfg["debug_overlay"]:
                        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
                        cv2.putText(frame, f"Hold ball at {cfg['reference_distance_cm']}cm, press 'c'",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"current radius: {radius}", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    distance_cm = tracker.estimate_distance(radius)
                    tracker.update_distance(distance_cm, now)

                    zone = tracker.zone_for_distance(distance_cm)
                    closing_rate = tracker.closing_rate_cm_s()
                    approaching = tracker.is_approaching(closing_rate)

                    if cfg["debug_overlay"]:
                        color = zone_colors[zone]
                        rate_txt = f"{closing_rate:+.0f}cm/s" if closing_rate is not None else "n/a"
                        cv2.circle(frame, (x, y), radius, color, 2)
                        cv2.putText(
                            frame,
                            f"zone={zone}  dist={int(distance_cm)}cm  rate={rate_txt}  "
                            f"approaching={approaching}  fps={fps:.0f}  "
                            f"{'AIRBORNE' if airborne else 'landed'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        )

                    if approaching and (now - tracker.last_reaction_time) > cfg["reaction_cooldown_s"]:
                        h, v = decide_direction(x, y, frame_width, frame_height)
                        rate_txt = f"{closing_rate:+.0f}cm/s" if closing_rate is not None else "n/a"
                        print(f">>> BALL CLOSING IN — MOVE {h.upper()} + {v.upper()}  "
                              f"(zone={zone}, rate={rate_txt})")
                        if airborne:
                            tracker.start_move(tello, h, v, cfg["move_speed"])
                        tracker.last_reaction_time = now
            else:
                tracker.clear()

            # `mask` is only the small ROI crop most frames now (that's the
            # frametime win) but imshow-ing a differently-sized image every
            # frame makes the window constantly resize/reposition itself,
            # which is exactly what looked like the screen "bugging out" and
            # was also stealing keyboard focus. Paste it onto a full-frame
            # canvas so the displayed image size never changes.
            mask_display = np.zeros((frame_height, frame_width), dtype=np.uint8)
            rx, ry = roi_origin
            rh, rw = mask.shape[:2]
            mask_display[ry:ry + rh, rx:rx + rw] = mask

            cv2.imshow("Tello Feed", frame)
            cv2.imshow("Mask", mask_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and ball_info:
                tracker.calibrate(ball_info[2])
            elif key == ord('t') and not airborne:
                tello.takeoff()
                airborne = True
            elif key == ord('l') and airborne:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                airborne = False

    finally:
        # Belt-and-braces shutdown: zero any residual rc command, land if
        # we're still up, and always release the stream/windows even if
        # something above raised.
        tello.send_rc_control(0, 0, 0, 0)
        if airborne:
            tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()