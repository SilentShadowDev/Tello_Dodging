import cv2
import numpy as np
from djitellopy import Tello
import time

# ---- Config ----
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_CONTOUR_AREA = 300
MIN_RADIUS = 10

REFERENCE_DISTANCE_CM = 100
RED_DISTANCE_CM = 40
ORANGE_DISTANCE_CM = 80

LOOKAHEAD_SECONDS = 0.3     # how far ahead we predict, in real time now (not frames)
HISTORY_MAX_AGE = 0.5       # drop position samples older than this
GRID_SIZE = 5
CENTER_CELL = (2, 2)

REPOSITION_COOLDOWN = 1.0
DODGE_COOLDOWN = 0.5
DODGE_SPEED = 60
REPOSITION_SPEED = 30
MOVE_DURATION = 0.4         # how long a movement burst should last

position_history = []       # list of (timestamp, x, y)
last_reposition_time = 0
last_dodge_time = 0
reference_radius = None

# NEW: non-blocking movement state
active_move_until = 0       # timestamp when current movement burst should stop
move_in_progress = False


def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def find_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None

    (x, y), radius = cv2.minEnclosingCircle(c)
    if radius < MIN_RADIUS:
        return None

    return (int(x), int(y), int(radius))


def estimate_distance(current_radius):
    if reference_radius is None or current_radius == 0:
        return None
    return (reference_radius * REFERENCE_DISTANCE_CM) / current_radius


def update_history(x, y):
    """Store timestamped positions, and drop anything too old to stay relevant."""
    now = time.time()
    position_history.append((now, x, y))
    while position_history and (now - position_history[0][0]) > HISTORY_MAX_AGE:
        position_history.pop(0)


def get_velocity():
    """Real pixels-per-second velocity, using actual elapsed time instead of frame count."""
    if len(position_history) < 2:
        return 0, 0

    t0, x0, y0 = position_history[0]
    t1, x1, y1 = position_history[-1]
    dt = t1 - t0

    if dt <= 0:
        return 0, 0  # guards against duplicate-timestamp edge case

    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    return vx, vy


def get_cell(x, y, frame_width, frame_height):
    col = min(max(int(x / (frame_width / GRID_SIZE)), 0), GRID_SIZE - 1)
    row = min(max(int(y / (frame_height / GRID_SIZE)), 0), GRID_SIZE - 1)
    return row, col


def predicts_collision(x, y, vx, vy, frame_width, frame_height):
    """Project position forward by LOOKAHEAD_SECONDS of real time, not frame count."""
    future_x = x + vx * LOOKAHEAD_SECONDS
    future_y = y + vy * LOOKAHEAD_SECONDS
    return get_cell(future_x, future_y, frame_width, frame_height) == CENTER_CELL


def get_state(distance_cm, on_collision_course):
    if distance_cm is None:
        return "SAFE"
    if distance_cm < RED_DISTANCE_CM:
        return "RED"
    elif distance_cm < ORANGE_DISTANCE_CM and on_collision_course:
        return "ORANGE"
    else:
        return "SAFE"


def decide_direction(x, y, frame_width, frame_height):
    h = "right" if x < frame_width // 2 else "left"
    v = "up" if y > frame_height // 2 else "down"
    return h, v


def start_move(tello, h_dir, v_dir, speed):
    """NEW: fire-and-forget movement — sends the command and records when to stop, no sleep()."""
    global active_move_until, move_in_progress

    left_right = speed if h_dir == "right" else -speed
    up_down = speed if v_dir == "up" else -speed
    tello.send_rc_control(left_right, 0, up_down, 0)

    active_move_until = time.time() + MOVE_DURATION
    move_in_progress = True


def update_move_state(tello):
    """NEW: called every loop iteration — stops the drone once its burst duration has elapsed."""
    global move_in_progress

    if move_in_progress and time.time() >= active_move_until:
        tello.send_rc_control(0, 0, 0, 0)
        move_in_progress = False


def draw_grid(frame, frame_width, frame_height):
    cell_w = frame_width // GRID_SIZE
    cell_h = frame_height // GRID_SIZE

    for i in range(1, GRID_SIZE):
        cv2.line(frame, (i * cell_w, 0), (i * cell_w, frame_height), (100, 100, 100), 1)
        cv2.line(frame, (0, i * cell_h), (frame_width, i * cell_h), (100, 100, 100), 1)

    cx1, cy1 = CENTER_CELL[1] * cell_w, CENTER_CELL[0] * cell_h
    cv2.rectangle(frame, (cx1, cy1), (cx1 + cell_w, cy1 + cell_h), (255, 255, 0), 2)


def calibrate(tello, frame_reader):
    global reference_radius

    print(f"Hold the ball at {REFERENCE_DISTANCE_CM}cm from the drone and press 'c' to calibrate...")

    while reference_radius is None:
        frame = frame_reader.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mask = get_mask(frame)
        ball_info = find_ball(mask)

        if ball_info:
            x, y, radius = ball_info
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
            cv2.putText(frame, f"radius: {radius} - press 'c' to lock in", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ball_info:
            reference_radius = ball_info[2]
            print(f"Calibrated! Reference radius = {reference_radius}px at {REFERENCE_DISTANCE_CM}cm")
        elif key == ord('q'):
            print("Calibration skipped/cancelled.")
            break

    cv2.destroyWindow("Calibration")


def main():
    global last_reposition_time, last_dodge_time

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_reader = tello.get_frame_read()

    calibrate(tello, frame_reader)

    if reference_radius is None:
        print("No calibration done, aborting.")
        tello.streamoff()
        return

    tello.takeoff()
    time.sleep(2)  # one-time startup delay is fine, this isn't in the reactive loop

    try:
        while True:
            frame = frame_reader.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_height, frame_width = frame.shape[:2]

            draw_grid(frame, frame_width, frame_height)

            # NEW: check every loop iteration whether an active dodge burst should end
            update_move_state(tello)

            mask = get_mask(frame)
            ball_info = find_ball(mask)

            if ball_info:
                x, y, radius = ball_info
                update_history(x, y)
                vx, vy = get_velocity()
                on_collision_course = predicts_collision(x, y, vx, vy, frame_width, frame_height)
                distance_cm = estimate_distance(radius)

                state = get_state(distance_cm, on_collision_course)
                color = {"SAFE": (0, 255, 0), "ORANGE": (0, 165, 255), "RED": (0, 0, 255)}[state]

                cv2.circle(frame, (x, y), radius, color, 2)
                cv2.putText(frame, f"{state}  dist={int(distance_cm)}cm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                now = time.time()

                if state == "RED" and (now - last_dodge_time) > DODGE_COOLDOWN:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f">>> EMERGENCY DODGE {h.upper()} + {v.upper()}")
                    start_move(tello, h, v, DODGE_SPEED)
                    last_dodge_time = now
                    # NOTE: no longer clearing position_history here — keeps velocity tracking alive

                elif state == "ORANGE" and (now - last_reposition_time) > REPOSITION_COOLDOWN:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f"repositioning {h} + {v}")
                    start_move(tello, h, v, REPOSITION_SPEED)
                    last_reposition_time = now
            # NOTE: no longer clearing position_history when ball isn't found either —
            # old samples age out naturally via HISTORY_MAX_AGE instead

            cv2.imshow("Tello Feed", frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                print("Manual land triggered")
                break

    finally:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
