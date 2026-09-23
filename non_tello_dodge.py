import cv2
import numpy as np
import time

# ---- Config ----
LOWER_YELLOW= np.array([20,100,100])
UPPER_YELLOW= np.array([35,255,255])
MIN_CONTOUR_AREA = 300
MIN_RADIUS = 2

REFERENCE_DISTANCE_CM = 100   # distance you'll hold the ball at during calibration
RED_DISTANCE_CM = 40          # closer than this = emergency dodge
ORANGE_DISTANCE_CM = 80       # closer than this + on collision course = reposition

LOOKAHEAD_FRAMES = 6
GRID_SIZE = 5
CENTER_CELL = (2, 2)

REPOSITION_COOLDOWN = 1.0
DODGE_COOLDOWN = 0.5

position_history = []
last_reposition_time = 0
last_dodge_time = 0
reference_radius = None   # set during calibration, None until then


def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
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
    """Convert pixel radius to estimated real-world distance using calibrated reference."""
    if reference_radius is None or current_radius == 0:
        return None
    return (reference_radius * REFERENCE_DISTANCE_CM) / current_radius


def update_history(x, y):
    position_history.append((x, y))
    if len(position_history) > LOOKAHEAD_FRAMES:
        position_history.pop(0)


def get_velocity():
    if len(position_history) < 2:
        return 0, 0
    dx = position_history[-1][0] - position_history[0][0]
    dy = position_history[-1][1] - position_history[0][1]
    n = len(position_history) - 1
    return dx / n, dy / n


def get_cell(x, y, frame_width, frame_height):
    col = min(int(x / (frame_width / GRID_SIZE)), GRID_SIZE - 1)
    row = min(int(y / (frame_height / GRID_SIZE)), GRID_SIZE - 1)
    return row, col


def predicts_collision(x, y, vx, vy, frame_width, frame_height):
    future_x = x + vx * LOOKAHEAD_FRAMES
    future_y = y + vy * LOOKAHEAD_FRAMES
    return get_cell(future_x, future_y, frame_width, frame_height) == CENTER_CELL


def get_state(distance_cm, on_collision_course):
    """Now gated by real-world distance, not raw pixel size — fixes false triggers from far balls."""
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


def draw_grid(frame, frame_width, frame_height):
    cell_w = frame_width // GRID_SIZE
    cell_h = frame_height // GRID_SIZE

    for i in range(1, GRID_SIZE):
        cv2.line(frame, (i * cell_w, 0), (i * cell_w, frame_height), (100, 100, 100), 1)
        cv2.line(frame, (0, i * cell_h), (frame_width, i * cell_h), (100, 100, 100), 1)

    cx1, cy1 = CENTER_CELL[1] * cell_w, CENTER_CELL[0] * cell_h
    cv2.rectangle(frame, (cx1, cy1), (cx1 + cell_w, cy1 + cell_h), (255, 255, 0), 2)


def main():
    global last_reposition_time, last_dodge_time, reference_radius

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        mask = get_mask(frame)

        draw_grid(frame, frame_width, frame_height)

        ball_info = find_ball(mask)

        if ball_info:
            x, y, radius = ball_info

            if reference_radius is None:
                # NEW: calibration mode - show instructions until 'c' is pressed
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
                cv2.putText(frame, f"Hold ball at {REFERENCE_DISTANCE_CM}cm, press 'c' to calibrate",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"current radius: {radius}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                update_history(x, y)
                vx, vy = get_velocity()
                on_collision_course = predicts_collision(x, y, vx, vy, frame_width, frame_height)
                distance_cm = estimate_distance(radius)

                state = get_state(distance_cm, on_collision_course)
                color = {"SAFE": (0, 255, 0), "ORANGE": (0, 165, 255), "RED": (0, 0, 255)}[state]

                cv2.circle(frame, (x, y), radius, color, 2)
                cv2.putText(frame, f"{state}  dist={int(distance_cm)}cm  collision={on_collision_course}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                now = time.time()

                if state == "RED" and (now - last_dodge_time) > DODGE_COOLDOWN:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f">>> EMERGENCY DODGE {h.upper()} + {v.upper()}")
                    last_dodge_time = now

                elif state == "ORANGE" and (now - last_reposition_time) > REPOSITION_COOLDOWN:
                    h, v = decide_direction(x, y, frame_width, frame_height)
                    print(f"repositioning {h} and {v} to get off its path")
                    last_reposition_time = now
        else:
            position_history.clear()

        cv2.imshow("Simulated Drone Feed", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and ball_info:
            reference_radius = ball_info[2]
            print(f"Calibrated! Reference radius = {reference_radius}px at {REFERENCE_DISTANCE_CM}cm")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()