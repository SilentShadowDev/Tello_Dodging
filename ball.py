import cv2
import numpy as np
import winsound
import threading

# ---- Config ----# ---- Config ----
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_CONTOUR_AREA = 300
MIN_RADIUS = 10
HISTORY_LENGTH = 8

radius_history = []
alarm_playing = False


def create_controls():
    """Create a window with live-adjustable sliders."""
    cv2.namedWindow("Controls")
    cv2.createTrackbar("Threshold", "Controls", 80, 300, lambda x: None)
    cv2.createTrackbar("Sensitivity", "Controls", 3, 20, lambda x: None)


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


def get_growth():
    """How much the radius changed from oldest to newest tracked frame."""
    if len(radius_history) < HISTORY_LENGTH:
        return 0
    return radius_history[-1] - radius_history[0]


def update_history(radius):
    radius_history.append(radius)
    if len(radius_history) > HISTORY_LENGTH:
        radius_history.pop(0)


def get_state(radius, growth, threshold, sensitivity):
    """Decide which of the 4 states we're in. Priority: too close > approaching > going away > safe."""
    if radius > threshold:
        return "TOO CLOSE", (0, 0, 255)       # red
    elif growth > sensitivity:
        return "APPROACHING", (0, 165, 255)   # orange
    elif growth < -sensitivity:
        return "GOING AWAY", (255, 0, 0)      # blue
    else:
        return "SAFE", (0, 255, 0)            # green


def play_siren():
    global alarm_playing
    alarm_playing = True
    for _ in range(4):
        winsound.Beep(800, 200)
        winsound.Beep(1200, 200)
    alarm_playing = False


def draw_info(frame, ball_info, state_label, color, growth, threshold, sensitivity):
    x, y, radius = ball_info

    cv2.circle(frame, (x, y), radius, color, 2)
    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # NEW: all the live numbers, stacked top-left
    cv2.putText(frame, f"radius: {radius}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"growth: {growth}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"threshold: {threshold}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"sensitivity: {sensitivity}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, state_label, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    global alarm_playing
    cap = cv2.VideoCapture(0)
    create_controls()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) 

        # NEW: read live slider values every frame
        threshold = cv2.getTrackbarPos("Threshold", "Controls")
        sensitivity = cv2.getTrackbarPos("Sensitivity", "Controls")

        mask = get_mask(frame)
        ball_info = find_ball(mask)

        if ball_info:
            x, y, radius = ball_info
            update_history(radius)
            growth = get_growth()
            state_label, color = get_state(radius, growth, threshold, sensitivity)

            draw_info(frame, ball_info, state_label, color, growth, threshold, sensitivity)

            if state_label == "TOO CLOSE" and not alarm_playing:
                threading.Thread(target=play_siren, daemon=True).start()
        else:
            radius_history.clear()

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()