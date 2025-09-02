import cv2
import numpy as np
import mediapipe as mp

# ---------------------- Setup ----------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Canvas to draw
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Default brush settings
brush_color = (0, 0, 255)  # Red
brush_thickness = 5
eraser_thickness = 50

# Previous finger coordinates
xp, yp = 0, 0

# ---------------------- Main Loop ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get index finger tip coordinates
            x1 = int(handLms.landmark[8].x * 1280)
            y1 = int(handLms.landmark[8].y * 720)
            # Get thumb tip coordinates
            x2 = int(handLms.landmark[4].x * 1280)
            y2 = int(handLms.landmark[4].y * 720)

            # Distance between index and thumb
            distance = np.hypot(x2 - x1, y2 - y1)

            # Pinch-to-draw (draw line if pinch detected)
            if distance < 40:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(canvas, (xp, yp), (x1, y1), brush_color, brush_thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

            # Draw hand landmarks for reference
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and webcam feed
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, canvas)

    # Display
    cv2.imshow("Air Canvas", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
