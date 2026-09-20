import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            p = hand.landmark

            # Index finger tip
            index = p[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Thumb tip
            thumb = p[mp_hands.HandLandmark.THUMB_TIP]

            # Distance between thumb + index finger
            distance = math.hypot(
                index.x - thumb.x,
                index.y - thumb.y
            )

            pinch = distance < 0.05

            # Draw the hand
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Show index position
            h, w, _ = frame.shape
            x, y = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            text = "PINCH!" if pinch else "OPEN"
            color = (0, 0, 255) if pinch else (0, 255, 0)

            cv2.putText(
                frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        cv2.imshow("Hand Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
