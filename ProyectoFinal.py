import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp.__version__
# ==========================
# MediaPipe Setup
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# Cámara
# ==========================
cap = cv2.VideoCapture(0)

# ==========================
# Suavizado
# ==========================
angle_buffer = deque(maxlen=5)

# ==========================
# Funciones
# ==========================

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def thumb_extended(hand_landmarks):
    """
    Detecta si el pulgar está extendido independientemente
    de la rotación de la mano.
    """

    lm = hand_landmarks.landmark

    thumb_tip = [lm[4].x, lm[4].y]
    thumb_mcp = [lm[2].x, lm[2].y]
    wrist = [lm[0].x, lm[0].y]

    # Distancia pulgar-centro
    d_tip = distance(thumb_tip, wrist)
    d_mcp = distance(thumb_mcp, wrist)

    # Si la punta está MUCHO más lejos que la base → extendido
    return d_tip > d_mcp * 1.35


def get_center(hand_landmarks, w, h):
    x = int(hand_landmarks.landmark[9].x * w)
    y = int(hand_landmarks.landmark[9].y * h)
    return (x, y)


# ==========================
# LOOP PRINCIPAL
# ==========================
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_centers = []
    thumbs = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            center = get_center(hand_landmarks, w, h)
            hand_centers.append(center)

            thumb = thumb_extended(hand_landmarks)
            thumbs.append(thumb)

            cv2.circle(frame, center, 8, (255,0,0), -1)

    # ==========================
    # Volante virtual
    # ==========================
    if len(hand_centers) == 2:

        p1, p2 = hand_centers

        cv2.line(frame, p1, p2, (0,255,0), 3)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        angle = math.degrees(math.atan2(dy, dx))

        angle_buffer.append(angle)
        smooth_angle = np.mean(angle_buffer)

        cv2.putText(
            frame,
            f"Steering: {smooth_angle:.2f}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # ==========================
        # Control pulgares
        # ==========================
        if thumbs[0]:
            cv2.putText(frame, "ACCELERATE", (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        if thumbs[1]:
            cv2.putText(frame, "BRAKE", (30,150),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("DIRT RALLY Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()