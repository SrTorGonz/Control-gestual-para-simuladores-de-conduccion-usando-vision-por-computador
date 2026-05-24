import cv2
import mediapipe as mp
import csv
import os
import numpy as np

DATASET_FILE = "dirt_rally_multilabel.csv"

# Crear CSV
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        header = []
        for i in range(21):
            header += [f"R_x{i}", f"R_y{i}", f"R_z{i}"]

        for i in range(21):
            header += [f"L_x{i}", f"L_y{i}", f"L_z{i}"]

        header += ["accelerate", "brake"]
        writer.writerow(header)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

accelerate = 0
brake = 0

print("A → toggle accelerate")
print("B → toggle brake")
print("N → neutral")
print("Q → quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("a"):
        accelerate = 1 - accelerate
    elif key == ord("b"):
        brake = 1 - brake
    elif key == ord("n"):
        accelerate = 0
        brake = 0
    elif key == ord("q"):
        break

    right_hand = np.zeros(63)
    left_hand = np.zeros(63)

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if handedness.classification[0].label == "Right":
                right_hand = np.array(coords)
            else:
                left_hand = np.array(coords)

    row = np.concatenate([right_hand, left_hand, [accelerate, brake]])

    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    cv2.putText(frame,
        f"ACC:{accelerate} BRAKE:{brake}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,(0,255,0),2)

    cv2.imshow("Dataset Creator", frame)

cap.release()
cv2.destroyAllWindows()