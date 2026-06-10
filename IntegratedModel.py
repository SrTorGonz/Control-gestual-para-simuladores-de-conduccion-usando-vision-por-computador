import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import vgamepad as vg
import math
import keyboard

# =====================================================
#  XBOX VIRTUAL CONTROL
# =====================================================

gamepad = vg.VX360Gamepad()

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.Sequential([

    tf.keras.layers.Dense(
        128,
        activation='relu',
        input_shape=(126,)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        2,
        activation='sigmoid'
    )
])

model.load_weights("dirt_rally.weights.h5")

print("✅ Modelo cargado.")

# =====================================================
# MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =====================================================
# CONFIG
# =====================================================

STEERING_SENSITIVITY = 0.8
DEADZONE = 2500

# =====================================================
# NORMALIZATION
# =====================================================

def normalize_hand(coords):

    coords = np.array(coords).reshape(21,3)

    wrist = coords[0]

    coords = coords - wrist

    return coords.flatten()

# =====================================================
# DIRECTION
# =====================================================

def calculate_steering(left_wrist, right_wrist):

    dx = right_wrist[0] - left_wrist[0]
    dy = right_wrist[1] - left_wrist[1]

    angle = math.atan2(dy, dx)

    angle_deg = math.degrees(angle)

    steering = np.interp(
        angle_deg,
        [-45, 45],
        [32767, -32768]
    )

    # Reducir sensibilidad
    steering *= STEERING_SENSITIVITY

    # Deadzone
    if abs(steering) < DEADZONE:
        steering = 0

    steering = int(np.clip(
        steering,
        -32768,
        32767
    ))

    return steering, angle_deg

# =====================================================
# WEBCAM
# =====================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# REDUCIR BUFFER
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("🎮 Dirt Rally Gesture Controller iniciado")
print("Presiona Q para salir")

# =====================================================
# LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # OPTIMIZATIONS
    rgb.flags.writeable = False

    results = hands.process(rgb)

    rgb.flags.writeable = True

    right_hand = np.zeros(63, dtype=np.float32)
    left_hand = np.zeros(63, dtype=np.float32)

    right_wrist = None
    left_wrist = None

    # =================================================
    # DETECT HANDS
    # =================================================

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            normalized = normalize_hand(coords)

            wrist = [coords[0], coords[1]]

            hand_label = handedness.classification[0].label

            if hand_label == "Right":

                right_hand = normalized
                right_wrist = wrist

            else:

                left_hand = normalized
                left_wrist = wrist

    # =================================================
    # IA
    # =================================================

    features = np.concatenate([
        right_hand,
        left_hand
    ]).astype(np.float32)

    features = np.expand_dims(features, axis=0)

    # MÁS RÁPIDO QUE predict()
    prediction = model(
        features,
        training=False
    ).numpy()[0]

    accelerate = prediction[0] > 0.5
    brake = prediction[1] > 0.5

    # =================================================
    # TRIGGERS
    # =================================================

    gamepad.right_trigger(
        value=255 if accelerate else 0
    )

    gamepad.left_trigger(
        value=255 if brake else 0
    )

    # =================================================
    # DIRECTION
    # =================================================

    if left_wrist and right_wrist:

        steering_value, steering_angle = calculate_steering(
            left_wrist,
            right_wrist
        )

        gamepad.left_joystick(
            x_value=steering_value,
            y_value=0
        )

    else:

        # Volver al centro si pierde manos
        gamepad.left_joystick(
            x_value=0,
            y_value=0
        )

    # -----------------------------------
    # BOTON RB (TECLA G)
    # -----------------------------------

    if keyboard.is_pressed("g"):

        gamepad.press_button(
            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
        )

    else:

        gamepad.release_button(
            button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
        )
    # =================================================
    # SEND INPUTS
    # =================================================

    gamepad.update()

    # =================================================
    # UI
    # =================================================

    state = "NEUTRAL"

    if accelerate and brake:
        state = "ACC + BRAKE"

    elif accelerate:
        state = "ACCELERATE"

    elif brake:
        state = "BRAKE"

    cv2.putText(
        frame,
        state,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "Dirt Rally Gesture Controller",
        frame
    )

    # =================================================
    # EXIT
    # =================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break

# =====================================================
# CLOSE
# =====================================================

cap.release()

cv2.destroyAllWindows()