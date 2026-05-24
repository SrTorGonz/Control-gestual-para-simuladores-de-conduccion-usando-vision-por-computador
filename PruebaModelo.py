import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# -----------------------------
# CARGAR MODELO
# -----------------------------
# -----------------------------
# RECONSTRUIR ARQUITECTURA
# -----------------------------
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

# -----------------------------
# CARGAR PESOS
# -----------------------------
model.load_weights("dirt_rally.weights.h5")

print("Modelo cargado correctamente.")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# FUNCION NORMALIZACION
# -----------------------------
def normalize_hand(coords):

    coords = np.array(coords).reshape(21,3)

    wrist = coords[0]

    coords = coords - wrist

    return coords.flatten()

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

print("Presiona Q para salir.")

# -----------------------------
# LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # -----------------------------------
    # VARIABLES MANOS
    # -----------------------------------
    right_hand = np.zeros(63)
    left_hand = np.zeros(63)

    # -----------------------------------
    # DETECCION MANOS
    # -----------------------------------
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

            # NORMALIZAR
            coords = normalize_hand(coords)

            # IDENTIFICAR MANO
            hand_label = handedness.classification[0].label

            if hand_label == "Right":
                right_hand = coords
            else:
                left_hand = coords

    # -----------------------------------
    # CREAR VECTOR FINAL
    # -----------------------------------
    features = np.concatenate([right_hand, left_hand])

    features = np.expand_dims(features, axis=0)

    # -----------------------------------
    # PREDICCION
    # -----------------------------------
    prediction = model.predict(features, verbose=0)[0]

    accelerate_prob = prediction[0]
    brake_prob = prediction[1]

    # Threshold
    accelerate = accelerate_prob > 0.5
    brake = brake_prob > 0.5

    # -----------------------------------
    # ESTADO FINAL
    # -----------------------------------
    if accelerate and brake:
        state = "ACCELERATE + BRAKE"

    elif accelerate:
        state = "ACCELERATE"

    elif brake:
        state = "BRAKE"

    else:
        state = "NEUTRAL"

    # -----------------------------------
    # MOSTRAR TEXTO
    # -----------------------------------
    cv2.putText(
        frame,
        state,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # Mostrar probabilidades
    cv2.putText(
        frame,
        f"ACC: {accelerate_prob:.2f}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"BRAKE: {brake_prob:.2f}",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Dirt Rally Gesture Control", frame)

    # Salir
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -----------------------------
# CERRAR
# -----------------------------
cap.release()
cv2.destroyAllWindows()