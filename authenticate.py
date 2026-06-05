import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load("hand_auth.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    text = "NO HAND"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            features = []

            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features)[0]

            if prediction == 1:
                text = "ACCESS GRANTED"
            else:
                text = "ACCESS DENIED"

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("HandAuth AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
