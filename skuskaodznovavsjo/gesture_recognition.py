import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()

        prediction = clf.predict([landmarks])[0]

        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
