import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# Try to load existing data, if any
file_name = "gesture_data.pkl"
if os.path.exists(file_name):
    with open(file_name, "rb") as f:
        data, labels = pickle.load(f)
else:
    data, labels = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("1"):
            data.append(landmarks)
            labels.append("gesture_1")#lajk
        elif key == ord("2"):
            data.append(landmarks)
            labels.append("gesture_2")#ukazovak hore
        elif key == ord("3"):
            data.append(landmarks)
            labels.append("gesture_3")#rozostreta dlan
        elif key == ord("4"):
            data.append(landmarks)
            labels.append("gesture_4")#zovreta past stop
        elif key == ord("5"):
            data.append(landmarks)
            labels.append("gesture_5")#rock
        elif key == ord("6"):
            data.append(landmarks)
            labels.append("gesture_6")#doleukazovacik
        elif key == ord("7"):
            data.append(landmarks)
            labels.append("gesture_7")#ok
        # Add more elif conditions for more gestures
        # ...

# Save collected data
with open("gesture_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print(labels)
cap.release()
cv2.destroyAllWindows()
