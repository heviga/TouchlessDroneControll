# gesta:
# 1 -like
# 2 - up
# 3 - palm
# 4 - fist
# 5 - rock
# 6 - down
# 7 - ok
# 8 - peace

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from djitellopy import Tello


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
    cv2.imshow("Frame", frame)
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()


    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("1"):
        data.append(landmarks)
        print("1")
        labels.append("like")#lajk
    elif key == ord("2"):
        data.append(landmarks)
        print("2")
        labels.append("up")#ukazovak hore
    elif key == ord("3"):
        print("3")
        data.append(landmarks)
        labels.append("palm")#rozostreta dlan
    elif key == ord("4"):
        data.append(landmarks)
        print("4")
        labels.append("fist")#zovreta past stop
    elif key == ord("5"):
        data.append(landmarks)
        print("5")
        labels.append("rock")#rock
    elif key == ord("6"):
        data.append(landmarks)
        print("6")
        labels.append("down")#doleukazovacik
    elif key == ord("7"):
        data.append(landmarks)
        print("7")
        labels.append("ok")#ok

    if key == ord("8"):
        data.append(landmarks)
        print("8")
        labels.append("peace")  # mier

# Save collected data
with open("gesture_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

print(labels)
cap.release()
cv2.destroyAllWindows()
