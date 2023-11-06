import cv2
import mediapipe as mp
import numpy as np
import pickle
from threading import Thread, Event

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)


def call_repeatedly(interval, func):
    stopped = Event()

    def loop():
        while not stopped.wait(interval):  # the first call is in `interval` secs
            func()

    Thread(target=loop).start()
    return stopped.set


def hand_landmarks():
    global frame, prediction
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()

        prediction = clf.predict([landmarks])[0]
        return prediction





# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)
prediction = '0'
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) == ord("s"):
        call_repeatedly(1, hand_landmarks)
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print(f"prediction = {prediction}")
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
