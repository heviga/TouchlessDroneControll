import cv2 as cv2
import mediapipe as mp
import numpy as np
import pickle
from threading import Thread, Event
import time
# import queue
mp_hands = mp.solutions.hands


hands = mp_hands.Hands(max_num_hands=1)
# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    clf = model_data['classifier']
    scaler = model_data['scaler']

stop_event = Event()

def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()

def hand_landmarks(scaler, clf, frame_cv):
    global prediction
    prediction = '0'
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark]).flatten()
        landmarks = scaler.transform([landmarks])  # Normalize the landmarks
        prediction = clf.predict(landmarks)[0]  # Make a prediction
        print(f"Prediction: {prediction}")  # Debugging line to print prediction
    else:
        print("No hand landmarks detected.")


# MOVE TELLO

def control(tello, prediction, frame_cv):

    # global prediction
    if prediction:
        if prediction == 'fist':
            tello.forward(30)
            time.sleep(1)
            tello.forward(0)
        elif prediction == 'ok':
            tello.backward(30)
            time.sleep(1)
            tello.backward(0)
        elif prediction == 'palm':
            time.sleep(5)
            tello.backward(0)
        elif prediction == 'like':
            tello.clockwise(30)
            time.sleep(1)
            tello.clockwise(0)
        elif prediction == 'rock':
            tello.flip_back()
        elif prediction == 'up':
            tello.up(30)
            time.sleep(1)
            tello.up(0)
        elif prediction == 'down':
            tello.down(30)
            time.sleep(1)
            tello.down(0)
        elif prediction == 'peace':
            cv2.imwrite("picture.png", frame_cv)


