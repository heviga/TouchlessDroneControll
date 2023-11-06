from djitellopy import Tello
import cv2
import mediapipe as mp
import time
import pickle
import numpy as np
from gesture_recognition import call_repeatedly, hand_landmarks, control, prediction
# Load the trained k-NN model
#
# with open('gesture_model.pkl', 'rb') as f:
#     knn = pickle.load(f)

# mp_hands = mp.solutions.hands
# hands = mp_hasssssssssssssssssnds.Hands(max_num_hands=1)

tello = Tello()
time.sleep(2.0) #waiting 2 seconds
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
tello.takeoff()
gesture_recognition_started = False
# call_repeatedly(1, hand_landmarks, frame_read.frame)

while True:
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if cv2.waitKey(1) == ord("e"):
        tello.land()
    if cv2.waitKey(1) == ord("s") and not gesture_recognition_started:
        call_repeatedly(1, hand_landmarks)
        call_repeatedly(1, lambda: control(tello))
        gesture_recognition_started = True
    elif cv2.waitKey(1) == ord("q"):
        break
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print(f"prediction = {prediction}")
    if prediction:
        print(f"predikcia{prediction}")

    cv2.imshow("drone", frame)



tello.land()
cv2.destroyAllWindows()
