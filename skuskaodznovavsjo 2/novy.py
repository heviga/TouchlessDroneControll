import cv2
import mediapipe as mp
import numpy as np
import pickle
from threading import Thread, Event
from tello import Tello
import cv2

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
def control(tello):

    global frame_read, prediction
    if prediction:
        if prediction == 'gesture_1':  # otockalajk
            tello.rotate_clockwise(30)
        elif prediction == 'gesture_2':  # UP
            tello.move_up(25)
        elif prediction == 'gesture_3':  # stop
            tello.forw_back_velocity = tello.up_down_velocity = (
                tello).left_right_velocity = tello.yaw_velocity = 0
        elif prediction == 'gesture_4':  # forward
            tello.forw_back_velocity(30)
        elif prediction == 'gesture_5':  # backwards
            tello.forw_back_velocity(-30)
        elif prediction == 'gesture_6':  # down
            (tello.move_down(25))
        elif prediction == 'gesture_7':  # ok fotka
            cv2.imwrite("picture.png", frame_read.frame)




def call_repeatedly(interval, func):
    stopped = Event()

    def loop():
        while not stopped.wait(interval):  # the first call is in `interval` secs
            func()

    Thread(target=loop).start()
    return stopped.set


def hand_landmarks():
    global frame, prediction
    prediction = '0'
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()

        prediction = clf.predict([landmarks])[0]
    # control(tello)





# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    clf = pickle.load(f)

prediction = '0'
while True:

    img = frame_read.frame
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1)
    if key == ord("s"):
        call_repeatedly(1, hand_landmarks)
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print(f"prediction = {prediction}")
    cv2.imshow("Frame", frame)

    if key == ord("q"):
        break

tello.land()
cv2.destroyAllWindows()
