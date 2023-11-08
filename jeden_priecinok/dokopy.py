import sys
import traceback
import tellopy
import av
import cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import mediapipe as mp
import pickle
from threading import Thread, Event



import cv2 as cv2
import mediapipe as mp
import numpy as np
import pickle
from threading import Thread, Event
import time
import queue
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1)
# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    clf = model_data['classifier']
    scaler = model_data['scaler']

stop_event = Event()
frame_cv = []
def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()

def hand_landmarks(scaler, clf):
    global frame_cv, prediction
    print("I am here")
    prediction = '0'
    frame = frame_cv
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.multi_hand_landmarks[0].landmark]).flatten()
        landmarks = scaler.transform([landmarks])  # Normalize the landmarks
        prediction = clf.predict(landmarks)[0]  # Make a prediction
        print(f"Prediction: {prediction}")  # Debugging line to print prediction
    else:
        print("No hand landmarks detected.")


# MOVE TELLO

def control(tello):
    global prediction

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



prediction = '0'




def main():
    global frame_cv, prediction
    with open("gesture_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    clf = model_data['classifier']
    scaler = model_data['scaler']

    tello = tellopy.Tello()
    gesture_recognition_started = False

    try:
        tello.connect()
        tello.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(tello.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        #tello.takeoff()
        frame_skip = 300


        while True:

            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                key = cv2.waitKey(1)
                frame_cv = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                if key == ord("q"):
                    #tello.land()
                    break

                if key == ord("s") and not gesture_recognition_started:
                    gesture_recognition_started = True
                    call_repeatedly(1, hand_landmarks, scaler, clf)

                if gesture_recognition_started:
                    #prediction = hand_landmarks(scaler, clf, frame_cv)
                    cv2.putText(frame_cv, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #print(f"Prediction: {prediction}")
                        #control(tello, current_prediction, frame_cv)
                cv2.imshow('Original', frame_cv)
            if key == ord("q"):
                break
        tello.land()


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        tello.land()
        tello.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()