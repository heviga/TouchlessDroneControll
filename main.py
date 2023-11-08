# import
import sys
import traceback
import tellopy
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import pickle
from threading import Thread, Event

stop_event = Event()

# initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

prediction = '0'
className = ''


f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

frame_cv = []


def hand_landmarks(model):
    global frame_cv, prediction, className
    frame = frame_cv
    x, y, c = frame_cv.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for handslms in results.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

    # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame_cv, handslms, mp_hands.HAND_CONNECTIONS)
        # Predict gesture
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]
    return className

def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()
def main():
    # Load the gesture recognizer model
    model = tf.keras.models.load_model('mp_hand_gesture')
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
                tello.land()
                break
            if key == ord("s") and not gesture_recognition_started:
                gesture_recognition_started = True
                call_repeatedly(1, hand_landmarks,model)

            if gesture_recognition_started:
                current_prediction = hand_landmarks(model)
                if current_prediction is not None:
                    cv2.putText(frame_cv, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Prediction: {current_prediction}")
                    # control(tello, current_prediction, frame_cv)
            cv2.imshow('Original', frame_cv)

            tello.land()

            if cv2.waitKey(1) == ord('q'):
                break
            if key == ord("s") and not gesture_recognition_started:
                gesture_recognition_started = True
                call_repeatedly(1, hand_landmarks, scaler, clf)

            if gesture_recognition_started:
                current_prediction = hand_landmarks(frame_cv)
                if current_prediction is not None:
                    cv2.putText(frame_cv, className, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Prediction: {current_prediction}")
                    # control(tello, current_prediction, frame_cv)
            cv2.imshow('Original', frame_cv)

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

