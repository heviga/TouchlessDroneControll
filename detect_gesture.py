import cv2
import mediapipe as mp
from model import KeyPointClassifier
import landmark_utils as u
import sys
import traceback
import tellopy
from threading import Thread, Event
import time
import numpy as np
import av
stop_event = Event()

def call_repeatedly(interval, func, *args):
    def loop():
        while not stop_event.is_set():
            func(*args)  # Pass the scaler and classifier to the function
            stop_event.wait(interval)
    Thread(target=loop).start()

prediction = 'None'
image = []


def control(tello):
    global prediction
    if prediction:
        if prediction in gestures.values:
            if prediction == 'Like':
                tello.forward(30)
                time.sleep(1)
                tello.forward(0)
            elif prediction == 'call me':
                tello.backward(30)
                time.sleep(1)
                tello.backward(0)
            elif prediction == 'stop':
                time.sleep(5)
                tello.backward(0)
            elif prediction == 'rock':
                tello.flip_forward()
            elif prediction == 'live long':
                tello.flip_back()
            elif prediction == 'thumbs up':
                tello.up(30)
                time.sleep(1)
                tello.up(0)
            elif prediction == 'thumbs down':
                tello.down(30)
                time.sleep(1)
                tello.down(0)
            elif prediction == 'peace':
                cv2.imwrite("picture.png", image)
        else:
            tello.hover()
    else:
        prediction = 'No Hand Detected'
        tello.hover()

    return prediction

def recognise_gesture():
    global prediction, image
    gestures = {
        0: "Like",
        1: "Up",
        2: "Palm",
        3: "Fist",
        4: "Rock"
    }
    mp_hands = mp.solutions.hands
    prediction = 'None'
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        # mp_hands = mp.solutions.hands
        # hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        # mpDraw = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles

        kpclf = KeyPointClassifier()

        results = hands.process(image)
        gesture_index = 4
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = u.calc_landmark_list(image, hand_landmarks)
                keypoints = u.pre_process_landmark(landmark_list)
                gesture_index = kpclf(keypoints)

                prediction = gestures[gesture_index]


def main():
    global prediction, image
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
                image = np.array(frame.to_image())

                if key == ord("g"):
                    call_repeatedly(1, recognise_gesture)
                    gesture_recognition_started = True

                if key == ord("s"):
                    control(tello)

                if key == ord('q'):
                    break
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image, prediction,
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
                cv2.imshow('MediaPipe Hands', image)
            if key == ord('q'):
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