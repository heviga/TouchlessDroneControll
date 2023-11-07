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


from gesture_recognition import call_repeatedly, hand_landmarks, control, stop_event
prediction = None
import queue



def main():


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

        # skip first 300 frames
        #tello.takeoff()
        frame_skip = 300


        while True:
            key = cv2.waitKey(1)
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

                frame_cv = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', frame_cv)

                if key == ord("q"):
                    tello.land()
                    break

                if key == ord("s") and not gesture_recognition_started:
                    call_repeatedly(1, hand_landmarks, scaler, clf, frame_cv)
                    print("ok ide to")

                    # call_repeatedly(1, control, tello, prediction, frame_cv)

                    gesture_recognition_started = True

                if gesture_recognition_started:
                    current_prediction = hand_landmarks(scaler, clf, frame_cv)
                    if current_prediction is not None:

                        cv2.putText(frame_cv, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"predikcia{current_prediction}")




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