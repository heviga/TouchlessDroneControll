from djitellopy import Tello
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained k-NN model
with open('gesture_model.pkl', 'rb') as f:
    knn = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
tello.takeoff()

while True:
    img = frame_read.frame
    # Process the image and get hand landmarks
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
        gesture = knn.predict(landmarks_array.reshape(1, -1))[0]


        if gesture == 'gesture_1':#otocka
            tello.rotate_clockwise(30)
        elif gesture == 'gesture_2':#UP
            tello.move_up(25)
        elif gesture == 'gesture_3':  # stop
            tello.forw_back_velocity = tello.up_down_velocity = (
                tello).left_right_velocity = tello.yaw_velocity = 0
        elif gesture == 'gesture_4':  # forward
            tello.forw_back_velocity(30)
        elif gesture == 'gesture_5':  # backwards
            tello.forw_back_velocity(-30)
        elif gesture == 'gesture_6':  # down
            tello.move_down(25)
        elif gesture == 'gesture_7':  # ok fotka
            cv2.imwrite("picture.png", frame_read.frame)




        # ... Add other gestures and their corresponding drone movements

    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESC
        break

tello.land()
cv2.destroyAllWindows()
