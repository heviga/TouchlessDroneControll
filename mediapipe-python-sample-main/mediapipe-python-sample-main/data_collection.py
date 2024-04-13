import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Prepare data collection
data = []
labels = []


def collect_data(image, face_landmarks, label):
    # Convert landmarks to a flat array
    landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark]).flatten()
    data.append(landmarks)
    labels.append(label)


def main():
    label = input("Enter person's name for this session: ")  # Get label for this session
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, image = cap.read()
        if not ret:
            break
        # Convert image to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks for visualization
                for lm in face_landmarks.landmark:
                    x = int(lm.x * image.shape[1])
                    y = int(lm.y * image.shape[0])
                    cv.circle(image, (x, y), 1, (0, 255, 0), -1)

        # Display the resulting frame
        cv.imshow('Frame', image)

        # Press 's' to save the data of the face, 'q' to quit
        key = cv.waitKey(1)
        if key == ord('s'):
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    collect_data(image, face_landmarks, label)
                    print(f"Data collected for {label}")
        elif key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # Save collected data
    with open('face_data.pkl', 'wb') as f:
        pickle.dump((data, labels), f)
        print("Data saved successfully.")


if __name__ == '__main__':
    main()
