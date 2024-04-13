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

def main():
    # Load the KNN model
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Start video capture
    cap = cv.VideoCapture(0)

    while True:
        ret, image = cap.read()
        if not ret:
            break
        # Process the image
        image_rgb = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Analyze the results
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks and predict the name
                landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark]).flatten()
                name = model.predict([landmarks])[0]
                print(f"Detected: {name}")
                # Display the name on the frame
                cv.putText(image, f'Detected: {name}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow('Frame', cv.cvtColor(image, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
