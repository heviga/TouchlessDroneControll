import cv2
from tellopy import Tello

# Initialize Tello
tello = Tello()
tello.connect()
container = tello.get_video_stream()
tello.start_video()

# Create an OpenCV window for displaying the video
cv2.namedWindow("Tello Video")

# Define a callback function to handle video frame
print(tello.subscribe(tello.EVENT_VIDEO_FRAME, videoFrameHandler))
while True:
    frame = data
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    cv2.imshow("Tello Video", frame_bgr)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
tello.stop_video()
tello.land()
cv2.destroyAllWindows()
