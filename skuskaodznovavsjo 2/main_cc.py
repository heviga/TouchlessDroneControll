import sys
import traceback
import tellopy
import av
import cv2 as cv2  # for avoidance of pylint error
import numpy
import time


def main():
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 300 frames
        drone.takeoff()
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', image)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                if key == 27:  # ESC
                    break
                elif key == ord('w'):
                    drone.forward(1)
                elif key == ord('s'):
                    drone.backward(30)
                elif key == ord('a'):
                    drone.left(30)
                elif key == ord('d'):
                    drone.right(30)
                elif key == ord('e'):
                    drone.clockwise(30)
                elif key == ord('q'):
                    drone.counter_clockwise(30)
                elif key == ord('r'):
                    drone.up(30)
                elif key == ord('f'):
                    drone.down(30)
                elif key == ord('p'):
                    cv2.imwrite("picture.png", image)
            if key == ord("q"):
                break
        drone.land()


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.land()
        drone.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()