import cv2
import time

def UserCam():
    # Initialize video capture
    cap = cv2.VideoCapture(2)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (1000, 800))

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the frame with the overlay
        cv2.imshow("Camera User", frame)

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

UserCam()
