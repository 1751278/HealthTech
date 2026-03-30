import cv2
def main():
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully.
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    print("Camera stream opened. Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was successfully read, display it
        if ret:
            cv2.imshow('Live Camera Feed', frame)
        else:
            print("Error: Failed to capture frame.")
            break

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and destroy all windows when done.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
