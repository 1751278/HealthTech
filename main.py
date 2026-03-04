import cv2 as cv
def main():
    # Open default camera
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Display frame
        cv.imshow('Camera Feed', frame)

        # Press 'q' to exit
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
