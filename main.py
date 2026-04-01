import cv2
import OCR #Gets the OCR File and its functions



def main():
    fnum = 0  # The Current Frame number
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
            print("\n Frame number:", fnum)
            fnum += 1
            
            img = cv2.resize(frame, (640,480)) # Resize the image to 640x480, this actually improves accuracy and fits our needs
            cv2.imshow('Live Camera Feed', img)
            OCR.read_text_from_image(img)  # Call the OCR function to read text from the current frame
            
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
