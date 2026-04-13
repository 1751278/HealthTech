import time
import cv2
import easyocr
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")

def main():
    print("Hello From HealthTech! \n")
    ocr = easyocr.Reader(['en'], gpu=True, quantize=True) # this is the OCR reader, it takes a list of languages to read. In this case, it's set to English. It can be modified to read other languages if needed.
    THRESH = 0.50 #probability threshhold for displaying prediction

    if not cap.isOpened():
        print("Failed to open webcam. Exiting.")
        return

    print("Webcam initialized. Starting video stream loop.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    cur_time = time.perf_counter()
    while True:
        prev_time = cur_time
        cur_time = time.perf_counter()
        success, frame = cap.read()

        if not success:
            print("Failed to grab frame. Exiting loop.")
            break

        pred = ocr.readtext(frame)

        for (bbox, text, prob) in pred:
            if prob > THRESH:
                ##gather info on bound box
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                ## make bound box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                ## put text on image
                cv2.putText(frame, text, (top_left[0]-20, top_left[1]), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 255), thickness = 2) #CV2 uses BGR
        
        cv2.putText(frame, f"FPS:{1/(cur_time-prev_time):.2f}", (5, 15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,255), thickness = 2)
        cv2.imshow("Live text detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("'q' key pressed. Stopping application.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped")


if __name__ == "__main__":
    main()
