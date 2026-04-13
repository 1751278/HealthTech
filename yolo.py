import cv2
from ultralytics import YOLO
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_object_detection():
    """
    Runs real-time object detection using YOLOv8 and a webcam feed.
    """
    logging.info("Starting YOLO object detection application.")

    try:
        model = YOLO("YoloModels/yolov11n.pt")
        logging.info("YOLOv11n model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(0)#Change to zero if not working
    if not cap.isOpened():
        logging.error("Failed to open webcam. Exiting.")
        return

    logging.info("Webcam initialized. Starting video stream loop.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        success, frame = cap.read()
        #frame = cv2.resize(frame, (360, 640))#vertical phone resolution
        if not success:
            logging.warning("Failed to grab frame. Exiting loop.")
            break

        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()

        cv2.imshow("AI Vision: Real-Time Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("'q' key pressed. Stopping application.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application stopped and resources released.")


if __name__ == "__main__":
    run_object_detection()
