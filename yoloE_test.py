import cv2
from yoloE import YOLOE
import yoloE

# Initialize the YOLOE model with the path to the checkpoint
model = YOLOE(model_path="YoloModels/yoloe-26s-seg.pt")

while True:
    # Capture a frame from the webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Run detection on the captured frame
    detections = model.detect(frame)

    # Draw bounding boxes and labels on the frame
    for detection in detections:
        bbox = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # Put label and confidence
        cv2.putText(frame, f"{label} ({confidence:.2f})", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLOE Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()