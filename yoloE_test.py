import cv2
from yoloE import YOLOE

# Initialize the YOLOE model with the path to the checkpoint
model = YOLOE(model_path="YoloModels/yoloe-26s-seg.pt")

# Open the webcam ONCE, outside the loop
cap = cv2.VideoCapture(0)

try:
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # Run detection + tracking on the frame (IDs persist across frames)
        detections = model.track(frame)

        # Draw bounding boxes and labels on the frame
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["class_name"]
            confidence = detection["confidence"]
            track_id = detection.get("track_id")  # None until the tracker assigns one
            id_text = f"ID {track_id}" if track_id is not None else "ID -"

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Put label, confidence, and track ID
            cv2.putText(frame, f"{label} ({confidence:.2f}) {id_text}", (bbox[0], max(bbox[1] - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("YOLOE Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Always release the camera and close windows, even on error
    cap.release()
    cv2.destroyAllWindows()
