import cv2
from ultralytics import YOLOE

# 1. Load a YOLOE checkpoint model (e.g., nano or small segment model)
# YOLOE checkpoints support open-vocabulary text or visual prompting
model = YOLOE("YoloModels/yoloe-26s-seg.pt")

# 2. Define the custom text prompts you want to detect in real-time
# You can change these words to look for literally anything!
target_classes = ["open floor", "stairs", "door", "doorway"]
model.set_classes(target_classes)

# 3. Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Starting YOLOE live detection for: {target_classes}")
print("Press 'q' to exit.")

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (720, 1280))#vertical phone resolution

    if not success:
        print("Error: Failed to grab frame.")
        break

    # Run YOLOE prediction on the current frame
    # stream=True optimizes memory for video processing
    results = model.predict(frame, stream=True, verbose=False)

    # Visualize the results on the frame
    for result in results:
        # plot() draws the bounding boxes, masks, and text prompt labels automatically
        annotated_frame = result.plot()

    annotated_frame = cv2.resize(annotated_frame, (360, 640))#vertical phone resolution
    # Display the live annotated frame
    cv2.imshow("YOLOE Live Prompt Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()