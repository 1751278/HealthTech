import cv2
import uuid

cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (360, 640))#vertical phone resolution
    if not ret:
        break

    # Show the live feed
    cv2.imshow("Press SPACE to take a photo", frame)

    key = cv2.waitKey(1)
    if key % 256 == 32:  # SPACE pressed
        id = uuid.uuid4()
        cv2.imwrite(f"DoorFrameData/train/{id}.jpg", frame)
        print(f"Image {id} saved!")
    elif key % 256 == 27:  # ESC pressed
        print("Closing...")
        break

cam.release()
cv2.destroyAllWindows()