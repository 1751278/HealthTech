#################
# collectData.py
# Created by Kenshi Kadarusman May 21 2026
# Last Updated: ?
# Description: Collects training data for the door frame detection model. It uses a live camera feed to capture images of door frames and saves them to a specified directory. The user can press the space bar to capture an image and the ESC key to exit the application.
#################
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
        success = cv2.imwrite(f"DoorFrameData/images/Train/{id}.jpg", frame)
        if success:
            print(f"Image {id} saved!")
        else:
            print(f"Failed to save image {id}.")
    elif key % 256 == 27:  # ESC pressed
        print("Closing...")
        break

cam.release()
cv2.destroyAllWindows()