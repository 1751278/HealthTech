import cv2
from python_orb_slam3 import ORBExtractor

def main():
    # Load a sample image
    image_path = "test.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image at '{image_path}'. Make sure the file exists.")
        return

    # Initialize the ORB-SLAM3 feature extractor
    orb_extractor = ORBExtractor()

    # Extract keypoints and descriptors
    keypoints, descriptors = orb_extractor.detectAndCompute(image)

    print(f"Successfully extracted {len(keypoints)} keypoints!")

if __name__ == "__main__":
    main()