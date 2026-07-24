import numpy as np
import cv2
import glob

CALIBRATION_FOLDER = "cameraCalibrationData/kenshiPhoneImg"  # Update this path to your calibration images folder
# 1. Define configuration parameters
# CHECKERBOARD size: (internal_corners_width, internal_corners_height)
# For an 8x6 square board, the internal corner count is 7x5
CHECKERBOARD = (8, 5)
SQUARE_SIZE = 28.2  # Real-world size of a square edge in mm (or your preferred unit)

# Termination criteria for sub-pixel corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 2. Prepare coordinate vectors
# Vector to store 3D points in real world space
objpoints = []
# Vector to store 2D points in image plane
imgpoints = []

# Prepare the template 3D object points based on grid geometry
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 3. Load calibration snapshots
# Make sure your target directory matches where you keep your image data
images = glob.glob(f"{CALIBRATION_FOLDER}/*.jpg")
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]  # Formatted as (width, height)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points and refined image points
    if ret == True:
        objpoints.append(objp)

        # Refine the pixel coordinates to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optional: Visualize corner detection step-by-step
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Detection", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 4. Compute Camera Intrinsics
if len(objpoints) > 0 and gray_shape is not None:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    # 5. Output values
    print("--- Calibration Successful ---")
    print(f"Reprojection Error (RMS): {ret:.4f} pixels")
    print("\nIntrinsic Camera Matrix (mtx):")
    print(mtx)
    print("\nDistortion Coefficients (dist):")
    print(dist)

    # Breakdown of the intrinsic matrix structure
    print("\n--- Extracted Intrinsic Fields ---")
    print(f"Focal Length X (fx): {mtx[0, 0]:.2f} px")
    print(f"Focal Length Y (fy): {mtx[1, 1]:.2f} px")
    print(f"Principal Point X (cx): {mtx[0, 2]:.2f} px")
    print(f"Principal Point Y (cy): {mtx[1, 2]:.2f} px")
else:
    print("Error: Pattern corners could not be extracted from any image files.")