import argparse
import glob
import os
import re
import sys
import threading
import cv2
import numpy as np
import matplotlib
# Set non-interactive backend to avoid needing a Tk main loop
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import serial
import time
from scipy.spatial.transform import Rotation as Rot

try:
    from python_orb_slam3 import ORBExtractor
except ImportError as e:
    raise ImportError(
        "Could not import 'python_orb_slam3'. Install it with:\n"
        "    pip install python-orb-slam3\n"
        "(pre-built wheels are only published for AMD64/x86_64; on other "
        "architectures you need to build it from source, see the project's "
        "GitHub page for build instructions)."
    ) from e

# --------------------------------------------------------------------------- #
# Frame source abstraction: webcam index, video file, or folder of images
# --------------------------------------------------------------------------- #
class FrameReader:
    """Unifies webcam / video file / image-sequence-folder into one interface."""

    def __init__(self, source):
        self.mode = None
        self.cap = None
        self.image_paths = []
        self.idx = 0

        if os.path.isdir(source):
            self.mode = "folder"
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            paths = []
            for e in exts:
                paths.extend(glob.glob(os.path.join(source, e)))
            self.image_paths = sorted(paths)
            if not self.image_paths:
                raise RuntimeError(f"No images found in folder: {source}")
        else:
            self.mode = "capture"
            # Webcam index like "0" or a video file path
            cam_source = int(source) if source.isdigit() else source
            self.cap = cv2.VideoCapture(cam_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        if self.mode == "folder":
            if self.idx >= len(self.image_paths):
                return False, None
            frame = cv2.imread(self.image_paths[self.idx])
            self.idx += 1
            return frame is not None, frame
        else:
            return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


# IMU reading loop
imu_data = None
imu_lock = threading.Lock()
is_connected_imu = False
is_connected_lock = threading.Lock()

def IMU_READER(main_thread):
    COM_PORT = 'COM3' # COM17 for KENSHI, COM3 for SAHIR
    BAUD_RATE = 115200

    print(f"Connecting to ESP32 on {COM_PORT}...")
    global is_connected_imu
    global imu_data
    try:
        # Open the serial port connection
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Allow connection to settle
        print("Connected successfully! Listening for data...")
        # read data from serial port to buffer any data that is corrupt
        for _ in range(10):
            if ser.in_waiting > 0:
                raw_data = ser.readline()
                decoded_data = raw_data.decode('utf-8', errors='ignore').strip()
                print(decoded_data)
                split_data = decoded_data.split("/")

                with imu_lock:
                    imu_data = [float(split_data[0]), float(split_data[1]), float(split_data[2])]

        with is_connected_lock:
            is_connected_imu = True
        while True:
            if ser.in_waiting > 0:
                # Read line, decode bytes to string, and strip extra whitespaces/newlines
                raw_data = ser.readline()
                decoded_data = raw_data.decode('utf-8', errors='ignore').strip()
                split_data = decoded_data.split("/")

                with imu_lock:
                    imu_data = [float(split_data[0]), float(split_data[1]), float(split_data[2])]
            if main_thread.is_alive() is not True:
                break
    except serial.SerialException as e:
        print(f"Error connecting to serial port: {e}")
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        with is_connected_lock:
            is_connected_imu = "Fail"
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
# --------------------------------------------------------------------------- #
# Core monocular VO
# --------------------------------------------------------------------------- #
class MonocularVO:
    def __init__(self, K, n_features=3000, min_matches=8, ratio=0.75):
        # consistently use the same CPU/GPU device.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.K = K
        # ORB-SLAM3's own feature extractor (quad-tree keypoint distribution),
        # API-compatible with cv2.ORB_create()'s detectAndCompute().
        self.orb = ORBExtractor(n_features=n_features, scale_factor=1.2, n_levels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Define LSH index parameters for binary descriptors
        FLANN_INDEX_LSH = 6
        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,       # Standard recommendation
            key_size=12,          # Standard recommendation
            multi_probe_level=1,   # Standard recommendation
        )
        search_params = dict(checks=50)
        # Initialize the matcher with LSH parameters
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_matches = min_matches
        self.n_features = n_features
        self.ratio = ratio

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        self.imu_angle = []
        self.imu_offset = []
        with is_connected_lock:
            self.imu_offset = imu_data

        # trajectory[i] is the 3x1 camera position at step i (arbitrary scale
        # unless an external scale is supplied every frame)
        self.trajectory = [self.cur_t.copy()]
        self.num_inlier_matches = 0

    @staticmethod
    def to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def _get_imu_data(self):
        with imu_lock:
            self.imu_angle = [imu_data[0], imu_data[1], -(imu_data[2] - self.imu_offset[2])]

    def _convert_to_R(self, angles):
        return Rot.from_euler("xzy", angles, degrees=True).as_matrix()
    
    def _detect(self, gray):
        return self.orb.detectAndCompute(gray, None)
    
    def _match(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        knn = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good
    def _keypoints_to_cv2(self, keypoints):
        """Convert a Tensor/NumPy array of keypoints (N, 2) to a list of cv2.KeyPoint."""
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        return [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in keypoints]

    def _detect_xFeat(self, frame, top_k=None):
        if top_k is None:
            top_k = self.n_features

        # XFeat inference does not require gradients, so disable autograd
        # to reduce memory usage and inference overhead.
        with torch.inference_mode():
            output = self.xfeat.detectAndCompute(frame, top_k=top_k)[0]

        # Keep keypoints as raw NumPy coordinates internally. Conversion to
        # cv2.KeyPoint objects is only needed later for visualization.
        kpts = output['keypoints']
        if isinstance(kpts, torch.Tensor):
            kpts = kpts.cpu().numpy()
        return kpts, output

    def _match_xFeat(self, feats0, feats1):
        # Return an empty index array if either descriptor set is unavailable.
        if feats0 is None or feats1 is None:
            return np.zeros((0, 2), dtype=np.int64)

        # Only transfer descriptors when they are not already on the correct
        # device, avoiding unnecessary CPU/GPU transfers.
        if feats0.device != self.device:
            feats0 = feats0.to(self.device)
        if feats1.device != self.device:
            feats1 = feats1.to(self.device)

        # XFeat matching is inference-only, so gradients are unnecessary.
        with torch.inference_mode():
            match = self.xfeat.match(feats0, feats1)

        if isinstance(match, (tuple, list)):
            idx0, idx1 = match
        else:
            match = torch.as_tensor(match)

            # Convert XFeat's possible match formats into two arrays of
            # corresponding descriptor indices.
            if match.ndim == 2 and match.shape[1] >= 2:
                idx0, idx1 = match[:, 0], match[:, 1]
            elif match.ndim == 1 and match.numel() % 2 == 0:
                idx0, idx1 = match[0::2], match[1::2]
            else:
                return np.zeros((0, 2), dtype=np.int64)

        if isinstance(idx0, torch.Tensor):
            idx0 = idx0.cpu().numpy()
            idx1 = idx1.cpu().numpy()
        idx0 = np.asarray(idx0, dtype=np.int64)
        idx1 = np.asarray(idx1, dtype=np.int64)
        if idx0.size == 0:
            return np.zeros((0, 2), dtype=np.int64)

        # Store matches as raw (N, 2) descriptor-index pairs instead of
        # constructing cv2.DMatch objects.
        return np.stack([idx0, idx1], axis=1)
    
    def process_frame(self, frame, scale=1.0):
        """
        Processes one frame, updates the accumulated pose, and returns
        (kp, matches) for visualization purposes.
        """
        
        gray = self.to_gray(frame)
        kp, des = self._detect(gray)

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return kp, []

        matches = self._match(self.prev_des, des)

        if len(matches) < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return kp, matches
        

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return kp, matches

        _, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask)
        self.num_inlier_matches = int(pose_mask.sum()) if pose_mask is not None else 0

        # Reject degenerate / low-inlier estimates
        if self.num_inlier_matches >= self.min_matches:
            self.cur_t = self.cur_t + scale * (self.cur_R @ t)

            print(np.asarray(self.cur_t, dtype=np.float64).reshape(3))# print 3d position 

            #self.cur_R = R @ self.cur_R
            self._get_imu_data()
            self.cur_R = self._convert_to_R(self.imu_angle)# from imu
            self.trajectory.append(self.cur_t.copy())

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return kp, matches


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def draw_trajectory_canvas(trajectory, canvas_size=600, world_scale=1.0):
    """Renders the X-Z trajectory onto a fresh top-down canvas each call."""
    canvas = np.full((canvas_size, canvas_size, 3), 30, dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2
    cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)  # origin marker

    pts = []
    for p in trajectory:
        x, z = float(p[0, 0]), float(p[2, 0])
        px = int(x * world_scale) + cx
        py = int(-z * world_scale) + cy #make z-axis go "up" in the image
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 0), 2)

    if pts:
        cv2.circle(canvas, pts[-1], 4, (255, 255, 0), -1)  # current position

    cv2.putText(canvas, "Top-down trajectory (X-Z)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def save_matplotlib_plot(trajectory, out_path="trajectory.png"):
    xs = [float(p[0, 0]) for p in trajectory]
    zs = [float(p[2, 0]) for p in trajectory]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, zs, "-b", linewidth=1.5)
    plt.scatter(xs[:1], zs[:1], c="green", label="start", zorder=5)
    plt.scatter(xs[-1:], zs[-1:], c="red", label="end", zorder=5)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Estimated camera trajectory (arbitrary scale unless supplied)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory plot to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
TRAJECTORY = None
CALIBRATION_PATH = "cameraCalibrationData/calibrationMetrics/sahir.txt"
CALIBRATION_VALS = []
with open(CALIBRATION_PATH, "r") as file:
    for line in file:
        # Regex to find integers and floating-point numbers
        pattern = r'[-+]?\d*\.\d+|\d+'
        if re.findall(pattern, line):
            CALIBRATION_VALS.append(float(re.findall(pattern, line)[0]))
print(CALIBRATION_VALS)

parser = argparse.ArgumentParser(description="Monocular Visual Odometry (ORB + Essential matrix)")
parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0), path to a video file, or path to a folder of image frames")
parser.add_argument("--fx", type=float, default=CALIBRATION_VALS[0]/2.0, help="Focal length x (pixels)")
parser.add_argument("--fy", type=float, default=CALIBRATION_VALS[1]/2.0, help="Focal length y (pixels)")
parser.add_argument("--cx", type=float, default=CALIBRATION_VALS[2]/2.0, help="Principal point x")
parser.add_argument("--cy", type=float, default=CALIBRATION_VALS[3]/2.0, help="Principal point y")
parser.add_argument("--scale", type=float, default=0.2,
                        help="Per-frame translation scale factor. Monocular VO has no absolute "
                            "scale; supply this from external info (e.g. constant speed * dt) "
                            "or leave at 1.0 for a scale-free trajectory shape.")
parser.add_argument("--n_features", type=int, default=3000, help="Max ORB features per frame")
parser.add_argument("--no_display", action="store_true",
                        help="Disable live OpenCV windows (useful for headless runs)")
parser.add_argument("--out", default="trajectory.png", help="Output path for the final trajectory plot")

args = parser.parse_args()
def main_loop():
    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0, 0, 1]], dtype=np.float64)

    print("Camera intrinsics K:\n", K)

    imu_connected = False
    while imu_connected == False:
        with is_connected_lock:
            imu_connected = is_connected_imu
        
    reader = FrameReader(args.source)
    vo = MonocularVO(K, n_features=args.n_features)

    FRAME_WINDOW = 1
    frame_count = 0
    try:
        kp, matches = None, None
        traj_canvas = None
        while True:
            with is_connected_lock:
                if is_connected_imu == False:
                    continue
                elif is_connected_imu == "Fail":
                    break

            ok, frame = reader.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (int(720*1/2.0), int(1280*1/2.0)))  # Resize for faster processing
            if frame_count % FRAME_WINDOW == 0:
                kp, matches = vo.process_frame(frame, scale=args.scale)
                traj_canvas = draw_trajectory_canvas(vo.trajectory)
            frame_count += 1

            if not args.no_display:
                vis = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
                cv2.putText(vis, f"frame {frame_count} | keypoints {len(kp)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(vis, f"matches {len(matches)} "
                                    f"| inliers {vo.num_inlier_matches}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                vis = cv2.resize(vis, (360, 640))  # Resize for display window
                cv2.imshow("Monocular VO - Frame", vis)

                cv2.imshow("Monocular VO - Trajectory", traj_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("ESC pressed, stopping.")
                    break
            else:
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")

    finally:
        reader.release()
        cv2.destroyAllWindows()

    print(f"Total frames processed: {frame_count}")
    print(f"Trajectory points: {len(vo.trajectory)}")
    TRAJECTORY = vo.trajectory
    save_matplotlib_plot(TRAJECTORY, out_path=args.out)

if __name__ == "__main__":
    main_thread = threading.Thread(target=main_loop)
    main_thread.start()

    imu_thread = threading.Thread(target=IMU_READER, args=(main_thread,))
    imu_thread.start()