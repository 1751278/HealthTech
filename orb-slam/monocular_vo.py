"""
monocular_vo.py
================
A monocular Visual Odometry (VO) pipeline that uses the `python-orb-slam3`
package (https://pypi.org/project/python-orb-slam3/) for feature extraction.

Important: `python-orb-slam3` (pip package `python_orb_slam3`, class
`ORBExtractor`) only wraps ORB-SLAM3's C++ ORB *feature extractor* -- it is
not a full SLAM/VO system and has no matcher, pose-estimation, or mapping
API. Its `ORBExtractor` is a drop-in replacement for `cv2.ORB_create()`: it
returns the same `cv2.KeyPoint` list + uint8 descriptor array, but uses
ORB-SLAM3's original extraction code (quad-tree keypoint distribution across
an image pyramid), which gives more evenly spread keypoints than OpenCV's
stock ORB. Matching, essential-matrix estimation, and pose recovery below
still use standard OpenCV, exactly as they would in ORB-SLAM3's own tracking
front-end.

Install:
    pip install python-orb-slam3 opencv-python numpy matplotlib
    (pre-built wheels are AMD64/x86_64 only; other platforms need a source build)

Pipeline per frame:
    1. Detect ORB keypoints + descriptors with python_orb_slam3.ORBExtractor.
    2. Match against the previous frame's descriptors (ratio test).
    3. Estimate the Essential matrix E with RANSAC (needs camera intrinsics K).
    4. Recover relative rotation R and translation direction t from E.
    5. Accumulate global pose:  R_global = R * R_global ;  t_global += s * R_global * t
       (monocular VO cannot recover absolute scale s from a single camera --
        you either supply it externally, e.g. from wheel odometry/GPS/IMU/known
        speed, or just visualize the trajectory up to an unknown scale factor.)

Visualization:
    - A live OpenCV window showing detected keypoints on the current frame.
    - A live OpenCV window showing the top-down (X-Z) trajectory being drawn.
    - A final matplotlib plot of the full trajectory, saved to trajectory.png.

Usage
-----
Webcam:
    python monocular_vo.py --source 0

Video file:
    python monocular_vo.py --source path/to/video.mp4 --fx 718.856 --fy 718.856 --cx 607.19 --cy 185.22

KITTI-style image sequence folder (numbered .png/.jpg frames):
    python monocular_vo.py --source path/to/image_folder --fx 718.856 --fy 718.856 --cx 607.19 --cy 185.22

Press ESC in the video window to quit early.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
from loopClosure.loop_closure import LoopClosure as lc
from accelerated_features.modules.xfeat import XFeat

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


# --------------------------------------------------------------------------- #
# Core monocular VO
# --------------------------------------------------------------------------- #
class MonocularVO:
    def __init__(self, K, n_features=3000, min_matches=8, ratio=0.75):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_features = n_features
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
        self.ratio = ratio

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        self.prev_feats = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # trajectory[i] is the 3x1 camera position at step i (arbitrary scale
        # unless an external scale is supplied every frame)
        self.trajectory = [self.cur_t.copy()]
        self.num_inlier_matches = 0

        self.xfeat = XFeat().to(self.device).eval()
        self.lc = lc(self._match,self.K,"orb", self.device)

    @staticmethod
    def to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _detect(self, gray):
        return self.orb.detectAndCompute(gray, None)
    
    def _convert_cv2_to_xfeat(self, cv_kpts, cv_descs, device="cpu"):
        """
        Converts OpenCV keypoints and descriptors to the XFeat dictionary format.
        """
        if not cv_kpts or cv_descs is None or len(cv_kpts) == 0:
            return {
                "keypoints": torch.empty((0, 2), dtype=torch.float32, device=device),
                "scores": torch.empty((0,), dtype=torch.float32, device=device),
                "descriptors": torch.empty((0, cv_descs.shape[1] if cv_descs is not None else 64), dtype=torch.float32, device=device)
            }

        # 1. Extract (x, y) coordinates from cv2.KeyPoint objects
        kpts_list = [kp.pt for kp in cv_kpts]
        xfeat_kpts = torch.tensor(kpts_list, dtype=torch.float32, device=device)

        # 2. Extract responses/scores from cv2.KeyPoint objects
        scores_list = [kp.response for kp in cv_kpts]
        xfeat_scores = torch.tensor(scores_list, dtype=torch.float32, device=device)

        # 3. Convert descriptors NumPy array to PyTorch Tensor
        # Ensure descriptors are float32 (ORB might be uint8, convert if needed)
        if cv_descs.dtype == np.uint8:
            cv_descs = cv_descs.astype(np.float32)
            
        xfeat_descs = torch.from_numpy(cv_descs).to(device=device, dtype=torch.float32)

        # Return the exact dictionary structure XFeat creates
        return {
            "keypoints": xfeat_kpts,      # Shape: (N, 2)
            "scores": xfeat_scores,        # Shape: (N,)
            "descriptors": xfeat_descs     # Shape: (N, D)
            }
    
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
    
    def process_frame(self, frame, frame_count, scale=1.0):
        """
        Processes one frame, updates the accumulated pose, and returns
        (kp, matches) for visualization purposes.
        """
        
        gray = self.to_gray(frame)
        kp, des = self._detect(gray)
        xFeat_format = self._convert_cv2_to_xfeat(kp, des)
        kp_full = xFeat_format['keypoints'].cpu().numpy()
        feats_full = xFeat_format['descriptors'].cpu().numpy()

        feats = feats_full[: self.n_features]
        kp_np = kp_full[: self.n_features]

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_des, self.prev_feats = gray, kp, des, feats
            return kp, []

        matches = self._match(self.prev_des, des)
        #matches = self._match_xFeat(self.prev_feats, feats)

        if len(matches) < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_des, self.prev_feats = gray, kp, des, feats
            return kp, matches
        

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            self.prev_gray, self.prev_kp, self.prev_des, self.prev_feats = gray, kp, des, feats
            return kp, matches

        _, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask)
        self.num_inlier_matches = int(pose_mask.sum()) if pose_mask is not None else 0

        if self.num_inlier_matches >= self.min_matches:
            self.cur_t = self.cur_t + scale * (self.cur_R @ t)

            #print(np.asarray(self.cur_t, dtype=np.float64).reshape(3))# print 3d position 

            self.cur_R = R @ self.cur_R
    
        # Record the pose on every frame, including frames where the pose
        # update was rejected. This keeps trajectory length 1:1 with frames.
        self.trajectory.append(self.cur_t.copy())

        traj = self.lc.process_loop_check(self.cur_R, self.cur_t, frame_count, kp_full, feats_full, kp, des, self.trajectory)

        if traj:
            self.trajectory = traj
            self.cur_R = self.lc.cur_R
            self.cur_t = self.lc.cur_t


        self.prev_gray, self.prev_kp, self.prev_des, self.prev_feats = gray, kp, des, feats
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
CALIBRATION_PATH = "cameraCalibrationData/calibrationMetrics/kenshi.txt"
CALIBRATION_VALS = []
RES_SCALE = 1/2.0
with open(CALIBRATION_PATH, "r") as file:
    for line in file:
        # Regex to find integers and floating-point numbers
        pattern = r'[-+]?\d*\.\d+|\d+'
        if re.findall(pattern, line):
            CALIBRATION_VALS.append(float(re.findall(pattern, line)[0]))
print(CALIBRATION_VALS)

def main():
    parser = argparse.ArgumentParser(description="Monocular Visual Odometry (ORB + Essential matrix)")
    parser.add_argument("--source", default="vo_videos/vid3.mp4",
                         help="Webcam index (e.g. 0), path to a video file, or path to a folder of image frames")
    parser.add_argument("--fx", type=float, default=CALIBRATION_VALS[0]*RES_SCALE, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=CALIBRATION_VALS[1]*RES_SCALE, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=CALIBRATION_VALS[2]*RES_SCALE, help="Principal point x")
    parser.add_argument("--cy", type=float, default=CALIBRATION_VALS[3]*RES_SCALE, help="Principal point y")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Per-frame translation scale factor. Monocular VO has no absolute "
                              "scale; supply this from external info (e.g. constant speed * dt) "
                              "or leave at 1.0 for a scale-free trajectory shape.")
    parser.add_argument("--n_features", type=int, default=3000, help="Max ORB features per frame")
    parser.add_argument("--no_display", action="store_true",
                         help="Disable live OpenCV windows (useful for headless runs)")
    parser.add_argument("--out", default="trajectory.png", help="Output path for the final trajectory plot")
    args = parser.parse_args()

    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0, 0, 1]], dtype=np.float64)

    print("Camera intrinsics K:\n", K)

    reader = FrameReader(args.source)
    vo = MonocularVO(K, n_features=args.n_features)

    FRAME_WINDOW = 1
    frame_count = 0
    old_frame = None
    try:
        kp, matches = None, None
        traj_canvas = None
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (int(720*RES_SCALE), int(1280*RES_SCALE)))  # Resize for faster processing
            if frame_count % FRAME_WINDOW == 0:
                kp, matches = vo.process_frame(frame, frame_count, scale=args.scale)
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
    save_matplotlib_plot(vo.trajectory, out_path=args.out)


if __name__ == "__main__":
    main()