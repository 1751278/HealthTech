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
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from superpoint_matcher import SuperPointMatcher

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

        self.K = K
        # ORB-SLAM3's own feature extractor (quad-tree keypoint distribution),
        # API-compatible with cv2.ORB_create()'s detectAndCompute().
        self.orb = ORBExtractor(n_features=n_features, scale_factor=1.2, n_levels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.sp_matcher = SuperPointMatcher(max_keypoints=2048)
        # Define LSH index parameters for binary descriptors
        FLANN_INDEX_LSH = 6
        FLANN_INDEX_KDTREE = 1
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,       # Standard recommendation
            key_size=12,          # Standard recommendation
            multi_probe_level=1,   # Standard recommendation
        )
        search_params = dict(checks=100)
        # Initialize the matcher with LSH parameters
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_matches = min_matches
        self.ratio = ratio

        self.prev_gray = None
        self.prev_kp = None
        self.prev_feats = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # trajectory[i] is the 3x1 camera position at step i (arbitrary scale
        # unless an external scale is supplied every frame)
        self.trajectory = [self.cur_t.copy()]
        self.num_inlier_matches = 0

    @staticmethod
    def to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _detect(self, gray):
        return self.orb.detectAndCompute(gray, None)
    
    def _detect_superpoint(self, frame):
        return self.sp_matcher.detect_and_compute(frame)
    
    def _match_superpoint(self, feats0, feats1, min_confidence=0.0):
        return self.sp_matcher.match(feats0, feats1, min_confidence=min_confidence)
    
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

    def process_frame(self, frame, scale=1.0):
        """
        Processes one frame, updates the accumulated pose, and returns
        (kp, matches) for visualization purposes.
        """
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
        """
        #with super points
        gray = self.to_gray(frame)
        kp, feats = self._detect_superpoint(frame)

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp, feats
            return kp, []
        
        matches = self._match_superpoint(self.prev_feats, feats)

        if len(matches) < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp, feats
            return kp, matches

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp, feats
            return kp, matches

        _, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask)
        self.num_inlier_matches = int(pose_mask.sum()) if pose_mask is not None else 0

        # Reject degenerate / low-inlier estimates
        if self.num_inlier_matches >= self.min_matches:
            self.cur_t = self.cur_t + scale * (self.cur_R @ t)
            self.cur_R = R @ self.cur_R
            self.trajectory.append(self.cur_t.copy())

        self.prev_gray, self.prev_kp, self.prev_feats = gray, kp, feats
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
        py = int(z * world_scale) + cy
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
def main():
    parser = argparse.ArgumentParser(description="Monocular Visual Odometry (ORB + Essential matrix)")
    parser.add_argument("--source", default="vo_videos/vid1.mp4",
                         help="Webcam index (e.g. 0), path to a video file, or path to a folder of image frames")
    parser.add_argument("--fx", type=float, default=990.57/2, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=991.07/2, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=372.83/2, help="Principal point x")
    parser.add_argument("--cy", type=float, default=644.54/2, help="Principal point y")
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

    frame_count = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (int(720*1/2), int(1280*1/2)))  # Resize for faster processing

            kp, matches = vo.process_frame(frame, scale=args.scale)
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

                traj_canvas = draw_trajectory_canvas(vo.trajectory)
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