import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from accelerated_features.modules.xfeat import XFeat


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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Initialize XFeat
        self.xfeat = XFeat().to(device).eval()

        self.K = K
        # ORB-SLAM3's own feature extractor (quad-tree keypoint distribution),
        # API-compatible with cv2.ORB_create()'s detectAndCompute().
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
    
    def _keypoints_to_cv2(self, keypoints):
        """Convert a Tensor/NumPy array of keypoints (N, 2) to a list of cv2.KeyPoint."""
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        return [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in keypoints]
    
    def _matches_to_cv2(self, matches_indices):
        """Convert matching indices to a list of cv2.DMatch."""
        
        # 1. If matches_indices is a tuple/list of 2 arrays (idx0, idx1), stack them into (N, 2)
        if isinstance(matches_indices, (tuple, list)) and len(matches_indices) == 2:
            idx0, idx1 = matches_indices
            if isinstance(idx0, torch.Tensor):
                matches_indices = torch.stack([idx0, idx1], dim=-1)
            else:
                matches_indices = np.stack([idx0, idx1], axis=-1)

        # 2. Convert PyTorch Tensor to NumPy
        if isinstance(matches_indices, torch.Tensor):
            matches_indices = matches_indices.cpu().numpy()

        # 3. Handle shape (2, N) -> transpose to (N, 2)
        if matches_indices.ndim == 2 and matches_indices.shape[0] == 2 and matches_indices.shape[1] != 2:
            matches_indices = matches_indices.T

        # 4. If shape is (N, 3) or higher (contains scores), keep only the first two columns (idx0, idx1)
        if matches_indices.ndim == 2 and matches_indices.shape[1] > 2:
            matches_indices = matches_indices[:, :2]

        # 5. Build OpenCV DMatch objects
        cv2_matches = []
        for idx0, idx1 in matches_indices:
            match = cv2.DMatch(_queryIdx=int(idx0), _trainIdx=int(idx1), _distance=0.0)
            cv2_matches.append(match)
            
        return cv2_matches
    
    def _detect_xFeat(self, frame):
        output = self.xfeat.detectAndCompute(frame, top_k=4096)[0]
        kpts = self._keypoints_to_cv2(output['keypoints'])
        return kpts, output['descriptors']
    
    def _match_xFeat(self, feats0, feats1):
        match = self.xfeat.match(feats0, feats1)
        return self._matches_to_cv2(match)
    
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
        #with super points
        gray = self.to_gray(frame)
        kp, feats = self._detect_xFeat(frame)

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp, feats
            return kp, []
        
        matches = self._match_xFeat(self.prev_feats, feats)

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
        py = int(-z * world_scale) + cy#make z-axis go "up" in the image
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
    parser.add_argument("--source", default="vo_videos/vid2.mp4",
                         help="Webcam index (e.g. 0), path to a video file, or path to a folder of image frames")
    parser.add_argument("--fx", type=float, default=483.30/1.5, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=483.69/1.5, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=360.41/1.5, help="Principal point x")
    parser.add_argument("--cy", type=float, default=639.01/1.5, help="Principal point y")
    parser.add_argument("--scale", type=float, default=0.2,
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
            frame = cv2.resize(frame, (int(720*1/1.5), int(1280*1/1.5)))  # Resize for faster processing

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