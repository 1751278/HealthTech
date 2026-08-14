import argparse
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from accelerated_features.modules.xfeat import XFeat
from loopClosure.loop_closure import LoopClosure as lc
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
        
        # Store the device so feature extraction, matching, and VLAD can
        # consistently use the same CPU/GPU device.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        



        # 1. Initialize XFeat
        self.xfeat = XFeat().to(self.device).eval()

        self.K = K

        self.lc = lc(self._match_xFeat,self.K,self.device)


        # Store the requested feature count so regular-frame detection
        # uses the same value instead of a hardcoded feature limit.
        self.n_features = n_features


        self.min_matches = min_matches
        self.ratio = ratio

        self.prev_gray = None
        self.prev_kp = None
        self.prev_feats = None



        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # trajectory[i] is the 3x1 camera position at frame i
        # Every frame appends the current pose, even if no new inliers were found.
        self.trajectory = []
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
    

    

    
    
    

    def process_frame(self, frame, frame_count, scale=1.0):
        """
        Processes one frame, updates the accumulated pose, and returns
        (kp, matches) for visualization purposes.
        """
        gray = self.to_gray(frame)

        # Run one dense XFeat detection pass and reuse its results for both
        # normal frame-to-frame tracking and keyframe creation. This avoids
        # running XFeat twice when a new keyframe is created.
        kp_full, output = self._detect_xFeat(gray, top_k=4096)
        feats_full = output['descriptors']

        # Use only the configured number of features for regular tracking,
        # while retaining the full dense detection for keyframe storage.
        kp_np = kp_full[: self.n_features]
        feats = feats_full[: self.n_features]

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp_np, feats

            # Record a trajectory point for the first frame so trajectory
            # length remains one-to-one with the number of processed frames.
            self.trajectory.append(self.cur_t.copy())
            return kp_np, np.zeros((0, 2), dtype=np.int64)

        matches = self._match_xFeat(self.prev_feats, feats)

        if matches.shape[0] < self.min_matches:
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp_np, feats

            # Even when tracking fails, record the unchanged pose so every
            # processed frame has exactly one corresponding trajectory point.
            self.trajectory.append(self.cur_t.copy())
            return kp_np, matches

        pts_prev = np.float32(self.prev_kp[matches[:, 0]])
        pts_cur = np.float32(kp_np[matches[:, 1]])

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            self.prev_gray, self.prev_kp, self.prev_feats = gray, kp_np, feats

            # Keep trajectory indexing synchronized even when no valid pose
            # can be recovered from the current frame.
            self.trajectory.append(self.cur_t.copy())
            return kp_np, matches

        _, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask)
        self.num_inlier_matches = int(pose_mask.sum()) if pose_mask is not None else 0

        # Reject degenerate / low-inlier estimates
        if self.num_inlier_matches >= self.min_matches:
            self.cur_t = self.cur_t + scale * (self.cur_R @ t)
            self.cur_R = R @ self.cur_R

        # Record the pose on every frame, including frames where the pose
        # update was rejected. This keeps trajectory length 1:1 with frames.
        self.trajectory.append(self.cur_t.copy())

        traj = self.lc.process_loop_check(self.cur_R, self.cur_t, frame_count, kp_full, feats_full,self.trajectory)

        if traj:
            self.trajectory = traj



        

        self.prev_gray, self.prev_kp, self.prev_feats = gray, kp_np, feats
        return kp_np, matches

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
    parser.add_argument("--source", default="vo_videos/vid1.mp4",
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

            kp_np, matches = vo.process_frame(frame, frame_count, scale=args.scale)
            frame_count += 1

            if not args.no_display:
                # Detection now returns raw NumPy keypoint coordinates rather
                # than cv2.KeyPoint objects. Convert them only when OpenCV
                # visualization actually requires that format.
                kp = vo._keypoints_to_cv2(kp_np)
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