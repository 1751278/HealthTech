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
    3. Fit BOTH an Essential matrix E and a Homography H with RANSAC.
    4. Model selection (as in ORB-SLAM's initializer): the Essential matrix
       E = [t]_x R is mathematically degenerate as the translation t -> 0,
       so pure/near-pure camera rotation (panning/turning in place, little
       or no translation) makes E-based rotation estimates noisy or wrong.
       A Homography stays well-conditioned in exactly that regime, so if H
       explains the matches at least as well as E does (by inlier ratio),
       rotation is recovered by decomposing H instead of E.
    5. Accumulate global pose:  R_global = R * R_global ;  t_global += s * R_global * t
       (monocular VO cannot recover absolute scale s from a single camera --
        you either supply it externally, e.g. from wheel odometry/GPS/IMU/known
        speed, or just visualize the trajectory up to an unknown scale factor.)
    6. Track how much rotation has actually occurred: per-frame relative
       rotation angle, a running cumulative total, and the current absolute
       yaw/pitch/roll, all printed/drawn so you can read the numbers instead
       of only seeing a (possibly flat) translation trace.

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
# Rotation utilities
# --------------------------------------------------------------------------- #
def rotation_angle_deg(R):
    """Magnitude of the rotation represented by R, in degrees (axis-angle norm).
    Sign-agnostic -- useful as a 'how much did it turn' odometer even when
    the axis of rotation varies frame to frame."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return float(np.degrees(np.linalg.norm(rvec)))


def rotation_matrix_to_euler_deg(R):
    """Yaw/pitch/roll in degrees for OpenCV camera axes (X-right, Y-down, Z-forward):
    yaw = turning left/right (rotation about Y), pitch = tilting up/down
    (rotation about X), roll = tilting sideways (rotation about Z)."""
    sy = np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[0, 2], R[2, 2])
        pitch = np.arctan2(-R[1, 2], sy)
        roll = np.arctan2(R[1, 0], R[1, 1])
    else:
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        pitch = np.arctan2(-R[1, 2], sy)
        roll = 0.0
    return tuple(float(v) for v in np.degrees([yaw, pitch, roll]))


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
    def __init__(self, K, n_features=3000, min_matches=8, ratio=0.75,
                 homography_ratio_thresh=0.45, min_parallax_px=0.75):
        self.K = K
        # ORB-SLAM3's own feature extractor (quad-tree keypoint distribution),
        # API-compatible with cv2.ORB_create()'s detectAndCompute().
        self.orb = ORBExtractor(n_features=n_features, scale_factor=1.2, n_levels=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.min_matches = min_matches
        self.ratio = ratio
        # If the homography inlier ratio SH/(SH+SE) exceeds this, prefer the
        # homography-derived rotation over the essential-matrix one (ORB-SLAM
        # uses ~0.45 here). This is what rescues rotation estimates during
        # near-pure-rotation motion, where the essential matrix is degenerate.
        self.homography_ratio_thresh = homography_ratio_thresh
        # Skip the pose update entirely if the median match displacement (px)
        # is below this -- there isn't enough signal to trust any estimate.
        self.min_parallax_px = min_parallax_px

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # trajectory[i] is the 3x1 camera position at step i (arbitrary scale
        # unless an external scale is supplied every frame)
        self.trajectory = [self.cur_t.copy()]
        self.orientations = [self.cur_R.copy()]
        self.num_inlier_matches = 0
        self.pose_model = None  # "essential" or "homography", whichever was used last
        self.last_relative_rotation_deg = 0.0
        self.cumulative_rotation_deg = 0.0  # running total, an "odometer" for turning

    @staticmethod
    def to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _detect(self, gray):
        return self.orb.detectAndCompute(gray, None)

    def _match(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good

    @staticmethod
    def _select_homography_solution(Rs, Ts, Ns, pts_prev, pts_cur):
        """Pick the physically valid homography decomposition. decomposeHomographyMat
        returns up to 4 mathematical solutions; filter by the positive-depth
        (cheirality) constraint when possible, then break any remaining tie by
        preferring the smallest translation norm -- correct in the near-pure-
        rotation regime this fallback targets, where the true translation is
        close to zero."""
        num = len(Rs)
        candidates = list(range(num))
        try:
            sol = cv2.filterHomographyDecompByVisibleRefpoints(
                Rs, Ns,
                pts_prev.reshape(-1, 1, 2).astype(np.float32),
                pts_cur.reshape(-1, 1, 2).astype(np.float32),
            )
            if sol is not None and len(sol) > 0:
                candidates = sol.flatten().tolist()
        except cv2.error:
            pass  # keep the unfiltered candidate list
        return min(candidates, key=lambda i: np.linalg.norm(Ts[i]))

    def _estimate_relative_pose(self, pts_prev, pts_cur):
        """Returns (R, t, model_name, num_inliers) or (None, None, None, 0)."""
        E, mask_e = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        H, mask_h = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, 3.0)

        inliers_e = int(mask_e.sum()) if mask_e is not None else 0
        inliers_h = int(mask_h.sum()) if mask_h is not None else 0

        if inliers_e + inliers_h == 0:
            return None, None, None, 0

        homography_ratio = inliers_h / max(inliers_e + inliers_h, 1)

        # --- Prefer homography when it explains the motion at least as well:
        # this is the case that rescues rotation during near-pure-rotation
        # motion, where the essential matrix is degenerate. ---
        if H is not None and homography_ratio > self.homography_ratio_thresh:
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, self.K)
            if num > 0:
                best = self._select_homography_solution(Rs, Ts, Ns, pts_prev, pts_cur)
                return Rs[best], Ts[best], "homography", inliers_h

        # --- Otherwise fall back to the essential matrix (needs real baseline) ---
        if E is not None and E.shape == (3, 3):
            _, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask_e)
            n_inliers = int(pose_mask.sum()) if pose_mask is not None else 0
            if n_inliers >= self.min_matches:
                return R, t, "essential", n_inliers

        return None, None, None, 0

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

        # Not enough parallax (near-static frames, duplicated frames, etc.) --
        # skip rather than accept a numerically unstable estimate.
        median_flow = float(np.median(np.linalg.norm(pts_cur - pts_prev, axis=1)))
        if median_flow < self.min_parallax_px:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return kp, matches

        R, t, model, n_inliers = self._estimate_relative_pose(pts_prev, pts_cur)
        self.num_inlier_matches = n_inliers
        self.pose_model = model

        if R is not None:
            # Per-frame and cumulative rotation, so "how much rotation has
            # occurred" is an explicit number instead of something you have
            # to infer from the trajectory shape.
            self.last_relative_rotation_deg = rotation_angle_deg(R)
            self.cumulative_rotation_deg += self.last_relative_rotation_deg

            self.cur_t = self.cur_t + scale * (self.cur_R @ t)
            self.cur_R = R @ self.cur_R
            self.trajectory.append(self.cur_t.copy())
            self.orientations.append(self.cur_R.copy())

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return kp, matches


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def draw_trajectory_canvas(trajectory, orientations=None, canvas_size=600, world_scale=1.0):
    """Renders the X-Z trajectory onto a fresh top-down canvas each call.
    If `orientations` is given, also draws an arrow for the current heading
    (the camera's forward/+Z axis projected onto the X-Z plane) so rotation
    is visible, not just position."""
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

        if orientations is not None and len(orientations) == len(pts):
            forward = orientations[-1] @ np.array([0.0, 0.0, 1.0])  # camera +Z in world
            heading = np.array([forward[0], forward[2]])
            norm = np.linalg.norm(heading)
            if norm > 1e-6:
                heading = heading / norm * 25  # fixed-length arrow
                tip = (int(pts[-1][0] + heading[0]), int(pts[-1][1] + heading[1]))
                cv2.arrowedLine(canvas, pts[-1], tip, (0, 165, 255), 2, tipLength=0.35)

    cv2.putText(canvas, "Top-down trajectory (X-Z), arrow = heading", (10, 20),
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
    parser.add_argument("--source", default="1",
                         help="Webcam index (e.g. 0), path to a video file, or path to a folder of image frames")
    parser.add_argument("--fx", type=float, default=1345/4, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=1345/4, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=360/2, help="Principal point x")
    parser.add_argument("--cy", type=float, default=640/2, help="Principal point y")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Per-frame translation scale factor. Monocular VO has no absolute "
                              "scale; supply this from external info (e.g. constant speed * dt) "
                              "or leave at 1.0 for a scale-free trajectory shape.")
    parser.add_argument("--n_features", type=int, default=3000, help="Max ORB features per frame")
    parser.add_argument("--homography_ratio_thresh", type=float, default=0.45,
                         help="Prefer homography-derived rotation over essential-matrix when "
                              "homography inliers / (homography+essential inliers) exceeds this. "
                              "Lower it if rotation still looks wrong during panning/turning.")
    parser.add_argument("--min_parallax_px", type=float, default=0.75,
                         help="Skip the pose update if the median matched-keypoint pixel "
                              "displacement is below this (not enough signal to trust any estimate).")
    parser.add_argument("--no_display", action="store_true",
                         help="Disable live OpenCV windows (useful for headless runs)")
    parser.add_argument("--out", default="trajectory.png", help="Output path for the final trajectory plot")
    args = parser.parse_args()

    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0, 0, 1]], dtype=np.float64)

    print("Camera intrinsics K:\n", K)

    reader = FrameReader(args.source)
    vo = MonocularVO(K, n_features=args.n_features,
                      homography_ratio_thresh=args.homography_ratio_thresh,
                      min_parallax_px=args.min_parallax_px)

    frame_count = 0
    try:
        while True:
            ok, frame = reader.read()
            frame = cv2.resize(frame, (360, 640))  # downsample for speed
            if not ok or frame is None:
                break

            kp, matches = vo.process_frame(frame, scale=args.scale)
            frame_count += 1

            if not args.no_display:
                yaw, pitch, roll = rotation_matrix_to_euler_deg(vo.cur_R)
                vis = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
                cv2.putText(vis, f"frame {frame_count} | keypoints {len(kp)} | matches {len(matches)} "
                                  f"| inliers {vo.num_inlier_matches} | model {vo.pose_model}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(vis, f"rotation: this-frame {vo.last_relative_rotation_deg:5.2f} deg | "
                                  f"cumulative {vo.cumulative_rotation_deg:7.2f} deg | "
                                  f"yaw {yaw:6.1f} pitch {pitch:6.1f} roll {roll:6.1f}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("Monocular VO - Frame", vis)

                traj_canvas = draw_trajectory_canvas(vo.trajectory, vo.orientations)
                cv2.imshow("Monocular VO - Trajectory", traj_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("ESC pressed, stopping.")
                    break
            else:
                if frame_count % 30 == 0:
                    yaw, pitch, roll = rotation_matrix_to_euler_deg(vo.cur_R)
                    print(f"frame {frame_count}: model={vo.pose_model} "
                          f"this-frame_rot={vo.last_relative_rotation_deg:.2f}deg "
                          f"cumulative_rot={vo.cumulative_rotation_deg:.2f}deg "
                          f"yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}")

    finally:
        reader.release()
        cv2.destroyAllWindows()

    print(f"Total frames processed: {frame_count}")
    print(f"Trajectory points: {len(vo.trajectory)}")
    final_yaw, final_pitch, final_roll = rotation_matrix_to_euler_deg(vo.cur_R)
    print(f"Cumulative rotation traveled: {vo.cumulative_rotation_deg:.2f} deg")
    print(f"Final absolute orientation -> yaw: {final_yaw:.2f}, pitch: {final_pitch:.2f}, roll: {final_roll:.2f}")
    save_matplotlib_plot(vo.trajectory, out_path=args.out)


if __name__ == "__main__":
    main()
