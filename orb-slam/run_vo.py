import argparse
import glob
import os
import sys


import zmq


import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
from monocular_vo import FrameReader as fr
from monocular_vo import MonocularVO as vo
from accelerated_features.modules.xfeat import XFeat


import numpy as np

ctx = zmq.Context()
sock = ctx.socket(zmq.PAIR)
sock.connect("tcp://127.0.0.1:5555")

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
    parser.add_argument("--source", default="vo_videos/vid1.mp4",
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
    K = np.array([[args.fx, 0, args.cx],[0, args.fy, args.cy],[0, 0, 1]], dtype=np.float64)
    
    print("Camera intrinsics K:\n", K)

    reader = fr(args.source)
    vo = vo(K, n_features=args.n_features)
    FRAME_WINDOW = 1
    frame_count = 0
    old_frame = None
    while True:
        #Don't wait for reply
        if sock.poll(timeout=100):
            md = sock.recv_json(zmq.RCVMORE if False else 0)
            msg = sock.recv(copy=False)
            frame = np.frombuffer(msg, dtype=md["dtype"]).reshape(md["shape"])

            if frame is None:
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
        




    