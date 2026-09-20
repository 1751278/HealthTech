#################
# navigation.py
# Created by Ethan September 17 2026
# Last Updated: September 20 2026 by Ethan
# Last Change:
# - Initiated merger between VO and navigation.
# Description: This module captures video from a camera, runs depth estimation and tells the user to navigate to the door.
# TODO:
# - Nothing really, should be done, the output of this file should be used in navigation.py
#################

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
from monocular_vo import FrameReader as fr #Change this with any vo class provided it has the same methods (Ex: IMU)
from monocular_vo import MonocularVO as voclass #Change this with any vo class provided it has the same methods (Ex: IMU)
from accelerated_features.modules.xfeat import XFeat


import numpy as np

ip_address = "127.0.0.1"
port1 = "5555"
port2 = "5556"

ctx = zmq.Context()

# --- RECIEVER --- #
reciever = ctx.socket(zmq.PULL)
reciever.setsockopt(zmq.CONFLATE, 1)
reciever.connect(f"tcp://{ip_address}:{port1}")

# --- SENDER --- #
sender = ctx.socket(zmq.PUB)
sender.bind(f"tcp://{ip_address}:{port2}")

print("Server ready... on ports: ", port1, " and ", port2)

FRAME_WINDOW = 1  # Process every Nth frame for VO
SEND_INTERVAL = 30  # Send trajectory every N frames


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
    vo = voclass(K, n_features=args.n_features)
    
    frame_count = 0
    old_frame = None
    while True:
        print("YO")
        
        frame_bytes = reciever.recv()


        
        np_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            break
        
        frame = cv2.resize(frame, (int(720*RES_SCALE), int(1280*RES_SCALE)))  # Resize for faster processing
        if frame_count % FRAME_WINDOW == 0:
            kp, matches = vo.process_frame(frame, frame_count, scale=args.scale)
            traj_canvas = draw_trajectory_canvas(vo.trajectory)
            if frame_count % SEND_INTERVAL == 0:
                # Send the trajectory canvas to the client
                sender.send(vo.trajectory.tobytes())
                print("Sent!")

        frame_count += 1

        if not args.no_display:
 

            cv2.imshow("Monocular VO - Trajectory", traj_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESC pressed, stopping.")
                break
    cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_count}")
    print(f"Trajectory points: {len(vo.trajectory)}")
    save_matplotlib_plot(vo.trajectory, out_path=args.out)


if __name__ == "__main__":
    main()