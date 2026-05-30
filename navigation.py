#################
# navigation.py
# Created by Sahir Abrar May 28 2026
# Last Updated: May 30 2026 by Sahir Abrar
# Description: This module captures video from a camera, runs depth estimation and tells the user to navigate to the door.
# TODO:
# - Use NCNN TFlight model for depth estimation (faster/more efficient than current DPT)
# - Make navigate() return a value
# - Add text-to-speech output
#################
 
import argparse
import sys
import collections
import cv2
import torch
import numpy as np
import matplotlib
 
sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
# --- Models ---
YOLO_MODEL_PATH    = "YoloModels/doorFrameModel.pt" # Using the door frame model so we can try and help navigate Users to the door.
DEPTH_MODEL_PATH   = "depthmodels/depth_anything_v2_vits.pth"
DEPTH_ENCODER      = 'vits'
DEPTH_FEATURES     = 64
DEPTH_OUT_CHANNELS = [48, 96, 192, 384]
 
# --- Capture ---
DEFAULT_SOURCE   = '1'    # Camera index or file path
FRAME_WIDTH      = 480
FRAME_HEIGHT     = 640
DEPTH_INFER_SIZE = 256    # Resolution passed to depth model inference
 
# --- Processing intervals (run every N frames) ---
DEFAULT_YOLO_INTERVAL  = 8
DEFAULT_DEPTH_INTERVAL = 3
 
# --- Zone weights ---
# Bottom zones are weighted more heavily than top zones because
# obstacles at foot level are more immediately dangerous.
TOP_WEIGHT = 0.4
BOT_WEIGHT = 0.6
 
# --- Steering thresholds ---
# All values are in depth_uint8 space (0–255), where higher = closer/more red.
 
# A zone's max must exceed this to be considered "extremely close" (bright red).
EXTREME_RED_THRESHOLD = 210
 
# Center column weighted-average must be this much lower than the best side
# column to confidently go forward (avoids wobbling when margins are tiny).
FORWARD_MARGIN = 15
 
# If ALL column weighted-averages exceed this, trigger the all-blocked panic.
ALL_BLOCKED_THRESHOLD = 160
 
# When turning, keep turning until center avg drops this far below the
# turning-side avg (like center is clearly becoming the best path.)
TURN_DONE_MARGIN = 10
 
# --- Hysteresis ---
# Number of consecutive frames that must agree on a new direction before
# we actually switch. Prevents flickering between commands.
HYSTERESIS_FRAMES = 4
 
# --- Visualization ---
DEPTH_COLORMAP  = 'Spectral_r'
STEER_COLOR     = (0, 255, 100)   # BGR
STEER_FONT_SCALE = 0.8
STEER_THICKNESS  = 2
BOX_COLOR_DOOR   = (0, 220, 220)  # BGR — door labels
BOX_COLOR_OTHER  = (0,  60, 220)  # BGR — non-door labels
 
# Debug: print the 12 zone values every depth frame
DEBUG_ZONES = True
 
# =============================================================================
# SETUP AND MODEL LOADING
# =============================================================================
 
parser = argparse.ArgumentParser()
parser.add_argument('--source',         default=DEFAULT_SOURCE)
parser.add_argument('--yolo-interval',  type=int, default=DEFAULT_YOLO_INTERVAL)
parser.add_argument('--depth-interval', type=int, default=DEFAULT_DEPTH_INTERVAL)
args   = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source
 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cmap   = matplotlib.colormaps.get_cmap(DEPTH_COLORMAP)
LUT = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
print("loading models...")
yolo = YOLO(YOLO_MODEL_PATH)
 
depth_model = DepthAnythingV2(
    encoder=DEPTH_ENCODER,
    features=DEPTH_FEATURES,
    out_channels=DEPTH_OUT_CHANNELS,
)
depth_model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
 
# =============================================================================
# STEERING ALGORITHM
# =============================================================================
 
def get_zone_stats(depth_uint8):
    """
    Split depth map into 6 zones (top/bottom × left/center/right) and
    return the 12 descriptive variables:
      - avg: mean depth value  (higher = closer)
      - max: peak depth value  (higher = more extreme red)
 
    Also returns a bottom-weighted column average for each of the 3 columns,
    blending top and bottom zones using TOP_WEIGHT / BOT_WEIGHT.
    """
    h, w = depth_uint8.shape
    top  = depth_uint8[:h//2, :]
    bot  = depth_uint8[h//2:, :]
 
    # Raw zone slices
    zones = {
        'tl': top[:, :w//3],
        'tc': top[:, w//3:2*w//3],
        'tr': top[:, 2*w//3:],
        'bl': bot[:, :w//3],
        'bc': bot[:, w//3:2*w//3],
        'br': bot[:, 2*w//3:],
    }
 
    stats = {k: {'avg': float(v.mean()), 'max': float(v.max())} for k, v in zones.items()}
 
    # Weighted column averages (bot counts more)
    col = {}
    for side in ('l', 'c', 'r'):
        t = stats[f't{side}']['avg']
        b = stats[f'b{side}']['avg']
        col[side] = TOP_WEIGHT * t + BOT_WEIGHT * b
 
    return stats, col
 
 
def get_steer(depth_uint8):
    """
    Compute steering direction from depth map using 12-variable zone analysis.
 
    Decision priority (highest → lowest):
      1. All-blocked panic   — everything is dangerously close → turn right
      2. Forward             — center weighted-avg is lowest by FORWARD_MARGIN AND no extreme reds in center zones
      3. Turn toward gap     — left or right column is lower; turn that way
      4. Fallback            — pick the least-blocked side
    """
    # Get zone stats and column averages
    stats, col = get_zone_stats(depth_uint8)
 
    # Debug printout of all zone values. Commented out by default.
    """
    if DEBUG_ZONES: 
        print(
            f"L={col['l']:.1f}  C={col['c']:.1f}  R={col['r']:.1f}  |  "
            f"tl_avg={stats['tl']['avg']:.0f} tc_avg={stats['tc']['avg']:.0f} tr_avg={stats['tr']['avg']:.0f}  "
            f"bl_avg={stats['bl']['avg']:.0f} bc_avg={stats['bc']['avg']:.0f} br_avg={stats['br']['avg']:.0f}  |  "
            f"tl_max={stats['tl']['max']:.0f} tc_max={stats['tc']['max']:.0f} tr_max={stats['tr']['max']:.0f}  "
            f"bl_max={stats['bl']['max']:.0f} bc_max={stats['bc']['max']:.0f} br_max={stats['br']['max']:.0f}",
            end="  ->  "
        )
    """
    
    # --- 1. All-blocked panic ---
    if all(col[s] > ALL_BLOCKED_THRESHOLD for s in ('l', 'c', 'r')):
        steer = ">> TURN RIGHT"  # default escape direction when fully boxed in
 
    # --- 2. Forward: center is clearly the best and has no extreme reds ---
    elif (
        col['c'] < col['l'] - FORWARD_MARGIN and
        col['c'] < col['r'] - FORWARD_MARGIN and
        stats['tc']['max'] < EXTREME_RED_THRESHOLD and
        stats['bc']['max'] < EXTREME_RED_THRESHOLD
    ):
        steer = "^ FORWARD"
 
    # --- 3. Turn toward the clearer side ---
    elif col['l'] < col['r']:
        # Left column is further away — turn left
        # Keep turning until center becomes best (handled by hysteresis in navigate())
        steer = "TURN LEFT <<"
    elif col['r'] < col['l']:
        # Right column is further away — turn right
        # Keep turning until center becomes best (handled by hysteresis in navigate())
        steer = ">> TURN RIGHT"
 
    # --- 4. Fallback: columns are roughly equal, prefer right ---
    else:
        steer = ">> TURN RIGHT"
 
    print(steer)
    return steer, col, stats
 
# =============================================================================
# MAIN LOOP
# =============================================================================
 
def navigate():
    """
    Main navigation loop for obstacle avoidance and door detection.
    Captures video, runs depth estimation and object detection at specified intervals, and applies the steering algorithm to decide on a direction.
    Displays the video feed with depth and detected boxes.
    
    """
    # Note: If Camo studio is not open, you may need to change source to (source - 1)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("webcam not found")
        exit()
 
    frame_num   = 0
    depth_color = None
    boxes       = []
    
    # Intialize smoothed depth for visualization (not used in steering)
    smoothed_depth = None
    ALPHA = 0.7  # how much to trust the new frame vs history
 
    # Hysteresis state
    committed_steer  = "^ FORWARD"   # the direction currently being acted on
    vote_buffer      = collections.deque(maxlen=HYSTERESIS_FRAMES) # buffer of recent steer decisions for hysteresis voting (hysteresis voting means we only switch to a new direction after several consecutive frames all agree on it, to avoid flickering)
 
    """
    Main loop
    For each frame, reads a camera frame, runs depth estimation every
depth_interval frames to compute steering direction, runs YOLO object
detection every yolo_interval frames to find doors, then draws bounding
boxes and the committed steering label onto the frame before displaying
both the raw camera feed and the depth colormap side by side.
Pressing 'q' exits the loop and releases the camera.
    """
    while True:
        ret, frame = cap.read()
        if not ret: # end of video file or camera error
            break
        
        # Resize the depth frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) 
        h, w  = frame.shape[:2]
 
        # --- Depth estimation ---
        if frame_num % args.depth_interval == 0:
            raw = depth_model.infer_image(frame, DEPTH_INFER_SIZE) # returns a 2D array of depth values (higher = closer)
            
            # Normalize raw depth to 0–255 for visualization and smoothing. Add small epsilon to denominator to avoid division by zero.
            raw_norm = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-6) * 255).astype(np.float32)
            
            # Apply smoothing
            if smoothed_depth is None:
                smoothed_depth = raw_norm
            else:
                smoothed_depth = ALPHA * raw_norm + (1 - ALPHA) * smoothed_depth
                
            # Convert to uint8 for steering algorithm and visualization
            depth_uint8 = smoothed_depth.astype(np.uint8)
             
            # raw_steer is the new candidate direction based on current depth map, col is the dict of weighted column averages, and stats is the dict of all zone stats.
            raw_steer, col, stats = get_steer(depth_uint8) 
 
            # Hysteresis: only commit to a new direction after HYSTERESIS_FRAMES
            # consecutive frames agree on it.
            vote_buffer.append(raw_steer)
            if len(vote_buffer) == HYSTERESIS_FRAMES and len(set(vote_buffer)) == 1:
                committed_steer = raw_steer
            

            # then inside the loop, just index into it
            depth_color = LUT[depth_uint8]  # pure numpy indexing, much faster
            
            # Resize depth frame for side-by-side display
            depth_color = cv2.resize(depth_color, (w, h))
 
        # --- Object detection ---
        if frame_num % args.yolo_interval == 0:
            results = yolo(frame, verbose=False)
            boxes   = results[0].boxes
 
        # --- Draw Frame ---
        for box in boxes:
            label           = yolo.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # bounding box coordinates
            color           = BOX_COLOR_DOOR if 'door' in label else BOX_COLOR_OTHER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # draw box
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) # draw label
 
        # draw steering label
        cv2.putText(frame, committed_steer, (w//2 - 90, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, STEER_FONT_SCALE, STEER_COLOR, STEER_THICKNESS)
 
        # display camera frame and depth side by side
        out = np.hstack([frame, depth_color]) if depth_color is not None else frame
        cv2.imshow('navigator', out)
 
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
        # Increment frame counter
        frame_num += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
 
# calling the function
navigate()
