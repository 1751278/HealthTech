#################
# navigation.py
# Created by Sahir Abrar May 28 2026
# Last Updated: June 8 2026 by Kenshi & Ethan
# Last Change:
# - Added a not annoying sound.
# Description: This module captures video from a camera, runs depth estimation and tells the user to navigate to the door.
# TODO:
# - Use NCNN TFlight model for depth estimation (faster/more efficient than current DPT) -> make model run faster K
# - Add text-to-speech output
# - Need to combine door path and avoidance path for guidance to the door S and E
#   -Improvements: Door decay (lose confidence over time if we don't see it), make a class or data structure to hold the door state (confidence, last seen, etc.), possibly redesign interpolation of combine_steer
#   -Possibilities: Incorporate future IMU (gyroscope) data. Pending confirmation...
# - some way to allow user to change it themselves
# - try to find a qunatized version of depth model
# - implement desk and chair avoidance with yolo model 26 S and E 
#################
 
import argparse
import sys
import collections
import cv2
import torch
import numpy as np
import matplotlib
import soundfile as sf
import sounddevice as sd
import math
import tensorflow as tf

sys.path.append('./Depth-Anything-V2')
import os

from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
# --- Models ---
YOLO_MODEL_PATH    = "YoloModels/DoorFrameModel26.pt" # Using the door frame model so we can try and help navigate Users to the door.
YOLOV26_MODEL_PATH = "YoloModels/Yolo26n.pt" # Using the door frame model so we can try and help navigate Users to the door.
DEPTH_MODEL_PATH   = "depthmodels/depth_anything_v2_vits.pth"
TFLITE_PATH = "depthAnythingModelFaster/midasDepth.tflite"
DEPTH_ENCODER      = 'vits'
DEPTH_FEATURES     = 64
DEPTH_OUT_CHANNELS = [48, 96, 192, 384]
 
# --- Capture ---
DEFAULT_SOURCE   = '1'    # Camera index or file path
FRAME_WIDTH      = 360
FRAME_HEIGHT     = 640
DEPTH_INFER_SIZE = 256    # Resolution passed to depth model inference
 
# --- Audio ---
print("loading audio... check the constants section to change the sound file.")
AUDIO_DATA, SAMPLE_RATE = sf.read("SoundAssets/jazz.mp3") #CHANGE THIS FOR DIFFERENT SOUND, I FOUND THIS ONLINE IM SORRY
audio_location = 0  # current position in the audio file (in samples)
# Audio params for non-blocking sounds
sample_rate = 44100
phase = 0.0

# Shared state that the main loop can modify dynamically
audio_state = {
    "left_freq": 440.0,
    "right_freq": 440.0,
    "left_vol": 0.0,  
    "right_vol": 0.0  
}

# --- Processing intervals (run every N frames) ---
DEFAULT_YOLO_INTERVAL  = 3
DOOR_YOLO_INTERVAL     = 5
DEFAULT_DEPTH_INTERVAL = 2


# Floor weight given to door_dir when the door was NOT detected this frame
# (i.e. we're relying on a stale last_door_direction). 0 = ignore stale door
# entirely, 1 = trust it as much as a live detection.
DOOR_STALE_WEIGHT = 0.25

DOOR_CONFIDENCE_THRESHOLD = 0.5  # minimum confidence to consider a door detection valid

OTHER_CONFIDENCE_THRESHOLD = 0.3  # minimum confidence to consider a non-door detection valid

door_state = {
    "last_door_direction": 90,  # last known direction to the door (degrees)
    "last_door_confidence": 0.0,  # confidence of the last detection
    "last_seen_frame": -1,
}


# Color for the final combined steering arrow (BGR)
COMBINED_STEER_COLOR = (0, 255, 255)  # yellow

#Params for the better_steer function
STEER_SENSITIVITY_DOOR = 0.5 # how strongly the direction responds to left-right differences when steering toward a door
STEER_SENSITIVITY = 0.5  #  how strongly the direction responds to left-right differences
BLOCKED_SENSITIVITY = 0.7 # how much the direction should be adjusted when the center is blocked
BLOCKED_THRESHOLD = 150   # above this center value, consider the forward path blocked and move away
#Params for avoiding large objects
OBJECT_AREA_THRESH_MAX = 150000 #above this value the object is just everything
OBJECT_AREA_THRESH_MIN = 50000 #below this value the object is too small
OBJECT_STEER_SENSITIVITY = 0.000001 #sensitivity of steering if large object detected
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
# Moving average to smoothen navigation
direction_history = np.array([0,0,0,0,0])#list with 5 items
direction_smoothen = np.array([0.05, 0.05, 0.1, 0.1, 0.7])

# =============================================================================
# SETUP AND MODEL LOADING
# =============================================================================
 
parser = argparse.ArgumentParser()
parser.add_argument('--source',         default=DEFAULT_SOURCE)
parser.add_argument('--yolo-door-interval',  type=int, default=DOOR_YOLO_INTERVAL)
parser.add_argument('--yolo-default-interval', type=int, default=DEFAULT_YOLO_INTERVAL)
parser.add_argument('--depth-interval', type=int, default=DEFAULT_DEPTH_INTERVAL)
args   = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source
 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cmap   = matplotlib.colormaps.get_cmap(DEPTH_COLORMAP)
LUT = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
print("loading models...")
yolo = YOLO(YOLO_MODEL_PATH)
yolo26 = YOLO(YOLOV26_MODEL_PATH)
#disable depthanything for now
"""
depth_model = DepthAnythingV2(
    encoder=DEPTH_ENCODER,
    features=DEPTH_FEATURES,
    out_channels=DEPTH_OUT_CHANNELS,
)
depth_model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
"""
# TFLITE
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=7)#match num cores with threads
interpreter.allocate_tensors()
# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_tflite_depth(frame):
    orig_h, orig_w, _ = frame.shape
    # Resize image to target dimensions and convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (DEPTH_INFER_SIZE, DEPTH_INFER_SIZE), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize using standard ImageNet values
    img_input = img_resized.astype(np.float32)
    img_input = img_input / 255.0  # Scale to [0, 1]
    #image net norm
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_input = (img_input - mean) / std

    # Add batch dimension: (518, 518, 3) -> (1, 518, 518, 3)
    img_input = np.expand_dims(img_input, axis=0)
    # Check if model expects channels_first (1, 3, 518, 518) or channels_last (1, 518, 518, 3)
    # PyTorch conversions via onnx2tf typically default to channels_first unless specified
    if input_details[0]['shape'][1] == 3:
        img_input = np.transpose(img_input, (0, 3, 1, 2))

    # 3. Execute inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # 4. Extract and post-process the output
    depth_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Clean up dimensions (remove batch/channel squeezing if necessary)
    depth_map = np.squeeze(depth_output)

    # Resize depth map back up to match your original input dimensions
    depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Normalize depth map values to a visualizable 0-255 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_img = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_img = np.zeros_like(depth_map, dtype=np.uint8)

    # Apply an absolute colormap for depth visualization (e.g., INFERNO or PLASMA)
    depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
    return depth_img
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
 
    #print(steer)
    return steer, col, stats



# figure out the angle to go based on the same zone stats. This is a more continuous value that can be used to draw an arrow or something, rather than discrete labels.
def get_better_steer(depth_uint8):
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
    # Calculate the differences
    left_diff = col['l'] - col['r'] # larger positive means go right
    forward_prob = col['c']

    const_dir = STEER_SENSITIVITY  # adjust for sensitivity
    const_foward = BLOCKED_SENSITIVITY
    direction = left_diff * const_dir  # only taking edge of image to calculate direction (still goes forward if center blocked)
    if forward_prob > BLOCKED_THRESHOLD:  # if center is blocked, adjust direction to turn more aggressively toward the clearer side
        if direction > 0:
            direction += forward_prob * const_foward #If center blocked, go more left based on how blocked it is.
        else:
            direction -= forward_prob * const_foward#If center blocked, go more right based on how blocked it is.
    
    direction = max(-90, min(90, direction))  # clamp to [-90, 90]

    if direction > 0:
        audio_state["left_vol"] = 0.0
        audio_state["right_vol"] = abs(direction)/90.0 # scale volume by how strong the turn is
    else:
        audio_state["right_vol"] = 0.0
        audio_state["left_vol"] = abs(direction)/90.0 # scale volume by how strong the turn is

    #print(f"Direction: {direction:.1f} degrees Forward Prob: {forward_prob:.1f}")
    return direction
# function for non-blocking audio playback using sounddevice's callback mechanism
def audio_callback(outdata, frames, time_info, status):
    """Background thread that reads the audio_state and generates the sound."""
    #global phase
    global audio_location
    if status:
        print(status)
        
    # Read the current values from our shared dictionary
    #f_left = audio_state["left_freq"]
    #f_right = audio_state["right_freq"]
    v_left = audio_state["left_vol"]
    v_right = audio_state["right_vol"]
    
    # Create the time array for this chunk
    t = (np.arange(frames) + audio_location) % len(AUDIO_DATA) #the len(AUDIO_DATA) part makes it loop


    #Takes a small "chunk" (hence the name chunk :D) of the audio file
    chunk = AUDIO_DATA[t]

    #convert stereo -> mono, Stereo being that it plays on multiple audio channels, mono only playing in one channel (I am not audio expert)
    #If your audio is not stereo, then prob comment it out or something idk.
    mono = chunk.mean(axis=1) #Axis one basically averages out along the channels
    # Takes the audio data and multiplies it by a volume factor
    left_side = mono * v_left
    right_side = mono * v_right
    
    # Write to the output channels
    outdata[:, 0] = left_side
    outdata[:, 1] = right_side
    
    # Update the audio location (Basically just phase but renamed, didn't realize that this is basically the same thing :/)
    audio_location = (audio_location + frames) % len(AUDIO_DATA)
    
    # Keep the wave continuous
    #phase += frames

def get_steer_from_objects(boxes, depth_uint8, direction, thresh=0.3, dis_thresh=180, steer_sensitivity=0.000003):
    dir_ection = direction
    if len(boxes) == 0:
        return direction
    for i, box in enumerate(boxes):
            label           = yolo26.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            color           = BOX_COLOR_OTHER
            conf26 = box.conf.item()
            if conf26 > thresh:
                cropped_box = depth_uint8[y1:y2, x1:x2]
                avg_distance = np.mean(cropped_box)
                if avg_distance < dis_thresh:
                    # Calculate the center of the box
                    center_x = (x1 + x2) / 2
                    # Determine if the box is on the left or right side of the frame
                    if abs(center_x - depth_uint8.shape[1] / 2) < 50:  # If the box is near the center
                        if direction<0: #If we are already turning left, then steer more left
                            dir_ection -= steer_sensitivity * avg_distance * cropped_box.shape[0]*cropped_box.shape[1] # scale by area of box to make it more sensitive to larger objects
                        else:
                            dir_ection += steer_sensitivity * avg_distance * cropped_box.shape[0]*cropped_box.shape[1] # scale by area of box to make it more sensitive to larger objects
                    elif center_x < depth_uint8.shape[1] / 2:
                        # Box is on the left side, steer right
                        dir_ection += steer_sensitivity * avg_distance *cropped_box.shape[0]*cropped_box.shape[1] # scale by area of box to make it more sensitive to larger objects
                    else:
                        # Box is on the right side, steer left
                        dir_ection -= steer_sensitivity * avg_distance * cropped_box.shape[0]*cropped_box.shape[1] # scale by area of box to make it more sensitive to larger objects
    dir_ection = max(-90, min(90, dir_ection))  # clamp to [-90, 90]
    return dir_ection

def get_door_steer(box, frame_width, yolo_names, thresh=0.2):
    # gets straight line steering to the door
    if box.conf.item() > thresh:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = (int((x1+x2)/2), int((y1+y2)/2))
        direction = -(frame_width/2 - center[0])*STEER_SENSITIVITY_DOOR # positive if door is on the left, negative if on the right
        direction = max(-90, min(90, direction))  # clamp to [-90, 90]
        door_state["last_door_direction"] = direction
        door_state["last_door_confidence"] = box.conf.item()
        return direction
    if door_state["last_door_direction"] is not None:
        return door_state["last_door_direction"]
    return 90
def combine_steer(obstacle_dir, door_dir, door_confidence, center_blocked):
    if center_blocked <= BLOCKED_THRESHOLD: #If the center of the screen is not blocked sufficiently, then we can trust the door direction more.
        safety_weight = 1.0
    elif center_blocked >= ALL_BLOCKED_THRESHOLD: #If the center of the screen is blocked, then we should not trust the door direction at all.
        safety_weight = 0.0
    else:
        span = ALL_BLOCKED_THRESHOLD - BLOCKED_THRESHOLD #So between the two thresholds, we can linearly interpolate the safety weight. The more blocked the center is, the less we trust the door direction.
        safety_weight = 1.0 - (center_blocked - BLOCKED_THRESHOLD) / span

    # --- Confidence weight: distrust a stale (undetected) door direction ---
    
    door_weight = safety_weight * door_confidence #Is the door safe? Is the door detected? Trust the door direction more depending on these factors.
    obstacle_weight = 1.0 - door_weight #The more we trust the door, the less we trust the obstacle direction. The more we trust the obstacle direction, the less we trust the door.

    combined = obstacle_weight * obstacle_dir + door_weight * door_dir
    return max(-90, min(90, combined))
# =============================================================================
# MAIN LOOP
# =============================================================================
        



def navigate():
    """
    Main navigation loop for obstacle avoidance and door detection.
    Captures video, runs depth estimation and object detection at specified intervals, and applies the steering algorithm to decide on a direction.
    Displays the video feed with depth and detected boxes.
    Returns a dict with frames_processed and exit_reason ('user_quit' or 'stream_ended').
    """
    # Note: If Camo studio is not open, you may need to change source to (source - 1)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Camo Studio not detected, trying default camera...")
        cap = cv2.VideoCapture(source - 1)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()
 
    frame_num   = 0
    depth_color = None
    boxes       = []
    boxes26     = []

    # FIX: initialize direction so the arrow doesn't crash before depth runs
    direction = 0.0

    # Initialize smoothed depth for visualization (not used in steering)
    smoothed_depth = None
    ALPHA = 0.7  # how much to trust the new frame vs history
 
    # Hysteresis state
    committed_steer  = "^ FORWARD"   # the direction currently being acted on
    vote_buffer      = collections.deque(maxlen=HYSTERESIS_FRAMES)  # buffer of recent steer decisions for hysteresis voting

    # FIX: track why the loop exited so the return value is accurate
    exit_reason = "user_quit"

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
        if not ret:  # end of video file or camera error
            exit_reason = "stream_ended"
            break
        
        # Resize the depth frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) 
        h, w  = frame.shape[:2]
 
        # --- Depth estimation ---
        if frame_num % args.depth_interval == 0:
            # Disable Depth Anything inference
            """
            raw = depth_model.infer_image(frame, DEPTH_INFER_SIZE)  # returns a 2D array of depth values (higher = closer)
            
            # Normalize raw depth to 0–255 for visualization and smoothing. Add small epsilon to denominator to avoid division by zero.
            raw_norm = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-6) * 255).astype(np.float32)
            
            # Apply smoothing
            if smoothed_depth is None:
                smoothed_depth = raw_norm
            else:
                smoothed_depth = ALPHA * raw_norm + (1 - ALPHA) * smoothed_depth
                
            # Convert to uint8 for steering algorithm and visualization
            depth_uint8 = smoothed_depth.astype(np.uint8)
            """
            depth_map = get_tflite_depth(frame)  # run TFLITE depth estimation
            depth_uint8 = depth_map.astype(np.uint8)
            # raw_steer is the new candidate direction based on current depth map
            raw_steer, col, stats = get_steer(depth_uint8)

            direction = get_better_steer(depth_uint8)
 
            # Hysteresis: only commit to a new direction after HYSTERESIS_FRAMES
            # consecutive frames agree on it.
            vote_buffer.append(raw_steer)
            if len(vote_buffer) == HYSTERESIS_FRAMES and len(set(vote_buffer)) == 1:
                if committed_steer != raw_steer:
                    #print(f"Steering change: {committed_steer} -> {raw_steer}")

                    # FIX: play audio from the start of the file, not 1 second in
                    #data, sr = sf.read(AUDIO_FORWARD)  # default to forward sound
                    if raw_steer == "TURN LEFT <<":
                        #data, sr = sf.read(AUDIO_LEFT)
                        pass
                    elif raw_steer == ">> TURN RIGHT":
                        #data, sr = sf.read(AUDIO_RIGHT)
                        pass
                    #sd.play(data, sr)

                committed_steer = raw_steer

            # Index into LUT for fast colormap application
            depth_color = LUT[depth_uint8]
            
            # Resize depth frame for side-by-side display
            depth_color = cv2.resize(depth_color, (w, h))
       
        
        
        # --- Door detection ---
        if frame_num % args.yolo_door_interval == 0:
            results = yolo(frame, verbose=False)
            boxes   = results[0].boxes

            """
            # FIX: use get_door_steer to bias committed_steer toward the door when one is visible
            door_side = get_door_steer(boxes, w, yolo.names)
            if door_side is not None:
                if door_side == 'left':
                    committed_steer = "TURN LEFT <<"
                elif door_side == 'right':
                    committed_steer = ">> TURN RIGHT"
                elif door_side == 'center':
                    committed_steer = "^ FORWARD"
                print(f"Door detected on {door_side}, overriding steer to: {committed_steer}")
            """
 
        # --- Draw Frame ---
        max_conf = 0
        index = -1
        #Pick the best door (highest confidence) and use that to steer toward the door
        for i, box in enumerate(boxes):
            label           = yolo.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            color           = BOX_COLOR_DOOR
            conf = box.conf.item()
            if conf > max_conf and conf > DOOR_CONFIDENCE_THRESHOLD and 'door' in label:
                max_conf = conf
                index = i
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw box
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # draw label
 
        #Get straightline steering to the door if one is detected with high confidence
        if index != -1:
            door_direction = get_door_steer(boxes[index], w, yolo.names)
            door_state["last_seen_frame"] = frame_num
        else:
            door_direction = door_state["last_door_direction"]  # keep going toward the last known door direction if we lose sight of it
            max_conf = door_state["last_door_confidence"]*math.exp(-0.01*(frame_num - door_state["last_seen_frame"])) #If we don't see a door, use the last known confidence to determine how much to trust the last known direction
            print(max_conf, " On frame: ", frame_num - door_state["last_seen_frame"], " Orignial confidence: ", door_state["last_door_confidence"])
            
        # -- Object Detection using yolo26 for desk and chair avoidance. WIP --
        #"""
        if frame_num % args.yolo_default_interval == 0:
            results26 = yolo26(frame, verbose=False)
            boxes26   = results26[0].boxes
        # --- Draw Frame ---
        for i, box in enumerate(boxes26):
            label           = yolo26.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            color           = BOX_COLOR_OTHER
            conf26 = box.conf.item()
            if conf26 > OTHER_CONFIDENCE_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw box
                cv2.putText(frame, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # draw label
        #"""
        # get steer with obect detection
        print(direction)
        direction = get_steer_from_objects(boxes26, depth_uint8, direction)
        print(direction)
        #Call combine steer and use what we have currently to determine the final direction
        

        combined_direction = combine_steer(direction, door_direction, max_conf, col['c'])

        #smoothen direction with moving average
        direction_history[:-1] = direction_history[1:]
        direction_history[-1] = direction
        directionArray = direction_history * direction_smoothen
        smooth_direction = np.sum(directionArray)
        ### Direction should be finalized at this point
        
        if smooth_direction > 0:
            audio_state["left_vol"] = 0.0
            audio_state["right_vol"] = abs(smooth_direction)/90.0 # scale volume by how strong the turn is
        else:
            audio_state["right_vol"] = 0.0
            audio_state["left_vol"] = abs(smooth_direction)/90.0 # scale volume by how strong the turn is

 

        ##################### Draw Arrows
        start_point = (w//2, h//2)

        end_point = (int(math.sin(math.radians(smooth_direction)) * 100 + w//2), int(-math.cos(math.radians(smooth_direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)  # draw green arrow for avoidance direction

        end_point = (int(math.sin(math.radians(door_direction)) * 100 + w//2), int(-math.cos(math.radians(door_direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)  # draws red arrow for door direction

        end_point = (int(math.sin(math.radians(combined_direction)) * 100 + w//2), int(-math.cos(math.radians(combined_direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # draws blue arrow for FINAL direction

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

    return {"frames_processed": frame_num, "exit_reason": exit_reason}
 
# calling the function
if __name__ == "__main__":
    #starts callback audio stream for non-blocking sound playback
    stream = sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate)
    with stream:
        navigate()