#################
# navigation.py
# Created by Sahir Abrar May 28 2026
# Last Updated: September 15 2026 by Ethan
# Last Change:
# - Initiated merger between VO and navigation.
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
import subprocess
import argparse
import sys
import collections
import time
import cv2
import torch
import numpy as np
import matplotlib
import soundfile as sf
import math
import tensorflow as tf
import json

# sounddevice is imported lazily (see _open_audio_stream) because importing it
# runs PortAudio's Pa_Initialize() at import time, which can block or fail on a
# machine with no/enumerating audio endpoints. A bad audio device must not stop
# the camera, depth, YOLO or LLM paths from starting.

import zmq
from zmq.decorators import socket
import trajectory_helper as traj_helper
from goal_nav import GoalNav, Speaker, VoiceGoal
from llm_vision import LLMVisionAssistant
from LLM_Yolo_integration import LLMProcessor
from action import Action

ip_address = "127.0.0.1"
port1 = "5555"
port2 = "5556"




ctx = zmq.Context()

# --- SENDER ---
sender = ctx.socket(zmq.PUSH)
sender.setsockopt(zmq.CONFLATE, 1)
sender.bind(f"tcp://{ip_address}:{port1}")
print("Sender ready... on port: ", port1)

poller_send = zmq.Poller()
poller_send.register(sender, zmq.POLLOUT)
BUFFER_TIME = 500 # milliseconds to buffer for sending the frame to the vo process
# --- Receiver ---
receiver = ctx.socket(zmq.SUB)
receiver.setsockopt(zmq.CONFLATE, 1)
receiver.connect(f"tcp://{ip_address}:{port2}") # Change to Server IP later
receiver.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all messages

print("Receiver ready... on port: ", port2)
sys.path.append('./Depth-Anything-V2')
import os

from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
from ultralytics import YOLOE
# =============================================================================
# CONFIGURATION
# =============================================================================

# --- LLM ---
GOAL_PROMPT = (
    "You are a visual navigation assistant. Given the current camera image "
    "and YOLO object detections, choose a useful visual target to guide "
    "short-distance movement.\n\n"
    "Return either a bbox in [x1, y1, x2, y2] format, MATCHING one "
    "of the PROVIDED bbox detections, or a turning angle if no suitable target "
    "is available (prefer bboxes over angles). You should only return a turn angle if there is nothing to track or it is a dead end. "
    "Choose targets that appear to lead toward open, navigable space or to the provided destination."
    "Targets can still be obstructed by minor obstacles "
    "because program can avoid using depth map. Prefer continuing toward a useful existing target when appropriate.\n\n"
    "For a bbox, return its coordinates in the image's pixel space. "
    "Moving straight involves moving in a straight line, naturally avoiding obstances (which is handled automatically), and searching for a object \n"
    "When giving the search object, please give a list of objects to search for in a string format. Stay general with these search terms, like door, sign, etc. \n"
    "For a turn, use degrees, with negative values meaning left and positive "
    "values meaning right.\n\n"
    'Output only a JSON object: {"action":"target","bbox":[x1,y1,x2,y2]} '
    'or {"action":"straight","search":[string of objects to search for]} '
    'or on occasion {"action":"turn","angle":degrees}.'
)


# --- Models ---
YOLO_MODEL_PATH    = "YoloModels/DoorFrameModel26.pt" # Using the door frame model so we can try and help navigate Users to the door.
YOLOV26_MODEL_PATH = "YoloModels/Yolo26n.pt" # Using the door frame model so we can try and help navigate Users to the door.
YOLOE_MODEL_PATH = "YoloModels/yoloe-26n-seg.pt"
DEPTH_MODEL_PATH   = "depthmodels/depth_anything_v2_vits.pth"
TFLITE_PATH = "depthAnythingModelFaster/midasDepth.tflite"
DEPTH_ENCODER      = 'vits'
DEPTH_FEATURES     = 64
DEPTH_OUT_CHANNELS = [48, 96, 192, 384]

# --- VO ---
VO_venv = os.path.abspath("./orb-slam/.venv/Scripts/python.exe")
VO_script = os.path.abspath("./orb-slam/run_vo.py")
result = subprocess.Popen(
    [VO_venv, "-u", VO_script],
    cwd=os.path.dirname(VO_script), #Make the cwd the same as the script so it can find the calibration data
    #stdout=subprocess.PIPE, #As far as I understand, this is for prints
    #stderr=subprocess.PIPE, #This is for errors
    text=True
)



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
DEFAULT_YOLO_INTERVAL  = 1
DOOR_YOLO_INTERVAL     = 1
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
direction_history = np.zeros(5, dtype=np.float64) # list with dtype int64 9
direction_smoothen = np.array([0.05, 0.05, 0.1, 0.1, 0.7])

# =============================================================================
# SETUP AND MODEL LOADING
# =============================================================================



poller_recv = zmq.Poller()
poller_recv.register(receiver, zmq.POLLIN)


print("Streaming video... Listening for feedback on port 5556.")


parser = argparse.ArgumentParser()
parser.add_argument('--source',         default=DEFAULT_SOURCE)
parser.add_argument('--yolo-door-interval',  type=int, default=DOOR_YOLO_INTERVAL)
parser.add_argument('--yolo-default-interval', type=int, default=DEFAULT_YOLO_INTERVAL)
parser.add_argument('--depth-interval', type=int, default=DEFAULT_DEPTH_INTERVAL)
parser.add_argument('--goal', default="the bathroom")
args   = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cmap   = matplotlib.colormaps.get_cmap(DEPTH_COLORMAP)
LUT = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
print("loading models...")
yolo = YOLO(YOLO_MODEL_PATH)
yolo26 = YOLO(YOLOV26_MODEL_PATH)
yoloe = YOLOE(YOLOE_MODEL_PATH)
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

speaker = Speaker()



llm = LLMVisionAssistant(provider="gemini", prompt=GOAL_PROMPT).start()   # key from GEMINI_API_KEY

llm_processor = LLMProcessor(llm) #Initialize LLM processor

## --- Action Handler ---
act = Action()


#nav = GoalNav(llm, speaker, sens=STEER_SENSITIVITY_DOOR, blocked_thresh=BLOCKED_THRESHOLD)
#voice = VoiceGoal(nav.set_goal, speaker)
#if args.goal:
    #nav.set_goal(args.goal)

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
        print("Camo Studio not detected, trying default camera... (" + str(source - 1) + ")")
        cap = cv2.VideoCapture(source - 1)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()
    t_angle = 0  # Initialize target angle for navigation
    frame_num   = 0
    depth_color = None
    boxes       = []
    boxes26     = []
    traj_canvas = None
    trajectory_raw = []
    trajectory = None
    old_llm_response = None #Stores the last llm_response to be compared
    # FIX: initialize direction so the arrow doesn't crash before depth runs
    direction = 0.0
    # Bound here so the depth block is the only place that assigns them, and
    # consumers downstream can test for None instead of risking NameError.
    depth_uint8 = None
    col = None

    # Initialize smoothed depth for visualization (not used in steering)
    smoothed_depth = None
    ALPHA = 0.7  # how much to trust the new frame vs history

    # Hysteresis state
    committed_steer  = "^ FORWARD"   # the direction currently being acted on
    vote_buffer      = collections.deque(maxlen=HYSTERESIS_FRAMES)  # buffer of recent steer decisions for hysteresis voting

    # FIX: track why the loop exited so the return value is accurate
    exit_reason = "user_quit"
    latest_traj = None

    """
    Main loop
    For each frame, reads a camera frame, runs depth estimation every
    depth_interval frames to compute steering direction, runs YOLO object
    detection every yolo_interval frames to find doors, then draws bounding
    boxes and the committed steering label onto the frame before displaying
    both the raw camera feed and the depth colormap side by side.
    Pressing 'q' exits the loop and releases the camera.
    """
    heading_degrees = 0
    vo_loop_closure_detected = False
    combined_direction = 0

    start_llm = False 
    time.sleep(5) # give time for vo to start up and send the first trajectory
    #yoloe.set_classes(["door", "doorway", "open floor"])
    while True:
        ret, frame = cap.read()
        obj_box_context = None #Establish this for use in the yolo section



        if not ret:  # end of video file or camera error
            print("Error: Could not read frame from video source.")
            exit_reason = "stream_ended"
            break

        # Resize the depth frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        h, w  = frame.shape[:2]
        save_frame = frame.copy() #Copies it for the original frame
        md = dict(shape=frame.shape, dtype=str(frame.dtype))
        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if encoded:
            socks = dict(poller_send.poll(BUFFER_TIME))
            if sender in socks and socks[sender] == zmq.POLLOUT:
                # Socket is ready; send without blocking
                sender.send(buffer.tobytes())
        #Check for feedback
        socks = dict(poller_recv.poll(timeout=0.1))
        if socks.get(receiver): #Probably working
            feedback_msg = receiver.recv()

            data = json.loads(feedback_msg.decode('utf-8'))

            heading_degrees = data['vo'][0]
            print(heading_degrees)
            loop_closure_detected = data['vo'][1]

            new_traj_point = data['trajectory'][-1:]  # Get the last point in the trajectory
            #print("New trajectory point received:", new_traj_point)
            new_point = (float(np.array(new_traj_point[0])[0,0]), float(np.array(new_traj_point[0])[2,0]))  # Convert to (x, z) tuple
            old_point = (float(np.array(trajectory_raw[-1])[0,0]), float(np.array(trajectory_raw[-1])[2,0])) if trajectory_raw else (0,0)
            if len(trajectory_raw) < 10 or math.dist(new_point, old_point) > 3:  # Only add if it's the first point or sufficiently far from the last
                trajectory_raw.append(new_traj_point[0])  # Add the new point to the trajectory

            #Data from vo processing
            trajectory = [(float(p[0, 0]), float(p[2, 0])) for p in np.array(trajectory_raw, dtype=np.float32)] #Get the data, convert it, then store as a list of (x, z) tuples
            #print(trajectory[-1)


            t_angle = traj_helper.find_best_traj_angle(trajectory, heading_degrees, prev_angle = t_angle) #Target angle to move
            #print(t_angle)
            traj_canvas = draw_trajectory_canvas(np.array(trajectory_raw, dtype=np.float32), heading_degrees, t_angle)

        # --- Depth estimation ---
        if frame_num % args.depth_interval == 0:
            # Disable Depth Anything inference

            depth_map = get_tflite_depth(frame)  # run TFLITE depth estimation
            depth_uint8 = depth_map.astype(np.uint8)

            direction = get_better_steer(depth_uint8)

            # Index into LUT for fast colormap application
            depth_color = LUT[depth_uint8]

            # Resize depth frame for side-by-side display
            depth_color = cv2.resize(depth_color, (w, h))

        
        labels_door = []
        #only run yolo e in main loop if action not in progress
        if act.get_action_state() != "in_progress":
            # --- Door detection ---
            if frame_num % args.yolo_door_interval == 0:
                yoloe.set_classes(["door", "doorway"])
                results = yoloe.predict(frame, verbose=False)
                boxes   = results[0].boxes
                obj_box_context_door = boxes.xyxy.cpu().round().int().tolist()

            # --- Draw Frame ---
            max_conf = 0
            index = -1
            #Pick the best door (highest confidence) and use that to steer toward the door
            for i, box in enumerate(boxes):
                label           = yoloe.names[int(box.cls[0])]
                labels_door.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
                color           = BOX_COLOR_DOOR
                conf = box.conf.item()
                if conf > max_conf and conf > DOOR_CONFIDENCE_THRESHOLD and 'door' in label:
                    max_conf = conf
                    index = i
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw box
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # draw label



        #disable yolo26 for now
        """
        # -- Object Detection using yolo26 for desk and chair avoidance. --
        labels_object = []
        if frame_num % args.yolo_default_interval == 0:
            results26 = yolo26(frame, verbose=False)
            boxes26   = results26[0].boxes
            obj_box_context_26 = boxes26.xyxy.cpu().round().int().tolist() #First, turn all the elements to integers, then convert to list, as it is originally a tensor; therefore, it is easier to work with in my opinion.
        # --- Draw Frame ---
        for i, box in enumerate(boxes26):
            label           = yolo26.names[int(box.cls[0])]
            labels_object.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            color           = BOX_COLOR_OTHER
            conf26 = box.conf.item()
            if conf26 > OTHER_CONFIDENCE_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw box
                cv2.putText(frame, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # draw label

        """

        # get steer with obect detection
        #direction = get_steer_from_objects(boxes26, depth_uint8, direction)
        labels_door = labels_door
        obj_box_context_door = obj_box_context_door

        dummy_answer = {'action': 'straight', 'search': ['door', 'hallway'], 'reasoning': ''}
        if act.get_action_state() != "in_progress" and start_llm:
            #LLM Handler
            #Should only be able to handle objects
            if not llm.is_busy() and start_llm:
                resized_frame, obj_box_context = llm_processor.preprocess(save_frame, obj_box_context_door) #Rescale the frame for the LLM
                llm_context = llm_processor.make_context(args.goal, resized_frame, obj_box_context,labels_door) #Formats the context for the LLM so that the LLM can add it to the end of the prompt
                accepted = llm_processor.process_frame(resized_frame, context=llm_context) #Send the frame to the LLM, if the LLM can query, it will
            while llm.is_busy():# wait for the LLM to finish processing to get result
                audio_state["left_vol"] = 0.0
                audio_state["right_vol"] = 0.0
            time.sleep(0.1) #Sleep for a bit 
            #LLM Reciever
            llm_response = llm_processor.get_result() #Gets the latest LLM result, not necessarily new from the last frame. Format is described in the prompt
            if llm_response is not None:
                if llm_response != old_llm_response: #Check if it is the same response as before
                    old_llm_response = llm_response
                    print(obj_box_context_door)
                    act.update_action(llm_response, heading_degrees, frame, llm_processor.max_size, llm_processor.xyxy_rescale, yoloe) #Updates the action state based on the LLM response
                else:
                    print("LLM response is the same as the previous one.")
                
        if act.get_action_state() == "in_progress":
            modified_frame, combined_direction = act.get_update_dir(frame, heading_degrees, trajectory)
            frame = modified_frame

        #smoothen direction with moving average
        direction_history[:-1] = direction_history[1:]
        direction_history[-1] = direction if combined_direction == 0 else combined_direction
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

        end_point = (int(math.sin(math.radians(combined_direction)) * 100 + w//2), int(-math.cos(math.radians(combined_direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # draws blue arrow for FINAL direction

        #nav.draw(frame)
    
        # display camera frame and depth side by side
        out = np.hstack([frame, depth_color,save_frame]) if depth_color is not None else frame
        cv2.imshow('navigator', out)
        if traj_canvas is not None:
            cv2.imshow('trajectory', traj_canvas)



        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            voice.listen_async()     # push-to-talk; saying "stop" cancels the goal
        elif key == ord('s'):
            start_llm = True
            act.action_state = "done" #Reset the action state so that the LLM can start processing again

        # Increment frame counter
        frame_num += 1

    # Cleanup
    llm.stop()
    speaker.stop()
    cap.release()
    if stream is not None:
        stream.close()
    cv2.destroyAllWindows()


    result.terminate()  # terminate the ZMQ context
    result.wait()
    print("ZMQ process terminated.")
    sender.setsockopt(zmq.LINGER, 0)
    sender.close()
    receiver.close()

    return {"frames_processed": frame_num, "exit_reason": exit_reason}

def _open_audio_stream():
    """Start the non-blocking cue stream, or return None if audio is unavailable.

    Sound is a comfort cue here; vision and speech still work without it, so an
    audio failure must never be fatal.
    """
    try:
        import sounddevice as sd
        return sd.OutputStream(channels=2, callback=audio_callback, samplerate=sample_rate)
    except Exception as e:
        print(f"[audio] cue stream unavailable, continuing without it: {e}")
        return None


def draw_trajectory_canvas(trajectory, current_angle, best_angle, distance = 30, canvas_size=600, world_scale=1.0):
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

    for i in range(1, len(pts), 2):  # draw lines between every other point for clarity
        cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 0), 2)

    if pts:
        cv2.circle(canvas, pts[-1], 4, (255, 255, 0), -1)  # current position
    
    # Draw the target angle line from the current position
    if pts:
        current_pos = pts[-1]
        target_x = int(current_pos[0] + distance * np.cos(np.radians(best_angle)))
        target_y = int(current_pos[1] + distance * np.sin(np.radians(best_angle)))
        cv2.line(canvas, current_pos, (target_x, target_y), (0, 0, 255), 2) # best angle line

        target_x = int(current_pos[0] + distance * np.cos(np.radians(current_angle - 90)))
        target_y = int(current_pos[1] + distance * np.sin(np.radians(current_angle - 90)))
        cv2.line(canvas, current_pos, (target_x, target_y), (0, 255, 0), 2)  # current angle line

    cv2.putText(canvas, "Top-down trajectory (X-Z): NAV.py", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas




# calling the function
if __name__ == "__main__":
    #starts callback audio stream for non-blocking sound playback
    stream = _open_audio_stream()
    if stream is not None:
        with stream:
            navigate()
    else:
        navigate()

