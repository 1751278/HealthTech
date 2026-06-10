#################
# navigation.py
# Created by Sahir Abrar May 28 2026
# Last Updated: June 8 2026 by Kenshi & Ethan
# Last Change:
# - Added a not annoying sound.
# Description: This module captures video from a camera, runs depth estimation and tells the user to navigate to the door.
# TODO:
# - Use NCNN TFlight model for depth estimation (faster/more efficient than current DPT)
# - Add text-to-speech output
# - Need to combine door path and avoidance path for guidance to the door
# - Change song please... Or maybe some way to allow user to change it themselves
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

sys.path.append('./Depth-Anything-V2')
import os

from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
# --- Models ---
YOLO_MODEL_PATH    = "YoloModels/doorFrameModel26.pt" # Using the door frame model so we can try and help navigate Users to the door.
DEPTH_MODEL_PATH   = "depthmodels/depth_anything_v2_vits.pth"
DEPTH_ENCODER      = 'vits'
DEPTH_FEATURES     = 64
DEPTH_OUT_CHANNELS = [48, 96, 192, 384]
 
# --- Capture ---
DEFAULT_SOURCE   = '1'    # Camera index or file path
FRAME_WIDTH      = 360
FRAME_HEIGHT     = 640
DEPTH_INFER_SIZE = 256    # Resolution passed to depth model inference
 
# --- Audio ---
print("loading audio... check the constants section to change the sound file. MB if it is bad. I just searched no copyright music")
AUDIO_DATA, SAMPLE_RATE = sf.read("SoundAssets/music.wav") #CHANGE THIS FOR DIFFERENT SOUND, I FOUND THIS ONLINE IM SORRY
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
DEFAULT_DEPTH_INTERVAL = 3
 
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


def get_door_steer(box, frame_width, yolo_names, thresh=0.2):
    # gets straight line steering to the door
    print(box.conf.item())
    if box.conf.item() > thresh:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = (int((x1+x2)/2), int((y1+y2)/2))
        direction = -(frame_width/2 - center[0])*STEER_SENSITIVITY_DOOR # positive if door is on the left, negative if on the right
        direction = max(-90, min(90, direction))  # clamp to [-90, 90]
        global last_door_direction
        last_door_direction = direction
        return direction
    if last_door_direction is not None:
        return last_door_direction
    return 90
def combine_steer(obstacle_dir, door_dir, depth_uint8):
     pass
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

    # FIX: initialize direction so the arrow doesn't crash before depth runs
    direction = 0.0
    #last door frame bbox
    global last_door_direction
    last_door_direction = 90 # default to right if we haven't seen a door yet

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

        ################################# Find contours
        #convert depth map to binary
        gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
        #crop_gray = gray[FRAME_HEIGHT//3 : 2*FRAME_HEIGHT//3, 0 : FRAME_WIDTH]

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 3. Find contours
        # Returns a list of contours and their structural hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        # 3. Calculate its numerical area value (if needed)
        max_area = cv2.contourArea(max_contour)
        avg_x = np.mean(max_contour[:, 0, 0])
        # 1. Create a black mask of the same size as your image
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        # 2. Draw the filled contour on the mask
        cv2.drawContours(mask, [max_contour], -1, 255, -1)

        # 3. Calculate the average BGR color using the mask
        avg_red = cv2.mean(frame, mask=mask)[2]

        print(max_area)
        #if max_area < OBJECT_AREA_THRESH_MAX and max_area > OBJECT_AREA_THRESH_MIN:
        if avg_x < FRAME_WIDTH//2+75 and avg_x > FRAME_WIDTH//2-75:
            pass
        elif avg_x < FRAME_WIDTH//2: 
            direction -= max_area * avg_red * OBJECT_STEER_SENSITIVITY
            direction = max(-90, min(90, direction))  # clamp to [-90, 90]

        else:
            direction += max_area * avg_red * OBJECT_STEER_SENSITIVITY
            direction = max(-90, min(90, direction))  # clamp to [-90, 90]
            

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        ##############################################
 
        # --- Object detection ---
        if frame_num % args.yolo_interval == 0:
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
        for i, box in enumerate(boxes):
            label           = yolo.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            color           = BOX_COLOR_DOOR if 'door' in label else BOX_COLOR_OTHER
            conf = box.conf.item()
            if conf > max_conf:
                max_conf = conf
                index = i
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw box
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # draw label
 
        #Get straghtline steering to the door if one is detected with high confidence
        if index != -1:
            door_direction = get_door_steer(boxes[index], w, yolo.names)
        else:
            door_direction = last_door_direction  # keep going toward the last known door direction if we lose sight of it
        
        start_point = (w//2, h//2)
        end_point = (int(math.sin(math.radians(direction)) * 100 + w//2), int(-math.cos(math.radians(direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)  # draw green arrow for avoidance direction
        start_point = (w//2, h//2)
        end_point = (int(math.sin(math.radians(door_direction)) * 100 + w//2), int(-math.cos(math.radians(door_direction)) * 100 + h//2))
        cv2.arrowedLine(frame, start_point, end_point, (0, 0, 255), 2)  # draws red arrow for door direction
 
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