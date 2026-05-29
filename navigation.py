# navigation.py reccomendations 
#################
# Start of navigation.py. Incomplete though. 
# Use the doorFrameModel
# NCNN TFlight model for depth estimation, much faster than DPT. (More efficent then currently)
# make return a value
# ADD FUNCTIONS
# turning if screen is fully red
# increase resolution
# start cutting down frames for other files (OCR)
# documentation of each part. (Similar to Gurveer's vo_mapper.py)
# text to speech
#################
import argparse

import cv2
import torch
import numpy as np
import matplotlib
import sys
 
sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
 
print("loading models...")
yolo = YOLO("YoloModels/yolov11n.pt")
 
depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load('depthmodels/depth_anything_v2_vits.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
parser = argparse.ArgumentParser(description='Visual Odometry 2D Mapper')

# source can be a number like 0 or 1 for webcam, or a url for phone camera
parser.add_argument('--source', default='1',
                    help='Camera index (e.g. 0) or stream URL (e.g. http://192.168.x.x:8080/video)')
parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--yolo-model', default='YoloModels/yolov11n.pt')

# dont run yolo every single frame, too slow. run it every N frames instead
parser.add_argument('--yolo-interval', type=int, default=2,
                    help='Run YOLO every N frames')

# depth is even heavier than yolo so run it less often
parser.add_argument('--depth-interval', type=int, default=5,
                    help='Run depth model every N frames')

args = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source


#Primary function
#return tyes: Direction "Left", "RIght", "Forward", "nil"
def navigate():
    cap = cv2.VideoCapture(source) # Note from Sahir. If this line says "source-1," change it to "source". I had to change it to "source-1" to get it working on my computer.
    if not cap.isOpened():
        print("webcam not found, try VideoCapture(0)")
        exit()
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        frame = cv2.resize(frame, (360, 640))
        h, w = frame.shape[:2]
    
        # depth every 3 frames
        if frame_num % 3 == 0:
            
            raw = depth_model.infer_image(frame, 128) # second paramater repersents the input resolution. 
            depth_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    
            # check left center right strips for steering
            strip = depth_norm[int(0.4*depth_norm.shape[0]):int(0.6*depth_norm.shape[0]), :]
            dw = strip.shape[1]
            left_avg   = strip[:, :dw//3].mean()
            center_avg = strip[:, dw//3:2*dw//3].mean()
            right_avg  = strip[:, 2*dw//3:].mean()
    
            if center_avg < 0.35:
                steer = "GO RIGHT >>" if right_avg < left_avg else "<< GO LEFT"
            else:
                steer = "^ FORWARD"
    
            depth_color = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth_color = cv2.resize(depth_color, (w, h))
    
        # yolo every 2 frames
        if frame_num % 2 == 0:
            results = yolo(frame, verbose=False)
            boxes = results[0].boxes
    
        annotated = frame.copy()
        for box in boxes:
            label = yolo.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 220, 220) if 'door' in label else (0, 60, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
        cv2.putText(annotated, steer, (w//2 - 70, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
    
        combined = np.hstack([annotated, depth_color])
        cv2.imshow('navigator', combined)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame_num += 1
    
    cap.release()
    cv2.destroyAllWindows()
navigate()