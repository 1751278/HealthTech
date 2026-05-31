#################
# vo_mapper.py
# Created by Gurveer Minhas April 27 2026
# Last Updated: ?
# Description: This is the visual odometry mapper module for HealthTech. It uses YOLO for object detection and Depth Anything V2 for depth estimation.
# TODO:
# - ?
# - ?
# - ?
#################
import argparse
import math
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# depth anything v2 is in a subfolder so we gotta tell python where to look
sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# how big the map window is in pixels
MAP_SIZE = 800

# how zoomed in the map starts, bigger = more zoomed in
TRAVEL_SCALE = 80

# controls how far away objects appear on the map, tweak this if stuff looks too close or far
DEPTH_DISPLAY_SCALE = 2.0

# converts pixel movement from the camera into map movement, smaller = less sensitive
DISPLACEMENT_SCALE = 0.005

# if we drop below this many tracked points we find new ones
MIN_FEATURES = 50

# max points we track at once
MAX_FEATURES = 200

# assumed camera field of view, used to figure out if something is to the left or right
CAMERA_FOV_DEG = 60

# if the path gets this close to the edge of the map, zoom out
AUTO_SCALE_MARGIN = 60

# colors for specific objects on the map (BGR format not RGB)
CATEGORY_COLORS = {
    'person':       (0,   0, 220),  # red
    'chair':        (220, 80,  0),  # blue
    'couch':        (220, 80,  0),  # blue
    'sofa':         (220, 80,  0),  # blue
    'bed':          (180, 60,  0),  # also blue-ish
    'dining table': (200,100,  0),  # blue-ish
    'table':        (200,100,  0),  # blue-ish
}

# anything not in the list above gets this color (yellow)
DEFAULT_COLOR = (0, 200, 255)

# settings for the optical flow tracker (how it follows points frame to frame)
LK_PARAMS = dict(
    winSize=(21, 21),   # how big of an area it looks at around each point
    maxLevel=3,         # how many image scales to use (more = handles fast movement better)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),  # when to stop iterating
)

# settings for finding good points to track in the first place
FEATURE_PARAMS = dict(
    maxCorners=MAX_FEATURES,  # max points to find
    qualityLevel=0.3,         # only keep pretty good corners (0-1, higher = stricter)
    minDistance=7,            # dont pick points too close together
    blockSize=7,              # size of area used to check if its a good corner
)

# depth anything v2 comes in different sizes, bigger = more accurate but slower
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536, 1536]},
}


def world_to_canvas(wx, wy, origin, scale):
    # converts a real world position (x, y) into a pixel position on the map canvas
    # origin is the center of the map, y is flipped bc screen y goes down but map y goes up
    return (
        int(origin[0] + wx * scale),
        int(origin[1] - wy * scale),
    )


def draw_map(path, objects, pose_x, pose_y, origin, scale):
    # make a dark grey blank canvas to draw everything on
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), 35, dtype=np.uint8)

    # draw a subtle grid so the map doesnt look completely empty
    for i in range(0, MAP_SIZE, 80):
        cv2.line(canvas, (i, 0), (i, MAP_SIZE), (50, 50, 50), 1)
        cv2.line(canvas, (0, i), (MAP_SIZE, i), (50, 50, 50), 1)

    # small dot to mark where u started
    cv2.circle(canvas, origin, 5, (80, 80, 80), -1)

    # draw the path as a trail of connected dots
    if len(path) > 1:
        pts = [world_to_canvas(x, y, origin, scale) for x, y in path]

        # connect each point to the next with a line
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (100, 100, 100), 1)

        # draw a small dot every 5 points so it doesnt look like just lines
        for pt in pts[::5]:
            if 0 <= pt[0] < MAP_SIZE and 0 <= pt[1] < MAP_SIZE:
                cv2.circle(canvas, pt, 2, (160, 160, 160), -1)

    # draw each detected object as a colored dot with a label
    for ox, oy, label, color in objects:
        cp = world_to_canvas(ox, oy, origin, scale)
        # only draw if its actually on screen
        if 0 <= cp[0] < MAP_SIZE and 0 <= cp[1] < MAP_SIZE:
            pass
            #cv2.circle(canvas, cp, 6, color, -1)
            #cv2.circle(canvas, cp, 6, (255, 255, 255), 1)  # white outline so its visible on dark bg
            #cv2.putText(canvas, label, (cp[0] + 9, cp[1] + 4),
                   #     cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # draw ur current position as a green dot (bigger than the trail dots)
    curr = world_to_canvas(pose_x, pose_y, origin, scale)
    cv2.circle(canvas, curr, 9, (0, 220, 0), -1)
    cv2.circle(canvas, curr, 9, (255, 255, 255), 2)  # white ring around it so its easy to spot

    return canvas


def main():
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

    # pick the best available hardware: nvidia gpu, apple silicon, or just cpu
    DEVICE = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')
    print(f"Device: {DEVICE}")

    print("Loading YOLO...")
    yolo = YOLO(args.yolo_model)

    print("Loading Depth Anything V2...")
    depth_model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    depth_model.load_state_dict(
        torch.load(f'depthmodels/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
    )
    depth_model = depth_model.to(DEVICE).eval()

    # if source is a number use it as a camera index, otherwise treat it as a url
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open source '{args.source}'")
        return

    # grab the first frame so we can set up the tracker before the main loop
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read first frame")
        cap.release()
        return

    h, w = frame.shape[:2]

    # convert first frame to grayscale, optical flow works on grayscale
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find good points to track in the first frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **FEATURE_PARAMS)

    # where we are on the map, starts at 0,0 (center)
    pose_x, pose_y = 0.0, 0.0

    # Direction the camera is facing in radians, starts facing "up" the map (towards negative y)
    heading = 0.0 
    
    # list of all positions we've been at, used to draw the trail
    path = [(0.0, 0.0)]

    # all objects we've spotted so far, each one is (world_x, world_y, label, color)
    objects = []

    # these get filled in during the loop, start as none so we know they arent ready yet
    depth_map = None
    detections = []

    frame_count = 0

    # map origin is the center of the canvas
    origin = (MAP_SIZE // 2, MAP_SIZE // 2)
    scale = TRAVEL_SCALE

    # blank canvas just so the variable exists before the loop
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), 35, dtype=np.uint8)

    print("Running — press 'q' to quit and save map.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optical flow needs grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # visual odometry: figure out how much the camera moved this frame
        if prev_pts is not None and len(prev_pts) >= MIN_FEATURES:

            # track where each point moved to in the new frame
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **LK_PARAMS
            )

            # status==1 means the point was tracked successfully
            good_prev = prev_pts[status.ravel() == 1]
            good_curr = curr_pts[status.ravel() == 1]

            if len(good_curr) > 0:
                M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
                if M is not None:
                    dx = M[0, 2]
                    dy = M[1, 2]
                    heading += np.arctan2(M[1, 0], M[0, 0])  # track rotation too
                    pose_x += dx * DISPLACEMENT_SCALE
                    pose_y -= dy * DISPLACEMENT_SCALE
            # keep the successfully tracked points for next frame
            prev_pts = (good_curr.reshape(-1, 1, 2)
                        if len(good_curr) >= MIN_FEATURES else None)
        else:
            prev_pts = None

        # if we lost too many points, find new ones to track
        if prev_pts is None or len(prev_pts) < MIN_FEATURES:
            prev_pts = cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)

        # save this frame as "previous" for next iteration
        prev_gray = gray.copy()

        # record where we are now
        path.append((pose_x, pose_y))

        # run yolo every N frames to detect objects
        if frame_count % args.yolo_interval == 0:
            results = yolo(frame, verbose=False)
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = yolo.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # store the center of the bounding box too, we need it for depth lookup
                detections.append((label, (x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2))

        # run depth model every N frames, slower than yolo so we do it less
        if frame_count % args.depth_interval == 0:
            raw = depth_model.infer_image(frame, 256)
            # normalize to 0-1 so its easier to work with
            dmin, dmax = raw.min(), raw.max()
            depth_map = (raw - dmin) / (dmax - dmin + 1e-6)  # +tiny number to avoid divide by zero

        # place detected objects onto the map using their depth
        if depth_map is not None and frame_count % args.yolo_interval == 0:
            dh, dw = depth_map.shape
            for label, cx_box, cy_box, x1, y1, x2, y2 in detections:

                # depth map might be a diff resolution than the camera frame, so scale the coords
                sx = int(np.clip(cx_box * dw / w, 0, dw - 1))
                sy = int(np.clip(cy_box * dh / h, 0, dh - 1))

                # how far away the object is (rough estimate, not real meters)
                d = float(depth_map[sy, sx]) * DEPTH_DISPLAY_SCALE + 0.3

                # figure out left/right angle based on where in the frame the object is
                horiz_angle = ((cx_box - w / 2) / w) * math.radians(CAMERA_FOV_DEG)

                # project object position onto the map relative to where we are
                obj_x = pose_x + d * math.sin(horiz_angle)
                obj_y = pose_y + d * math.cos(horiz_angle)

                color = CATEGORY_COLORS.get(label, DEFAULT_COLOR)
                objects.append((obj_x, obj_y, label, color))

        # auto zoom out if the path is getting close to the edge of the map
        for wx, wy in path[-50:] + [(o[0], o[1]) for o in objects[-20:]]:
            cpx, cpy = world_to_canvas(wx, wy, origin, scale)
            if (cpx < AUTO_SCALE_MARGIN or cpx > MAP_SIZE - AUTO_SCALE_MARGIN
                    or cpy < AUTO_SCALE_MARGIN or cpy > MAP_SIZE - AUTO_SCALE_MARGIN):
                scale = max(8, scale * 0.92)  # shrink scale a bit, stop at 8 so it doesnt get too tiny
                break

        # redraw the map with everything on it
        canvas = draw_map(path, objects, pose_x, pose_y, origin, scale)

        # draw yolo boxes on the camera feed
        annotated = frame.copy()
        for label, cx_box, cy_box, x1, y1, x2, y2 in detections:
            color = CATEGORY_COLORS.get(label, DEFAULT_COLOR)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # show both windows
        cv2.imshow('Camera Feed', annotated)
        cv2.imshow('Live Map', canvas)

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # save the final map when done
    cv2.imwrite('map_output.png', canvas)
    print("Map saved to map_output.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
