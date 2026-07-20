import cv2
import ncnn
import numpy as np

PARAM_PATH = "depthAnythingModelFaster/depth_anything_v2_vits_sim.ncnn.param"
BIN_PATH = "depthAnythingModelFaster/depth_anything_v2_vits_sim.ncnn.bin"

CAMERA_SOURCE = 1
FRAME_WIDTH = 360
FRAME_HEIGHT = 640
MODEL_INPUT_SIZE = 266 # Must match the static size chosen during ONNX export

# 1. Load the NCNN network
net = ncnn.Net()

# Enable multi-threading for faster CPU performance
net.opt.num_threads = 8  #match number of cores in your cpu

net.load_param(PARAM_PATH)
net.load_model(BIN_PATH)

def main_loop():
    # Depth Anything V2 standard ImageNet normalization
    # Note: NCNN uses a known typo in its API mapping: 'substract' instead of 'subtract'
    mean_vals = [123.675, 116.28, 103.53]
    norm_vals = [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375]
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Camo Studio not detected, trying default camera...")
        cap = cv2.VideoCapture(CAMERA_SOURCE - 1)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

    while True:
        ret, frame = cap.read()
        if not ret:  # end of video file or camera error
            exit_reason = "stream_ended"
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Convert BGR to RGB, resize, and pack into an NCNN Mat structure
        mat_in = ncnn.Mat.from_pixels_resize(
            frame, 
            ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
            FRAME_WIDTH, FRAME_HEIGHT, 
            MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )
        mat_in.substract_mean_normalize(mean_vals, norm_vals) 

        # 3. Execute inference
        ex = net.create_extractor()
        
        # Use the specific layer names specified during your ONNX export step
        ex.input("in0", mat_in)
        retval, mat_out = ex.extract("out0")
        
        if retval != 0:
            print("Error encountered during feature extraction.")
            return

        # 4. Post-process the output Matrix
        # Convert NCNN Mat layer into a standard NumPy array
        out_np = np.array(mat_out).squeeze()

        # Resize depth map back to match your original input dimensions
        depth_map = cv2.resize(out_np, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Normalize depth map values to a visualizable 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_img = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_img = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply an absolute colormap for depth visualization (e.g., INFERNO or PLASMA)
        depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
        
        # display camera frame and depth side by side
        out = np.hstack([frame, depth_colormap]) if depth_colormap is not None else frame
        cv2.imshow('depth', out)
 
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()