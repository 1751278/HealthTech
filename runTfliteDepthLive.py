import cv2
import tensorflow as tf
import numpy as np

TFLITE_PATH = "depthAnythingModelFaster/midasDepth.tflite"

CAMERA_SOURCE = 1
FRAME_WIDTH = 360
FRAME_HEIGHT = 640
MODEL_INPUT_SIZE = 256 # Must match the static size chosen during ONNX export


def main_loop():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=8)#match num cores with threads
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

        # 1. Get input tensor propertie
    print("--- Input Details ---")
    print(input_details)
    print("\n--- Output Details ---")
    print(output_details)

    # Depth Anything V2 standard ImageNet normalization
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
        orig_h, orig_w, _ = frame.shape
        # Resize image to target dimensions and convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

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