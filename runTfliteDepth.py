import cv2
import numpy as np
import tensorflow as tf

def run_tflite_inference(image_path, model_path, output_path="depth_tflite_result.jpg"):
    # 1. Load the TFLite model and allocate tensors
    # We specify number of threads to optimize CPU performance
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Read and preprocess the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    orig_h, orig_w, _ = img.shape
    target_size = 266  # Must match the static size used during export

    # Resize image to target dimensions and convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize using standard ImageNet values
    img_input = img_resized.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
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

    # Apply a high-contrast colormap for depth visualization (e.g., INFERNO)
    depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

    # 5. Save the final rendering
    cv2.imshow('Depth Map', depth_colormap)
    cv2.waitKey(0)

if __name__ == "__main__":
    # Target your standard FP32 or optimized INT8 model
    run_tflite_inference(
        image_path="TestImage/name.jpg",
        model_path="depthAnythingModelFaster\int8Tflite\depth_anything_v2_vits_sim_float32.tflite" 
    )