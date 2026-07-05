import cv2
import ncnn
import numpy as np

def run_ncnn_inference(image_path, param_path, bin_path, output_path="depth_result.jpg"):
    # 1. Load the NCNN network
    net = ncnn.Net()
    
    # Enable multi-threading for faster CPU performance
    net.opt.num_threads = 4  
    
    net.load_param(param_path)
    net.load_model(bin_path)

    # 2. Read and prepare the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    orig_h, orig_w, _ = img.shape
    target_size = 518  # Must match the static size chosen during ONNX export

    # Depth Anything V2 standard ImageNet normalization
    # Note: NCNN uses a known typo in its API mapping: 'substract' instead of 'subtract'
    mean_vals = [123.675, 116.28, 103.53]
    norm_vals = [1.0 / 58.395, 1.0 / 57.12, 1.0 / 57.375]

    # Convert BGR to RGB, resize, and pack into an NCNN Mat structure
    mat_in = ncnn.Mat.from_pixels_resize(
        img, 
        ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
        orig_w, orig_h, 
        target_size, target_size
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
    depth_map = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Normalize depth map values to a visualizable 0-255 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_img = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_img = np.zeros_like(depth_map, dtype=np.uint8)

    # Apply an absolute colormap for depth visualization (e.g., INFERNO or PLASMA)
    depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

    # 5. Save the final rendering
    cv2.imshow('Display Window',depth_colormap)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_ncnn_inference(
        image_path="TestImage/name.jpg",
        param_path="depthAnythingModelFaster/depth_anything_v2_vits_sim.ncnn.param",
        bin_path="depthAnythingModelFaster/depth_anything_v2_vits_sim.ncnn.bin"
    )