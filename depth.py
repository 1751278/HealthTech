import argparse
import cv2
import matplotlib
import numpy as np
import torch
import sys
sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Live Camera')
    
    # Changed to camera index (usually 0)
    parser.add_argument('--camera-id', type=int, default=0, help='ID of the camera/webcam device')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'depthmodels/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Open Camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        exit()

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("Starting stream... Press 'q' to quit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Inference
        depth = depth_anything.infer_image(raw_frame, args.input_size)

        # Normalize depth for visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Prepare output frame
        if args.pred_only:
            combined_frame = depth
        else:
            split_region = np.ones((raw_frame.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])

        # Display results
        cv2.imshow('Depth Anything V2 - Camera Feed', combined_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
