"""
To convert to NCNN must use the following steps:
1. Export the PyTorch model to ONNX format using the provided script (convertDepthAnythingv2.py).
2. run this to simplify the ONNX model:onnxsim depth_anything_v2_vits.onnx depth_anything_v2_vits_sim.onnx
3. run this to convert to NCNN format(Must install pnnx first by going into default terminal and run: pip3 install pnnx): 
pnnx depth_anything_v2_vits_sim.onnx inputshape=[1,3,266,266] fp16=1 optlevel=2
4. transfer all the output files to the depthAnythingModelFaster folder and run runNCNNDepth.py or runNCNNDepthLive.py
"""
"""
Converting onnx into .h5:
onnx2tf -i depthAnythingModelFaster\depth_anything_v2_vits_sim.onnx -o depthAnythingModelFaster --output_h5
"""

import torch
import sys
# Ensure you have cloned the Depth-Anything-V2 repo and it is in your path
sys.path.append('Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


DEPTH_MODEL_PATH   = "depthmodels/depth_anything_v2_vits.pth"
ONNX_PATH = "depthAnythingModelFaster\depth_anything_v2_vits_sim.onnx"
TFOUTPUT_DIR = "depthAnythingModelFaster\depth_anything_v2_vits_tf"

def convertDepthAnythingV2ToONNX():
    # 1. Define model config for ViT-S (Small)
    config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    model = DepthAnythingV2(**config)

    # 2. Load your weights
    model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location='cpu'))
    model.eval()

    # 3. Create dummy static input (Batch size=1, Channels=3, H=266, W=266)
    dummy_input = torch.randn(1, 3, 266, 266)#Dimensions must be multiples of 14

    # 4. Export to ONNX (Using opset 16 or 17 for robust Transformer support)
    torch.onnx.export(
        model,
        dummy_input,
        "depth_anything_v2_vits.onnx",
        input_names=["image"],
        output_names=["depth"],
        opset_version=17
    )
    print("ONNX export successful!")


if __name__ == "__main__":
    convertDepthAnythingV2ToONNX()