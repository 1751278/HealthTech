import cv2
from ultralytics import YOLOE

# 1. Load a YOLOE checkpoint model (e.g., nano or small segment model)
# YOLOE checkpoints support open-vocabulary text or visual prompting
model = YOLOE("YoloModels/yoloe-26s-seg.pt")

# 2. Define the custom text prompts you want to detect in real-time
# You can change these words to look for literally anything!
target_classes = ["open floor", "stairs", "door", "doorway"]
model.set_classes(target_classes)

model.export(format="ncnn", half=True, imgsz=640)