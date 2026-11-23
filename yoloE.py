#################
# vo_mapper.py
# Created by Sahir Abrar 27 2026
# Last Updated: ?
# Description: Updated yolo model to YOLO-E for open-vocabulary detection of doors and doorways.
# TODO:
# - intergrate with navigation.py
# - ?
# - ?
#################

from typing import Optional, Sequence

import numpy as np
from ultralytics import YOLOE as UltralyticsYOLOE


# args: model_path, classes, device, confidence
# description: YOLO-E wrapper for open-vocabulary detection of doors and doorways. Can be used in navigation.py or other modules.
class YOLOE:
    """Reusable wrapper around an Ultralytics YOLO-E segmentation model."""

    def __init__(
        self,
        model_path: str = "YoloModels/yoloe-26s-seg.pt", # default model path for YOLO-E. If broken, check if it exists.
        classes: Optional[Sequence[str]] = None, # default classes for open-vocabulary detection. If None, defaults to ["door", "doorway"].
        device: Optional[str] = None, 
        confidence: float = 0.25, # adjust based on how relient you want to be on yolo-e's predictions.
    ) -> None:
        self.model_path = model_path # 
        self.device = device
        self.confidence = confidence
        self.classes = list(classes) if classes is not None else ["door", "doorway"] # ADJUST HERE FOR DEFAULT CLASSES
        self.model = UltralyticsYOLOE(model_path)
        self.set_classes(self.classes)
    # args: classes
    # description: Set the text prompts used for open-vocabulary detection. Raises ValueError if classes is empty.
    def set_classes(self, classes: Sequence[str]) -> None:
        """Set the text prompts used for open-vocabulary detection."""
        if not classes:
            raise ValueError("classes must contain at least one text prompt")

        self.classes = list(classes)
        self.model.set_classes(self.classes)
    # args: frame, confidence, verbose
    # description: Detect objects in one image frame and return normalized detection dicts. Raises Value
    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Detect objects in one image frame and return normalized detection dicts."""
        if frame is None:
            raise ValueError("frame cannot be None")

        conf = self.confidence if confidence is None else confidence
        predict_kwargs = {
            "conf": conf,
            "verbose": verbose,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(frame, **predict_kwargs)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks = results[0].masks
        detections = []
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            score = float(box.conf[0].item())
            class_name = (
                str(results[0].names.get(class_id, class_id))
                if isinstance(results[0].names, dict)
                else str(results[0].names[class_id])
            )

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
            }

            if masks is not None:
                detection["mask"] = masks.data[index].detach().cpu().numpy()

            detections.append(detection)

        return detections

    # args: frame, confidence, verbose
    # description: Run prediction and return the raw Ultralytics result list.
    def predict(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        verbose: bool = False,
    ):
        """Run prediction and return the raw Ultralytics result list."""
        predict_kwargs = {
            "conf": self.confidence if confidence is None else confidence,
            "verbose": verbose,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        return self.model.predict(frame, **predict_kwargs)