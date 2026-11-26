#################
# yoloE.py
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


# args: model_path, classes, device, confidence, tracker
# description: YOLO-E wrapper for open-vocabulary detection of doors and doorways. Can be used in navigation.py or other modules.
class YOLOE:
    """Reusable wrapper around an Ultralytics YOLO-E segmentation model."""

    def __init__(
        self,
        model_path: str = "YoloModels/yoloe-26s-seg.pt", # default model path for YOLO-E. If broken, check if it exists.
        classes: Optional[Sequence[str]] = None, # default classes for open-vocabulary detection. If None, defaults to ["door", "doorway"].
        device: Optional[str] = None,
        confidence: float = 0.25, # adjust based on how relient you want to be on yolo-e's predictions.
        tracker: str = "bytetrack.yaml", # tracker config used by track(). Also supports "botsort.yaml".
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.tracker = tracker
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

    # args: confidence, verbose
    # description: Build the kwargs shared by predict() and track().
    def _kwargs(self, confidence: Optional[float], verbose: bool) -> dict:
        kwargs = {
            "conf": self.confidence if confidence is None else confidence,
            "verbose": verbose,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    # args: result, with_ids
    # description: Convert one Ultralytics result into a list of normalized detection dicts.
    def _parse_result(self, result, with_ids: bool = False) -> list[dict]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks = result.masks

        # boxes.id is None when the tracker has not confirmed any track yet
        # (e.g. first frames). Missing IDs become None instead of crashing.
        ids = None
        if with_ids and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()

        detections = []
        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            score = float(box.conf[0].item())
            class_name = (
                str(result.names.get(class_id, class_id))
                if isinstance(result.names, dict)
                else str(result.names[class_id])
            )

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
            }

            if with_ids:
                detection["track_id"] = ids[index] if ids is not None else None

            if masks is not None:
                detection["mask"] = masks.data[index].detach().cpu().numpy()

            detections.append(detection)

        return detections

    # args: frame, confidence, verbose
    # description: Detect objects in one image frame and return normalized detection dicts. Raises ValueError if frame is None.
    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Detect objects in one image frame and return normalized detection dicts."""
        if frame is None:
            raise ValueError("frame cannot be None")

        results = self.model.predict(frame, **self._kwargs(confidence, verbose))
        if not results:
            return []

        return self._parse_result(results[0])

    # args: frame, confidence, verbose, tracker, persist
    # description: Like detect(), but runs tracking so each detection also has a "track_id" (None if not yet assigned).
    #              Call once per frame, in order, on the same instance so IDs stay consistent. Raises ValueError if frame is None.
    def track(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        verbose: bool = False,
        tracker: Optional[str] = None,
        persist: bool = True,
    ) -> list[dict]:
        """Detect and track objects in one frame; detections include "track_id"."""
        if frame is None:
            raise ValueError("frame cannot be None")

        kwargs = self._kwargs(confidence, verbose)
        kwargs["tracker"] = self.tracker if tracker is None else tracker
        kwargs["persist"] = persist  # keep tracker state between calls -> stable IDs

        results = self.model.track(frame, **kwargs)
        if not results:
            return []

        return self._parse_result(results[0], with_ids=True)

    # description: Clear tracker state so new IDs start fresh (e.g. between videos or scenes).
    def reset_tracking(self) -> None:
        """Reset tracker state, if a tracker has been started."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    # args: frame, confidence, verbose
    # description: Run prediction and return the raw Ultralytics result list.
    def predict(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        verbose: bool = False,
    ):
        """Run prediction and return the raw Ultralytics result list."""
        return self.model.predict(frame, **self._kwargs(confidence, verbose))
