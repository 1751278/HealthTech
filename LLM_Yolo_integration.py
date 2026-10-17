import collections
import math
import queue
import threading
import time
 
import cv2
import numpy as np

#Simplistic (for now) LLM processor for integrating with YOLO detection, simply runs the llm on a frame

class LLMProcessor:
    def __init__(self, llm, max_size = 512):
        self.llm = llm
        self.last_call_time = time.time()
        self.min_seconds_between_calls = 5  # Adjust this value as needed
        self.max_size = max_size #Max size for the longer dimension
        
    def start(self):
        self.llm.start()

    def stop(self):
        self.llm.stop()

    def preprocess(self, frame, obj_box_context): #Rescale equally
        # Perform any necessary preprocessing on the frame and context
        h, w = frame.shape[:2]
        scale = self.max_size / max(h, w) # Determine scaling factor
        resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        resized_bboxes = []
        if obj_box_context:
            #rescale the object bounding boxes
            for bbox in obj_box_context:
                resized_bboxes.append(self.xyxy_rescale(bbox[0], bbox[1], bbox[2], bbox[3], scale))
        return resized_frame, resized_bboxes

    def get_result(self): #Return the latest LLM result as a dictionary.
        return self.llm.latest_description()

    def xyxy_rescale(self, x1, y1, x2, y2, scale):
        return [int(coord * scale) for coord in [x1, y1, x2, y2]]
    
    def make_context(self, Goal, obj_box_context = None): #Returns the context string to be used later
        return f"Goal: {Goal}\nDetected object bounding boxes: {obj_box_context} (choose [x1, y1, x2, y2] bbox inside list)"
    
    def evaluate(self, frame, context=""): #Determine if a frame should be processed, more complicated functionality should come later, for now it is just time based.
        if time.time() - self.last_call_time < self.min_seconds_between_calls:
            return False
        self.last_call_time = time.time()
        return True

    def process_frame(self, frame, context=""): #Returns if a frame was accepted, also sends it to be queried
        if self.evaluate(frame, context):
            accepted = self.llm.force_submit(frame, context=context)
            return accepted
        return False

    def get_result(self): #Return the latest LLM result as a dictionary.
        return self.llm.latest_description()