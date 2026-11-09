import collections
import math
import queue
import threading
import time
 
import cv2
import numpy as np
import context_point
from context_point import contextPoint
#Simplistic (for now) LLM processor for integrating with YOLO detection, simply runs the llm on a frame

class LLMProcessor:
    def __init__(self, llm, max_size = 512, utilize_past_responses = True, max_context_points = 5):
        self.llm = llm
        self.last_call_time = time.time()
        self.min_seconds_between_calls = 5  # Adjust this value as needed
        self.max_size = max_size #Max size for the longer dimension
        self.last_response = None
        self.last_frame = None
        self.image_history = [] # Oldest images go in the front (0)
        self.last_context = ""
        self.context_points = [] #Stores context point objects (from the last [self.max_context_points] responses) that are then evaluated to create a wider context
        self.max_context_points = max_context_points
        self.utilize_past_responses = utilize_past_responses
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

    def xyxy_rescale(self, x1, y1, x2, y2, scale):
        return [int(coord * scale) for coord in [x1, y1, x2, y2]]
    
    def make_context(self, Goal, frame, obj_box_context = None,labels_context = None): #Returns the context string to be used later
        labeled_context = []
        for bbox, label in zip(obj_box_context, labels_context):
            labeled_context.append(self.convert_bbox_to_definition(frame, bbox, label))
        self.last_context = f"Goal: {Goal}\nDetected object bounding boxes: {labeled_context}"
        self.last_frame = frame
        context = self.last_context + " (choose [x1, y1, x2, y2] bbox inside list)"
        return context
    
    def evaluate(self, frame, context=""): #Determine if a frame should be processed, more complicated functionality should come later, for now it is just time based.
        if time.time() - self.last_call_time < self.min_seconds_between_calls:
            return False
        self.last_call_time = time.time()
        return True

    def process_frame(self, frame, context=""): #Returns if a frame was accepted, also sends it to be queried
        if self.evaluate(frame, context):
            accepted = self.llm.force_submit(frame, context=context, history=(self._make_history(), self.image_history))
            return accepted
        return False

    def get_result(self): #Return the latest LLM result as a dictionary.
        if self.last_response != self.llm.latest_description(): #If these two don't match, update the last response and store a new context object
            self.last_response = self.llm.latest_description()
            if self.utilize_past_responses:
                self._make_context_point()
                self.image_history.append(self.last_frame)
                if len(self.image_history) > self.max_context_points:
                    self.image_history.pop(0) #Remove oldest image if the list is greater than the maximum allowed number of images
        return self.last_response

    def _make_history(self): #Creates a list of past queries and decisions, serving as captions to previous images
        history = []
        for context_point in self.context_points:
            history.append(context_point.get_context())
        return history


    def _make_context_point(self):
        self.context_points.append(contextPoint(self.last_response, self.last_context, self.last_frame))
        if len(self.context_points) > self.max_context_points:
            self.context_points.pop(0) #Remove oldest context point if the list is greater than the maximum allowed number of context points

    def convert_bbox_to_definition(self, frame, bbox, label): #Turns a bbox into a definition EX: "Door on top-left"
        return f"{label} at {bbox}"

