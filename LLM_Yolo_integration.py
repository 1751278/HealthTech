import collections
import math
import queue
import threading
import time
 
import cv2
import numpy as np

#Simplistic (for now) LLM processor for integrating with YOLO detection, simply runs the llm on a frame

class LLMProcessor:
    def __init__(self, llm):
        self.llm = llm

    def start(self):
        self.llm.start()

    def stop(self):
        self.llm.stop()

    def process_frame(self, frame, context=""): #Returns if a frame was accepted
        return self.llm.force_submit(frame, context=context)

    def get_result(self): #Return the latest LLM result as a dictionary.
        return self.llm.latest_description()