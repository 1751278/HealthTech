import collections
import math
import queue
import threading
import time
 
import cv2
import numpy as np



class LLMProcessor:
    def __init__(self, llm):
        self.llm = llm

    def start(self):
        self.llm.start()

    def stop(self):
        self.llm.stop()

    def process_frame(self, frame, context=""):
        """Submit a frame for processing. Returns whether it was accepted."""
        return self.llm.force_submit(frame, context=context)

    def get_result(self):
        """Return the latest LLM result as a dictionary."""
        return self.llm.latest_description()