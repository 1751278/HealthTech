import threading
import time
import pyttsx3
from ultralytics import engine
import cv2

class Action:
    def __init__(self):
        self.llm_dict = None
        self.action_type = None
        self.starting_heading_degrees = None
        self.heading_degrees = None
        self.start_time = None
        self.action_state = "done"  # Can be "in_progress" or "done"

        self.bounding_box = None
        self.is_tracking = False
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT_create()
    
    def speak_worker(self, text):
        voice = pyttsx3.init()
        voice.setProperty('rate', 150)    # Speed of speech (words per minute)
        voice.setProperty('volume', 1.0)       
        voice.say(text)
        voice.runAndWait()
        return

    def speak_non_blocking(self, text):
        """Starts a new thread to speak without blocking the main program."""
        thread = threading.Thread(target=self.speak_worker, args=(text,), daemon=True)
        thread.start()

    def update_action(self, llm_dict, heading_degrees, frame):
        self.llm_dict = llm_dict
        self.action_type = llm_dict["action"]
        self.starting_heading_degrees = heading_degrees
        self.heading_degrees = heading_degrees
        self.start_time = None
        self.action_state = "in_progress"  # Reset action state when updating
        if self.action_type == "turn":
            pass
            #self.speak_non_blocking(f"Turning {llm_dict['angle']} degrees.")
        if self.action_type == "target":
            self.bounding_box = llm_dict["bbox"]
            self.bounding_box_tracker = (self.bounding_box[0], self.bounding_box[1], self.bounding_box[2]-self.bounding_box[0], self.bounding_box[3]-self.bounding_box[1])  # Convert to (x, y, w, h)
            self.tracker.init(frame, self.bounding_box_tracker)        

    def turn(self):
        #Determine the direction to turn based on the heading_degrees and the action specified in llm_dict
        target_direction = self.starting_heading_degrees + self.llm_dict["angle"]
        turn_error = target_direction - self.heading_degrees
        #print(turn_error)
        if abs(turn_error)%360 < 8:  # If the error is less than 8 degrees, consider it done
            if self.start_time is None:
                self.start_time = time.time()  # Start the timer
            if self.timer(3):  # Wait for 3 seconds to ensure stability
                self.action_state = "done"
                self.start_time = None  # Reset the timer for future actions
        return turn_error
    def track(self, frame):
        #track object and go to object based on bounding box
        success, bbox = self.tracker.update(frame)
        if success:
            bbox_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            direction_error = bbox_center[0] - frame_center[0]
            return bbox, direction_error
        else:
            self.action_state = "done"  # If tracking fails, consider the action done
            return [0,0,0,0], 0  # Return None if tracking fails

    def timer(self, duration):
        #Simple timer function to check if the specified duration has passed
        current_time = time.time()
        print(current_time - self.start_time)
        if (current_time - self.start_time) >= duration:
            return True
        return False

    def get_action_state(self):
        return self.action_state #returns if action is done or in progress
    
    def get_update_dir(self, frame, heading_degrees):
            self.heading_degrees = heading_degrees
            if self.action_type == "turn":
                turn_angle = self.turn()
                return [0,0,0,0], max(-90, min(90, turn_angle))
            if self.action_type == "target":
                bbox, direction_error = self.track(frame)
                bbox_cv2 = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]  # Convert to [x1, y1, x2, y2] format
                return bbox_cv2, max(-90, min(90, direction_error))

        
