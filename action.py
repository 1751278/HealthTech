import threading
import time
import pyttsx3
from ultralytics import engine

class Action:
    def __init__(self):
        self.llm_dict = None
        self.action_type = None
        self.starting_heading_degrees = None
        self.heading_degrees = None
        self.start_time = None
        self.action_state = "done"  # Can be "in_progress" or "done"
    
    def speak_worker(self, text):
        self.voice = pyttsx3.init()
        self.voice.setProperty('rate', 150)    # Speed of speech (words per minute)
        self.voice.setProperty('volume', 1.0)       
        self.voice.say(text)
        self.voice.runAndWait()

    def speak_non_blocking(self, text):
        """Starts a new thread to speak without blocking the main program."""
        thread = threading.Thread(target=self.speak_worker, args=(text,), daemon=True)
        thread.start()

    def update_action(self, llm_dict, heading_degrees):
        self.llm_dict = llm_dict
        self.action_type = llm_dict["action"]
        self.starting_heading_degrees = heading_degrees
        self.heading_degrees = heading_degrees
        self.start_time = None
        self.action_state = "in_progress"  # Reset action state when updating
        if self.action_type == "turn":
            pass
            #self.speak_non_blocking(f"Turning {llm_dict['angle']} degrees.")

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
                return max(-90, min(90, turn_angle))

        