import threading
import time
import sounddevice as sd
import soundfile as sf
from ultralytics import engine
import cv2
import numpy as np
import trajectory_helper as traj_helper
import time

class Action:
    def __init__(self):
        self.llm_dict = None
        self.action_type = None
        self.starting_heading_degrees = None
        self.heading_degrees = None
        self.start_time = None
        self.action_state = "done"  # Can be "in_progress" or "done"

        #tracking
        self.bounding_box = None
        self.is_tracking = False
        #search
        self.search_objects = None
        self.yoloE = None 
        self.floor_seg = None
        self.detections = None
        self.detected = [False, 0] #(is something detected, class, num frames detected)
        self.search_turning = [False, 0] #first value is whether to turn to search and second number is number of frames since turn begin
        self.traj_angle = 0

        self.modified_frame = None
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT_create()
    
    def speak_worker(self, file):
        data, fs = sf.read(file)
        sd.play(data, fs)
        sd.wait()
        return

    def speak_non_blocking(self, file):
        """Starts a new thread to speak without blocking the main program."""
        thread = threading.Thread(target=self.speak_worker, args=(file,), daemon=True)
        thread.start()

    #Any time LLM calls for a new action call this function
    def update_action(self, llm_dict, heading_degrees, frame, max_size, resize_func, yoloE):
        self.llm_dict = llm_dict
        self.action_type = llm_dict["action"]
        self.starting_heading_degrees = heading_degrees
        self.heading_degrees = heading_degrees
        self.start_time = None
        self.action_state = "in_progress"  # Reset action state when updating
        if self.action_type == "turn":
            self.speak_non_blocking("SoundAssets/turn.mp3")

        self.max_size = max_size  # Store the max_size for later use in tracking   
        self.resize_func = resize_func  # Store the resize function for later use in tracking 
        h, w = frame.shape[:2]
        self.scale_back = max(h, w)/self.max_size
        if self.action_type == "target":
            self.speak_non_blocking("SoundAssets/go_to_target.mp3")
            self.bounding_box = llm_dict["bbox"]
            self.bounding_box = self.resize_func(self.bounding_box[0], self.bounding_box[1], 
                                                 self.bounding_box[2], self.bounding_box[3], self.scale_back)
            print(self.bounding_box)
            self.bounding_box_tracker = (self.bounding_box[0], self.bounding_box[1], self.bounding_box[2]-self.bounding_box[0], self.bounding_box[3]-self.bounding_box[1])  # Convert to (x, y, w, h)
            self.tracker.init(frame, self.bounding_box_tracker)
        if self.action_type == "straight":
            self.search_objects = llm_dict['search']
            self.yoloE = yoloE
            self.search_objects.insert(0, 'open floor')# adding open floor to search space to help with navigation
            self.yoloE.set_classes(self.search_objects)
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
        bbox = self.resize_func(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], 1.0)
        if success:
            bbox_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            direction_error = bbox_center[0] - frame_center[0]
            return bbox, direction_error
        else:
            self.action_state = "done"  # If tracking fails, consider the action done
            return [0,0,0,0], 0  # Return None if tracking fails
        
    def search(self, frame, trajectory, confidence_thresh=0.6, detect_num_frame_thresh=10):
        self.modified_frame = frame.copy()
        h, w = frame.shape[:2]
        results = self.yoloE.predict(frame, retina_masks=True, verbose=False)
        result_dict = self.get_yolo_results(results)
        self.floor_seg = None
        self.detections = []
        for detection in result_dict:
            if detection['class_name'] == "open floor" and detection['confidence'] > confidence_thresh-0.4:
                self.floor_seg = detection['mask']
                self.modified_frame = self.apply_mask(self.modified_frame, detection)

            elif detection['confidence'] > confidence_thresh:
                self.detections.append((detection['class_name'], detection['bbox']))
                self.modified_frame = self.apply_bbox(self.modified_frame, detection['bbox'], detection['class_name'])
                if self.detections:
                    self.detected[1]+=1#add to total detection count
                else:
                    self.detected = [False, 0]
        # When something is detected turn towards it 
        if self.detected[1]>detect_num_frame_thresh:
            if self.detected[0] == False:
                self.speak_non_blocking("SoundAssets/turn.mp3")
            self.detected[0] = True
            bboxs = np.array([data[1] for data in self.detections])
            avr_bbox = np.mean(bboxs, axis=0)
            box_center = (int((avr_bbox[0] + avr_bbox[2]) / 2), int((avr_bbox[1] + avr_bbox[3]) / 2))
            frame_center = (w//2, h//2)
            diff = frame_center[0] - box_center[0]
            if abs(diff) < 20:#within pixel range
                if self.start_time is None:
                    self.start_time = time.time()  # Start the timer
                if self.timer(3):  # Wait for 3 seconds to ensure stability
                    self.action_state = "done"
                    self.start_time = None  # Reset the timer for future actions
                    self.detected = [False, 0]
            return frame_center[0] - box_center[0]

        #If no floor then turn to look around
        if self.floor_seg is None:
            self.search_turning[0] = True
            if self.search_turning[1] == 0:
                self.speak_non_blocking("SoundAssets/turn.mp3")
            self.search_turning[1] += 1
            return 90 #turn 90 degrees
        else: #find best direction to go in 
            self.traj_angle = traj_helper.find_best_traj_angle(trajectory, self.heading_degrees, prev_angle = self.traj_angle)
            self.seg_angle = traj_helper.find_best_navigation_direction(frame, self.floor_seg, self.traj_angle-(self.heading_degrees-90))
            if self.seg_angle is not None:
                if self.search_turning[0]==True:
                    self.speak_non_blocking("SoundAssets/now_exploring.mp3")
                    time.sleep(1)
                self.search_turing = (False, 0)
                return self.seg_angle
            else:
                self.search_turning[0] = True
                if self.search_turning[1] == 0:
                    self.speak_non_blocking("SoundAssets/turn.mp3")
                self.search_turning[1] += 1
                return 90 #turn 90 degrees

    def timer(self, duration):
        #Simple timer function to check if the specified duration has passed
        current_time = time.time()
        print(current_time - self.start_time)
        if (current_time - self.start_time) >= duration:
            return True
        return False

    def get_action_state(self):
        return self.action_state #returns if action is done or in progress
    
    def get_update_dir(self, frame, heading_degrees, trajectory):
            self.heading_degrees = heading_degrees
            if self.action_type == "turn":
                turn_angle = self.turn()
                return frame, max(-90, min(90, turn_angle))
            if self.action_type == "target":
                bbox, direction_error = self.track(frame)
                bbox_cv2 = bbox #[int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]  # Convert to [x1, y1, x2, y2] format
                self.modified_frame = frame.copy()
                self.modified_frame = self.apply_bbox(self.modified_frame, bbox_cv2, "tracking")
                return self.modified_frame, max(-90, min(90, direction_error))
            if self.action_type == "straight":
                best_angle = self.search(frame, trajectory)
                return self.modified_frame, best_angle

    # parse through yolo e results
    def get_yolo_results(self, results):
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
                detection["mask"] = masks.xy[index]#sending as polygon contour

            detections.append(detection)

        return detections
    def apply_mask(self, frame, result):
        x1, y1, x2, y2 = map(int, result['bbox'])
        contour = result['mask'].astype(np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], color=(0, 255, 0))  # Green mask
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        cv2.putText(frame, result['class_name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    def apply_bbox(self, frame, bbox, class_name):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # Blue box
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame
