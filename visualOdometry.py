import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, max_features=100):
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(21, 21), 
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.max_features = max_features
        self.old_gray = None
        self.p0 = None
        
        # Path tracking
        self.current_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.path_history = [self.current_pos.copy()]
        
        # Visualization canvas (600x600)
        self.trajectory_map = np.zeros((600, 600, 3), dtype=np.uint8)
        self.offset = np.array([300, 300]) # Center of the map

    def update(self, frame):
        """Processes a new frame and returns the current (x, y) coordinates."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialization logic for the first frame
        if self.old_gray is None:
            self.old_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, 
                                              maxCorners=self.max_features, 
                                              qualityLevel=0.3, minDistance=7)
            return self.current_pos

        # 1. Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is not None and len(p1[st == 1]) > 0:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            # 2. Estimate 2D Translation (mean displacement)
            diff = good_new - good_old
            avg_diff = np.mean(diff, axis=0)

            # 3. Accumulate position
            self.current_pos += avg_diff
            self.path_history.append(self.current_pos.copy())

            # 4. Draw trajectory
            self._draw_trajectory()
            
            # Prepare for next frame
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
            # Redetect if feature count drops
            if len(self.p0) < self.max_features // 2:
                self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, 
                                                  maxCorners=self.max_features, 
                                                  qualityLevel=0.3, minDistance=7)

        return self.current_pos

    def _draw_trajectory(self):
        """Internal method to update the 2D path visualization."""
        x, y = self.current_pos.astype(int) + self.offset
        # Ensure drawing stays within canvas boundaries
        if 0 <= x < 600 and 0 <= y < 600:
            cv2.circle(self.trajectory_map, (x, y), 1, (0, 255, 0), 2)

    def get_trajectory_view(self):
        return self.trajectory_map

# --- Main Execution ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    tracker = VisualOdometry(max_features=150)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (360, 640)) # vertical phone resolution

        # Process frame
        coords = tracker.update(frame)
        
        # Print coordinates to console
        print(f"Current XY: {coords[0]:.2f}, {coords[1]:.2f}")

        # Display results
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Path Traveled", tracker.get_trajectory_view())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()