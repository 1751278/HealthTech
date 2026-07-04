#################
# liveOCR.py
# Created by Kenshi Kadarusman April 1 2026
# Last Updated: July 4 2026 by Sahir Abrar
# Description: This is the live OCR module for HealthTech. It uses EasyOCR
# to read text from a live camera feed. The current implementation displays
# the detected text on the video feed, but it can be modified to return the
# detected text for further processing in the future.
# Notes:
# - May have to change cv2.VideoCapture(1) to cv2.VideoCapture(0) depending
#   on the system. If you have multiple cameras, try different indices.
# TODO:
# - Add to Navigation.py
# - Return detected text for further processing.
#################
 
import time
import threading
import cv2
import easyocr
import torch
 
print(f"CUDA Available: {torch.cuda.is_available()}")  # just checking the GPU shows up
 
 
class OCRWorker:
    """
    This handles the OCR part on a separate thread, so it doesn't freeze
    up the video while it's thinking. Basic idea:
    - main loop calls submit() to hand it the newest camera frame
    - this thread reads that frame and runs OCR on it whenever it's free
    - main loop calls get_results() to grab whatever text it found last
    If the thread is still busy and a newer frame comes in, we just
    throw away the old one and use the newest — we don't want to fall behind.
    """
 
    def __init__(self, reader, thresh=0.35):
        self.reader = reader              # the OCR reader
        self.thresh = thresh              # only keep results we're at least this confident about
        self.lock = threading.Lock()      # keeps the two threads from stepping on each other
        self.latest_frame = None          # newest frame waiting to be read
        self.latest_results = []          # newest text/boxes we found
        self.running = True               # keep the thread running
        self.thread = threading.Thread(target=self._loop, daemon=True) # run in the background
        self.thread.start() # start the thread right away
  
    def submit(self, frame):
        """Main loop calls this to hand over a new frame to read."""
        with self.lock:
            self.latest_frame = frame
 
    def _loop(self):
        """This just runs forever in the background, reading whatever frame it's given."""
        while self.running:
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None  # clear it so we don't read the same frame twice
 
            if frame is None:
                time.sleep(0.005)  # nothing to do yet, chill for a bit
                continue
 
            # This is the slow part — actually finding + reading the text
            preds = self.reader.readtext(
                frame,
                decoder="greedy",
                canvas_size=640,
                mag_ratio=1.5,
                batch_size=4,
                workers=0,
                link_threshold=0.5,   # lower = less likely to chop words into pieces
            )
 
            # toss out low-confidence junk before saving the results
            with self.lock:
                self.latest_results = [p for p in preds if p[2] > self.thresh]
 
    def get_results(self):
        """Main loop calls this to get whatever text was found most recently."""
        with self.lock:
            return self.latest_results
    def stop(self):
        self.running = False
        self.thread.join()
 
 
def main():
    print("Hello From HealthTech! \n")
 
    # gpu=True means use the graphics card. quantize is a CPU-only speed
    # trick, so we turn it off here since we're already using the GPU.
    ocr = easyocr.Reader(['en'], gpu=True, quantize=False)
    worker = OCRWorker(ocr, thresh=0.35)  # how confident it needs to be before we show a result
 
    cap = cv2.VideoCapture(1)  # Change to 0 if this doesn't find the right camera
    # Set the camera resolution ourselves instead of using whatever default
    # it picks (which is usually bigger than we need and just slows things down).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
 
    if not cap.isOpened():
        print("Failed to open webcam. Exiting.")
        return
 
    print("Webcam initialized. Starting video stream loop.")
 
    # CLAHE fixes up lighting/contrast — helps a lot with glare or shadows
    # on the text, way better than just using plain grayscale.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 
    cur_time = time.perf_counter()  # DEBUG - used to calculate FPS
    frame_idx = 0  # counts frames so we know when it's time to send one to OCR
 
    while True:
        success, frame = cap.read()
        if not success:
            # Check this right away, before doing anything else with the
            # frame, so we don't crash trying to process an empty frame.
            print("Failed to grab frame. Exiting loop.")
            break
 
        prev_time = cur_time
        cur_time = time.perf_counter()
 
        # Get the frame ready for OCR: grayscale + fix up the lighting.
        # (Not resizing it ourselves anymore — EasyOCR already resizes
        # internally, so doing it twice was just wasted work and the old
        # method was actually making the text look worse, not better.)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
 
        # Only send every 3rd frame over to OCR. OCR is the slow part,
        # so this keeps the video itself running smooth while OCR still
        # gets updated often enough to feel live.
        if frame_idx % 3 == 0:
            worker.submit(gray)
        frame_idx += 1
 
        # Draw whatever text OCR found most recently. It might be a frame
        # or two behind, but that's fine — video stays smooth either way.
        for (bbox, text, prob) in worker.get_results():
            # Get the corners of the box around the text
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
 
            # Draw the box
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
 
            # Draw the text above the box (CV2 uses BGR, not RGB, for colors)
            cv2.putText(
                frame, text, (top_left[0] - 20, top_left[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 255), thickness=2
            )
 
        # This FPS number is now the actual video speed, not the OCR speed,
        # since OCR runs separately and doesn't hold up the video anymore.
        cv2.putText(
            frame, f"FPS:{1/(cur_time-prev_time):.2f}", (5, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 0, 255), thickness=2
        )
 
        cv2.imshow("Live text detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("'q' key pressed. Stopping application.")
            break
 
    worker.stop()   # tell the OCR thread to stop nicely
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped")
 
 
if __name__ == "__main__":
    main()
 