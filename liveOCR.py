import time
import threading
import cv2
import easyocr
import torch

class OCRWorker:
    def __init__(self, reader, thresh=0.35):
        self.reader = reader
        self.thresh = thresh
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_results = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.latest_frame = frame

    def _loop(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue
            preds = self.reader.readtext(
                frame, decoder="greedy", canvas_size=640,
                mag_ratio=1.5, batch_size=4, workers=0,
                link_threshold=0.5
            )
            with self.lock:
                self.latest_results = [p for p in preds if p[2] > self.thresh]

    def get_results(self):
        with self.lock:
            return self.latest_results

    def stop(self):
        self.running = False
        self.thread.join()


def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    ocr = easyocr.Reader(['en'], gpu=True, quantize=False)
    worker = OCRWorker(ocr, thresh=0.35)

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        print("Failed to open webcam. Exiting.")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cur_time = time.perf_counter()
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Exiting loop.")
            break

        prev_time = cur_time
        cur_time = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        # only hand off every 3rd frame to the OCR thread
        if frame_idx % 3 == 0:
            worker.submit(gray)
        frame_idx += 1

        for (bbox, text, prob) in worker.get_results():
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0] - 20, top_left[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS:{1/(cur_time-prev_time):.2f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Live text detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()