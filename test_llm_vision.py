"""
test_llm_vision.py
Standalone smoke test for llm_vision.py — verifies your API key and the
request/response plumbing work before wiring it into navigation.py.

Controls:
  d  -> request a description of the current frame
  q  -> quit

Run:
    python test_llm_vision.py
"""

import cv2

from llm_vision import LLMVisionAssistant
import os
from dotenv import load_dotenv
load_dotenv()
print(repr(os.environ.get("GEMINI_API_KEY")))

def main():
    source = 1
    cap = cv2.VideoCapture(source)
    
    camera_name = cap.getBackendName()
    if not cap.isOpened():
        print("Camo Studio not detected, trying default camera...")
        cap = cv2.VideoCapture(source - 1)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

    # interval_frames is effectively disabled here since we trigger manually
    assistant = LLMVisionAssistant(provider="gemini", interval_frames=10**9)
    assistant.start()

    last_shown = ""
    print("Press 'd' to describe the current frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (360, 640))  # Resize for display purposes
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            if assistant.force_submit(frame):
                print("Requested description...")
            else:
                print("Still waiting on the previous request.")
        elif key == ord("q"):
            break

        desc = assistant.latest_description()
        if desc != last_shown:
            print("LLM:", desc)
            last_shown = desc

        cv2.putText(frame, desc[:70], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.imshow("llm_vision test", frame)

    cap.release()
    cv2.destroyAllWindows()
    assistant.stop()


if __name__ == "__main__":
    main()