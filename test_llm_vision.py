#################
# test_llm_vision.py
# Created by Sahir Abrar July 2026
# Last Updated: July 2026 by Sahir Abrar
# Last Change:
# - Updated to display new hazards/objects dict format from llm_vision.py
# Description: Standalone smoke test for llm_vision.py — verifies your API key
# and the request/response plumbing work before wiring it into navigation.py.
# Controls:
#   d  -> request a description of the current frame
#   q  -> quit
# Run:
#   python test_llm_vision.py
################

import cv2

from llm_vision import LLMVisionAssistant
import os 
from dotenv import load_dotenv
load_dotenv()
print(repr(os.environ.get("GEMINI_API_KEY")))


def format_desc(desc):
    """Turn the {"hazards": [...], "objects": [...]} dict into a short
    display string."""
    hazards = desc.get("hazards", [])
    objects = desc.get("objects", [])
    parts = []
    if hazards:
        parts.append("HAZARD: " + "; ".join(hazards))
    if objects:
        parts.append("Objects: " + "; ".join(objects))
    return " | ".join(parts) if parts else "Nothing notable"

def main():
    source = 1
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Camo Studio not detected, trying default camera...")
        cap = cv2.VideoCapture(source - 1)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

    assistant = LLMVisionAssistant(provider="gemini", interval_frames=10**9)
    assistant.start()

    last_shown = {}
    print("Press 'd' to describe the current frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (360, 640))

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

        display_text = format_desc(desc)
        cv2.putText(frame, display_text[:70], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.imshow("llm_vision test", frame)

    cap.release()
    cv2.destroyAllWindows()
    assistant.stop()


if __name__ == "__main__":
    main()