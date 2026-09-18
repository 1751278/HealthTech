#################
# combined_perception.py
# Description: Combines liveOCR.py's OCRWorker and llm_vision.py's
# LLMVisionAssistant into one capture loop. Neither source file is modified —
# this just imports both classes and feeds each one the same frame.
# Created by Sahir Abrar 9/17/2026
#
# OCR runs continuously (cheap, every Nth frame). LLM vision only fires
# when you press 'd' — no automatic interval-based calls, since it hits a
# rate-limited network API.
#
# Controls:
#   d -> force a new LLM description request right now
#   q -> quit
# Run:
#   python combined_perception.py --source 1
#################

import argparse
import time

import cv2
import easyocr
from dotenv import load_dotenv

from liveOCR import OCRWorker          # reused as-is
from llm_vision import LLMVisionAssistant  # reused as-is

load_dotenv()


def format_desc(desc):
    hazards = desc.get("hazards", [])
    objects = desc.get("objects", [])
    parts = []
    if hazards:
        parts.append("HAZARD: " + "; ".join(hazards))
    if objects:
        parts.append("Objects: " + "; ".join(objects))
    return " | ".join(parts) if parts else "Nothing notable"


def draw_ocr_results(frame, results):
    for (bbox, text, prob) in results:
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(
            frame, text, (top_left[0] - 20, top_left[1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 0, 255), thickness=2
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=1)
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--llm-model", default=None,
                         help="override the default model; leave unset to use llm_vision.py's default")
    parser.add_argument("--ocr-interval", type=int, default=3,
                         help="submit every Nth frame to OCR (higher = less CPU contention, less frequent text updates)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        print(f"Failed to open source {args.source}, trying {args.source - 1}...")
        cap = cv2.VideoCapture(args.source - 1)
        if not cap.isOpened():
            print("Error: Could not open any video source.")
            return

    ocr = easyocr.Reader(['en'], gpu=True, quantize=False)
    ocr_worker = OCRWorker(ocr, thresh=0.35)

    llm_assistant = LLMVisionAssistant(
        provider=args.llm_provider,
        model=args.llm_model,
    )
    llm_assistant.start()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_idx = 0
    cur_time = time.perf_counter()
    last_llm_desc = {}

    print("Press 'd' to force an LLM description, 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Exiting loop.")
            break

        prev_time = cur_time
        cur_time = time.perf_counter()

        # --- OCR channel: gets a CLAHE-adjusted grayscale frame ---
        # Only submit every Nth frame — OCR is CPU-heavy, and submitting
        # every single frame starves the main render loop (this is the
        # bug liveOCR.py's comment describes but its code never actually
        # did: `frame_idx % 1 == 0` is always true).
        if frame_idx % args.ocr_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            ocr_worker.submit(gray)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            if llm_assistant.force_submit(frame):
                print("Requested LLM description...")
            else:
                print("LLM assistant still busy with the previous request.")
        elif key == ord("q"):
            print("'q' key pressed. Stopping application.")
            break

        # --- draw OCR boxes ---
        draw_ocr_results(frame, ocr_worker.get_results())

        # --- draw LLM hazards/objects ---
        desc = llm_assistant.latest_description()
        if desc != last_llm_desc:
            print("LLM:", desc)
            last_llm_desc = desc
        cv2.putText(
            frame, format_desc(desc)[:80], (10, frame.shape[0] - 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 255, 255), thickness=2
        )

        cv2.putText(
            frame, f"FPS:{1/(cur_time-prev_time):.2f}", (5, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 0, 255), thickness=2
        )

        cv2.imshow("Combined perception (OCR + LLM vision)", frame)

    ocr_worker.stop()
    llm_assistant.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped")


if __name__ == "__main__":
    main()