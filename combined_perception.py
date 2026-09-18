#################
# combined_perception.py
# Description: Combines liveOCR.py's OCRWorker, llm_vision.py's
# LLMVisionAssistant, and ocr_reasoning.py's OCRReasoningAssistant into one
# capture loop. None of those source files are modified — this just imports
# all three classes and feeds each one the same frame/OCR results.
# Created by Sahir Abrar 9/17/2026
# Last Updated: 9/17/2026 by Sahir Abrar
# Last Change:
# - Wired in OCRReasoningAssistant as a third channel: passive debounced
#   submission every frame via maybe_submit(), plus a manual 'r' keypress
#   for forced testing, mirroring the existing 'd' (LLM vision) pattern.
#
# OCR runs continuously (cheap, every Nth frame). LLM vision only fires
# when you press 'd' — no automatic interval-based calls, since it hits a
# rate-limited network API. OCR reasoning runs passively in the background
# (debounced on OCR text content changes) and can also be forced with 'r'.
#
# Controls:
#   d -> force a new LLM description request right now
#   r -> force a new OCR reasoning request right now
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
from ocr_device import resolve_ocr_reader_kwargs
from ocr_reasoning import OCRReasoningAssistant  # reused as-is

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


def format_ocr_summary(summary):
    if summary.get("is_actionable") and summary.get("summary"):
        return f"[{summary['category']}] {summary['summary']}"
    return "No actionable text"


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

    _reader_kwargs = resolve_ocr_reader_kwargs()
    ocr = easyocr.Reader(['en'], gpu=_reader_kwargs["gpu"], quantize=_reader_kwargs["quantize"])
    ocr_worker = OCRWorker(ocr, thresh=0.35)

    llm_assistant = LLMVisionAssistant(
        provider=args.llm_provider,
        model=args.llm_model,
    )
    llm_assistant.start()

    ocr_reasoner = OCRReasoningAssistant(provider="gemini")
    ocr_reasoner.start()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_idx = 0
    cur_time = time.perf_counter()
    last_llm_desc = {}
    last_ocr_summary = {}

    print("Press 'd' to force an LLM description, 'r' to force OCR reasoning, 'q' to quit.")

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
        elif key == ord("r"):
            if ocr_reasoner.force_submit(ocr_worker.get_results()):
                print("Requested OCR reasoning...")
            else:
                print("OCR reasoning still busy with the previous request.")
        elif key == ord("q"):
            print("'q' key pressed. Stopping application.")
            break

        # --- draw OCR boxes ---
        ocr_results = ocr_worker.get_results()
        draw_ocr_results(frame, ocr_results)

        # --- passive/debounced OCR reasoning submission (no keypress needed) ---
        ocr_reasoner.maybe_submit(ocr_results)

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

        # --- draw OCR reasoning summary ---
        summary = ocr_reasoner.latest_summary()
        if summary != last_ocr_summary:
            print("OCR reasoning:", summary)
            last_ocr_summary = summary
        cv2.putText(
            frame, format_ocr_summary(summary)[:80], (10, frame.shape[0] - 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(255, 255, 0), thickness=2
        )

        cv2.putText(
            frame, f"FPS:{1/(cur_time-prev_time):.2f}", (5, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(0, 0, 255), thickness=2
        )

        cv2.imshow("Combined perception (OCR + LLM vision)", frame)

    ocr_worker.stop()
    llm_assistant.stop()
    ocr_reasoner.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Application stopped")


if __name__ == "__main__":
    main()