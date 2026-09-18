"""Manual (real-inference) benchmark, covering BOTH axes of the OCR.py speed
optimization end-to-end:

  1. Reader construction: OLD hardcoded `gpu=False, quantize=True` vs NEW
     `resolve_ocr_reader_kwargs()` (cuda>mps>cpu-aware, quantize only on cpu).
  2. readtext() call params: OLD (no overrides, EasyOCR's untuned defaults)
     vs NEW (`TUNED_READTEXT_KWARGS`, reused from liveOCR.py's OCRWorker).

A prior version of this script only isolated axis 1, leaving axis 2's
speedup claim (made in tests/test_ocr_readtext_params.py's docstring)
unbacked by a rerunnable artifact. This version measures OCR.py's actual
end-to-end before/after behavior (OLD reader + OLD readtext() call vs NEW
reader + NEW readtext() call) alongside the two isolated axes, so every
speedup number quoted anywhere in this repo is reproducible by running
this one script.

Run on the actual TestImage/*.jpg images, resized to 640x480 exactly like
OCR.py does.

This is NOT a pytest-collected fast test — it does real model loading and
real inference on purpose, to prove/disprove speedup and accuracy parity.

Run directly:
    .venv/bin/python tests/manual_benchmark_ocr_device.py
"""

import sys
import time
from pathlib import Path

import cv2
import easyocr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr_device import TUNED_READTEXT_KWARGS, resolve_ocr_reader_kwargs  # noqa: E402

IMAGES = ["indoorSigns.jpg", "name.jpg", "signs.jpg"]


def load_image(name: str):
    path = REPO_ROOT / "TestImage" / name
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"{path} not found")
    return cv2.resize(img, (640, 480))


def build_reader(label: str, kwargs: dict):
    t0 = time.perf_counter()
    reader = easyocr.Reader(["en"], **kwargs)
    t1 = time.perf_counter()
    print(f"[{label}] Reader construction (model load) time: {t1 - t0:.4f}s  kwargs={kwargs}")
    return reader, (t1 - t0)


def run_readtext(reader, img, **kwargs):
    t0 = time.perf_counter()
    result = reader.readtext(img, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def summarize(result):
    """Return (set_of_texts, list_of_(text, conf))."""
    texts = [text for (_bbox, text, _prob) in result]
    confs = [(text, prob) for (_bbox, text, prob) in result]
    return set(texts), confs


def main():
    print("=" * 100)
    print("OCR.py END-TO-END BENCHMARK: OLD (original code) vs NEW (optimized code)")
    print("Also isolates each axis individually: device/quantize selection, and readtext() tuning.")
    print("=" * 100)

    new_kwargs_full = resolve_ocr_reader_kwargs()
    new_kwargs = {k: v for k, v in new_kwargs_full.items() if k in ("gpu", "quantize")}
    print(f"\nresolve_ocr_reader_kwargs() -> {new_kwargs_full}")
    print(f"NEW reader will be built with: {new_kwargs}")
    print(f"TUNED_READTEXT_KWARGS -> {TUNED_READTEXT_KWARGS}\n")

    # Two reader instances: OLD mirrors OCR.py's original hardcoded reader
    # construction; NEW mirrors what OCR.py builds today via ocr_device.py.
    old_reader, old_build_time = build_reader("OLD", {"gpu": False, "quantize": True})
    new_reader, new_build_time = build_reader("NEW", new_kwargs)

    print()
    # Three readtext() calls per image, forming a consistent A -> B -> C chain
    # so the two axes decompose cleanly instead of being measured against
    # inconsistent baselines:
    #   A = OLD reader, no readtext kwargs      (original OCR.py, pre-optimization)
    #   B = NEW reader, no readtext kwargs      (after Stage 2 device fix only)
    #   C = NEW reader, TUNED_READTEXT_KWARGS   (after Stage 2+3, i.e. OCR.py today)
    # device_only = A -> B (device/quantize axis, readtext params held at "none")
    # params_only = B -> C (readtext-tuning axis, device held at "new", matching
    #                       what OCR.py actually runs on today -- the tuned
    #                       canvas_size/mag_ratio values were chosen for this
    #                       device path, so isolating them against the OLD
    #                       CPU-quantized reader would understate their effect)
    # end_to_end  = A -> C (the real OCR.py before/after change a user sees)
    totals = {"end_to_end_old": 0.0, "end_to_end_new": 0.0,
              "device_only_old": 0.0, "device_only_new": 0.0,
              "params_only_old": 0.0, "params_only_new": 0.0}

    per_image_results = []

    for name in IMAGES:
        print("-" * 100)
        print(f"IMAGE: {name}")
        img = load_image(name)

        a_result, a_time = run_readtext(old_reader, img)
        b_result, b_time = run_readtext(new_reader, img)
        c_result, c_time = run_readtext(new_reader, img, **TUNED_READTEXT_KWARGS)

        totals["end_to_end_old"] += a_time
        totals["end_to_end_new"] += c_time
        totals["device_only_old"] += a_time
        totals["device_only_new"] += b_time
        totals["params_only_old"] += b_time
        totals["params_only_new"] += c_time

        e2e_old_texts, e2e_old_confs = summarize(a_result)
        e2e_new_texts, e2e_new_confs = summarize(c_result)
        lost = e2e_old_texts - e2e_new_texts   # present in old, missing in new -> regression signal
        gained = e2e_new_texts - e2e_old_texts  # present in new, not in old

        print(f"  [end-to-end]  OLD {a_time:.4f}s -> NEW {c_time:.4f}s "
              f"({a_time / c_time:.2f}x)" if c_time > 0 else "")
        print(f"  [device only] OLD {a_time:.4f}s -> NEW {b_time:.4f}s "
              f"({a_time / b_time:.2f}x)" if b_time > 0 else "")
        print(f"  [params only] OLD {b_time:.4f}s -> NEW {c_time:.4f}s "
              f"({b_time / c_time:.2f}x)" if c_time > 0 else "")

        print(f"  OLD detected text set ({len(e2e_old_texts)}): {sorted(e2e_old_texts)}")
        print(f"  NEW detected text set ({len(e2e_new_texts)}): {sorted(e2e_new_texts)}")
        print(f"  TEXT LOST (in OLD, missing from NEW): {sorted(lost) if lost else 'NONE'}")
        print(f"  TEXT GAINED (in NEW, not in OLD): {sorted(gained) if gained else 'NONE'}")

        print("  OLD confidences:")
        for text, prob in e2e_old_confs:
            print(f"    {text!r}: {prob:.4f}")
        print("  NEW confidences:")
        for text, prob in e2e_new_confs:
            print(f"    {text!r}: {prob:.4f}")

        per_image_results.append({"name": name, "lost": lost, "gained": gained})

    print("-" * 100)
    print("\nSUMMARY")
    print("=" * 100)
    print(f"Reader construction (model load): OLD {old_build_time:.4f}s -> NEW {new_build_time:.4f}s")
    for axis, label in [("end_to_end", "END-TO-END: A(old reader, no kwargs) -> C(new reader, tuned kwargs) -- the real OCR.py before/after"),
                         ("device_only", "DEVICE AXIS: A(old reader, no kwargs) -> B(new reader, no kwargs)"),
                         ("params_only", "PARAMS AXIS: B(new reader, no kwargs) -> C(new reader, tuned kwargs)")]:
        old_t, new_t = totals[f"{axis}_old"], totals[f"{axis}_new"]
        print(f"\n[{label}]")
        print(f"  OLD total across {len(IMAGES)} images: {old_t:.4f}s (avg {old_t/len(IMAGES):.4f}s/image)")
        print(f"  NEW total across {len(IMAGES)} images: {new_t:.4f}s (avg {new_t/len(IMAGES):.4f}s/image)")
        if new_t > 0:
            print(f"  TOTAL speedup ratio (old/new): {old_t / new_t:.2f}x")

    # Known, already-investigated case: on signs.jpg, OLD misreads a stylized
    # word as "EASHION" (0.60 confidence); NEW correctly reads it as "FASHION"
    # (0.69 confidence, higher). A strict set-difference sees this as "EASHION
    # lost", but it's the same text region being read MORE correctly, not a
    # detection that vanished -- flagged here explicitly so this known,
    # already-investigated case doesn't read as a new, unexplained regression.
    KNOWN_CORRECTED_MISREADS = {("signs.jpg", "EASHION"): "FASHION"}

    any_regression = any(r["lost"] for r in per_image_results)
    print("\nACCURACY REGRESSION CHECK (end-to-end OLD vs NEW):")
    if any_regression:
        print("  Text present in OLD results is missing from NEW results:")
        for r in per_image_results:
            for lost_text in sorted(r["lost"]):
                key = (r["name"], lost_text)
                if key in KNOWN_CORRECTED_MISREADS:
                    print(f"    Image {r['name']}: {lost_text!r} -> "
                          f"{KNOWN_CORRECTED_MISREADS[key]!r} (KNOWN CORRECTED MISREAD, "
                          f"not a real loss -- same text region, now read correctly "
                          f"with higher confidence; see 'TEXT GAINED' above)")
                else:
                    print(f"    Image {r['name']}: LOST -> {lost_text!r} "
                          f"(UNEXPLAINED -- investigate, this is not a known case)")
    else:
        print("  No regression: every text string detected by OLD was also detected by NEW "
              "(NEW may additionally detect more, see 'TEXT GAINED' above).")


if __name__ == "__main__":
    main()
