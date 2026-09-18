"""Real-inference regression guard for OCR.py's readtext() tuning.

Background: Stage 3 of the OCR speed-optimization plan changed both
`ocr.readtext(...)` call sites in OCR.py (inside `read_text_from_image()`
and inside `main()`) to pass `decoder="greedy", canvas_size=640,
mag_ratio=1.5, link_threshold=0.5` -- params reused from liveOCR.py's
already-tuned OCRWorker -- instead of calling `readtext()` with no
overrides (which fell back to EasyOCR's internal defaults of
canvas_size=2560, mag_ratio=1.0).

tests/manual_benchmark_ocr_device.py reproduces this claim on demand (its
"PARAMS AXIS" section): on TestImage/{indoorSigns,name,signs}.jpg resized
to 640x480 (the same resize OCR.py's own main() does), the new tuned
params produce same-or-better detected text than the old defaults (one
case even improved: a misread "EASHION" at 0.60 confidence became the
correctly-read "FASHION" at 0.69 confidence on signs.jpg), alongside real
readtext()-only speedups of roughly 1.2x-3.2x per image (run the script
yourself for current numbers -- they vary run to run with system load).

This test is intentionally NOT mocked at the inference layer: it is the
accuracy-sensitive stage, so it must exercise actual EasyOCR inference on
actual images to be worth anything. It is slow (real model load + real
inference) by design -- that cost buys a real regression guard instead of
a guard that would pass even if EasyOCR's output silently degraded.

It does two independent things:
  1. Locks in a baseline set of expected detected text per image (a
     known-good superset of what the OLD defaults detected, i.e. what
     NEW currently detects), so a future change that regresses detection
     quality on these images gets caught here.
  2. Spies (via unittest.mock.patch.object with wraps=, so the real call
     still executes) on `ocr.readtext` to assert the tuned kwargs
     (canvas_size=640, mag_ratio=1.5, link_threshold=0.5) are actually
     being passed by `read_text_from_image()`. This guards against
     someone silently reverting the tuning later while the accuracy
     assertions above might not catch it (e.g. if a regression only
     shows up on harder images not in this fixed set).

Run directly (this file only):
    .venv/bin/python -m pytest tests/test_ocr_readtext_params.py -v

This file takes several seconds because importing OCR constructs a real
easyocr.Reader at module level (existing OCR.py behavior) and each test
runs real inference on a real image -- that's expected for an
integration test at this accuracy-sensitive stage.
"""

from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from OCR import ocr, read_text_from_image

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_test_image(name: str):
    """Load a TestImage/*.jpg and resize to 640x480, matching exactly what
    OCR.py's own main() does before calling readtext()."""
    path = REPO_ROOT / "TestImage" / name
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"{path} not found")
    return cv2.resize(img, (640, 480))


def detected_texts(result):
    """Extract just the recognized text strings from an easyocr readtext()
    result (list of (bbox, text, prob) tuples)."""
    return {text for (_bbox, text, _prob) in result}


@pytest.mark.slow
def test_indoor_signs_detects_expected_baseline_text():
    img = load_test_image("indoorSigns.jpg")
    result = read_text_from_image(img)
    texts = detected_texts(result)

    expected = {"EXIT", "LAB", "OFFICE", "RESTROOM", "WAREHOUSE"}
    missing = expected - texts
    assert not missing, f"Missing expected text {missing} in detected {texts}"


@pytest.mark.slow
def test_name_detects_expected_baseline_text():
    img = load_test_image("name.jpg")
    result = read_text_from_image(img)
    texts = detected_texts(result)

    # 'MR', 'MS', 'GUNDERSON', 'KULICK' are all high-confidence (>0.9) and
    # stable. We deliberately skip asserting on the low-confidence
    # '2r25153'-like token (observed at ~0.04 confidence): it's OCR noise
    # over a stylized/illegible portion of the image, not meaningful
    # detected text, and asserting on its exact garbled spelling would
    # make this test brittle without guarding anything real.
    expected = {"MR", "MS", "GUNDERSON", "KULICK"}
    missing = expected - texts
    assert not missing, f"Missing expected text {missing} in detected {texts}"


@pytest.mark.slow
def test_signs_detects_expected_baseline_text():
    img = load_test_image("signs.jpg")
    result = read_text_from_image(img)
    texts = detected_texts(result)

    # Deliberately not asserting on the EASHION/FASHION token: OCR text on
    # ambiguous/stylized signage can legitimately vary slightly with minor
    # environment differences (EasyOCR/torch version, etc.), and that
    # specific token's exact spelling isn't what this test is guarding.
    # The other tokens below are stable, unambiguous signage/street text.
    expected = {"AVE", "WAY", "W30", "7"}
    missing = expected - texts
    assert not missing, f"Missing expected text {missing} in detected {texts}"


@pytest.mark.slow
def test_read_text_from_image_passes_tuned_readtext_kwargs():
    """Guards against someone accidentally reverting the readtext() tuning.

    Wraps the real ocr.readtext with a spy (wraps=... so it still executes
    the real call and this stays a real-inference test) and inspects the
    kwargs it was actually invoked with.
    """
    img = load_test_image("indoorSigns.jpg")

    with patch.object(ocr, "readtext", wraps=ocr.readtext) as spy:
        read_text_from_image(img)

    spy.assert_called_once()
    _args, kwargs = spy.call_args
    assert kwargs.get("decoder") == "greedy"
    assert kwargs.get("canvas_size") == 640
    assert kwargs.get("mag_ratio") == 1.5
    assert kwargs.get("link_threshold") == 0.5
