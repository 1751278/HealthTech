"""Tests for the OCR frame-skip logic in liveOCR.py.

Background: Stage 1 of the OCR speed-optimization plan replaced a dead
`frame_idx % 1 == 0` condition in main()'s capture loop -- an expression
that is *always* True, so it silently submitted every single captured
frame to OCR regardless of any "only OCR every Nth frame" intent -- with
a named constant `OCR_FRAME_INTERVAL` and a pure helper function
`should_submit_frame()`. These tests lock in that fix so it can't
silently regress back to submitting every frame.

These tests are intentionally narrow and fast: they only exercise the
pure `should_submit_frame` helper. They never call liveOCR.main(), never
touch cv2.VideoCapture/imshow/waitKey, and never instantiate a real
easyocr.Reader -- all of that requires a camera/GUI/heavy model load
that isn't available (or desirable) in this test environment.
"""
from liveOCR import OCR_FRAME_INTERVAL, should_submit_frame


def test_ocr_frame_interval_default_value():
    """Locks in the validated default interval (matches
    combined_perception.py's --ocr-interval default of 3)."""
    assert OCR_FRAME_INTERVAL == 3


def test_not_every_frame_regression():
    """Regression guard for the original bug.

    The original code was `if frame_idx % 1 == 0: worker.submit(gray)`.
    Since anything mod 1 is always 0, that condition was always True,
    meaning EVERY frame was submitted to OCR -- defeating the entire
    point of a frame-skip optimization. With the fix (mod
    OCR_FRAME_INTERVAL, i.e. mod 3), only every 3rd frame should be
    submitted. Over frames 0..29 (30 frames), that's exactly 10 frames
    (indices 0, 3, 6, ..., 27).
    """
    submitted = [f for f in range(30) if should_submit_frame(f)]
    assert len(submitted) == 10
    # If the old `% 1` bug were reintroduced, this would be 30, not 10.
    assert len(submitted) != 30


def test_known_good_points_default_interval():
    assert should_submit_frame(0) is True
    assert should_submit_frame(1) is False
    assert should_submit_frame(2) is False
    assert should_submit_frame(3) is True


def test_custom_interval_override():
    assert should_submit_frame(5, interval=5) is True
    assert should_submit_frame(4, interval=5) is False
