#################
# test_ocr_reasoning_smoke.py
# Created by Sahir Abrar September 2026
# Last Updated: 9/17/2026 by Sahir Abrar
# Last Change:
# - Initial version.
# Description: Standalone smoke test for ocr_reasoning.py's
# OCRReasoningAssistant — verifies your GEMINI_API_KEY and the
# request/response schema plumbing work end-to-end against the REAL Gemini
# API before wiring OCRReasoningAssistant into combined_perception.py.
#
# WARNING: This hits the REAL Gemini API using GEMINI_API_KEY from .env and
# costs real API quota. It is intentionally NOT part of the pytest suite
# (it lives at the repo root, not in tests/, and is not collected by
# `pytest tests/`) — run it manually/on-demand only:
#   python test_ocr_reasoning_smoke.py
# Makes exactly ONE real API call.
#################

import os

from dotenv import load_dotenv

from ocr_reasoning import OCRReasoningAssistant

load_dotenv()


def main():
    # Presence-only check -- never print the raw key (avoids leaking it into
    # terminal scrollback/CI logs; unlike test_llm_vision.py's `repr(...)`,
    # which prints the literal secret and shouldn't be copied here).
    print("GEMINI_API_KEY loaded:", "yes" if os.environ.get("GEMINI_API_KEY") else "MISSING")

    assistant = OCRReasoningAssistant(provider="gemini")

    # Real OCR fragments from this repo's own test data (the name.jpg
    # detection), minus the noisy "2r25153" fragment.
    ocr_results = [
        (None, "MR", 0.9875),
        (None, "GUNDERSON", 0.9996),
        (None, "MS", 0.9125),
        (None, "KULICK", 0.9866),
    ]

    print("Calling OCRReasoningAssistant.reason() synchronously (ONE real API call)...")
    result = assistant.reason(ocr_results)

    print("Full result dict:")
    print(result)

    required_types = {
        "summary": str,
        "category": str,
        "resolved_text": str,
        "is_actionable": bool,
        "confidence": str,
        "discarded_fragments": list,
    }

    for key, expected_type in required_types.items():
        assert key in result, f"Missing required key: {key!r}"
        assert isinstance(result[key], expected_type), (
            f"Key {key!r} has type {type(result[key]).__name__}, "
            f"expected {expected_type.__name__}"
        )

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
