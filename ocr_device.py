###########
# Description: Small shared helper that centralizes the device-selection
# policy for constructing an easyocr.Reader. Both OCR.py and liveOCR.py
# previously duplicated (and disagreed on) this logic.
#
# EasyOCR's own Reader.__init__ already auto-selects cuda > mps > cpu
# whenever gpu=True is passed (falling back to cpu safely if neither
# accelerator is available). This module mirrors that same detection
# order purely so it can decide whether `quantize` should be enabled
# (quantization is a CPU-only speed trick and has no effect on cuda/mps).
###########

import torch

# Tuned readtext() kwargs shared by OCR.py's two single-image calls and
# liveOCR.py's OCRWorker._loop live-frame call, so the tuning can't drift
# apart between call sites. (batch_size/workers are intentionally NOT part
# of this shared dict -- those are live-stream-specific and only apply to
# liveOCR.py's call; see liveOCR.py's OCRWorker._loop.)
TUNED_READTEXT_KWARGS = {
    "decoder": "greedy",
    "canvas_size": 640,
    "mag_ratio": 1.5,
    "link_threshold": 0.5,
}


def _resolve_device() -> str:
    """Mirror EasyOCR's own cuda > mps > cpu selection order."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_ocr_reader_kwargs(gpu: bool = True) -> dict:
    """Determine the gpu/quantize kwargs to pass to easyocr.Reader.

    - gpu=False: explicit CPU-quantized override, in case a caller ever
      wants to force CPU regardless of what's available.
    - gpu=True (default): let EasyOCR do its own cuda>mps>cpu selection
      via gpu=True, but only enable quantize if the resolved device is
      cpu, since quantization isn't meaningful on cuda or mps.

    Returns a dict with at least "gpu" and "quantize" keys (safe to pass
    straight into easyocr.Reader(['en'], **kwargs) after dropping any
    extra informational keys), plus an informational "resolved_device" key.
    """
    if not gpu:
        return {"gpu": False, "quantize": True, "resolved_device": "cpu"}

    resolved_device = _resolve_device()
    return {
        "gpu": True,
        "quantize": resolved_device == "cpu",
        "resolved_device": resolved_device,
    }
