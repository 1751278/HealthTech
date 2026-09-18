"""Unit tests for ocr_device.resolve_ocr_reader_kwargs.

These tests mock torch's cuda/mps availability checks so they run fast,
deterministically, and without needing any real GPU/accelerator hardware.
"""

from unittest.mock import patch

from ocr_device import resolve_ocr_reader_kwargs


def test_cuda_and_mps_available_prefers_cuda():
    """cuda wins priority over mps when both are available."""
    with patch("ocr_device.torch.cuda.is_available", return_value=True), \
         patch("ocr_device.torch.backends.mps.is_available", return_value=True):
        kwargs = resolve_ocr_reader_kwargs()

    assert kwargs["resolved_device"] == "cuda"
    assert kwargs["quantize"] is False
    assert kwargs["gpu"] is True


def test_cuda_unavailable_mps_available_resolves_mps():
    with patch("ocr_device.torch.cuda.is_available", return_value=False), \
         patch("ocr_device.torch.backends.mps.is_available", return_value=True):
        kwargs = resolve_ocr_reader_kwargs()

    assert kwargs["resolved_device"] == "mps"
    assert kwargs["quantize"] is False
    assert kwargs["gpu"] is True


def test_cuda_and_mps_unavailable_resolves_cpu_and_quantizes():
    with patch("ocr_device.torch.cuda.is_available", return_value=False), \
         patch("ocr_device.torch.backends.mps.is_available", return_value=False):
        kwargs = resolve_ocr_reader_kwargs()

    assert kwargs["resolved_device"] == "cpu"
    assert kwargs["quantize"] is True
    assert kwargs["gpu"] is True


def test_gpu_false_always_short_circuits_to_cpu_quantized():
    """gpu=False is an explicit override: it must return the fixed
    CPU-quantized dict regardless of what cuda/mps report as available."""
    with patch("ocr_device.torch.cuda.is_available", return_value=True), \
         patch("ocr_device.torch.backends.mps.is_available", return_value=True):
        kwargs = resolve_ocr_reader_kwargs(gpu=False)

    assert kwargs == {"gpu": False, "quantize": True, "resolved_device": "cpu"}
