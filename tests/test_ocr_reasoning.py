"""Unit tests for ocr_reasoning.py (Stage 1): pure fragment extraction plus
the synchronous Gemini text-reasoning call.

Everything Gemini-related is mocked here -- google.genai.Client is patched
before any OCRReasoningAssistant is constructed, and generate_content is
replaced with a MagicMock that returns a fake response object. No real
network call to Gemini happens in this file (that's covered separately by
the Stage 3 smoke test).
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ocr_reasoning import OCRReasoningAssistant, extract_fragments


# ---------------------------------------------------------------------------
# extract_fragments -- pure function, no mocking needed
# ---------------------------------------------------------------------------


class TestExtractFragments:
    def test_boundary_prob_exactly_at_thresh_excluded(self):
        """Strict '>' comparison: prob == conf_thresh must be excluded."""
        result = extract_fragments([(None, "BOUNDARY", 0.35)], conf_thresh=0.35)
        assert result == []

    def test_boundary_prob_just_above_thresh_included(self):
        result = extract_fragments([(None, "BOUNDARY", 0.3501)], conf_thresh=0.35)
        assert result == [{"text": "BOUNDARY", "conf": 0.35}]

    def test_real_repo_example(self):
        ocr_results = [
            (None, "AV", 0.5402),
            (None, "7", 1.0),
            (None, "M7", 0.1003),
            (None, "AVE", 0.9761),
            (None, "WAY", 0.9757),
            (None, "EASHION", 0.6021),
            (None, "W30", 0.7144),
        ]
        result = extract_fragments(ocr_results)

        # M7 (prob=0.1003) falls below the default 0.35 threshold and must
        # be dropped; everything else survives in original encounter order.
        assert [f["text"] for f in result] == ["AV", "7", "AVE", "WAY", "EASHION", "W30"]
        assert len(result) == 6

        expected_confidences = {
            "AV": 0.54,
            "7": 1.0,
            "AVE": 0.98,
            "WAY": 0.98,
            "EASHION": 0.6,
            "W30": 0.71,
        }
        for frag in result:
            assert frag["conf"] == expected_confidences[frag["text"]]

        # bbox must never leak into the output dicts.
        for frag in result:
            assert "bbox" not in frag
            assert set(frag.keys()) == {"text", "conf"}

    def test_dedup_keeps_first_occurrence_confidence(self):
        ocr_results = [
            (None, "EXIT", 0.9),
            (None, "EXIT", 0.5),
            (None, "LAB", 0.99),
        ]
        result = extract_fragments(ocr_results)

        assert len(result) == 2
        exit_frag = next(f for f in result if f["text"] == "EXIT")
        assert exit_frag["conf"] == 0.9

    def test_empty_input_returns_empty_list(self):
        assert extract_fragments([]) == []

    def test_all_filtered_out_returns_empty_list(self):
        ocr_results = [
            (None, "A", 0.1),
            (None, "B", 0.2),
            (None, "C", 0.34),
        ]
        assert extract_fragments(ocr_results, conf_thresh=0.35) == []


# ---------------------------------------------------------------------------
# OCRReasoningAssistant.__init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError):
            OCRReasoningAssistant(provider="openai")

    def test_gemini_provider_constructs_mocked_client_no_network_call(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance

            assistant = OCRReasoningAssistant(provider="gemini", api_key="fake-key")

            mock_client_cls.assert_called_once_with(api_key="fake-key")
            assert assistant._client is mock_client_instance
            assert assistant.model == "gemini-flash-latest"

    def test_gemini_provider_custom_model(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()

            assistant = OCRReasoningAssistant(
                provider="gemini", model="gemini-2.5-pro", api_key="fake-key"
            )

            assert assistant.model == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# OCRReasoningAssistant._call_api
# ---------------------------------------------------------------------------


def _make_mocked_assistant(**kwargs):
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        assistant = OCRReasoningAssistant(provider="gemini", api_key="fake-key", **kwargs)
    return assistant


class TestCallApi:
    def test_valid_json_response_returns_matching_dict(self):
        assistant = _make_mocked_assistant()

        fake_response = MagicMock()
        fake_response.text = (
            '{"summary": "Exit sign ahead.", "category": "exit_sign", '
            '"resolved_text": "EXIT", "is_actionable": true, '
            '"confidence": "high", "discarded_fragments": []}'
        )
        assistant._client.models.generate_content = MagicMock(return_value=fake_response)

        result = assistant._call_api([{"text": "EXIT", "conf": 0.9}])

        assert result == {
            "summary": "Exit sign ahead.",
            "category": "exit_sign",
            "resolved_text": "EXIT",
            "is_actionable": True,
            "confidence": "high",
            "discarded_fragments": [],
        }

    def test_malformed_json_falls_back_to_safe_default(self):
        assistant = _make_mocked_assistant()

        fake_response = MagicMock()
        fake_response.text = "not valid json{{{"
        assistant._client.models.generate_content = MagicMock(return_value=fake_response)

        result = assistant._call_api([{"text": "EXIT", "conf": 0.9}])

        assert result["summary"] == ""
        assert result["category"] == "none"
        assert result["resolved_text"] == ""
        assert result["is_actionable"] is False
        assert result["confidence"] == "low"
        assert result["discarded_fragments"] == []
        assert result["_raw"] == "not valid json{{{"

    def test_generate_content_called_with_expected_model_and_config(self):
        from google.genai import types

        assistant = _make_mocked_assistant(model="gemini-2.5-pro")

        fake_response = MagicMock()
        fake_response.text = (
            '{"summary": "", "category": "none", "resolved_text": "", '
            '"is_actionable": false, "confidence": "low", "discarded_fragments": []}'
        )
        mock_generate = MagicMock(return_value=fake_response)
        assistant._client.models.generate_content = mock_generate

        assistant._call_api([{"text": "EXIT", "conf": 0.9}])

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-pro" == assistant.model

        config = call_kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# OCRReasoningAssistant.reason -- integration of extract_fragments + _call_api
# ---------------------------------------------------------------------------


class TestReason:
    def test_reason_calls_call_api_with_extracted_fragments(self):
        assistant = _make_mocked_assistant()

        canned = {
            "summary": "Exit sign ahead.",
            "category": "exit_sign",
            "resolved_text": "EXIT",
            "is_actionable": True,
            "confidence": "high",
            "discarded_fragments": [],
        }
        with patch.object(assistant, "_call_api", return_value=canned) as mock_call_api:
            ocr_results = [
                (None, "EXIT", 0.9),
                (None, "EXIT", 0.5),
                (None, "NOISE", 0.1),
            ]
            result = assistant.reason(ocr_results)

        expected_fragments = extract_fragments(ocr_results, conf_thresh=assistant.conf_thresh)
        mock_call_api.assert_called_once_with(expected_fragments)
        assert result == canned


# ---------------------------------------------------------------------------
# OCRReasoningAssistant.maybe_submit -- debounce decision logic (Stage 2)
#
# These tests patch force_submit directly via unittest.mock.patch.object, so
# no real threading/queueing is exercised here -- just the pure decision
# logic in maybe_submit() itself: content-change detection, the
# min_new_text_fragments threshold, and the min_seconds_between_calls
# cooldown, tested independently of each other.
# ---------------------------------------------------------------------------


class TestMaybeSubmitDebounce:
    def test_first_call_with_fragments_submits(self):
        """Fresh _last_submitted_texts=set() differs from any non-empty content,
        so the very first call should submit (given enough fragments to clear
        the default min_new_text_fragments=2)."""
        assistant = _make_mocked_assistant(min_seconds_between_calls=0)

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            assistant.maybe_submit([(None, "A", 0.9), (None, "B", 0.9)])

        mock_force_submit.assert_called_once()

    def test_identical_fragments_submitted_twice_does_not_resubmit(self):
        assistant = _make_mocked_assistant(min_seconds_between_calls=0)
        ocr_results = [(None, "A", 0.9), (None, "B", 0.9)]

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            assistant.maybe_submit(ocr_results)
            assert mock_force_submit.call_count == 1

            # Same content again -- current_texts == _last_submitted_texts, so
            # this should be a no-op without even reaching force_submit.
            assistant.maybe_submit(ocr_results)
            assert mock_force_submit.call_count == 1

    def test_new_fragment_count_below_min_new_text_fragments_blocks_submit(self):
        assistant = _make_mocked_assistant(min_seconds_between_calls=0, min_new_text_fragments=3)

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            # Baseline call: 3 fragments clears the min_new_text_fragments=3 bar.
            assistant.maybe_submit([(None, "A", 0.9), (None, "B", 0.9), (None, "C", 0.9)])
            assert mock_force_submit.call_count == 1

            # Only one new fragment ("D") vs. the baseline -- below the
            # required 3 new fragments, so this must NOT submit.
            assistant.maybe_submit(
                [(None, "A", 0.9), (None, "B", 0.9), (None, "C", 0.9), (None, "D", 0.9)]
            )
            assert mock_force_submit.call_count == 1

    def test_cooldown_not_elapsed_blocks_submit_despite_enough_new_content(self):
        assistant = _make_mocked_assistant(min_seconds_between_calls=100.0, min_new_text_fragments=2)

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            # Baseline call: _last_call_time starts at 0.0, so the 100s cooldown
            # trivially "elapsed" (time.time() - 0.0 is huge) and this submits.
            assistant.maybe_submit([(None, "A", 0.9), (None, "B", 0.9)])
            assert mock_force_submit.call_count == 1

            # Simulate the worker having just processed that call (this is
            # normally done by _worker_loop, not maybe_submit/force_submit
            # directly, so we set it explicitly here to test the cooldown
            # check in isolation).
            assistant._last_call_time = time.time()

            # Enough new content (2 new fragments), but the 100s cooldown has
            # not elapsed -- must NOT submit.
            assistant.maybe_submit(
                [(None, "A", 0.9), (None, "B", 0.9), (None, "C", 0.9), (None, "D", 0.9)]
            )
            assert mock_force_submit.call_count == 1

    def test_cooldown_elapsed_with_enough_new_content_submits(self):
        assistant = _make_mocked_assistant(min_seconds_between_calls=0, min_new_text_fragments=2)

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            assistant.maybe_submit([(None, "A", 0.9), (None, "B", 0.9)])
            assert mock_force_submit.call_count == 1

            # Explicitly force the cooldown to read as elapsed, independent of
            # real wall-clock timing.
            assistant._last_call_time = 0.0

            assistant.maybe_submit(
                [(None, "A", 0.9), (None, "B", 0.9), (None, "C", 0.9), (None, "D", 0.9)]
            )
            assert mock_force_submit.call_count == 2

    def test_fragment_below_conf_thresh_never_triggers_submit_on_its_own(self):
        """A fragment at prob=0.1 (below the default 0.35 conf_thresh) must not
        appear in current_texts at all, matching extract_fragments' own
        filtering semantics."""
        assistant = _make_mocked_assistant(min_seconds_between_calls=0, min_new_text_fragments=1)

        with patch.object(assistant, "force_submit", return_value=True) as mock_force_submit:
            assistant.maybe_submit([(None, "WEAK", 0.1)])

        mock_force_submit.assert_not_called()
        # current_texts must have been the empty set, equal to the initial
        # _last_submitted_texts -- not a 1-fragment set that merely failed
        # the min_new_text_fragments check.
        assert assistant._last_submitted_texts == set()

    def test_last_submitted_texts_not_updated_when_force_submit_returns_false(self):
        """force_submit() returning False (e.g. dropped because the worker was
        busy) must NOT update _last_submitted_texts, so the same content is
        attempted again on a later call rather than being silently
        suppressed forever."""
        assistant = _make_mocked_assistant(min_seconds_between_calls=0, min_new_text_fragments=2)
        ocr_results = [(None, "A", 0.9), (None, "B", 0.9)]

        with patch.object(assistant, "force_submit", return_value=False) as mock_force_submit:
            assistant.maybe_submit(ocr_results)
            assert mock_force_submit.call_count == 1
            assert assistant._last_submitted_texts == set()

            # Same content again: since _last_submitted_texts was never
            # updated, this must attempt to submit again rather than treating
            # it as unchanged content.
            assistant.maybe_submit(ocr_results)
            assert mock_force_submit.call_count == 2


# ---------------------------------------------------------------------------
# OCRReasoningAssistant threading lifecycle (Stage 2) -- real threads, but
# the Gemini client is mocked at construction and reason() is monkeypatched
# so nothing here ever touches the network.
# ---------------------------------------------------------------------------


class TestThreadingLifecycle:
    def test_start_then_force_submit_updates_summary_and_clears_busy(self):
        assistant = _make_mocked_assistant(min_seconds_between_calls=0)
        fake_result = {
            "summary": "Fake summary.",
            "category": "informational",
            "resolved_text": "FAKE",
            "is_actionable": True,
            "confidence": "high",
            "discarded_fragments": [],
        }

        assistant.start()
        try:
            with patch.object(assistant, "reason", return_value=fake_result):
                submitted = assistant.force_submit([(None, "FAKE", 0.9)])
                assert submitted is True

                # Poll (bounded, no fixed sleep) until the worker has both
                # written the new summary AND cleared _busy -- avoids any
                # race on exactly when the busy flag flips relative to the
                # summary write inside _worker_loop.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if assistant.latest_summary() == fake_result and not assistant.is_busy():
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail("worker did not process force_submit within 5s")

                assert assistant.latest_summary() == fake_result
                assert assistant.is_busy() is False
        finally:
            assistant.stop()
            assistant._thread.join(timeout=2.0)

    def test_stop_causes_background_thread_to_exit(self):
        assistant = _make_mocked_assistant()
        assistant.start()
        assert assistant._thread.is_alive()

        assistant.stop()

        deadline = time.time() + 5.0
        while assistant._thread.is_alive() and time.time() < deadline:
            time.sleep(0.01)

        assert assistant._thread.is_alive() is False

    def test_exception_in_reason_is_caught_and_thread_stays_alive(self):
        """An exception raised inside reason() must be swallowed by
        _worker_loop's try/except, with _busy still reset to False in the
        finally block -- and the thread must remain usable for a subsequent
        submit afterward."""
        assistant = _make_mocked_assistant(min_seconds_between_calls=0)
        called_event = threading.Event()

        def raising_reason(ocr_results):
            called_event.set()
            raise RuntimeError("boom")

        assistant.start()
        try:
            with patch.object(assistant, "reason", side_effect=raising_reason):
                submitted = assistant.force_submit([(None, "X", 0.9)])
                assert submitted is True

                assert called_event.wait(timeout=5.0), "worker never invoked reason()"

                deadline = time.time() + 5.0
                while assistant.is_busy() and time.time() < deadline:
                    time.sleep(0.01)

                assert assistant.is_busy() is False
                assert assistant._thread.is_alive()

            # The thread must still be functional afterward: a subsequent
            # submit with a working reason() should process normally.
            fake_result = {
                "summary": "Recovered.",
                "category": "informational",
                "resolved_text": "RECOVERED",
                "is_actionable": True,
                "confidence": "high",
                "discarded_fragments": [],
            }
            with patch.object(assistant, "reason", return_value=fake_result):
                assistant.force_submit([(None, "Y", 0.9)])

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if assistant.latest_summary() == fake_result and not assistant.is_busy():
                        break
                    time.sleep(0.01)
                else:
                    pytest.fail("worker did not recover after a swallowed exception within 5s")

                assert assistant.latest_summary() == fake_result
        finally:
            assistant.stop()
            assistant._thread.join(timeout=2.0)

    def test_force_submit_returns_false_and_does_not_block_when_busy(self):
        assistant = _make_mocked_assistant()
        # Bypassing the lock here is fine for a test: we only need _busy to
        # read True from force_submit's perspective.
        assistant._busy = True

        result = assistant.force_submit([(None, "X", 0.9)])

        assert result is False


# ---------------------------------------------------------------------------
# Fresh (unstarted) OCRReasoningAssistant -- initial state before any submit
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_fresh_assistant_reports_safe_default_summary_and_is_not_busy(self):
        assistant = _make_mocked_assistant()

        assert assistant.latest_summary() == {
            "summary": "",
            "category": "none",
            "resolved_text": "",
            "is_actionable": False,
            "confidence": "low",
            "discarded_fragments": [],
        }
        assert assistant.is_busy() is False
