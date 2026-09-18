#################
# ocr_reasoning.py
# Created by Sahir Abrar September 2026
# Last Updated: September 2026 by Sahir Abrar
# Last Change:
# - Stage 2: added threaded/debounced async wrapper (maybe_submit/
#   force_submit, background worker loop, latest_summary/is_busy/start/stop)
#   mirroring llm_vision.py's LLMVisionAssistant. reason()/_call_api()/
#   extract_fragments() are unchanged and still directly callable
#   synchronously.
# - Initial version (Stage 1): pure fragment extraction + synchronous Gemini
#   text-reasoning call.
# Description: Turns noisy/fragmented EasyOCR readtext() output into
# structured, actionable reasoning for a blind/low-vision navigation
# assistant. Unlike llm_vision.py (which uploads a JPEG frame to Gemini's
# vision API), this module makes a text-only Gemini call over the OCR text
# fragments themselves, which is much cheaper — no image bytes are sent.
#################


import json
import os
import queue
import threading
import time

DEFAULT_REASONING_PROMPT = (
    "You are reasoning over text fragments extracted by OCR from a scene, "
    "for a blind or low-vision person walking indoors or outdoors. The "
    "fragments come from a live camera feed and are often noisy, garbled, "
    "partial, or duplicated — individual fragments may be missing letters, "
    "split across multiple entries, or simply wrong. "
    "Your job: "
    "1) Merge and correct the fragments into the single best-guess coherent "
    "phrase, but only where you are reasonably confident — do not invent "
    "words or meaning that the fragments do not support. "
    "2) Classify what the resolved text represents. "
    "3) Judge whether this is worth surfacing to the user right now (i.e. "
    "actionable, relevant to navigation or safety, not stale or irrelevant "
    "clutter). "
    "4) Produce one short, calm, spoken-style sentence suitable for "
    "text-to-speech. "
    "Never output anything alarming or exaggerated. If the fragments do not "
    "resolve to anything meaningful — for example they are just noise, "
    "random characters, or unrelated scraps — say so plainly: return an "
    "empty summary, an empty resolved_text, category 'none', and "
    "is_actionable false. Do not hallucinate meaning from garbage OCR noise."
)


def extract_fragments(ocr_results, conf_thresh=0.35):
    """ocr_results: list of (bbox, text, prob) tuples, e.g. EasyOCR's readtext() output.
    Returns a deduplicated list of {"text": str, "conf": float} dicts, filtered to
    prob > conf_thresh, confidence rounded to 2 decimals, in original encounter order
    (first occurrence wins on dedup)."""
    seen = set()
    fragments = []
    for _bbox, text, prob in ocr_results:
        if not (prob > conf_thresh):
            continue
        if text in seen:
            continue
        seen.add(text)
        fragments.append({"text": text, "conf": round(prob, 2)})
    return fragments


class OCRReasoningAssistant:
    def __init__(
        self,
        provider="gemini",
        model=None,
        api_key=None,
        prompt=DEFAULT_REASONING_PROMPT,
        conf_thresh=0.35,
        min_seconds_between_calls=4.0,
        min_new_text_fragments=2,
    ):
        self.provider = provider
        self.prompt = prompt
        self.conf_thresh = conf_thresh
        # Higher than LLMVisionAssistant's 2.0 default — text reasoning here
        # is less time-critical than hazard vision, per the architecture plan.
        self.min_seconds_between_calls = min_seconds_between_calls
        # Debounce trigger threshold: how many *new* fragment texts (vs. the
        # last submitted set) are required before we bother calling the API.
        self.min_new_text_fragments = min_new_text_fragments

        # --- Stage 2 threading/debounce state (mirrors LLMVisionAssistant) ---
        self._frame_queue = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        # Safe-default shape, matching _call_api's own fallback dict minus "_raw".
        self._summary = {
            "summary": "",
            "category": "none",
            "resolved_text": "",
            "is_actionable": False,
            "confidence": "low",
            "discarded_fragments": [],
        }
        self._busy = False
        self._last_call_time = 0.0
        self._stop_flag = False
        self._thread = None
        # Tracks the fragment-text-set last actually sent to the API, for
        # content-change debounce in maybe_submit().
        self._last_submitted_texts = set()

        if provider == "gemini":
            from google import genai
            self.model = model or "gemini-flash-latest"
            self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider!r} (expected 'gemini')")

    # --- lifecycle -------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_flag = True
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        # NOTE: LLMVisionAssistant.stop() doesn't join its thread either — not
        # joining here mirrors that exactly and keeps shutdown behavior
        # consistent across the codebase's two LLM assistants. A caller that
        # wants a hard guarantee the thread has exited can still do
        # `assistant._thread.join(timeout=...)` after calling stop().

    # --- public API ------------------------------------------------------

    def reason(self, ocr_results):
        """Convenience wrapper: extract_fragments() + _call_api() in one synchronous call."""
        fragments = extract_fragments(ocr_results, conf_thresh=self.conf_thresh)
        return self._call_api(fragments)

    def force_submit(self, ocr_results):
        """Queue raw ocr_results for background reasoning, bypassing debounce.

        Unlike LLMVisionAssistant.force_submit(frame), which JPEG-encodes the
        frame before queuing (an expensive step worth doing once up front),
        there's no equivalent expensive preprocessing for OCR results here —
        extract_fragments() is cheap and runs inside the worker loop via
        reason(), not here — so we just queue the raw list to keep this
        method cheap/non-blocking.
        """
        with self._state_lock:
            if self._busy:
                return False
        try:
            self._frame_queue.put_nowait(ocr_results)
            return True
        except queue.Full:
            return False

    def maybe_submit(self, ocr_results, frame_num=None):
        """Debounced submit: only calls force_submit() when there's enough new
        text content and the cooldown has elapsed.

        frame_num is accepted only for API-shape symmetry with
        LLMVisionAssistant.maybe_submit(frame, frame_num) — this design
        intentionally debounces on content change + a time cooldown rather
        than a frame-count interval, per the architecture plan, so frame_num
        is unused here.
        """
        # Cheap text-only extraction for comparison purposes; the full
        # extract_fragments() dict-building work happens once inside the
        # worker after a real submit (via reason()).
        current_texts = {text for (_bbox, text, prob) in ocr_results if prob > self.conf_thresh}

        if current_texts == self._last_submitted_texts:
            return  # nothing new

        new_fragment_count = len(current_texts - self._last_submitted_texts)
        if new_fragment_count < self.min_new_text_fragments:
            return  # not enough new content yet

        if time.time() - self._last_call_time < self.min_seconds_between_calls:
            return  # cooldown not elapsed

        submitted = self.force_submit(ocr_results)
        if submitted:
            # Only update tracking state on a successful submit, not a
            # dropped-due-to-busy one, so a busy-drop doesn't silently
            # suppress a legitimate future retry of the same content.
            self._last_submitted_texts = current_texts

    def latest_summary(self):
        """Returns the latest reasoning result dict (see _call_api's schema)."""
        with self._state_lock:
            return self._summary

    def is_busy(self):
        with self._state_lock:
            return self._busy

    # --- internals -------------------------------------------------------

    def _worker_loop(self):
        while not self._stop_flag:
            ocr_results = self._frame_queue.get()
            if ocr_results is None:
                break
            with self._state_lock:
                self._busy = True
            self._last_call_time = time.time()
            try:
                summary = self.reason(ocr_results)
                with self._state_lock:
                    self._summary = summary
            except Exception as e:
                print(f"[ocr_reasoning] API call failed: {e}")
            finally:
                with self._state_lock:
                    self._busy = False

    def _call_api(self, fragments):
        # NOTE on STRING enums: google.genai.types.Schema *does* support a real
        # enum constraint for STRING fields (verified by reading types.Schema
        # in the installed google-genai package: it has an `enum: Optional[list[str]]`
        # field, used together with `format: "enum"` — e.g.
        # {"type": "STRING", "format": "enum", "enum": ["EAST", "NORTH", ...]}
        # per that field's docstring). This is real, working enum support, not
        # a guess — so it's used below for `category` and `confidence` rather
        # than relying on description text alone.
        schema = {
            "type": "OBJECT",
            "properties": {
                "summary": {
                    "type": "STRING",
                    "description": "One short spoken-style sentence for text-to-speech, or an empty string if the fragments don't resolve to anything meaningful.",
                },
                "category": {
                    "type": "STRING",
                    "format": "enum",
                    "enum": [
                        "exit_sign",
                        "room_label",
                        "hazard_warning",
                        "directional_sign",
                        "informational",
                        "none",
                    ],
                    "description": "What the resolved text represents. Must be one of: exit_sign, room_label, hazard_warning, directional_sign, informational, none (use 'none' if nothing meaningful was resolved).",
                },
                "resolved_text": {
                    "type": "STRING",
                    "description": "The merged/corrected best-guess phrase built from the fragments, or an empty string if nothing meaningful was resolved.",
                },
                "is_actionable": {
                    "type": "BOOLEAN",
                    "description": "True if this should be surfaced to the user right now.",
                },
                "confidence": {
                    "type": "STRING",
                    "format": "enum",
                    "enum": ["low", "medium", "high"],
                    "description": "The model's own confidence in this reasoning. Must be one of: low, medium, high.",
                },
                "discarded_fragments": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Fragments judged to be noise and excluded from resolved_text, for debugging/logging.",
                },
            },
            "required": [
                "summary",
                "category",
                "resolved_text",
                "is_actionable",
                "confidence",
                "discarded_fragments",
            ],
        }

        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=[self.prompt, json.dumps(fragments)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
            ),
        )
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            return {
                "summary": "",
                "category": "none",
                "resolved_text": "",
                "is_actionable": False,
                "confidence": "low",
                "discarded_fragments": [],
                "_raw": response.text,
            }
