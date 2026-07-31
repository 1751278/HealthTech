#################
# llm_vision.py
# Created by Sahir Abrar July 2026
# Last Updated: July 2026 by Sahir Abrar
# Last Change:
# - Switched to structured JSON output (hazards/objects) via Gemini response_schema
# - Narrowed prompt to hazards + notable objects only, dropped free-text description
# Description: Non-blocking vision-LLM assistant for the HealthTech navigation
# stack. Runs image-description queries against Gemini's (or OpenAI's/Anthropic's)
# vision APIs on a background thread so a slow network call never stalls the
# main capture/steering loop in navigation.py.
# TODO:
# - long api keys
#################


import base64
import json
import os
import queue
import threading
import time

import cv2

DEFAULT_PROMPT = (
    "You are a real-time navigation assistant for a blind or low-vision "
    "person walking indoors. Look at the image and identify only what is "
    "clearly visible and relevant to safely moving forward. "
    "Do not mention colors, lighting, or aesthetics. "
    "Do not estimate exact distances — use only 'near' or 'far' and a "
    "rough direction (left/right/ahead) if relevant."
)


class LLMVisionAssistant:
    def __init__(
        self,
        provider="openai",
        model=None,
        api_key=None,
        prompt=DEFAULT_PROMPT,
        interval_frames=90,
        min_seconds_between_calls=2.0,
        jpeg_quality=70,
        max_side=512,
    ):
        self.provider = provider
        self.prompt = prompt
        self.interval_frames = interval_frames
        self.min_seconds_between_calls = min_seconds_between_calls
        self.jpeg_quality = jpeg_quality
        self.max_side = max_side

        self._frame_queue = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._description = {"hazards": [], "objects": []}
        self._busy = False
        self._last_call_time = 0.0
        self._stop_flag = False
        self._thread = None

        if provider == "gemini":
            from google import genai
            self.model = model or "gemini-flash-latest"
            self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider!r} (expected 'openai', 'anthropic', or 'gemini')")

    # --- lifecycle -----------------------------------------------------

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

    # --- public API ------------------------------------------------------

    def maybe_submit(self, frame, frame_num):
        if frame_num % self.interval_frames != 0:
            return
        if time.time() - self._last_call_time < self.min_seconds_between_calls:
            return
        self.force_submit(frame)

    def force_submit(self, frame):
        with self._state_lock:
            if self._busy:
                return False
        try:
            self._frame_queue.put_nowait(self._preprocess(frame))
            return True
        except queue.Full:
            return False

    def latest_description(self):
        """Returns a dict: {"hazards": [...], "objects": [...]}"""
        with self._state_lock:
            return self._description

    def is_busy(self):
        with self._state_lock:
            return self._busy

    # --- internals -------------------------------------------------------

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    def _worker_loop(self):
        while not self._stop_flag:
            frame = self._frame_queue.get()
            if frame is None:
                break
            with self._state_lock:
                self._busy = True
            self._last_call_time = time.time()
            try:
                description = self._call_api(frame)
                with self._state_lock:
                    self._description = description
            except Exception as e:
                print(f"[llm_vision] API call failed: {e}")
            finally:
                with self._state_lock:
                    self._busy = False

    def _encode_jpeg_b64(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf).decode("utf-8")

    def _call_api(self, frame):
        b64 = self._encode_jpeg_b64(frame)

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=60,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=60,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": self.prompt},
                    ],
                }],
            )
            return response.content[0].text.strip()

        elif self.provider == "gemini":
            from google.genai import types
            image_bytes = base64.b64decode(b64)

            schema = {
                "type": "OBJECT",
                "properties": {
                    "hazards": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Immediate dangers: steps, drop-offs, low overhangs, obstacles in the direct path, wet/uneven surfaces.",
                    },
                    "objects": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Notable non-hazardous objects relevant to navigation: doors, furniture, signage, walls, open paths.",
                    },
                },
                "required": ["hazards", "objects"],
            }

            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    self.prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            try:
                return json.loads(response.text)
            except (json.JSONDecodeError, TypeError):
                return {"hazards": [], "objects": [], "_raw": response.text}