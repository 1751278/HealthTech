"""
llm_vision.py
Non-blocking vision-LLM assistant for the HealthTech navigation stack.
 
Runs image-description queries against OpenAI's or Anthropic's vision APIs
on a background thread so a slow network call never stalls the main
capture/steering loop in navigation.py.
 
Usage (see navigation.py integration notes for the full wiring):
 
    from llm_vision import LLMVisionAssistant
 
    assistant = LLMVisionAssistant(provider="openai", interval_frames=90)
    assistant.start()
 
    # inside the main while-loop, once per frame:
    assistant.maybe_submit(frame, frame_num)
    description = assistant.latest_description()
 
    # on shutdown:
    assistant.stop()
"""
 
import base64
import os
import queue
import threading
import time
 
import cv2
 
DEFAULT_PROMPT = (
    "In one short sentence, describe what's directly ahead that a person "
    "navigating indoors without sight would need to know. Mention hazards, "
    "doorways, stairs, or signage if visible. Be concise and concrete."
)
 
 
class LLMVisionAssistant:
    def __init__(
        self,
        provider="openai",              # "openai" or "anthropic"
        model=None,
        api_key=None,
        prompt=DEFAULT_PROMPT,
        interval_frames=90,             # min frames between auto-triggered calls
        min_seconds_between_calls=2.0,  # hard floor regardless of frame interval
        jpeg_quality=70,
        max_side=512,                   # downscale long edge before sending
    ):
        self.provider = provider
        self.prompt = prompt
        self.interval_frames = interval_frames
        self.min_seconds_between_calls = min_seconds_between_calls
        self.jpeg_quality = jpeg_quality
        self.max_side = max_side
 
        self._frame_queue = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self._description = ""
        self._busy = False
        self._last_call_time = 0.0
        self._stop_flag = False
        self._thread = None
 

        if provider == "gemini":
            from google import genai
            self.model = model or "gemini-3.5-flash"
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
            self._frame_queue.put_nowait(None)  # wake the worker so it can exit
        except queue.Full:
            pass
 
    # --- public API ------------------------------------------------------
 
    def maybe_submit(self, frame, frame_num):
        """Call once per loop iteration. Internally decides whether it's time
        to actually fire a request, so it's always safe to call unconditionally."""
        if frame_num % self.interval_frames != 0:
            return
        if time.time() - self._last_call_time < self.min_seconds_between_calls:
            return
        self.force_submit(frame)
 
    def force_submit(self, frame):
        """Submit a frame right now, bypassing the interval check (e.g. for a
        manual 'describe what's in front of me' keypress/voice command)."""
        with self._state_lock:
            if self._busy:
                return False
        try:
            self._frame_queue.put_nowait(self._preprocess(frame))
            return True
        except queue.Full:
            return False  # previous request still in flight, dropped this one
 
    def latest_description(self):
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
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    self.prompt,
                ],
            )
            return response.text.strip()