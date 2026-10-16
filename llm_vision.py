#################
# llm_vision.py
# Created by Sahir Abrar July 2026
# Last Updated: October 2026 by Sahir Abrar
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

# HOW THIS FILE WORKS (the big picture):
# The main loop in navigation.py grabs camera frames many times per second.
# Asking an AI model "what do you see?" over the internet can take a second or
# more, so if we did that in the main loop, the whole program would freeze
# while waiting. Instead we use a "producer / consumer" setup:
#   - The main loop (producer) drops a frame into a tiny queue.
#   - A background thread (consumer) picks it up, sends it to the AI, and
#     stores the answer when it comes back.
#   - The main loop can check for the newest answer at any time without waiting.


# --- Imports -------------------------------------------------------------
import base64  # Turns raw image bytes into text so they can be sent over the internet
import json  # Lets us turn the AI's text reply into a Python dictionary
import os  # Used to read environment variables (like our API key)
import queue  # A thread-safe "line" for passing frames between threads
import threading  # Lets us run code in the background
import time  # Used for timing how long calls take and for rate limiting
from dotenv import load_dotenv  # Reads secrets out of a .env file
from pathlib import Path  # A clean way to work with file paths

import cv2  # OpenCV, used here to resize and compress images

# Load the .env file that sits next to this script. This is where
# GEMINI_API_KEY lives, so we never have to hard-code the key in our code.
# (Path(__file__) = this file, .resolve().parent = the folder it's in)
load_dotenv(Path(__file__).resolve().parent / ".env")

# The instructions we give the AI along with every image. This is basically
# the "personality" and rules for the model. Each sentence limits what it can
# say so the answers stay short, useful, and safe for someone who can't see.
DEFAULT_PROMPT = (
    "You are a real-time navigation assistant for a blind or low-vision "
    "person walking indoors. Look at the image and identify only what is "
    "clearly visible and relevant to safely moving forward. "
    "Do not mention colors, lighting, or aesthetics. "
    "Do not estimate exact distances — use only 'near' or 'far' and a "
    "rough direction (left/right/ahead) if relevant."
)



class LLMVisionAssistant:
    # A class bundles data (like the latest answer) and functions (like
    # start/stop) into one object, so navigation.py can just make one
    # LLMVisionAssistant and call its methods.
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
        # These are the settings you can tweak when you create the assistant.
        # Everything below just saves them on `self` so other methods can use them.
        self.provider = provider  # Which AI company to use ("gemini" is the one actually wired up)
        self.prompt = prompt  # The instructions sent with each image
        self.interval_frames = interval_frames  # Only try a call every N frames (90 frames is ~3 sec at 30 fps)
        self.min_seconds_between_calls = min_seconds_between_calls  # Hard cooldown so we don't spam the API
        self.jpeg_quality = jpeg_quality  # 0-100, lower = smaller file = faster upload, but blurrier
        self.max_side = max_side  # Longest image side in pixels; bigger images get shrunk to this

        # A queue is a line of items waiting to be processed. maxsize=1 means
        # only ONE frame can wait at a time. That's on purpose: if the AI is
        # slow we don't want a pile of stale frames, we only care about the newest one.
        self._frame_queue = queue.Queue(maxsize=1)

        # A lock stops two threads from touching the same variable at the same
        # moment (which can cause weird bugs). Anytime we read/write
        # _description or _busy we wrap it in `with self._state_lock:`.
        self._state_lock = threading.Lock()

        # The most recent answer from the AI. Starts out empty so the main
        # loop always has something valid to read, even before the first call.
        self._description = None

        self._busy = False  # True while an API call is in progress
        self._last_call_time = 0.0  # Timestamp of the last call (0 = "a long time ago")
        self._stop_flag = False  # Set to True to tell the background thread to quit
        self._thread = None  # Will hold the background thread once start() is called

        # Set up the connection to the AI service.
        if provider == "gemini":
            # Imported here (not at the top) so people who don't use Gemini
            # don't need the library installed.
            from google import genai
            self.model = model or "gemini-3.5-flash-lite" #For latest model use "gemini-flash-latest"
            # Use the key passed in, or fall back to the one from the .env file
            self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        else:
            # NOTE: right now only "gemini" is set up in __init__, so any other
            # provider ends up here and crashes on purpose with a clear message.
            raise ValueError(f"Unknown provider: {provider!r} (expected 'openai', 'anthropic', or 'gemini')")

    # --- lifecycle -----------------------------------------------------
    # These two methods turn the background worker on and off.

    def start(self):
        # daemon=True means this thread will automatically die when the main
        # program exits, so it can't keep the program hanging around.
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        return self  # Returning self lets you chain it: assistant = LLMVisionAssistant(...).start()

    def stop(self):
        self._stop_flag = True
        # The worker is usually sleeping, waiting for a frame to arrive. We
        # drop a `None` in the queue as a "wake up and quit" signal.
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            # If the queue is already full the worker will wake up anyway,
            # and then it'll see _stop_flag, so we can safely ignore this.
            pass

    # --- public API ------------------------------------------------------
    # These are the methods navigation.py actually calls.

    def maybe_submit(self, frame, frame_num):
        # Call this on EVERY frame. It decides whether it's actually time to
        # send one to the AI.

        # `%` is the remainder after dividing. This is only 0 once every
        # `interval_frames` frames (e.g. frames 0, 90, 180...), so we skip all the others.
        if frame_num % self.interval_frames != 0:
            return
        # Even on the right frame, skip if we called the API too recently.
        if time.time() - self._last_call_time < self.min_seconds_between_calls:
            return
        self.force_submit(frame)

    def force_submit(self, frame, context="", request_id=None):
        # Sends a frame to the AI right now (ignoring the frame-count timer).
        # `context` is extra text to add to the prompt, and `request_id` is a
        # tag so the caller can match an answer back to the question it asked.
        # Returns True if the frame was accepted, False if it was rejected.

        # If the AI is already working on something, don't pile on.
        with self._state_lock:
            if self._busy:
                print("already busy")
                return False
        try:
            # put_nowait = "add to the queue, but don't wait if it's full".
            # We pack everything the worker needs into one tuple.
            self._frame_queue.put_nowait((frame, context, request_id))
            return True
        except queue.Full:
            # Another frame is already waiting, so we drop this one.
            print("queue is full")
            return False

    def latest_description(self):
        """Returns a dict: {"hazards": [...], "objects": [...]}"""
        # Hand back the newest answer. The lock makes sure we never read it
        # halfway through the worker thread updating it.
        with self._state_lock:
            return self._description

    def is_busy(self):
        # Lets the main loop ask "is the AI still thinking?"
        with self._state_lock:
            return self._busy

    # --- internals -------------------------------------------------------
    # Methods starting with an underscore are "private": they're only meant
    # to be used inside this class, not called from other files.

    def _preprocess(self, frame):
        # Shrinks big images so they upload faster and cost less.
        # NOTE: this isn't currently called anywhere in _call_api, so frames
        # are sent at full size. Call self._preprocess(frame) there to use it.
        h, w = frame.shape[:2]  # Image height and width in pixels
        scale = self.max_side / max(h, w)  # How much we'd need to shrink the longest side
        if scale < 1.0:  # Only shrink if the image is bigger than max_side, never enlarge
            # Multiply both sides by the same scale so the picture doesn't get squished
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    def _worker_loop(self):
        # This is the function the background thread runs forever (until stopped).
        while not self._stop_flag:
            # Get the next frame from the queue
            # .get() WAITS here until something shows up, so the thread uses
            # basically no CPU while idle.
            item = self._frame_queue.get()
            if item is None:
                # That's the "quit" signal from stop()
                break
            # Process the frame, grabbing the necessary information
            # Unpack the tuple we built in force_submit
            frame, context, request_id = item
            with self._state_lock:
                self._busy = True  # Tell everyone we're now working
            # Chained assignment: both _last_call_time and t0 get the same
            # timestamp. t0 is just for measuring how long the call takes.
            self._last_call_time = t0 = time.time()
            try:
                print(f"[llm_vision] Calling Gemini model={self.model}")
                description = self._call_api(frame, context)  # The slow network call happens here
                print("[llm_vision] response:", description)
                # Attach the request_id so the caller knows which request this answer belongs to
                description["request_id"] = request_id
                print(f"[llm_vision] {time.time() - t0:.2f}s")  # How long the call took, to 2 decimals
                # Save the new answer where the main loop can read it
                with self._state_lock:
                    self._description = description
            except Exception as e:
                # If the network is down, the key is wrong, etc., we print the
                # error but keep the thread alive so the next frame can still work.
                print(f"[llm_vision] API call failed: {e}")
            finally:
                # `finally` ALWAYS runs, whether the call worked or crashed.
                # That guarantees we never get stuck permanently "busy".
                with self._state_lock:
                    self._busy = False

    def _encode_jpeg_b64(self, frame):
        # APIs can't take a raw OpenCV image, so we convert it in two steps:
        # 1) compress it to JPEG bytes, 2) turn those bytes into a base64 text string.
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf).decode("utf-8")  # bytes -> base64 -> normal string

    def _call_api(self, frame, context=""):
        # Sends one image to the chosen AI provider and returns its answer.
        b64 = self._encode_jpeg_b64(frame)

        # NOTE: the openai and anthropic branches below can't run right now,
        # because __init__ only creates a client for "gemini". They're left
        # in as a starting point if you add those providers later.
        if self.provider == "openai":
            # OpenAI wants the image as a "data URL" (text that contains the whole image)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=60,  # Cap the reply length so it stays short and fast
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            # Anthropic takes the base64 image directly, with its type labelled
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
            # Gemini wants real image bytes, so we undo the base64 step
            image_bytes = base64.b64decode(b64)
            # Add any extra context to the end of the prompt (blank line in between)
            prompt = self.prompt + ("\n\n" + context if context else "")
            # The "schema" is a template that forces the AI to reply in an exact
            # JSON shape, instead of free-form text. That makes the answer easy
            # for our code to read and use.
            schema = { #Schematic LLM will follow
                "type": "OBJECT",
                "properties": {
                    "action": {
                        "type": "String",
                        "description": "Either: 'turn' or 'target'",
                    },
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "description": "Bounding box coordinates [x1, y1, x2, y2] for the selected target. Preferred over angle. SHOULD MATCH A BBOX GIVEN IN THE PROMPT",
                    },
                    "angle": {
                        "type": "NUMBER",
                        "description": "The angle to turn, in degrees. Negative values mean left, positive values mean right.",
                    },
                    "reasoning": {
                        "type": "String",
                        "description": "The reasoning behind the selected action.",
                    }
                },
            }
            # NOTE: this schema asks for action/bbox/angle/reasoning, but the
            # header (and latest_description) talk about "hazards"/"objects".
            # Those two should be kept in sync or the output won't match.

            # "required" means the AI MUST include "action" in every reply
            schema["required"] = ["action"]

            # Actually send the image + prompt to Gemini and wait for the reply
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig( #Constraints the LLM to this format
                    response_mime_type="application/json",  # Reply as JSON, not plain text
                    response_schema=schema,  # ...and follow the template above
                    temperature=0.1,  # Low = more predictable/consistent, high = more creative. We want consistent.
                ),
            )
            try:
                print("description received")
                # response.text is a JSON string; json.loads converts it to a Python dict
                return json.loads(response.text)
            except (json.JSONDecodeError, TypeError):
                # If the reply isn't valid JSON, play it safe and tell the
                # robot/user to stop rather than guess.
                return {"action": "Stop"} #If the LLM fails to return a valid response