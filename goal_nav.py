#################
# goal_nav.py
# Description: Goal-directed navigation layer for navigation.py.
#   1. Gather context (depth zones, VO, IMU stub) and ask the LLM for the next waypoint.
#   2. Track the waypoint's bounding box with an OpenCV tracker (catching up on the
#      frames that passed while the LLM was thinking).
#   3. Hand (direction, confidence) to combine_steer(); depth avoidance stays the safety layer.
#   4. When the waypoint is reached (box grew + depth close) or lost, ask the LLM again.
# Also contains: Speaker (pyttsx3 TTS thread) and VoiceGoal (push-to-talk goal capture).
# Deps: pip install opencv-contrib-python pyttsx3 SpeechRecognition pyaudio
#################
 
import collections
import math
import queue
import threading
import time
 
import cv2
import numpy as np
print("[DEBUG] goal_nav.py MODULE LOADED")
# ---------------------------------------------------------------- tunables ---
TRACK_SCALE = 0.5            # trackers run on half-res frames (cheaper)
MAX_REPLAY_PER_FRAME = 6     # tracker updates per main-loop frame while catching up
LOST_LIMIT = 8               # consecutive failed tracker updates before we requery
WAIT_TIMEOUT_S = 8.0         # give up on an LLM call after this long
REQUERY_S = 10.0             # sanity requery while tracking
RETRY_S = 1.5                # delay before re-asking when the target isn't visible
ARRIVE_GROW = 2.5            # waypoint box area vs. its initial area
ARRIVE_FRAC = 0.30           # ...or box covers this fraction of the frame
ARRIVE_DEPTH = 170           # mean depth inside box (0-255, higher = closer)
ARRIVE_FRAMES = 10           # consecutive frames the arrival test must hold
TARGET_MAX_WEIGHT = 0.85     # never let the target fully override obstacle avoidance
DECAY = 0.01                 # per-frame confidence decay when the target is lost
MOVE_EPS = 5.0               # VO "user is moving" threshold (VO units are arbitrary; tune)
 
# + = right, same convention as get_door_steer()
HEADING_DEG = {"left": -60, "slight_left": -25, "ahead": 0,
               "slight_right": 25, "right": 60, "turn_around": 90}
 
 
def make_tracker():
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for mod in (cv2, getattr(cv2, "legacy", None)):
            fn = getattr(mod, name, None) if mod is not None else None
            if fn is not None:
                return fn()
    raise RuntimeError("No OpenCV tracker available; pip install opencv-contrib-python")
 
 
# ------------------------------------------------------------------- speech ---
class Speaker:
    """Non-blocking TTS. Newest line wins; repeats within min_gap seconds are dropped."""
 
    def __init__(self, rate=190):
        self._q = queue.Queue(maxsize=2)
        self._last = ("", 0.0)
        self._stop = False
        self._rate = rate
        threading.Thread(target=self._run, daemon=True).start()
 
    def say(self, text, min_gap=4.0):
        text = (text or "").strip()
        if not text:
            return
        last, t = self._last
        if text == last and time.time() - t < min_gap:
            return
        self._last = (text, time.time())
        try:
            while True:
                self._q.get_nowait()  # drop stale lines
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass
 
    def _run(self):
        import pyttsx3  # engine must be created on the thread that uses it
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        while not self._stop:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[speaker] {e}")
 
    def stop(self):
        self._stop = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
 
 
class VoiceGoal:
    """Push-to-talk goal capture. Call listen_async() (e.g. on a key press)."""
 
    def __init__(self, on_goal, speaker=None):
        import speech_recognition as sr
        self._sr = sr
        self._rec = sr.Recognizer()
        self._on_goal = on_goal
        self._speaker = speaker
        self._busy = False
 
    def listen_async(self):
        if self._busy:
            return
        threading.Thread(target=self._run, daemon=True).start()
 
    def _run(self):
        self._busy = True
        try:
            if self._speaker:
                self._speaker.say("Where to?")
            with self._sr.Microphone() as src:
                self._rec.adjust_for_ambient_noise(src, duration=0.3)
                audio = self._rec.listen(src, timeout=6, phrase_time_limit=6)
            text = self._rec.recognize_google(audio)  # swap for faster-whisper to go offline
            print(f"[voice] heard: {text!r}")
            self._on_goal(text)
        except Exception as e:
            print(f"[voice] {e}")
            if self._speaker:
                self._speaker.say("Sorry, I didn't catch that.")
        finally:
            self._busy = False
 
 
# ---------------------------------------------------------------------- VO ---
class VOState:
    """Scale-free summary of the VO trajectory (N,3,1): X-Z position + travel heading."""
 
    def __init__(self, window=20, eps=1e-3):
        self.window, self.eps = window, eps
        self.xz = None
        self.heading = None  # degrees, atan2(dx, dz); positive = toward +X (right)
 
    def update(self, traj):
        if traj is None or len(traj) < 2:
            return
        pts = traj[:, [0, 2], 0]
        self.xz = pts[-1].copy()
        d = pts[-1] - pts[max(0, len(pts) - self.window)]
        if float(np.hypot(d[0], d[1])) > self.eps:
            self.heading = math.degrees(math.atan2(d[0], d[1]))
 
    def snapshot(self):
        return (None if self.xz is None else self.xz.copy(), self.heading)
 
    def since(self, snap):
        xz0, h0 = snap
        if xz0 is None or self.xz is None:
            return None
        dx = self.xz - xz0
        dist = float(np.hypot(dx[0], dx[1]))
        turn = None
        if h0 is not None and self.heading is not None:
            turn = (self.heading - h0 + 180.0) % 360.0 - 180.0
        return dist, turn
 
 
# --------------------------------------------------------------- GoalNav ---
class GoalNav:
    def __init__(self, llm, speaker, sens=0.5, blocked_thresh=150):
        self.llm, self.speaker = llm, speaker
        self.sens, self.blocked_thresh = sens, blocked_thresh
        self.vo = VOState()
        self.goal = None
        self._new_goal = None
        self.pending_id = None
        self.req_time = 0.0
        self.req_frame = None
        self.req_snap = None
        self.next_query = 0.0
        self.last_query_t = 0.0
        self.last_instruction = ""
        self.step_done = False
        self.frame_num = 0
        self.pending_frames = collections.deque(maxlen=300)
        self._reset_target()
 
    @property
    def active(self):
        return self.goal is not None
 
    # ---- goal management (set_goal is safe to call from other threads) ----
    def set_goal(self, text):
        self._new_goal = text
 
    def _apply_goal(self, text):
        text = text.strip()
        self._reset_target()
        self.pending_id = None
        self.pending_frames.clear()
        self.step_done = False
        self.next_query = 0.0
        if text.lower() in ("stop", "cancel", "stop navigation", "never mind"):
            self.goal = None
            self.speaker.say("Navigation stopped.")
            return
        self.goal = text
        self.speaker.say(f"Going to {text}.")
 
    def _reset_target(self):
        self.tracker = None
        self.box = None               # (x, y, w, h) full-res pixels
        self.box0_area = 1.0
        self.feed = collections.deque(maxlen=300)  # frames the live tracker hasn't seen yet
        self.lost = 0
        self.arrive_count = 0
        self.last_good_frame = -1
        self.coarse_dir = None
        self.coarse_frame = -1
 
    # ---- per-frame update --------------------------------------------------
    def update(self, frame, depth_uint8, frame_num, traj, col):
        """Call once per main-loop frame with the CLEAN frame (before any drawing)."""
        self.frame_num = frame_num
        if self._new_goal is not None:
            goal, self._new_goal = self._new_goal, None
            self._apply_goal(goal)
        if not self.active:
            return
        self.vo.update(traj)
        small = cv2.resize(frame, None, fx=TRACK_SCALE, fy=TRACK_SCALE)
 
        if self.tracker is not None:
            self.feed.append(small)
        if self.pending_id is not None:
            self.pending_frames.append(small)
            self._poll_result()   # may swap in a new tracker whose feed already includes `small`
        if self.tracker is not None:
            self._pump()
 
        if self.pending_id is None and time.time() >= self.next_query:
            need = False
            if self.tracker is None:
                need = True
            elif time.time() - self.last_query_t > REQUERY_S:
                need = True
            elif self._arrived(depth_uint8):
                # Reached this waypoint: stop pulling toward it, let depth steering
                # drive until the LLM names the next one.
                self.step_done = True
                self.tracker, self.box, self.coarse_dir = None, None, None
                need = True
            if need and col is not None:
                self._send_query(frame, small, col)
 
    # ---- LLM round trip ----------------------------------------------------
    def _send_query(self, frame, small, col):
        ctx = self._context(col)
        rid = self.frame_num
        print(f"[goal_nav] Submitting LLM request for goal={self.goal!r}, frame={rid}")
        accepted = self.llm.force_submit(frame, ctx, rid)
        if not accepted:
            return
        print(f"[goal_nav] Request accepted: {accepted}")
        self.pending_id = rid
        self.req_time = time.time()
        self.req_frame = small
        self.req_snap = self.vo.snapshot()
        self.pending_frames.clear()
        self.step_done = False
 
    def _context(self, col):
        lines = [f"GOAL: {self.goal}"]
        if self.step_done:
            lines.append("The previous waypoint was just reached.")
        if self.last_instruction:
            lines.append(f"Your previous instruction was: {self.last_instruction!r}")
        blocked = col["c"] > self.blocked_thresh
        lines.append(
            f"Depth sensor (0=far, 255=near): left {col['l']:.0f}, center {col['c']:.0f}, "
            f"right {col['r']:.0f}; path straight ahead is {'BLOCKED' if blocked else 'clear'}."
        )
        d = self.vo.since(self.req_snap) if self.req_snap is not None else None
        if d is not None:
            dist, turn = d
            s = f"Visual odometry since your last look: user is {'moving' if dist > MOVE_EPS else 'not moving'}"
            if turn is not None and abs(turn) > 10:
                s += f", travel heading changed ~{abs(turn):.0f} degrees to the {'right' if turn > 0 else 'left'}"
            lines.append(s + ".")
        # IMU: append yaw / step count here once the VO process exposes it.
        return "\n".join(lines)
 
    def _poll_result(self):
        res = self.llm.latest_description()
        if res.get("request_id") == self.pending_id:
            self.pending_id = None
            self._handle(res)
        elif time.time() - self.req_time > WAIT_TIMEOUT_S:
            print("[goal_nav] LLM request timed out")
            self.pending_id = None
            self.next_query = time.time() + RETRY_S
 
    def _handle(self, res):
        self.last_query_t = time.time()
        text = res.get("instruction", "")
        if text:
            self.last_instruction = text
            self.speaker.say(text)
        if res.get("arrived"):
            self.speaker.say("You have arrived.")
            self.goal = None
            self._reset_target()
            return
        t = res.get("target") or {}
        if res.get("target_visible") and all(k in t for k in ("ymin", "xmin", "ymax", "xmax")):
            if self._start_tracker(t):
                return
        self._coarse(res.get("heading"))
 
    def _start_tracker(self, t):
        H, W = self.req_frame.shape[:2]
        x1 = max(0.0, min(1000.0, float(t["xmin"]))) / 1000.0 * W
        x2 = max(0.0, min(1000.0, float(t["xmax"]))) / 1000.0 * W
        y1 = max(0.0, min(1000.0, float(t["ymin"]))) / 1000.0 * H
        y2 = max(0.0, min(1000.0, float(t["ymax"]))) / 1000.0 * H
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            return False
        tr = make_tracker()
        tr.init(self.req_frame, (int(x1), int(y1), int(w), int(h)))
        s = 1.0 / TRACK_SCALE
        self.tracker = tr
        self.box = (x1 * s, y1 * s, w * s, h * s)
        self.box0_area = w * h * s * s
        self.lost = 0
        self.arrive_count = 0
        self.coarse_dir = None
        self.last_good_frame = self.frame_num
        # Replay every frame since the request frame; _pump() catches up within a budget.
        self.feed = self.pending_frames
        self.pending_frames = collections.deque(maxlen=300)
        return True
 
    def _coarse(self, heading):
        self.tracker, self.box = None, None   # a visible-target miss means any old lock is suspect
        self.coarse_dir = HEADING_DEG.get(heading, 0)
        self.coarse_frame = self.frame_num
        self.next_query = time.time() + RETRY_S
 
    # ---- tracking ----------------------------------------------------------
    def _pump(self):
        n = 0
        while self.feed and n < MAX_REPLAY_PER_FRAME:
            img = self.feed.popleft()
            ok, b = self.tracker.update(img)
            n += 1
            if ok and self._sane(b, img.shape):
                s = 1.0 / TRACK_SCALE
                self.box = (b[0] * s, b[1] * s, b[2] * s, b[3] * s)
                self.lost = 0
                self.last_good_frame = self.frame_num
            else:
                self.lost += 1
        if self.lost >= LOST_LIMIT:
            self.tracker = None   # keep self.box: steering decays toward zero confidence
 
    @staticmethod
    def _sane(b, shape):
        H, W = shape[:2]
        x, y, w, h = b
        cx, cy = x + w / 2, y + h / 2
        return w >= 6 and h >= 6 and 0 <= cx <= W and 0 <= cy <= H and w * h < 0.8 * W * H
 
    def _arrived(self, depth_uint8):
        if self.box is None or depth_uint8 is None:
            return False
        H, W = depth_uint8.shape[:2]
        x, y, w, h = self.box
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            self.arrive_count = 0
            return False
        big = w * h >= ARRIVE_GROW * self.box0_area or w * h >= ARRIVE_FRAC * W * H
        close = float(depth_uint8[y1:y2, x1:x2].mean()) >= ARRIVE_DEPTH
        self.arrive_count = self.arrive_count + 1 if (big and close) else 0
        return self.arrive_count >= ARRIVE_FRAMES
 
    # ---- outputs -----------------------------------------------------------
    def steering(self, frame_w):
        """(direction_deg, confidence); + = right. Drop-in for (door_direction, max_conf)."""
        if not self.active:
            return 0.0, 0.0
        if self.box is not None:
            cx = self.box[0] + self.box[2] / 2
            d = max(-90.0, min(90.0, (cx - frame_w / 2) * self.sens))
            if self.tracker is not None:
                conf = 1.0 if (self.lost == 0 and len(self.feed) <= 2) else 0.6
            else:
                conf = math.exp(-DECAY * (self.frame_num - self.last_good_frame))
            return d, min(conf, TARGET_MAX_WEIGHT)
        if self.coarse_dir is not None:
            return float(self.coarse_dir), 0.5 * math.exp(-DECAY * (self.frame_num - self.coarse_frame))
        return 0.0, 0.0
 
    def draw(self, frame):
        if self.box is not None:
            x, y, w, h = map(int, self.box)
            color = (255, 0, 255) if self.tracker is not None else (128, 0, 128)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "WAYPOINT", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)