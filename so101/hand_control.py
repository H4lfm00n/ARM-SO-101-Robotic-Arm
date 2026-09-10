"""
SO-101 hand control: move the arm by moving your hand in front of a camera.

    python hand_control.py --port /dev/tty.usbmodem5B415332511 --id my_follower

Try it without the arm first:

    python hand_control.py --preview

Hand tracking runs inside your browser (MediaPipe's WebAssembly build), and this
script turns the tracked hand into smooth arm motion. The first run downloads the
tracking files (about 20 MB) into a hand_assets folder next to this script.

How the hand maps to the arm (relative to where your hand was when following started):
    hand left / right       -> base
    hand up / down          -> shoulder
    hand closer / farther   -> elbow
    tip fingers toward cam  -> wrist flex
    twist your forearm      -> wrist roll
    pinch thumb + index     -> gripper

Keep this file in the same folder as arm_joystick.py; it reuses that motion controller.
"""

import argparse
import io
import json
import math
import os
import ssl
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from arm_joystick import JOINTS, NAMES, ArmController, SO101Follower, SO101FollowerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
]
TASKS_VISION_VERSION = "0.10.21"
ASSET_DIR = os.path.join(HERE, "hand_assets", f"tasks-vision-{TASKS_VISION_VERSION}")
NPM_URL = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{TASKS_VISION_VERSION}.tgz"

# joint: (hand feature, degrees of joint per unit of feature, sign)
RELATIVE_MAP = {
    "shoulder_pan": ("x", 150.0, 1),  # full frame width = 150°
    "shoulder_lift": ("y", 130.0, -1),  # full frame height = 130°, hand up = positive
    "elbow_flex": ("reach", 90.0, 1),  # hand twice as close ≈ 62°
    "wrist_flex": ("pitch", 1.0, 1),  # degrees of hand tilt
    "wrist_roll": ("twist", 1.0, 1),  # degrees of forearm twist
}
PINCH_CLOSED = 0.25  # thumb-index gap relative to palm length
PINCH_OPEN = 1.0
LOST_AFTER_S = 0.25  # hand missing this long pauses following
PAGE_TIMEOUT_S = 1.0  # following stops if the page stops sending
GOAL_DEADBAND = 0.2
PALM = [0, 5, 9, 17]


# ---------------------------------------------------------------- downloads

def _download(url):
    """Fetch a URL, trying certifi's certificates first and then the system's."""
    contexts = []
    try:
        import certifi

        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    contexts.append(ssl.create_default_context())
    last = None
    for ctx in contexts:
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
                return r.read()
        except urllib.error.URLError as e:
            last = e
            if not isinstance(getattr(e, "reason", None), ssl.SSLError):
                break
    raise last


def ensure_assets():
    if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000):
        print("Downloading the hand tracking model (about 8 MB)...")
        last = None
        for url in MODEL_URLS:
            try:
                data = _download(url)
                with open(MODEL_PATH, "wb") as f:
                    f.write(data)
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            print(f"Couldn't download the model: {last}")
            print(f"Download it in a browser from:\n  {MODEL_URLS[0]}\nand save it next to this script.")
            sys.exit(1)

    bundle = os.path.join(ASSET_DIR, "vision_bundle.mjs")
    if not os.path.exists(bundle):
        print(f"Downloading the browser hand tracker (MediaPipe tasks-vision {TASKS_VISION_VERSION}, about 10 MB)...")
        try:
            data = _download(NPM_URL)
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't download it: {e}")
            sys.exit(1)
        os.makedirs(os.path.join(ASSET_DIR, "wasm"), exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if not member.isfile():
                    continue
                if name == "package/vision_bundle.mjs":
                    dest = bundle
                elif name.startswith("package/wasm/") and "/" not in name[len("package/wasm/"):]:
                    dest = os.path.join(ASSET_DIR, "wasm", os.path.basename(name))
                else:
                    continue
                with tar.extractfile(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
        print("Saved.")


# ---------------------------------------------------------------- hand math

def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class OneEuro:
    """One Euro filter: smooth when still, responsive when moving fast."""

    def __init__(self, mincutoff, beta, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        a_d = self._alpha(dt, self.dcutoff)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(dt, self.mincutoff + self.beta * abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x


def hand_features(lm, world, aspect):
    """Hand measurements from 21 image landmarks [x, y, z] and 21 world landmarks [x, y, z]."""
    cx = sum(lm[i][0] for i in PALM) / len(PALM)
    cy = sum(lm[i][1] for i in PALM) / len(PALM)
    size = math.hypot((lm[9][0] - lm[0][0]) * aspect, lm[9][1] - lm[0][1])

    def sub(a, b):
        return [world[a][k] - world[b][k] for k in range(3)]

    d = sub(9, 0)  # wrist -> middle knuckle
    across = sub(17, 5)  # index knuckle -> pinky knuckle
    palm_len = max(math.sqrt(sum(v * v for v in d)), 1e-6)
    pinch = math.sqrt(sum(v * v for v in sub(4, 8))) / palm_len
    return {
        "x": cx,
        "y": cy,
        "reach": math.log(max(size, 1e-4)),
        "pitch": math.degrees(math.atan2(-d[2], -d[1])),
        "twist": math.degrees(math.atan2(across[2], across[0])),
        "pinch": pinch,
    }


class SimRobot:
    """Stand-in arm for --preview: follows commands perfectly."""

    class _Cal:
        def __init__(self, lo, hi):
            self.range_min, self.range_max = lo, hi

    class _Bus:
        pass

    def __init__(self):
        self.bus = self._Bus()
        self.bus.calibration = {j: self._Cal(800, 3300) for j in JOINTS}
        self.pos = {j: 0.0 for j in JOINTS}
        self.pos["gripper"] = 50.0

    def get_observation(self):
        return {f"{j}.pos": v for j, v in self.pos.items()}

    def send_action(self, action):
        for k, v in action.items():
            self.pos[k[: -len(".pos")]] = float(v)
        return action

    def disconnect(self):
        pass


# ---------------------------------------------------------------- following

class HandFollower:
    """Turns hand measurements from the browser into arm goals."""

    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.Lock()
        self.running = True

        self.following = False
        self.ref = None
        self.arm_ref = None
        self.hand_visible = False
        self.last_seen = 0.0
        self.last_msg = time.monotonic()
        self.last_goal = {}
        self.latest = None

        self.enabled = {j: True for j in JOINTS}
        self.flip = {j: False for j in JOINTS}
        self.sensitivity = 1.0

        self.filters = {
            "x": OneEuro(1.2, 2.0),
            "y": OneEuro(1.2, 2.0),
            "reach": OneEuro(1.0, 1.0),
            "pitch": OneEuro(0.8, 0.02),
            "twist": OneEuro(0.8, 0.02),
            "pinch": OneEuro(1.5, 0.5),
        }
        self._unwrap = {}
        self.thread = threading.Thread(target=self._watchdog, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    # ---- from the page ----

    def set_following(self, on):
        with self.lock:
            self.following = bool(on)
            self.ref = None
        if not on:
            self._stop_smoothly()

    def configure(self, data):
        with self.lock:
            for j, v in (data.get("enabled") or {}).items():
                if j in self.enabled:
                    self.enabled[j] = bool(v)
            for j, v in (data.get("flip") or {}).items():
                if j in self.flip:
                    self.flip[j] = bool(v)
            if data.get("sensitivity") is not None:
                self.sensitivity = max(0.25, min(3.0, float(data["sensitivity"])))
            self.ref = None  # re-anchor so a settings change never makes the arm jump
        if data.get("speed") is not None:
            with self.controller.lock:
                self.controller.speed = max(5.0, min(90.0, float(data["speed"])))

    def update(self, msg):
        now = time.monotonic()
        self.last_msg = now
        lm, world = msg.get("lm"), msg.get("world")
        valid = (
            msg.get("visible")
            and isinstance(lm, list) and len(lm) == 21
            and isinstance(world, list) and len(world) == 21
        )
        if not valid:
            self.hand_visible = False
            return
        try:
            raw = hand_features(lm, world, float(msg.get("aspect", 16 / 9)))
            t = float(msg.get("t", now))
        except (TypeError, ValueError, IndexError):
            self.hand_visible = False
            return
        feats = self._filtered(raw, t)
        self.hand_visible = True
        self.last_seen = now
        self.latest = feats
        self._apply(feats)

    def state(self):
        with self.lock:
            if not self.following:
                tracking = "idle"
            elif self.ref is None:
                tracking = "waiting"
            else:
                tracking = "following"
            anchor = {"x": self.ref["x"], "y": self.ref["y"]} if self.ref else None
            return {
                "tracking": tracking,
                "hand_visible": self.hand_visible,
                "anchor": anchor,
                "enabled": self.enabled,
                "flip": self.flip,
                "sensitivity": self.sensitivity,
            }

    # ---- internals ----

    def _watchdog(self):
        while self.running:
            now = time.monotonic()
            if now - self.last_msg > PAGE_TIMEOUT_S:
                self.hand_visible = False
                if self.following:
                    self.set_following(False)
            if now - self.last_seen > LOST_AFTER_S:
                with self.lock:
                    was_anchored = self.ref is not None
                    self.ref = None
                if was_anchored or self.filters["x"].x is not None:
                    self._reset_filters()
                if was_anchored:
                    self._stop_smoothly()
            time.sleep(0.05)

    def _stop_smoothly(self):
        c = self.controller
        with c.lock:
            for j in JOINTS:
                c.goal[j] = None
                c.inputs[j] = 0.0

    def _filtered(self, raw, t):
        out = {}
        for k, v in raw.items():
            if k in ("pitch", "twist"):  # keep angles continuous across ±180°
                prev = self._unwrap.get(k)
                v = v if prev is None else prev[1] + wrap180(v - prev[0])
                self._unwrap[k] = (raw[k], v)
            out[k] = self.filters[k](v, t)
        return out

    def _reset_filters(self):
        for f in self.filters.values():
            f.reset()
        self._unwrap = {}

    def _apply(self, feats):
        c = self.controller
        with self.lock:
            if not self.following:
                return
            if self.ref is None:
                self.ref = dict(feats)
                with c.lock:
                    self.arm_ref = dict(c.pos)
                self.last_goal = {}
            ref, arm_ref = self.ref, self.arm_ref
            enabled, flip, sens = dict(self.enabled), dict(self.flip), self.sensitivity

        goals = {}
        for j, (feature, gain, sign) in RELATIVE_MAP.items():
            if enabled[j]:
                direction = -1 if flip[j] else 1
                goals[j] = arm_ref[j] + sign * direction * gain * sens * (feats[feature] - ref[feature])
        if enabled["gripper"]:
            lo, hi = c.limits["gripper"]
            t = max(0.0, min(1.0, (feats["pinch"] - PINCH_CLOSED) / (PINCH_OPEN - PINCH_CLOSED)))
            goals["gripper"] = hi - t * (hi - lo) if flip["gripper"] else lo + t * (hi - lo)

        for j, value in goals.items():
            if abs(value - self.last_goal.get(j, float("inf"))) >= GOAL_DEADBAND:
                c.set_goal(j, value)
                self.last_goal[j] = value


# ---------------------------------------------------------------- web server

CONTENT_TYPES = {".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm"}


def make_handler(controller, follower, preview):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, content_type, code=200, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code=200):
            self._send(json.dumps(payload).encode(), "application/json", code)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return {}

        def _state(self):
            s = follower.state()
            s["arm"] = controller.state()
            s["preview"] = preview
            return s

        def _file(self, path, content_type):
            try:
                with open(path, "rb") as f:
                    self._send(f.read(), content_type, cache=True)
            except OSError:
                self._json({"error": "not found"}, 404)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/assets/vision_bundle.mjs":
                self._file(os.path.join(ASSET_DIR, "vision_bundle.mjs"), "text/javascript")
            elif path.startswith("/assets/wasm/"):
                name = os.path.basename(path)
                ext = os.path.splitext(name)[1]
                if ext in CONTENT_TYPES:
                    self._file(os.path.join(ASSET_DIR, "wasm", name), CONTENT_TYPES[ext])
                else:
                    self._json({"error": "not found"}, 404)
            elif path == "/assets/hand_landmarker.task":
                self._file(MODEL_PATH, "application/octet-stream")
            elif path == "/api/state":
                self._json(self._state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            data = self._body()
            follower.last_msg = time.monotonic()  # any request from the page counts as a heartbeat
            if self.path == "/api/hand":
                follower.update(data)
            elif self.path == "/api/follow":
                follower.set_following(data.get("on", False))
            elif self.path == "/api/hold":
                follower.set_following(False)
                controller.hold()
            elif self.path == "/api/home":
                follower.set_following(False)
                controller.go_home()
            elif self.path == "/api/config":
                follower.configure(data)
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json(self._state())

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Control the SO-101 follower with your hand")
    parser.add_argument("--port", help="follower port from lerobot-find-port")
    parser.add_argument("--id", default="my_follower", help="the --robot.id you calibrated with")
    parser.add_argument("--preview", action="store_true", help="run without the arm")
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.preview and not args.port:
        parser.error("pass --port for the arm, or --preview to try it without the arm")

    ensure_assets()

    # Grab the web port first, so a busy port fails before the motors are powered.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.web_port), BaseHTTPRequestHandler)
    except OSError:
        print(f"Port {args.web_port} is already in use. Close the other copy, or pass --web-port 8767.")
        sys.exit(1)

    if args.preview:
        robot = SimRobot()
    else:
        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=15.0))
        if not robot.calibration:
            server.server_close()
            print(f"No calibration found for id '{args.id}'. Run lerobot-calibrate for the follower first.")
            sys.exit(1)
        print("Connecting to the arm...")
        robot.connect()

    controller = ArmController(robot)
    controller.speed = 45.0
    if controller.locked:
        names = ", ".join(NAMES[j] for j in controller.locked)
        print(f"\nWarning: {names} still show almost no calibrated range and won't move.")

    follower = HandFollower(controller)
    server.RequestHandlerClass = make_handler(controller, follower, args.preview)
    controller.start()
    follower.start()

    url = f"http://127.0.0.1:{args.web_port}"
    print(f"\nHand control ready at {url}")
    print("Use Chrome for the most reliable camera and tracking support.")
    print("Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        follower.set_following(False)
        follower.stop()
        server.server_close()
        if args.preview:
            controller.stop()
            print("\nStopped.")
            return
        print("\nReturning to the starting pose...")
        controller.go_home()
        controller.wait_until_home()
        controller.stop()
        try:
            input("Support the arm with your hand (torque turns off next), then press ENTER...")
        except (KeyboardInterrupt, EOFError):
            pass
        robot.disconnect()
        print("Disconnected.")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 hand control</title>
<style>
:root {
  --body: #C9D0D6;
  --plate: #E3E7EA;
  --ink: #1C2731;
  --muted: #55616C;
  --live: #0A66C2;
  --target: #B07F00;
  --stop: #E4570B;
  --num: "DIN Alternate", "Bahnschrift", "Roboto Condensed", "Arial Narrow", sans-serif;
  --text: "Avenir Next", "Segoe UI", system-ui, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--body); color: var(--ink); font: 15px/1.45 var(--text); }
body { min-height: 100vh; display: flex; flex-direction: column; }

header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px 28px;
  padding: 14px 28px; background: var(--ink); color: #EDF1F4;
}
h1 { margin: 0; font: 600 24px/1 var(--num); }
.status { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #C3CCD4; }
.dot { flex: none; width: 10px; height: 10px; border-radius: 50%; background: #7C8791; }
.dot.ok { background: #3FB950; }
.dot.bad { background: var(--stop); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; margin-left: auto; }
.slider { display: flex; align-items: center; gap: 10px; font-size: 14px; }
.slider output { font: 600 18px/1 var(--num); min-width: 54px; font-variant-numeric: tabular-nums; }
.slider input { width: 120px; accent-color: #8FB8E6; }

button, select {
  font: 600 15px/1 var(--text); color: var(--ink); background: var(--plate);
  border: 0; border-radius: 8px; padding: 11px 18px; cursor: pointer;
}
button:hover { filter: brightness(1.05); }
button:focus-visible, input:focus-visible, select:focus-visible { outline: 3px solid var(--live); outline-offset: 3px; }
button.hold { background: var(--stop); color: #fff; padding-inline: 22px; }
kbd {
  font: 600 13px/1 var(--num); background: var(--plate); border-radius: 4px;
  padding: 3px 6px; box-shadow: inset 0 -2px 0 rgba(28,39,49,.25);
}

.banner {
  margin: 20px 28px 0; padding: 12px 18px; border-radius: 10px; max-width: 90ch;
  background: #FBE3D4; color: #5A2208; border-left: 6px solid var(--stop);
}
.banner.info { background: #DCE8F5; color: #0D2E52; border-left-color: var(--live); }
.banner[hidden] { display: none; }

main {
  flex: 1; display: grid; gap: 28px; padding: 24px 28px;
  grid-template-columns: minmax(420px, 1.6fr) minmax(340px, 1fr); align-items: start;
}

/* Camera feed */
.feed { position: relative; background: var(--ink); border-radius: 18px; padding: 10px; }
.feed canvas { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; border-radius: 10px; background: #26313B; }
.feed video { display: none; }
.chip {
  position: absolute; left: 24px; top: 24px; display: flex; align-items: center; gap: 10px;
  padding: 10px 16px; border-radius: 999px; background: rgba(28,39,49,.86); color: #EDF1F4;
  font-size: 15px; max-width: calc(100% - 48px);
}
.chip .dot { width: 12px; height: 12px; }
.chip.following .dot { background: #3FB950; box-shadow: 0 0 0 4px rgba(63,185,80,.3); }
.chip.waiting .dot { background: #E3B341; }
.chip.error .dot { background: var(--stop); }
.feedbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 12px 6px 4px; color: #C3CCD4; font-size: 14px; }
.feedbar select { margin-left: auto; padding: 8px 12px; font-size: 14px; max-width: 60%; }

/* Panel */
.panel { display: flex; flex-direction: column; gap: 18px; }
.follow {
  width: 100%; padding: 22px; font: 600 22px/1 var(--num); border-radius: 14px;
  background: var(--live); color: #fff; box-shadow: 0 6px 0 #084E94;
}
.follow:active { transform: translateY(3px); box-shadow: 0 3px 0 #084E94; }
.follow[aria-pressed="true"] { background: var(--ink); box-shadow: 0 6px 0 #0B1117; }
.follow:disabled { opacity: .5; cursor: default; }
.hint { margin: 0; color: var(--muted); font-size: 14px; max-width: 60ch; }

.joints { background: var(--plate); border-radius: 14px; padding: 8px 16px; }
.row {
  display: grid; grid-template-columns: 128px 1fr 56px auto auto; align-items: center; gap: 12px;
  padding: 11px 0; border-bottom: 1px solid rgba(28,39,49,.1);
}
.row:last-child { border-bottom: 0; }
.row .jname { font: 600 16px/1.2 var(--num); }
.row .how { display: block; font: 12px/1.3 var(--text); color: var(--muted); }
.bar { position: relative; height: 14px; border-radius: 7px; background: #BCC4CB; }
.bar .mk { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; translate: -50% -50%; }
.bar .present { background: var(--live); box-shadow: 0 0 0 2px var(--plate); }
.bar .target { width: 8px; height: 22px; border-radius: 3px; background: var(--target); }
.row .num { font: 600 16px/1 var(--num); text-align: right; font-variant-numeric: tabular-nums; }
.row label { display: flex; align-items: center; gap: 5px; font-size: 13px; cursor: pointer; }
.row .flip { padding: 6px 10px; font-size: 12px; background: transparent; box-shadow: inset 0 0 0 1px rgba(28,39,49,.25); }
.row .flip[aria-pressed="true"] { background: var(--ink); color: #fff; box-shadow: none; }
.row.off .bar, .row.off .num { opacity: .35; }
.row.locked { opacity: .5; }
.legend { display: flex; gap: 18px; font-size: 13px; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.legend .l-live::before { background: var(--live); }
.legend .l-tgt::before { background: var(--target); border-radius: 2px; }

footer {
  display: flex; flex-wrap: wrap; gap: 10px 32px; padding: 14px 28px 20px;
  font-size: 14px; color: var(--muted); border-top: 1px solid rgba(28,39,49,.12);
}
footer p { margin: 0; line-height: 2; }

@media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>SO-101 hand control</h1>
  <div class="status" role="status"><span class="dot" id="dot"></span><span id="statusText">Connecting…</span></div>
  <div class="controls">
    <label class="slider">Speed
      <input type="range" id="speed" min="5" max="90" step="5" value="45">
      <output id="speedOut">45°/s</output>
    </label>
    <button id="homeBtn" title="Shortcut: H">Go home</button>
    <button id="holdBtn" class="hold" title="Shortcut: Space">Hold position</button>
  </div>
</header>

<div class="banner info" id="previewBanner" hidden>Preview mode: no arm is connected. The bars show what the arm would do.</div>
<div class="banner" id="lockBanner" hidden></div>

<main>
  <section aria-label="Camera">
    <div class="feed">
      <video id="video" playsinline muted></video>
      <canvas id="view" width="1280" height="720" aria-label="Camera view with your tracked hand"></canvas>
      <div class="chip" id="chip"><span class="dot"></span><span id="chipText">Loading hand tracking…</span></div>
      <div class="feedbar">
        <span id="camText">Starting camera…</span>
        <select id="camSelect" aria-label="Camera"></select>
      </div>
    </div>
  </section>

  <section class="panel" aria-label="Controls">
    <button class="follow" id="followBtn" aria-pressed="false" title="Shortcut: F" disabled>Start following</button>
    <p class="hint">Hold your hand up, palm toward the camera, about an arm's length away. When following starts, the arm stays where it is and then moves as your hand moves from that spot (the amber ring). Pinch your thumb and index finger to close the gripper. Move your hand out of view to pause, and bring it back to continue.</p>

    <label class="slider">Sensitivity
      <input type="range" id="sens" min="0.25" max="3" step="0.25" value="1">
      <output id="sensOut">1×</output>
    </label>

    <div class="joints" id="joints"></div>
    <div class="legend"><span class="l-live">Where the joint is</span><span class="l-tgt">Where it's headed</span></div>
  </section>
</main>

<footer>
  <p>Keys: <kbd>F</kbd> start or stop following, <kbd>Space</kbd> hold, <kbd>H</kbd> home</p>
  <p>If a joint moves the wrong way, press Flip. Turn off any joint that's too jittery.</p>
</footer>

<script type="module">
import { FilesetResolver, HandLandmarker } from "/assets/vision_bundle.mjs";

const JOINTS = [
  { id: "shoulder_pan", name: "Base", how: "Hand left and right", unit: "°" },
  { id: "shoulder_lift", name: "Shoulder", how: "Hand up and down", unit: "°" },
  { id: "elbow_flex", name: "Elbow", how: "Hand closer and farther", unit: "°" },
  { id: "wrist_flex", name: "Wrist flex", how: "Tip fingers toward camera", unit: "°" },
  { id: "wrist_roll", name: "Wrist roll", how: "Twist your forearm", unit: "°" },
  { id: "gripper", name: "Gripper", how: "Pinch thumb and index", unit: "" },
];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const PALM = [0, 5, 9, 17];
const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const video = $("video"), canvas = $("view"), ctx = canvas.getContext("2d");
let landmarker = null, stream = null, built = false, following = false;
let lastState = null, lastVideoTime = -1, lastHand = null, trackError = null;
let fps = 0, lastDetectT = 0, inflight = false, lastSend = 0;

/* ---------- server ---------- */

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
const send = (path, body) => post(path, body).then(render).catch(() => {});

function setStatus(ok, text) {
  $("dot").className = "dot " + (ok ? "ok" : "bad");
  $("statusText").textContent = text;
}

/* ---------- joint panel ---------- */

function build(s) {
  const wrap = $("joints");
  for (const j of JOINTS) {
    const row = document.createElement("div");
    const locked = s.arm.locked.includes(j.id);
    row.className = "row" + (locked ? " locked" : "");
    row.innerHTML = `
      <span class="jname">${j.name}<span class="how">${j.how}</span></span>
      <div class="bar" role="img" aria-label="${j.name} position"><div class="mk target"></div><div class="mk present"></div></div>
      <span class="num">0</span>
      <label><input type="checkbox" ${locked ? "disabled" : ""}> On</label>
      <button class="flip" aria-pressed="false">Flip</button>`;
    wrap.appendChild(row);
    const box = row.querySelector("input");
    box.addEventListener("change", () => send("/api/config", { enabled: { [j.id]: box.checked } }));
    const flip = row.querySelector(".flip");
    flip.addEventListener("click", () => send("/api/config", { flip: { [j.id]: flip.getAttribute("aria-pressed") !== "true" } }));
    j.el = { row, box, flip, present: row.querySelector(".present"), target: row.querySelector(".target"), num: row.querySelector(".num") };
  }
  if (s.arm.locked.length) {
    const names = s.arm.locked.map(id => JOINTS.find(j => j.id === id).name).join(", ");
    $("lockBanner").textContent = `${names} can't move because calibration recorded almost no range. Run lerobot-calibrate again and move each one all the way to both ends.`;
    $("lockBanner").hidden = false;
  }
  $("previewBanner").hidden = !s.preview;
  built = true;
}

const CHIP = {
  idle: "Not following. Press Start following.",
  waiting: "Show your hand to start. The arm picks up from where it is now.",
  following: "Following your hand",
};

function render(s) {
  lastState = s;
  if (!built) build(s);
  following = s.tracking !== "idle";
  $("followBtn").setAttribute("aria-pressed", String(following));
  $("followBtn").textContent = following ? "Stop following" : "Start following";

  if (trackError) {
    $("chip").className = "chip error";
    $("chipText").textContent = trackError;
  } else if (landmarker) {
    $("chip").className = "chip " + s.tracking;
    $("chipText").textContent = s.tracking === "idle" && !lastHand ? "Not following. No hand in view." : CHIP[s.tracking];
  }

  for (const j of JOINTS) {
    const [lo, hi] = s.arm.limits[j.id], span = hi - lo || 1;
    const p = s.arm.present[j.id], t = s.arm.target[j.id];
    j.el.present.style.left = clamp((p - lo) / span, 0, 1) * 100 + "%";
    j.el.target.style.left = clamp((t - lo) / span, 0, 1) * 100 + "%";
    j.el.num.textContent = p.toFixed(0) + j.unit;
    j.el.box.checked = s.enabled[j.id];
    j.el.row.classList.toggle("off", !s.enabled[j.id]);
    j.el.flip.setAttribute("aria-pressed", String(s.flip[j.id]));
  }

  if (s.arm.error) setStatus(false, s.arm.error + " Check the power and cables.");
  else setStatus(true, s.preview ? "Preview, no arm" : "Arm connected");
}

/* ---------- camera ---------- */

function isOrbbec(label) { return /orbbec|femto/i.test(label || ""); }

async function listCameras(activeId) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  const sel = $("camSelect");
  sel.innerHTML = "";
  devices.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId;
    o.textContent = d.label || `Camera ${i + 1}`;
    if (d.deviceId === activeId) o.selected = true;
    sel.appendChild(o);
  });
  return devices;
}

async function startCamera(deviceId) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  const video_c = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video_c.deviceId = { exact: deviceId };
  stream = await navigator.mediaDevices.getUserMedia({ video: video_c, audio: false });
  video.srcObject = stream;
  await video.play();
  lastVideoTime = -1;
  const settings = stream.getVideoTracks()[0].getSettings();
  return settings.deviceId;
}

async function initCamera() {
  try {
    let activeId = await startCamera(null);
    const devices = await listCameras(activeId);
    const current = devices.find(d => d.deviceId === activeId);
    const orbbec = devices.find(d => isOrbbec(d.label));
    if (orbbec && !(current && isOrbbec(current.label))) {
      activeId = await startCamera(orbbec.deviceId);
      await listCameras(activeId);
    }
    trackError = null;
  } catch (e) {
    trackError = e.name === "NotAllowedError"
      ? "Camera access is blocked. Allow it with the camera icon in the address bar, then reload."
      : e.name === "NotFoundError" ? "No camera found. Plug one in, then reload."
      : "The camera didn't start (" + e.message + "). Pick another camera or reload.";
    $("camText").textContent = "No camera";
  }
}

$("camSelect").addEventListener("change", async e => {
  try {
    const id = await startCamera(e.target.value);
    await listCameras(id);
    trackError = null;
  } catch (err) {
    trackError = "That camera didn't start (" + err.message + "). Pick another one.";
  }
});

/* ---------- tracking ---------- */

async function initTracker() {
  const fileset = await FilesetResolver.forVisionTasks("/assets/wasm");
  const options = delegate => ({
    baseOptions: { modelAssetPath: "/assets/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (e) {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
  $("followBtn").disabled = false;
}

function draw(hand) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);  // mirror, so moving right moves right on screen
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const s = Math.max(1, w / 1280);
  if (lastState && lastState.anchor) {
    const ax = lastState.anchor.x * w, ay = lastState.anchor.y * h;
    ctx.strokeStyle = "#E3A300"; ctx.lineWidth = 2.5 * s;
    ctx.beginPath(); ctx.arc(ax, ay, 26 * s, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ax - 36 * s, ay); ctx.lineTo(ax + 36 * s, ay); ctx.moveTo(ax, ay - 36 * s); ctx.lineTo(ax, ay + 36 * s); ctx.stroke();
  }
  if (!hand) return;

  const pts = hand.lm.map(p => [p[0] * w, p[1] * h]);
  ctx.lineCap = "round";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.strokeStyle = "rgba(28,39,49,.85)"; ctx.lineWidth = 7 * s;
    ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
    ctx.strokeStyle = "#F4F6F8"; ctx.lineWidth = 2.5 * s;
    ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
  }
  const wl = hand.world;
  const palm = Math.hypot(wl[9][0] - wl[0][0], wl[9][1] - wl[0][1], wl[9][2] - wl[0][2]) || 1;
  const pinch = Math.hypot(wl[4][0] - wl[8][0], wl[4][1] - wl[8][1], wl[4][2] - wl[8][2]) / palm;
  ctx.strokeStyle = pinch < 0.6 ? "#E4570B" : "#E3A300"; ctx.lineWidth = 4 * s;
  ctx.beginPath(); ctx.moveTo(...pts[4]); ctx.lineTo(...pts[8]); ctx.stroke();

  const cx = PALM.reduce((a, i) => a + pts[i][0], 0) / 4, cy = PALM.reduce((a, i) => a + pts[i][1], 0) / 4;
  ctx.fillStyle = "#0A66C2"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5 * s;
  ctx.beginPath(); ctx.arc(cx, cy, 12 * s, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
}

async function sendHand(hand, t) {
  if (inflight) return;
  inflight = true;
  lastSend = performance.now();
  try {
    const body = hand
      ? { visible: true, lm: hand.lm, world: hand.world, aspect: video.videoWidth / video.videoHeight, t }
      : { visible: false, t };
    render(await post("/api/hand", body));
  } catch (e) {
    setStatus(false, "Lost connection to the script. Check the terminal, then reload this page.");
  } finally {
    inflight = false;
  }
}

function frame() {
  const now = performance.now();
  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    let hand = null;
    try {
      const res = landmarker.detectForVideo(video, now);
      if (res.landmarks && res.landmarks.length && res.worldLandmarks && res.worldLandmarks.length) {
        hand = {
          lm: res.landmarks[0].map(p => [1 - p.x, p.y, p.z]),
          world: res.worldLandmarks[0].map(p => [-p.x, p.y, p.z]),
        };
      }
    } catch (e) {
      trackError = "Hand tracking stopped (" + e.message + "). Reload the page.";
    }
    lastHand = hand;
    if (lastDetectT) fps = fps ? 0.9 * fps + 0.1 * (1000 / (now - lastDetectT)) : 1000 / (now - lastDetectT);
    lastDetectT = now;
    draw(hand);
    sendHand(hand, now / 1000);

    $("camText").textContent = `${Math.round(fps)} fps` + (fps && fps < 12 ? ". That's slow, so try another camera." : "");
  } else if (now - lastSend > 100) {
    sendHand(null, now / 1000);  // keeps the arm's safety timer alive while the video is paused
  }
  requestAnimationFrame(frame);
}

/* ---------- wiring ---------- */

$("followBtn").addEventListener("click", () => send("/api/follow", { on: !following }));
$("holdBtn").addEventListener("click", () => send("/api/hold"));
$("homeBtn").addEventListener("click", () => send("/api/home"));
$("speed").addEventListener("input", e => {
  $("speedOut").textContent = e.target.value + "°/s";
  send("/api/config", { speed: parseFloat(e.target.value) });
});
$("sens").addEventListener("change", e => send("/api/config", { sensitivity: parseFloat(e.target.value) }));
$("sens").addEventListener("input", e => { $("sensOut").textContent = e.target.value + "×"; });

addEventListener("keydown", e => {
  if (e.target.matches("input, select") || e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === "f" && landmarker) send("/api/follow", { on: !following });
  else if (k === " " && !e.target.matches("button")) { e.preventDefault(); send("/api/hold"); }
  else if (k === "h") send("/api/home");
});

fetch("/api/state").then(r => r.json()).then(render).catch(() => setStatus(false, "Can't reach the script. Check the terminal, then reload this page."));
requestAnimationFrame(frame);
await initCamera();
try {
  await initTracker();
} catch (e) {
  trackError = "Hand tracking didn't load (" + e.message + "). Check the terminal, then reload.";
  $("chip").className = "chip error";
  $("chipText").textContent = trackError;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
