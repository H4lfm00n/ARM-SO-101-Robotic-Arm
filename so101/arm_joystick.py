"""
SO-101 follower: one-joystick control panel.

One on-screen joystick drives the whole arm through three modes (Arm, Elbow and
wrist, Hand). Joints accelerate and brake smoothly, slow down before their
limits, and coast to a stop when you let go.

    python arm_joystick.py --port /dev/tty.usbmodem5B415332511 --id my_follower

Needs a good follower calibration (lerobot-calibrate). Ctrl+C in the terminal
stops it: the arm returns to its starting pose, then waits for you to support
it before torque turns off.
"""

import argparse
import json
import logging
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES = {
    "shoulder_pan": "Base",
    "shoulder_lift": "Shoulder",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist flex",
    "wrist_roll": "Wrist roll",
    "gripper": "Gripper",
}
MOTOR_RESOLUTION = 4095
MIN_RANGE_TICKS = 200  # a calibrated range smaller than this (~18°) means calibration missed the joint

LOOP_HZ = 50
DEADMAN_S = 0.35  # inputs drop to zero if the browser goes quiet this long
ACCEL_FACTOR = 4.0  # max acceleration = speed x this, so 0 to full speed in 0.25 s
MAX_LEAD = 25.0  # how far the command may run ahead of a blocked joint
GRIPPER_SPEED_SCALE = 1.5  # gripper is on a 0-100 scale, not degrees
LIMIT_MARGIN = 1.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ArmController:
    """Owns the robot. Only the control thread talks to the motor bus."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = threading.Lock()
        self.ranges, self.limits, self.locked = self._inspect_calibration()

        present = self._read()
        self.present = dict(present)
        self.pos = {j: clamp(present[j], *self.limits[j]) for j in JOINTS}  # commanded position
        self.vel = {j: 0.0 for j in JOINTS}  # commanded velocity
        self.goal = {j: None for j in JOINTS}  # position goal (home, sliders)
        self.inputs = {j: 0.0 for j in JOINTS}  # joystick velocity request, -1..1
        self.home = dict(self.pos)

        self.speed = 40.0
        self.last_cmd = 0.0
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _inspect_calibration(self):
        ranges, limits, locked = {}, {}, []
        for j in JOINTS:
            cal = self.robot.bus.calibration[j]
            ticks = cal.range_max - cal.range_min
            ranges[j] = ticks * 360 / MOTOR_RESOLUTION
            if ticks < MIN_RANGE_TICKS:
                locked.append(j)
            if j == "gripper":
                limits[j] = (0.0, 100.0)
            else:
                half = ticks / 2 * 360 / MOTOR_RESOLUTION
                margin = min(LIMIT_MARGIN, half / 2)
                limits[j] = (-half + margin, half - margin)
        return ranges, limits, locked

    def _read(self):
        obs = self.robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in JOINTS}

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    # ---- motion physics ----

    def _step_joint(self, j, present, dt):
        lo, hi = self.limits[j]
        vmax = self.speed * (GRIPPER_SPEED_SCALE if j == "gripper" else 1.0)
        amax = vmax * ACCEL_FACTOR
        p, v, u = self.pos[j], self.vel[j], self.inputs[j]

        if u:
            self.goal[j] = None
            v_des = u * vmax
        elif self.goal[j] is not None:
            d = self.goal[j] - p
            if abs(d) < 0.05 and abs(v) < 2.0:
                self.pos[j], self.vel[j], self.goal[j] = self.goal[j], 0.0, None
                return
            # Fastest speed that can still brake to a stop exactly on the goal.
            v_des = math.copysign(min(vmax, math.sqrt(2 * amax * abs(d))), d)
        else:
            v_des = 0.0

        # Brake early so the joint eases into its limits instead of slamming.
        v_des = min(v_des, math.sqrt(2 * amax * max(0.0, hi - p)))
        v_des = max(v_des, -math.sqrt(2 * amax * max(0.0, p - lo)))

        # If the real joint is stuck, stop pushing further in that direction.
        if p - present[j] > MAX_LEAD:
            v_des = min(v_des, 0.0)
        if present[j] - p > MAX_LEAD:
            v_des = max(v_des, 0.0)

        v += clamp(v_des - v, -amax * dt, amax * dt)
        p += v * dt
        if p <= lo or p >= hi:
            p = clamp(p, lo, hi)
            v = 0.0
        self.pos[j], self.vel[j] = p, v

    def _loop(self):
        dt = 1 / LOOP_HZ
        while self.running:
            t0 = time.perf_counter()
            try:
                present = self._read()
            except Exception as e:  # noqa: BLE001
                self.error = f"Couldn't read the motors: {e}"
                time.sleep(dt)
                continue

            with self.lock:
                self.present = present
                if time.monotonic() - self.last_cmd > DEADMAN_S:
                    self.inputs = {j: 0.0 for j in JOINTS}
                for j in JOINTS:
                    if j not in self.locked:
                        self._step_joint(j, present, dt)
                action = {f"{j}.pos": self.pos[j] for j in JOINTS if j not in self.locked}

            if action:
                try:
                    self.robot.send_action(action)
                    self.error = None
                except Exception as e:  # noqa: BLE001
                    self.error = f"Couldn't send the move: {e}"

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    # ---- called from web requests ----

    def state(self):
        with self.lock:
            at_limit = [
                j for j in JOINTS
                if j not in self.locked
                and (self.pos[j] <= self.limits[j][0] + 0.3 or self.pos[j] >= self.limits[j][1] - 0.3)
            ]
            return {
                "present": self.present,
                "target": self.pos,
                "limits": self.limits,
                "locked": self.locked,
                "at_limit": at_limit,
                "speed": self.speed,
                "error": self.error,
            }

    def tick(self, vel, speed):
        with self.lock:
            for j in JOINTS:
                self.inputs[j] = clamp(float(vel.get(j, 0.0)), -1.0, 1.0)
            if speed is not None:
                self.speed = clamp(float(speed), 5.0, 90.0)
            self.last_cmd = time.monotonic()

    def set_goal(self, joint, value):
        if joint not in JOINTS or joint in self.locked:
            return
        with self.lock:
            self.goal[joint] = clamp(float(value), *self.limits[joint])

    def go_home(self):
        with self.lock:
            for j in JOINTS:
                self.inputs[j] = 0.0
                self.goal[j] = self.home[j]

    def hold(self):
        with self.lock:
            for j in JOINTS:
                self.inputs[j] = 0.0
                self.goal[j] = None
                self.vel[j] = 0.0
                self.pos[j] = clamp(self.present[j], *self.limits[j])

    def wait_until_home(self, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                done = all(self.goal[j] is None for j in JOINTS)
            if done:
                return
            time.sleep(0.05)


def make_handler(holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            controller = holder["controller"]
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(controller.state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            controller = holder["controller"]
            data = self._body()
            if self.path == "/api/tick":
                controller.tick(data.get("vel", {}), data.get("speed"))
            elif self.path == "/api/set":
                controller.set_goal(data.get("joint"), data.get("value", 0))
            elif self.path == "/api/home":
                controller.go_home()
            elif self.path == "/api/hold":
                controller.hold()
            else:
                self._json({"error": "not found"}, 404)
                return
            self._json(controller.state())

    return Handler


def main():
    parser = argparse.ArgumentParser(description="One-joystick control panel for the SO-101 follower arm")
    parser.add_argument("--port", required=True, help="follower port from lerobot-find-port")
    parser.add_argument("--id", default="my_follower", help="the --robot.id you calibrated with")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=15.0)
    robot = SO101Follower(config)
    if not robot.calibration:
        print(f"No calibration found for id '{args.id}'. Run lerobot-calibrate for the follower first.")
        sys.exit(1)

    # Grab the web port first, so a busy port fails before the motors are powered.
    holder = {}
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.web_port), make_handler(holder))
    except OSError:
        print(f"Port {args.web_port} is already in use. Close the other copy, or pass --web-port 8766.")
        sys.exit(1)

    print("Connecting to the arm...")
    robot.connect()  # press ENTER here if asked to reuse the calibration file

    controller = ArmController(robot)
    holder["controller"] = controller

    print("\nCalibrated range of motion:")
    for j in JOINTS:
        flag = "   <- almost no range, this joint is locked" if j in controller.locked else ""
        print(f"  {NAMES[j]:11s} {controller.ranges[j]:6.1f}°{flag}")
    if controller.locked:
        names = ", ".join(NAMES[j] for j in controller.locked)
        print(
            f"\n{names}: calibration recorded almost no movement, so these joints can't move."
            "\nStop this script, run lerobot-calibrate again, and move each of them all the way"
            "\nto both ends before pressing ENTER the second time."
        )

    controller.start()
    url = f"http://127.0.0.1:{args.web_port}"
    print(f"\nJoystick ready at {url}")
    print("Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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
<title>SO-101 joystick</title>
<style>
:root {
  --body: #C9D0D6;
  --plate: #E3E7EA;
  --well-hi: #F3F5F7;
  --well-lo: #99A4AD;
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
h1 { margin: 0; font: 600 24px/1 var(--num); letter-spacing: .01em; }
.status { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #C3CCD4; max-width: 50ch; }
.dot { flex: none; width: 10px; height: 10px; border-radius: 50%; background: #7C8791; }
.dot.ok { background: #3FB950; }
.dot.bad { background: var(--stop); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; margin-left: auto; }
.speed { display: flex; align-items: center; gap: 10px; font-size: 14px; }
.speed output { font: 600 18px/1 var(--num); min-width: 58px; font-variant-numeric: tabular-nums; }
.speed input { width: 140px; accent-color: #8FB8E6; }

button {
  font: 600 15px/1 var(--text); color: var(--ink); background: var(--plate);
  border: 0; border-radius: 8px; padding: 11px 18px; cursor: pointer;
}
button:hover { filter: brightness(1.05); }
button:focus-visible, input:focus-visible { outline: 3px solid var(--live); outline-offset: 3px; }
button.hold { background: var(--stop); color: #fff; padding-inline: 22px; }
kbd {
  font: 600 13px/1 var(--num); background: var(--plate); border-radius: 4px;
  padding: 3px 6px; box-shadow: inset 0 -2px 0 rgba(28,39,49,.25);
}

.banner {
  margin: 20px 28px 0; padding: 14px 18px; border-radius: 10px;
  background: #FBE3D4; color: #5A2208; border-left: 6px solid var(--stop); max-width: 90ch;
}
.banner[hidden] { display: none; }
.banner strong { font-weight: 600; }

main {
  flex: 1; display: grid; gap: 36px; padding: 28px;
  grid-template-columns: minmax(340px, 1fr) minmax(420px, 1.15fr); align-items: start;
}

/* Mode switch */
.driver { display: flex; flex-direction: column; align-items: center; gap: 18px; }
.modes { display: flex; background: var(--ink); border-radius: 12px; padding: 4px; gap: 4px; }
.modes button { background: transparent; color: #C3CCD4; padding: 10px 16px; border-radius: 9px; }
.modes button span { font: 600 12px/1 var(--num); opacity: .6; margin-right: 6px; }
.modes button[aria-pressed="true"] { background: var(--plate); color: var(--ink); }

/* The joystick */
.stick { position: relative; width: min(100%, 440px); padding: 0 0 30px 30px; }
.axis { position: absolute; font-size: 14px; color: var(--muted); white-space: nowrap; }
.axis.x { left: calc(50% + 15px); bottom: 0; transform: translateX(-50%); }
.axis.y { top: calc(50% - 15px); left: 0; transform: translate(-50%, -50%) rotate(-90deg) translateY(12px); }
.bezel {
  padding: 12px; border-radius: 50%;
  background: repeating-conic-gradient(#A7B0B8 0 2deg, #C4CBD1 2deg 4deg);
  box-shadow: 0 2px 0 rgba(255,255,255,.55), inset 0 1px 2px rgba(28,39,49,.25);
}
.well {
  position: relative; aspect-ratio: 1; border-radius: 50%; touch-action: none; cursor: grab;
  background:
    radial-gradient(circle, transparent 0 59.5%, rgba(28,39,49,.16) 60% 60.4%, transparent 60.9%),
    radial-gradient(circle, transparent 0 29.5%, rgba(28,39,49,.10) 30% 30.4%, transparent 30.9%),
    radial-gradient(circle at 35% 30%, var(--well-hi), var(--well-lo));
  box-shadow: inset 0 8px 22px rgba(28,39,49,.35), inset 0 -3px 6px rgba(255,255,255,.5);
}
.well.active { cursor: grabbing; }
.well svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.cross { stroke: rgba(28,39,49,.18); stroke-width: .006; }
.boot { fill: #3A4652; }
.shaft { stroke: url(#shaftGrad); stroke-width: .075; stroke-linecap: round; }
.cap {
  position: absolute; left: 50%; top: 50%; width: 30%; aspect-ratio: 1; border-radius: 50%;
  translate: -50% -50%; pointer-events: none;
  background: radial-gradient(circle at 38% 32%, #56636F, var(--ink) 68%);
}
.cap::after {
  content: ""; position: absolute; inset: 20%; border-radius: 50%;
  background: repeating-radial-gradient(circle, rgba(255,255,255,.08) 0 2px, transparent 2px 5px);
}
.well.active .cap { outline: 4px solid rgba(10,102,194,.35); }
.readout { display: flex; gap: 22px; font-size: 14px; color: var(--muted); min-height: 22px; }
.readout b { font: 600 18px/1 var(--num); color: var(--ink); font-variant-numeric: tabular-nums; }
.hint { margin: 0; color: var(--muted); font-size: 14px; text-align: center; max-width: 44ch; }

/* Gauges */
.gauges { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.gauge {
  background: var(--plate); border-radius: 14px; padding: 12px 14px 14px;
  display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 2px 8px;
  box-shadow: 0 0 0 0 var(--live); transition: box-shadow .15s;
}
.gauge.active { box-shadow: 0 0 0 3px var(--live); }
.gauge h3 { margin: 0; font: 600 16px/1.2 var(--num); }
.gauge .flip { padding: 6px 10px; font-size: 12px; background: transparent; box-shadow: inset 0 0 0 1px rgba(28,39,49,.25); }
.gauge .flip[aria-pressed="true"] { background: var(--ink); color: #fff; box-shadow: none; }
.gauge svg { grid-column: 1 / -1; width: 100%; height: auto; display: block; }
.gauge .track { fill: none; stroke: #BCC4CB; stroke-width: 7; stroke-linecap: round; }
.gauge .fill { fill: none; stroke: var(--live); stroke-width: 7; stroke-linecap: round; opacity: .25; }
.gauge .needle { stroke: var(--live); stroke-width: 3; stroke-linecap: round; }
.gauge .hub { fill: var(--ink); }
.gauge .tgt { fill: var(--target); stroke: var(--plate); stroke-width: 2; }
.gauge .val { font: 600 22px var(--num); fill: var(--ink); text-anchor: middle; font-variant-numeric: tabular-nums; }
.gauge .sub { font: 12px var(--text); fill: var(--muted); text-anchor: middle; }
.gauge input[type=range] { grid-column: 1 / -1; width: 100%; accent-color: var(--target); }
.gauge.limit .track { stroke: #F2B48F; }
.gauge.limit .sub { fill: #A8410A; }
.gauge.locked { opacity: .6; }
.gauge.locked .sub { fill: #A8410A; }
.legend { display: flex; gap: 20px; margin: 14px 2px 0; font-size: 13px; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.legend .l-live::before { background: var(--live); }
.legend .l-tgt::before { background: var(--target); }

footer {
  display: flex; flex-wrap: wrap; gap: 10px 32px; padding: 14px 28px 20px;
  font-size: 14px; color: var(--muted); border-top: 1px solid rgba(28,39,49,.12);
}
footer p { margin: 0; line-height: 2; }

@media (max-width: 980px) {
  main { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  main { padding: 18px; }
  header { padding: 12px 18px; }
  .controls { margin-left: 0; }
  .gauges { grid-template-columns: repeat(2, 1fr); }
  .modes button { padding: 10px 10px; }
}
</style>
</head>
<body>
<header>
  <h1>SO-101 follower</h1>
  <div class="status" role="status"><span class="dot" id="dot"></span><span id="statusText">Connecting to the arm…</span></div>
  <div class="controls">
    <label class="speed">Speed
      <input type="range" id="speed" min="5" max="90" step="5" value="30">
      <output id="speedOut">30°/s</output>
    </label>
    <button id="homeBtn" title="Shortcut: H">Go home</button>
    <button id="holdBtn" class="hold" title="Shortcut: Space">Hold position</button>
  </div>
</header>

<div class="banner" id="banner" role="alert" hidden></div>

<main>
  <section class="driver" aria-label="Joystick">
    <div class="modes" role="group" aria-label="What the joystick moves" id="modes"></div>
    <div class="stick">
      <div class="bezel">
        <div class="well" id="well" aria-label="Joystick">
          <svg viewBox="-1 -1 2 2" aria-hidden="true">
            <defs>
              <linearGradient id="shaftGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="#8C98A3"/><stop offset="1" stop-color="#2B3641"/>
              </linearGradient>
            </defs>
            <line class="cross" x1="-1" y1="0" x2="1" y2="0"/>
            <line class="cross" x1="0" y1="-1" x2="0" y2="1"/>
            <circle class="boot" r=".13"/>
            <line class="shaft" id="shaft" x1="0" y1="0" x2="0" y2="0"/>
          </svg>
          <div class="cap" id="cap"></div>
        </div>
      </div>
      <span class="axis x" id="axisX">◀ Base ▶</span>
      <span class="axis y" id="axisY">◀ Shoulder ▶</span>
    </div>
    <div class="readout" id="readout"></div>
    <p class="hint">Drag the stick and let go to stop. The arm speeds up and slows down smoothly, so it coasts a little after you release. Press Hold position to stop instantly.</p>
  </section>

  <section aria-label="Joint positions">
    <div class="gauges" id="gauges"></div>
    <div class="legend"><span class="l-live">Where the joint is</span><span class="l-tgt">Where it's headed</span></div>
  </section>
</main>

<footer>
  <p>Keys: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> or arrows move the stick, <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> switch mode, <kbd>Z</kbd><kbd>X</kbd> gripper in any mode, <kbd>Space</kbd> hold, <kbd>H</kbd> home</p>
  <p id="padText">No gamepad detected. Plug one in and press any button to use it.</p>
</footer>

<script>
const JOINTS = [
  { id: "shoulder_pan", name: "Base", unit: "°" },
  { id: "shoulder_lift", name: "Shoulder", unit: "°" },
  { id: "elbow_flex", name: "Elbow", unit: "°" },
  { id: "wrist_flex", name: "Wrist flex", unit: "°" },
  { id: "wrist_roll", name: "Wrist roll", unit: "°" },
  { id: "gripper", name: "Gripper", unit: "" },
];
const JOINT = Object.fromEntries(JOINTS.map(j => [j.id, j]));
const MODES = [
  { name: "Arm", x: "shoulder_pan", y: "shoulder_lift" },
  { name: "Elbow and wrist", x: "wrist_flex", y: "elbow_flex" },
  { name: "Hand", x: "wrist_roll", y: "gripper" },
];
const TRAVEL = 0.62;       // how far the cap travels, as a fraction of the well radius
const DEADZONE = 0.07;

const ui = {
  mode: 0,
  drag: null,              // {x, y} while the pointer holds the stick
  keys: new Set(),
  flip: Object.fromEntries(JOINTS.map(j => [j.id, false])),
  sliding: {},
  speed: 30,
  built: false,
  locked: [],
  pad: { name: null, prev: {} },
  phys: { x: 0, y: 0, vx: 0, vy: 0 },
};
const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* ---------- input ---------- */

function keyVector() {
  const k = ui.keys;
  let x = (k.has("d") || k.has("arrowright") ? 1 : 0) - (k.has("a") || k.has("arrowleft") ? 1 : 0);
  let y = (k.has("s") || k.has("arrowdown") ? 1 : 0) - (k.has("w") || k.has("arrowup") ? 1 : 0);
  const m = Math.hypot(x, y);
  return m > 1 ? { x: x / m, y: y / m } : { x, y };
}

function readGamepad() {
  const out = { x: 0, y: 0, grip: 0 };
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  ui.pad.name = null;
  for (const p of pads) {
    if (!p) continue;
    ui.pad.name = p.id;
    const btn = i => (p.buttons[i] ? p.buttons[i].value : 0);
    out.x = p.axes[0] || 0;
    out.y = p.axes[1] || 0;
    out.grip = btn(7) - btn(6);
    const edge = (i, fn) => {
      const down = btn(i) > 0.5;
      if (down && !ui.pad.prev[i]) fn();
      ui.pad.prev[i] = down;
    };
    edge(0, hold);
    edge(1, home);
    edge(4, () => setMode((ui.mode + 2) % 3));
    edge(5, () => setMode((ui.mode + 1) % 3));
    break;
  }
  return out;
}

/* Where the stick is being pushed right now: x right, y down, inside the unit circle */
function stickInput(gp) {
  let x = 0, y = 0;
  if (ui.drag) { x = ui.drag.x; y = ui.drag.y; }
  else {
    const kv = keyVector();
    x = kv.x + gp.x; y = kv.y + gp.y;
  }
  const m = Math.hypot(x, y);
  if (m > 1) { x /= m; y /= m; }
  return { x, y };
}

/* Radial deadzone plus an expo curve, so small pushes give fine control */
function shape(v) {
  const m = Math.hypot(v.x, v.y);
  if (m < DEADZONE) return { x: 0, y: 0 };
  const scaled = Math.pow((m - DEADZONE) / (1 - DEADZONE), 1.7);
  return { x: v.x / m * scaled, y: v.y / m * scaled };
}

function velocities(inp, gp) {
  const vel = Object.fromEntries(JOINTS.map(j => [j.id, 0]));
  const mode = MODES[ui.mode];
  const s = shape(inp);
  vel[mode.x] = s.x;
  vel[mode.y] = -s.y;
  const gripKeys = (ui.keys.has("x") ? 1 : 0) - (ui.keys.has("z") ? 1 : 0) + gp.grip;
  if (gripKeys) vel.gripper = clamp(gripKeys, -1, 1);
  for (const j of JOINTS) if (ui.flip[j.id]) vel[j.id] = -vel[j.id];
  return vel;
}

/* ---------- stick physics: a sprung, damped lever ---------- */

function stepPhysics(dt, target) {
  const p = ui.phys;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) { p.x = target.x; p.y = target.y; p.vx = p.vy = 0; return; }
  const held = target.x !== 0 || target.y !== 0 || ui.drag;
  // Held: stiff, critically damped follow. Released: softer spring that overshoots a little.
  const k = held ? 900 : 260;
  const c = held ? 2 * Math.sqrt(k) : 2 * 0.3 * Math.sqrt(k);
  const n = 8, h = dt / n;
  for (let i = 0; i < n; i++) {
    p.vx += (k * (target.x - p.x) - c * p.vx) * h;
    p.vy += (k * (target.y - p.y) - c * p.vy) * h;
    p.x += p.vx * h;
    p.y += p.vy * h;
  }
  const m = Math.hypot(p.x, p.y);
  if (m > 1.06) { p.x *= 1.06 / m; p.y *= 1.06 / m; }
}

function drawStick() {
  const p = ui.phys, well = $("well");
  const R = well.clientWidth / 2 * TRAVEL;
  const m = Math.hypot(p.x, p.y);
  $("cap").style.transform = `translate(${(p.x * R).toFixed(1)}px, ${(p.y * R).toFixed(1)}px) scale(${(1 + m * 0.06).toFixed(3)})`;
  $("cap").style.boxShadow =
    `${(5 + p.x * 9).toFixed(1)}px ${(12 + p.y * 9).toFixed(1)}px ${(16 + m * 12).toFixed(0)}px rgba(28,39,49,.45), inset 0 2px 2px rgba(255,255,255,.18)`;
  const sh = $("shaft");
  sh.setAttribute("x2", (p.x * TRAVEL).toFixed(3));
  sh.setAttribute("y2", (p.y * TRAVEL).toFixed(3));
}

function setupWell() {
  const well = $("well");
  let pid = null;
  const move = e => {
    const r = well.getBoundingClientRect();
    const R = r.width / 2;
    let x = (e.clientX - r.left - R) / (R * TRAVEL);
    let y = (e.clientY - r.top - R) / (R * TRAVEL);
    const m = Math.hypot(x, y);
    if (m > 1) { x /= m; y /= m; }
    ui.drag = { x, y };
  };
  well.addEventListener("pointerdown", e => {
    pid = e.pointerId; well.setPointerCapture(pid); well.classList.add("active"); move(e);
  });
  well.addEventListener("pointermove", e => { if (e.pointerId === pid) move(e); });
  const end = e => {
    if (e.pointerId !== pid) return;
    pid = null; ui.drag = null; well.classList.remove("active");
  };
  well.addEventListener("pointerup", end);
  well.addEventListener("pointercancel", end);
}

/* ---------- modes ---------- */

function buildModes() {
  const wrap = $("modes");
  MODES.forEach((m, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span>${i + 1}</span>${m.name}`;
    b.addEventListener("click", () => setMode(i));
    wrap.appendChild(b);
  });
}

function setMode(i) {
  ui.mode = i;
  const m = MODES[i];
  [...$("modes").children].forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
  $("axisX").textContent = `◀ ${JOINT[m.x].name} ▶`;
  $("axisY").textContent = `◀ ${JOINT[m.y].name} ▶`;
  $("well").setAttribute("aria-label", `Joystick moving ${JOINT[m.x].name} left and right, ${JOINT[m.y].name} up and down`);
  for (const j of JOINTS) if (j.el) j.el.card.classList.toggle("active", j.id === m.x || j.id === m.y);
}

/* ---------- gauges ---------- */

const CX = 60, CY = 58, RAD = 44, SWEEP = 270;
function polar(frac, r = RAD) {
  const a = (-SWEEP / 2 + SWEEP * frac - 90) * Math.PI / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}
function arcPath(f0, f1) {
  const [x0, y0] = polar(f0), [x1, y1] = polar(f1);
  const large = (f1 - f0) * SWEEP > 180 ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${RAD} ${RAD} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

function buildGauges(s) {
  const wrap = $("gauges");
  ui.locked = s.locked || [];
  for (const j of JOINTS) {
    const [lo, hi] = s.limits[j.id];
    const locked = ui.locked.includes(j.id);
    const card = document.createElement("div");
    card.className = "gauge" + (locked ? " locked" : "");
    card.innerHTML = `
      <h3>${j.name}</h3>
      <button class="flip" aria-pressed="false" title="Reverse the joystick direction for this joint">Flip</button>
      <svg viewBox="0 0 120 116" role="img" aria-label="${j.name} position">
        <path class="track" d="${arcPath(0, 1)}"></path>
        <path class="fill"></path>
        <line class="needle" x1="${CX}" y1="${CY}" x2="${CX}" y2="${CY - 30}"></line>
        <circle class="hub" cx="${CX}" cy="${CY}" r="4"></circle>
        <circle class="tgt" r="5"></circle>
        <text class="val" x="${CX}" y="${CY + 36}">0</text>
        <text class="sub" x="${CX}" y="${CY + 52}">target 0</text>
      </svg>
      <input type="range" min="${lo}" max="${hi}" step="0.5" aria-label="Send ${j.name} to a position" ${locked ? "disabled" : ""}>`;
    wrap.appendChild(card);

    const flip = card.querySelector(".flip");
    flip.addEventListener("click", () => {
      ui.flip[j.id] = !ui.flip[j.id];
      flip.setAttribute("aria-pressed", String(ui.flip[j.id]));
    });

    const slider = card.querySelector("input");
    slider.addEventListener("pointerdown", () => { ui.sliding[j.id] = true; });
    slider.addEventListener("input", () => {
      ui.sliding[j.id] = true;
      post("/api/set", { joint: j.id, value: parseFloat(slider.value) }).catch(() => {});
    });
    const release = () => { ui.sliding[j.id] = false; };
    slider.addEventListener("change", release);
    slider.addEventListener("pointerup", release);
    slider.addEventListener("blur", release);

    j.el = {
      card, slider, lo, hi,
      fill: card.querySelector(".fill"),
      needle: card.querySelector(".needle"),
      tgt: card.querySelector(".tgt"),
      val: card.querySelector(".val"),
      sub: card.querySelector(".sub"),
    };
  }

  if (ui.locked.length) {
    const names = ui.locked.map(id => JOINT[id].name).join(", ");
    $("banner").innerHTML = `<strong>${names} can't move.</strong> Calibration recorded almost no movement for ${ui.locked.length > 1 ? "these joints" : "this joint"}, so ${ui.locked.length > 1 ? "they're" : "it's"} locked. Stop the script with Ctrl+C, run lerobot-calibrate again, and move each one all the way to both ends before pressing Enter the second time.`;
    $("banner").hidden = false;
  }
  ui.built = true;
  setMode(ui.mode);
}

function render(s) {
  if (!ui.built) buildGauges(s);
  for (const j of JOINTS) {
    const e = j.el, span = e.hi - e.lo || 1;
    const p = s.present[j.id], t = s.target[j.id];
    const fp = clamp((p - e.lo) / span, 0, 1), ft = clamp((t - e.lo) / span, 0, 1);
    const [nx, ny] = polar(fp, RAD - 12);
    e.needle.setAttribute("x2", nx.toFixed(2));
    e.needle.setAttribute("y2", ny.toFixed(2));
    const zero = clamp((0 - e.lo) / span, 0, 1);
    const a = Math.min(zero, fp), b = Math.max(zero, fp);
    e.fill.setAttribute("d", b - a > 0.002 ? arcPath(a, b) : "");
    const [tx, ty] = polar(ft);
    e.tgt.setAttribute("cx", tx.toFixed(2));
    e.tgt.setAttribute("cy", ty.toFixed(2));
    e.val.textContent = p.toFixed(1) + j.unit;
    const atLimit = (s.at_limit || []).includes(j.id);
    e.card.classList.toggle("limit", atLimit);
    e.sub.textContent = ui.locked.includes(j.id) ? "Needs calibration"
      : atLimit ? "At its limit" : "target " + t.toFixed(1) + j.unit;
    if (!ui.sliding[j.id]) e.slider.value = t;
  }
}

/* ---------- network ---------- */

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

function setStatus(kind, text) {
  $("dot").className = "dot " + kind;
  $("statusText").textContent = text;
}

function hold() { post("/api/hold").then(render).catch(() => {}); }
function home() { post("/api/home").then(render).catch(() => {}); }

let latest = { inp: { x: 0, y: 0 }, gp: { x: 0, y: 0, grip: 0 } };
let inflight = false;
async function tick() {
  if (inflight) return;
  inflight = true;
  try {
    const vel = document.hidden ? {} : velocities(latest.inp, latest.gp);
    const s = await post("/api/tick", { vel, speed: ui.speed });
    render(s);
    if (s.error) setStatus("bad", s.error + " Check the power and cables.");
    else setStatus("ok", "Connected");
  } catch (err) {
    setStatus("bad", "Lost connection to the script. Check the terminal, then reload this page.");
  } finally {
    inflight = false;
  }
}

/* ---------- frame loop ---------- */

function pct(v) { return (v < 0 ? "−" : "") + Math.round(Math.abs(v) * 100) + "%"; }

let lastT = performance.now();
function frame(t) {
  const dt = Math.min(0.05, (t - lastT) / 1000);
  lastT = t;
  const gp = readGamepad();
  const inp = stickInput(gp);
  latest = { inp, gp };

  stepPhysics(dt, inp);
  drawStick();

  const m = MODES[ui.mode], vel = velocities(inp, gp);
  $("readout").innerHTML =
    `<span>${JOINT[m.x].name} <b>${pct(vel[m.x])}</b></span><span>${JOINT[m.y].name} <b>${pct(vel[m.y])}</b></span>` +
    (m.y !== "gripper" && vel.gripper ? `<span>Gripper <b>${pct(vel.gripper)}</b></span>` : "");

  $("padText").textContent = ui.pad.name
    ? "Gamepad connected. Left stick drives, bumpers switch mode, triggers work the gripper, A holds, B goes home."
    : "No gamepad detected. Plug one in and press any button to use it.";
  requestAnimationFrame(frame);
}

/* ---------- wiring ---------- */

buildModes();
setupWell();
setMode(0);

$("speed").addEventListener("input", e => {
  ui.speed = parseInt(e.target.value, 10);
  $("speedOut").textContent = ui.speed + "°/s";
});
$("holdBtn").addEventListener("click", hold);
$("homeBtn").addEventListener("click", home);

const MOVE_KEYS = new Set(["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright", "z", "x"]);
addEventListener("keydown", e => {
  if (e.target.matches("input")) return;
  const k = e.key.toLowerCase();
  if (MOVE_KEYS.has(k)) { ui.keys.add(k); e.preventDefault(); }
  else if (k === "1" || k === "2" || k === "3") setMode(parseInt(k, 10) - 1);
  else if (k === " " && !e.target.matches("button")) { e.preventDefault(); hold(); }
  else if (k === "h" && !e.repeat) home();
});
addEventListener("keyup", e => ui.keys.delete(e.key.toLowerCase()));

function releaseEverything() {
  ui.keys.clear();
  ui.drag = null;
  $("well").classList.remove("active");
}
addEventListener("blur", releaseEverything);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseEverything(); });

fetch("/api/state").then(r => r.json()).then(s => {
  render(s);
  setStatus(s.error ? "bad" : "ok", s.error || "Connected");
}).catch(() => setStatus("bad", "Can't reach the script. Check the terminal, then reload this page."));

setInterval(tick, 50);
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
