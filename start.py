#!/usr/bin/env python3
import os, sys, json, warnings, math, argparse, threading, time, signal
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="google.protobuf")

for _var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
    _val = os.environ.get(_var, "")
    if "site-packages/cv2/qt/plugins" in _val:
        os.environ.pop(_var, None)

try:
    import uinput
    HAVE_UINPUT = True
except Exception:
    HAVE_UINPUT = False

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    HAVE_QT = True
except Exception:
    HAVE_QT = False

import cv2

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    HAVE_MEDIAPIPE = False


def _prepare_qt_environment():
    if not HAVE_QT:
        return

    try:
        from PyQt5.QtCore import QLibraryInfo
        plugins_path = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
        if plugins_path:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path
    except Exception:
        pass

    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

CFG_PATH = os.path.expanduser("~/.virtualsteer.json")
DEFAULT_CFG = {
    "neutral_deg": 0.0,
    "invert": False,
    "max_deg": 45.0,
    "mincutoff": 2.5,
    "beta": 0.08,
    "dcutoff": 1.0,
}

def load_cfg(path: str = CFG_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = {**DEFAULT_CFG, **data}
        return merged
    except Exception:
        return DEFAULT_CFG.copy()

def save_cfg(cfg: dict, path: str = CFG_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    except Exception:
        pass

class LowPass:
    def __init__(self):
        self.s = None
    def f(self, x, a):
        if self.s is None:
            self.s = x
            return x
        self.s = a * x + (1 - a) * self.s
        return self.s

def _alpha(dt, cutoff):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(1e-6, dt))

class OneEuro:
    def __init__(self, min_cutoff=2.5, beta=0.08, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx = LowPass()
        self.x = LowPass()
        self.prev_x = None
        self.prev_t = None
    def __call__(self, x, t):
        if self.prev_t is None:
            self.prev_t = t; self.prev_x = x
            return x
        dt = t - self.prev_t; self.prev_t = t
        if dt <= 0: dt = 1e-3
        dx = (x - self.prev_x) / dt; self.prev_x = x
        dx_hat = self.dx.f(dx, _alpha(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        return self.x.f(x, _alpha(dt, cutoff))

def clamp(v, lo, hi): return max(lo, min(hi, v))

_prev_angle = 0.0
def deg_from_vec(dx, dy):
    global _prev_angle
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle - _prev_angle) > 180:
        angle += 360 if angle < _prev_angle else -360
    _prev_angle = angle
    return angle

def landmark_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _unwrap_continuous(prev, raw_deg):
    if prev is None: return raw_deg
    prev_p = ((prev + 180) % 360) - 180
    diff = raw_deg - prev_p
    if diff <= -180: diff += 360 * ((-diff)//360 + 1)
    if diff > 180: diff -= 360 * (diff//360 + 1)
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return prev + diff

def _angle_from_two_hands(w, h, left_lm, right_lm):
    left_idx = landmark_xy(left_lm.landmark[5], w, h)
    right_idx = landmark_xy(right_lm.landmark[5], w, h)
    dx, dy = (right_idx - left_idx)
    return deg_from_vec(dx, dy)

def _angle_from_single_hand(w, h, hand_lm, label: str):
    wr = landmark_xy(hand_lm.landmark[0], w, h)
    idx = landmark_xy(hand_lm.landmark[5], w, h)
    dx, dy = (idx - wr)
    return deg_from_vec(dx, dy)

def mp_angle_from_result(w, h, res):
    if not getattr(res, 'multi_hand_landmarks', None):
        return 0.0, False, 0
    hands_lm = res.multi_hand_landmarks
    hands_h = getattr(res, 'multi_handedness', None)
    left_lm = right_lm = None
    if hands_h and len(hands_h) == len(hands_lm):
        for lm, hd in zip(hands_lm, hands_h):
            label = hd.classification[0].label
            if label == 'Left' and left_lm is None: left_lm = lm
            elif label == 'Right' and right_lm is None: right_lm = lm
    if (left_lm is None or right_lm is None) and len(hands_lm) >= 2:
        pairs = []
        for hand in hands_lm[:2]:
            idx = landmark_xy(hand.landmark[5], w, h)
            pairs.append((idx, hand))
        pairs.sort(key=lambda x: x[0][0])
        left_lm = pairs[0][1]; right_lm = pairs[1][1]
    if left_lm is not None and right_lm is not None:
        ang = _angle_from_two_hands(w, h, left_lm, right_lm)
        ang = ((ang + 180) % 360) - 180
        return ang, True, 2
    hand_lm = left_lm or right_lm or hands_lm[0]
    ang = _angle_from_single_hand(w, h, hand_lm, 'Left' if hand_lm is left_lm else 'Right')
    ang = ((ang + 180) % 360) - 180
    return ang, True, 1

def map_angle_to_axis(angle_deg, neutral_deg, max_deg, invert):
    rel = angle_deg - neutral_deg
    if invert: rel = -rel
    rel = clamp(rel, -max_deg, +max_deg)
    return clamp(int((rel / max_deg) * 32767), -32768, 32767)

def draw_hud(v, width=700, height=200, fps_txt=None, deg_txt=None, status_txt=None, help_on=False):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.rectangle(img, (10,10), (width-10, height-10), (60,60,60), 2)

    bx, by, bw, bh = 40, height // 2 - 14, width - 80, 28
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
    norm = (v + 32768) / 65535.0
    cx = int(bx + norm * bw)
    cv2.line(img, (cx, by), (cx, by + bh), (0, 255, 0), 3)

    cv2.putText(img, f"axis {v:+6d}", (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 230), 2)
    if deg_txt is not None:
        cv2.putText(img, deg_txt, (40, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if fps_txt is not None:
        cv2.putText(img, fps_txt, (width - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)
    if status_txt is not None:
        cv2.putText(img, status_txt, (40, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 200, 200), 1)
    return img

def parse_args():
    p = argparse.ArgumentParser(description="VirtualSteer GUI")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--proc-width", type=int, default=320)
    p.add_argument("--proc-height", type=int, default=180)
    p.add_argument("--emit", action="store_true", help="Emit joystick via uinput")
    p.add_argument("--stdout", action="store_true", help="Emit 't,angle,axis' to stdout")
    p.add_argument("--show-fps", action="store_true")
    p.add_argument("--headless", action="store_true", help="Disable GUI/HUD")
    p.add_argument("--gui", action="store_true", help="Force GUI (default if PyQt5 available)")
    p.add_argument("--max-deg", type=float, default=None)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--no-invert", action="store_true")
    p.add_argument("--neutral", type=float, default=None)
    p.add_argument("--mincutoff", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--dcutoff", type=float, default=None)

    p.add_argument("--save-config", action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()

@dataclass
class SharedState:
    frame: Optional[np.ndarray] = None
    angle: float = 0.0
    ok: bool = False
    hands_count: int = 0

class CoreRunner:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.stop = threading.Event()
        self.frame_lock = threading.Lock()
        self.shared = SharedState()

        self.euro = OneEuro(cfg["mincutoff"], cfg["beta"], cfg["dcutoff"])
        self._filter_params = (cfg["mincutoff"], cfg["beta"], cfg["dcutoff"])

        self.angle_ema = None
        self.angle_unwrap = None
        self.last_hands = 0
        self.last_ok = False
        self.last_emit = 0.0
        self.ts = deque(maxlen=120)

        self.udev = None
        self.cap = None
        self.hands = None

        self.tcap = None
        self.tinf = None

    def _maybe_update_filter(self):
        cur = (self.cfg["mincutoff"], self.cfg["beta"], self.cfg["dcutoff"])
        if cur != self._filter_params:
            self.euro = OneEuro(*cur)
            self._filter_params = cur

    def open(self):
        if not HAVE_MEDIAPIPE:
            raise RuntimeError("mediapipe not installed")

        self.cap = cv2.VideoCapture(self.args.camera, cv2.CAP_ANY)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        mp_h = mp.solutions.hands
        try:
            self.hands = mp_h.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        except TypeError:
            self.hands = mp_h.Hands(False, 2, 0.6, 0.6)

        if self.args.emit and HAVE_UINPUT:
            self.udev = self._device_make_uinput()

        self.stop.clear()
        self.tcap = threading.Thread(target=self._capture_loop, daemon=True)
        self.tinf = threading.Thread(target=self._infer_loop, daemon=True)
        self.tcap.start()
        self.tinf.start()

    def close(self):
        self.stop.set()
        try:
            if self.tcap is not None:
                self.tcap.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.tinf is not None:
                self.tinf.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def _device_make_uinput(self):
        absr = (-32768, 32767, 0, 0)
        events = [
            uinput.ABS_X + absr, uinput.ABS_Y + absr, uinput.ABS_RX + absr, uinput.ABS_RY + absr,
            uinput.ABS_Z + absr, uinput.ABS_RZ + absr,
            uinput.ABS_HAT0X + (-1, 1, 0, 0), uinput.ABS_HAT0Y + (-1, 1, 0, 0),
            uinput.BTN_SOUTH, uinput.BTN_EAST, uinput.BTN_NORTH, uinput.BTN_WEST,
            uinput.BTN_TL, uinput.BTN_TR, uinput.BTN_SELECT, uinput.BTN_START,
            uinput.BTN_THUMBL, uinput.BTN_THUMBR,
        ]
        try:
            return uinput.Device(
                events,
                name="Virtual Camera Gamepad",
                vendor=0x045E,
                product=0x028E,
                version=1
            )
        except Exception as e:
            print(f"[WARN] uinput create failed: {e}", file=sys.stderr)
            return None

    def _capture_loop(self):
        target = 1.0 / max(1, self.args.fps)
        last = time.perf_counter()
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                with self.frame_lock:
                    self.shared.frame = frame
            now = time.perf_counter()
            dt = now - last
            delay = max(0.0, target - dt)
            time.sleep(delay)
            last = now

    def _infer_loop(self):
        target = 1.0 / max(1, self.args.fps)
        pw, ph = self.args.proc_width, self.args.proc_height
        while not self.stop.is_set():
            with self.frame_lock:
                f = self.shared.frame
            if f is None:
                time.sleep(target)
                continue

            small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            ang, ok, hands_count = mp_angle_from_result(pw, ph, res)
            with self.frame_lock:
                self.shared.angle = ang
                self.shared.ok = ok
                self.shared.hands_count = hands_count

            time.sleep(target)

    def step(self):
        t = time.perf_counter()
        with self.frame_lock:
            ang = self.shared.angle
            ok = self.shared.ok
            hands_count = self.shared.hands_count

        if not ok or (self.last_ok and hands_count != self.last_hands):
            self.angle_unwrap = None
        self.last_hands = hands_count
        self.last_ok = ok

        self._maybe_update_filter()

        if ok:
            self.angle_unwrap = _unwrap_continuous(self.angle_unwrap, ang)
            if self.angle_unwrap is not None and abs(self.angle_unwrap) > 540:
                self.angle_unwrap = ang
            disp = self.euro(self.angle_unwrap, t)
            self.angle_ema = disp
        else:
            if self.angle_ema is not None:
                self.angle_ema *= 0.985
            disp = self.angle_ema or 0.0

        mapped = map_angle_to_axis(disp, self.cfg["neutral_deg"], self.cfg["max_deg"], self.cfg["invert"])

        if self.args.stdout:
            sys.stdout.write(f"{t:.6f},{disp:.3f},{mapped}\n")
            sys.stdout.flush()

        if self.udev is not None and ok:
            now = t
            if now - self.last_emit >= 1.0 / 120.0:
                try:
                    self.udev.emit(uinput.ABS_X, int(mapped), syn=False)
                    self.udev.syn()
                except Exception:
                    pass
                self.last_emit = now

        self.ts.append(t)
        fps = (len(self.ts) - 1) / (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0

        deg_txt = f"deg {(disp - self.cfg['neutral_deg']):+5.1f} (neutral {self.cfg['neutral_deg']:+.1f})"
        fps_txt = f"FPS {fps:.1f}" if self.args.show_fps else None
        stat = (
            f"{'HANDS' if ok else '    -'} x{hands_count} | "
            f"invert={'on' if self.cfg['invert'] else 'off'} | "
            f"max_deg={self.cfg['max_deg']:.1f} | "
            f"emit={'on' if self.udev else 'off'}"
        )
        hud = draw_hud(mapped, fps_txt=fps_txt, deg_txt=deg_txt, status_txt=stat)
        return hud, fps, disp, mapped, ok, hands_count

if HAVE_QT:
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, core: CoreRunner):
            super().__init__()
            self.core = core
            self.setWindowTitle("VirtualSteer")
            self.resize(980, 560)
            self._build_ui()
            self._apply_dark()

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.timer.start(1000 // 60)

        def _build_ui(self):
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)

            left = QtWidgets.QVBoxLayout()
            self.preview = QtWidgets.QLabel()
            self.preview.setMinimumSize(700, 200)
            self.preview.setAlignment(QtCore.Qt.AlignCenter)
            self.preview.setStyleSheet("background: #121212; border: 1px solid #333; border-radius: 8px;")
            left.addWidget(self.preview, 1)

            btn_row = QtWidgets.QHBoxLayout()
            self.btn_start = QtWidgets.QPushButton("Start")
            self.btn_stop = QtWidgets.QPushButton("Stop")
            self.btn_center = QtWidgets.QPushButton("Center (C)")
            self.btn_reset = QtWidgets.QPushButton("Reset")
            self.btn_save = QtWidgets.QPushButton("Save")
            btn_row.addWidget(self.btn_start)
            btn_row.addWidget(self.btn_stop)
            btn_row.addStretch(1)
            btn_row.addWidget(self.btn_center)
            btn_row.addWidget(self.btn_reset)
            btn_row.addWidget(self.btn_save)
            left.addLayout(btn_row)

            layout.addLayout(left, 3)

            right = QtWidgets.QVBoxLayout()

            self.chk_invert = QtWidgets.QCheckBox("Invert")
            self.chk_invert.setChecked(self.core.cfg["invert"])
            right.addWidget(self.chk_invert)

            self.spn_maxdeg = QtWidgets.QDoubleSpinBox()
            self.spn_maxdeg.setRange(5.0, 90.0)
            self.spn_maxdeg.setSingleStep(1.0)
            self.spn_maxdeg.setValue(self.core.cfg["max_deg"])
            self._add_form_row(right, "Max deg", self.spn_maxdeg)

            self.spn_min = QtWidgets.QDoubleSpinBox()
            self.spn_min.setRange(0.1, 10.0)
            self.spn_min.setSingleStep(0.1)
            self.spn_min.setValue(self.core.cfg["mincutoff"])
            self._add_form_row(right, "mincutoff", self.spn_min)

            self.spn_beta = QtWidgets.QDoubleSpinBox()
            self.spn_beta.setRange(0.0, 2.0)
            self.spn_beta.setSingleStep(0.01)
            self.spn_beta.setValue(self.core.cfg["beta"])
            self._add_form_row(right, "beta", self.spn_beta)

            self.spn_dc = QtWidgets.QDoubleSpinBox()
            self.spn_dc.setRange(0.1, 10.0)
            self.spn_dc.setSingleStep(0.1)
            self.spn_dc.setValue(self.core.cfg["dcutoff"])
            self._add_form_row(right, "dcutoff", self.spn_dc)

            right.addStretch(1)
            layout.addLayout(right, 2)

            self.status = QtWidgets.QStatusBar(self)
            self.setStatusBar(self.status)
            self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
            self.lbl_hands = QtWidgets.QLabel("Hands: 0")
            self.lbl_emit = QtWidgets.QLabel(f"Emit: {'on' if self.core.args.emit and HAVE_UINPUT else 'off'}")
            self.status.addPermanentWidget(self.lbl_fps)
            self.status.addPermanentWidget(self.lbl_hands)
            self.status.addPermanentWidget(self.lbl_emit)

            self.btn_start.clicked.connect(self._start)
            self.btn_stop.clicked.connect(self._stop)
            self.btn_center.clicked.connect(self._center)
            self.btn_reset.clicked.connect(self._reset)
            self.btn_save.clicked.connect(self._save)
            self.chk_invert.stateChanged.connect(self._invert_changed)
            self.spn_maxdeg.valueChanged.connect(self._params_changed)
            self.spn_min.valueChanged.connect(self._params_changed)
            self.spn_beta.valueChanged.connect(self._params_changed)
            self.spn_dc.valueChanged.connect(self._params_changed)

            QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self._center)
            QtWidgets.QShortcut(QtGui.QKeySequence("I"), self, activated=lambda: self.chk_invert.setChecked(not self.chk_invert.isChecked()))
            QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self._save)
            QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self._reset)
            QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self.close)

        def _apply_dark(self):
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(18, 18, 18))
            pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor(24, 24, 24))
            pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 30, 30))
            pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(28, 28, 28))
            pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(45, 140, 255))
            pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            self.setPalette(pal)

        def _add_form_row(self, layout, label, widget):
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(label)
            row.addWidget(lab)
            row.addWidget(widget, 1)
            layout.addLayout(row)

        def _start(self):
            try:
                self.core.open()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

        def _stop(self):
            self.core.close()

        def _center(self):
            if not self.core.cap:
                return
            _, _, disp, _, _, _ = self.core.step()
            self.core.cfg["neutral_deg"] = disp

        def _reset(self):
            self.core.cfg["neutral_deg"] = 0.0

        def _save(self):
            save_cfg(self.core.cfg)

        def _invert_changed(self, state):
            self.core.cfg["invert"] = (state == QtCore.Qt.Checked)

        def _params_changed(self, _=None):
            self.core.cfg["max_deg"] = float(self.spn_maxdeg.value())
            self.core.cfg["mincutoff"] = float(self.spn_min.value())
            self.core.cfg["beta"] = float(self.spn_beta.value())
            self.core.cfg["dcutoff"] = float(self.spn_dc.value())

        def _tick(self):
            if not self.core.cap:
                img = np.zeros((200, 700, 3), dtype=np.uint8)
                cv2.putText(img, "Click Start to begin", (60, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                self._update_preview(img)
                return

            hud, fps, disp, mapped, ok, hands = self.core.step()
            self._update_preview(hud)
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
            self.lbl_hands.setText(f"Hands: {hands}")

        def _update_preview(self, bgr):
            h, w, _ = bgr.shape
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.preview.width(),
                self.preview.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.preview.setPixmap(pix)

        def closeEvent(self, e):
            try:
                self.core.close()
            except Exception:
                pass
            super().closeEvent(e)

def main():
    if not HAVE_MEDIAPIPE:
        print("ERROR: mediapipe module not found. pip install mediapipe", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    cfg = load_cfg()

    if args.max_deg is not None: cfg["max_deg"] = float(args.max_deg)
    if args.neutral is not None: cfg["neutral_deg"] = float(args.neutral)
    if args.mincutoff is not None: cfg["mincutoff"] = float(args.mincutoff)
    if args.beta is not None: cfg["beta"] = float(args.beta)
    if args.dcutoff is not None: cfg["dcutoff"] = float(args.dcutoff)
    if args.invert and not args.no_invert: cfg["invert"] = True
    if args.no_invert and not args.invert: cfg["invert"] = False

    use_gui = (HAVE_QT and not args.headless)
    if args.gui:
        if not HAVE_QT:
            print("ERROR: --gui requested but PyQt5 is not installed. pip install PyQt5", file=sys.stderr)
            sys.exit(3)
        use_gui = True

    runner = CoreRunner(args, cfg)

    if use_gui:
        _prepare_qt_environment()

        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow(runner)
        win.show()
        rc = app.exec_()
        if args.save_config and not args.no_save:
            save_cfg(cfg)
        sys.exit(rc)
    else:
        try:
            runner.open()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        try:
            while True:
                runner.step()
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            runner.close()
            if args.save_config and not args.no_save:
                save_cfg(cfg)

if __name__ == "__main__":
    main()
