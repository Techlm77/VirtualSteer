# VirtualSteer

Camera-based steering input system for games and simulators on Linux.  
Track your hands via webcam using MediaPipe and output a smooth, low-latency virtual joystick using `uinput`.  
Designed for **BeamNG.drive**, but also may works with games that support multiple controllers such as **Assetto Corsa**, **DiRT Rally 2.0**, **CarX Drift Racing Online**, and **Euro Truck Simulator 2**.

---

## Features

- Hand tracking via MediaPipe (one or both hands)
- Smooth, low-latency steering using a OneEuro filter
- Virtual joystick output via `uinput`
- Configurable parameters (cutoff, beta, max angle, neutral)
- GUI and command-line modes
- Persistent configuration saved in `~/.virtualsteer.json`

---

## Requirements

Install the dependencies:

```bash
sudo apt install python3 python3-pip python3-opencv libopencv-dev libuinput-dev
pip install mediapipe PyQt5 numpy
```

If joystick creation fails, add your user to the input group:

```bash
sudo usermod -aG input $USER
sudo modprobe uinput
```

Then log out and back in.

---

## Running VirtualSteer

### GUI Mode (recommended)
```bash
python3 start.py --emit --gui --show-fps
```

### CLI / Headless Mode
```bash
python3 start.py --emit --headless
```

### Optional Arguments
| Flag | Description |
|------|--------------|
| `--camera N` | Select camera index (default `0`) |
| `--max-deg X` | Maximum steering angle (default `45.0`) |
| `--invert` | Invert steering direction |
| `--neutral N` | Manually set neutral center |
| `--mincutoff` | OneEuro filter minimum cutoff |
| `--beta` | OneEuro filter beta |
| `--dcutoff` | OneEuro filter derivative cutoff |
| `--stdout` | Output telemetry (`time,angle,axis`) to stdout |
| `--show-fps` | Show FPS overlay |
| `--save-config` | Force save parameters after run |

---

## GUI Controls

| Action | Shortcut |
|---------|-----------|
| Start camera | `Start` |
| Stop camera | `Stop` |
| Center (set neutral) | `C` |
| Reset neutral | `R` |
| Invert steering | `I` |
| Save config | `S` |
| Quit | `Q` |

---

## Setup Example

1. Launch the GUI version.
2. Make sure your hands are visible in front of the camera.
3. Press **C** to set a neutral position.
4. Tune filter parameters (mincutoff, beta, dcutoff) for stability.
5. Start BeamNG.drive or another racing sim.
6. In-game, bind the "Steering Axis" to **Virtual Camera Gamepad**.
7. Adjust in-game deadzone and sensitivity as needed.

---

## Troubleshooting

**Camera busy or unavailable?**  
Another process (like OBS or Discord) may be using `/dev/video0`. Close it first.

**Virtual joystick not created?**  
Run `lsmod | grep uinput`. If missing, enable it with `sudo modprobe uinput`.

**Qt GUI errors or missing plugins?**  
Install:  
```bash
sudo apt install libxcb-xinerama0 qtwayland5
```

**Unstable tracking?**  
Lower `--proc-width` / `--proc-height` or reduce FPS to 30.

---

## Config File

Location:
```
~/.virtualsteer.json
```

Example:
```json
{
  "neutral_deg": 0.0,
  "invert": false,
  "max_deg": 45.0,
  "mincutoff": 2.5,
  "beta": 0.08,
  "dcutoff": 1.0
}
```

---

Enjoy! :)
