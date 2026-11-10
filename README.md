# the_mouse_eye

Control your mouse with your eyes using a normal laptop webcam (MediaPipe FaceMesh iris tracking + PyAutoGUI).

## Install

```bash
pip install -r requirements.txt
python eye_mouse.py
```

Python 3.9–3.12 recommended. On Linux PyAutoGUI may need `sudo apt install python3-tk python3-dev scrot` (X11 session); on macOS grant Camera and Accessibility permission to your terminal.

## How it works

On first run a full-screen **9-point calibration** shows dots; look at each one. The calibration is saved to `calibration.json` and reused next time.

| Action | Gesture |
|---|---|
| Move cursor | Look where you want it to go |
| Left click | Wink **left** eye (~0.3 s) or blink **both** eyes (~0.5 s) |
| Right click | Wink **right** eye (~0.3 s) |
| Scroll up / down | Look at the **top / bottom 7 %** strip of the screen (further in = faster) |

Quick natural blinks are ignored, you have to hold the blink slightly longer.

Preview window keys: `c` recalibrate, `p` pause/resume, `q` or `Esc` quit.

## Options

```
--camera N          camera index (default 0)
--recalibrate       force calibration
--swap-eyes         swap left/right wink buttons if they feel reversed
--smooth 0.8        lower = smoother but laggier cursor
--wink-time 0.30    --blink-time 0.50    --closed-ratio 0.62
--scroll-zone 0.07  --scroll-speed 60
```

## Tips

- Good, even front lighting; camera at eye level; keep your head roughly where it was during calibration (recalibrate with `c` if you move).
- If clicks fire too easily, raise `--wink-time` or lower `--closed-ratio`; if they never fire, raise `--closed-ratio` (e.g. 0.7).
- Use `p` to pause when you want to type or look away.
