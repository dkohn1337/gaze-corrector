# Gaze Corrector

Real-time gaze correction for macOS. Makes you appear to look directly at the camera during video calls, running 100% locally on your Mac.

Works as a virtual camera that any app (Zoom, Teams, Google Meet, etc.) can use.

## How It Works

```
┌──────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Webcam  │────▶│  Gaze Corrector  │────▶│  OBS Virtual Cam │
│          │     │                  │     │                  │
│          │     │  1. Face detect  │     │  Appears as a    │
│          │     │  2. Iris track   │     │  normal camera   │
│          │     │  3. Warp eyes    │     │  in Zoom/Teams   │
└──────────┘     └──────────────────┘     └──────────────────┘
```

1. Captures your webcam feed
2. Detects face and iris landmarks using MediaPipe (478 points including iris)
3. Estimates where you're looking (gaze direction + head pose via solvePnP)
4. Warps the eye region to redirect your gaze toward the camera
5. Outputs the corrected feed to a virtual camera (OBS Virtual Camera)
6. Intelligently disengages when you look away (notes, second monitor) to preserve natural behavior

## Preview Mode

Run with `--preview --no-vcam` to see the debug overlay:

```
┌─────────────────────────────────────────┐
│ Gaze angle: 4.2 deg                    │
│ Head yaw: -2.1  pitch: 3.4             │
│ Blend: 1.00                            │
│ Strength: 70%                          │
│ Enabled: True                          │
│                                        │
│         ╭───╮       ╭───╮              │
│        │ ◉→ │     │ ◉→ │              │
│         ╰───╯       ╰───╯              │
│     [green eye contours with           │
│      iris dots and red correction      │
│      arrows showing the shift]         │
│                                        │
└─────────────────────────────────────────┘
```

The green outlines show detected eye contours, green dots mark iris positions, and red arrows show correction vectors.

## Requirements

- macOS 12+ (Intel or Apple Silicon)
- A webcam
- Internet connection (for initial setup only)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/dkohn1337/gaze-corrector.git ~/gaze_corrector
cd ~/gaze_corrector

# 2. Install everything (Homebrew, Python, OBS, dependencies)
./setup.sh

# 3. Test with preview window
./run.sh --preview --no-vcam

# 4. Run for real (virtual camera for Zoom/Teams)
./run.sh
```

Then select **"OBS Virtual Camera"** in your video call app's camera dropdown.

## Usage

### Basic — virtual camera output

```bash
./run.sh
```

A menu bar icon (👁) appears with controls. Select "OBS Virtual Camera" in Zoom/Teams/Meet.

### Debug / preview mode

```bash
./run.sh --preview --no-vcam
```

Shows a window with landmark overlays, gaze vectors, and FSM state. Press `q` to quit.

### All options

```
./run.sh [OPTIONS]

  --preview       Show debug preview window with overlays
  --no-vcam       Disable virtual camera output
  --no-tray       Disable macOS menu bar icon
  --strength 0.7  Correction strength, 0.0-1.0 (default: 0.7)
  --piecewise     Use piecewise affine warp (higher quality, slower)
  --camera 0      Camera device index (default: 0)
```

### Examples

```bash
# Preview mode for testing/debugging
./run.sh --preview --no-vcam

# Virtual camera with subtle correction
./run.sh --strength 0.5

# Full correction + preview to see what's happening
./run.sh --preview --strength 1.0

# Higher quality warp (uses more CPU)
./run.sh --piecewise

# Use external webcam
./run.sh --camera 1
```

## Menu Bar Controls

When running normally (without `--preview`), a 👁 menu bar icon appears:

- **Correction: ON/OFF** — toggle gaze correction
- **Strength** — submenu: 30%, 50%, **70% (default)**, 85%, 100%
- **Quit** — stop and exit

## Behavior Detection

The system uses a 4-state machine to avoid correcting when you're intentionally looking away:

```
  gaze near         gaze away
 ┌────────┐       ┌────────────┐
 │ENGAGED │──────▶│DISENGAGING │
 │ (ON)   │◀──────│ (fading)   │──── 0.4s ────▶ DISENGAGED (OFF)
 └────────┘       └────────────┘                      │
      ▲                                               │
      │              ┌──────────────┐                  │
      └── 0.2s ──────│ RE_ENGAGING  │◀── gaze near ───┘
                      │  (fading in) │
                      └──────────────┘
```

| State | Correction | Triggers |
|-------|-----------|----------|
| Engaged | Full | Looking near the camera |
| Disengaging | Fading out over 0.4s | Gaze > 25° or head yaw > 20° |
| Disengaged | Off | Looking at notes / other monitor |
| Re-engaging | Fading in over 0.2s | Gaze returns < 15° |

Head turns > 20° yaw or > 15° pitch always disengage immediately.

## Architecture

Three-thread pipeline connected by bounded drop-oldest queues:

```
[Capture Thread]        [Processing Thread]         [Output Thread]
cv2.VideoCapture  →     FaceMesh (478 landmarks) →  pyvirtualcam.send()
                        Gaze estimation (solvePnP)   + optional preview
                        Behavior FSM
                        Eye warping + blending
```

## Performance

Targets 30fps on both Intel and Apple Silicon Macs:

| Stage | Time |
|-------|------|
| Capture | ~1ms |
| MediaPipe FaceMesh | ~8-12ms |
| Gaze estimation + solvePnP | ~1-2ms |
| Gaze correction (affine) | ~2-3ms |
| Blending | ~1ms |
| Virtual camera output | ~0.5ms |
| **Total** | **~14-20ms** |

## Uninstall

```bash
./uninstall.sh
```

This removes the Python virtual environment. OBS Studio is kept (you may use it for other things).

To fully remove everything:

```bash
./uninstall.sh
brew uninstall --cask obs  # optional
rm -rf ~/gaze_corrector
```

## License

MIT
