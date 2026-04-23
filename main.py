"""Gaze Corrector — entry point with argument parsing and orchestration."""

import argparse
import signal
import sys

import config
from pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time gaze correction for macOS virtual camera",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show debug preview window with landmarks and gaze vectors",
    )
    parser.add_argument(
        "--no-vcam", action="store_true",
        help="Disable virtual camera output (useful for testing)",
    )
    parser.add_argument(
        "--no-tray", action="store_true",
        help="Disable macOS menu bar tray icon",
    )
    parser.add_argument(
        "--strength", type=float, default=config.DEFAULT_CORRECTION_STRENGTH,
        help=f"Correction strength 0.0-1.0 (default: {config.DEFAULT_CORRECTION_STRENGTH})",
    )
    parser.add_argument(
        "--piecewise", action="store_true",
        help="Use piecewise affine warp (higher quality, slower)",
    )
    parser.add_argument(
        "--camera", type=int, default=config.CAMERA_INDEX,
        help=f"Camera device index (default: {config.CAMERA_INDEX})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Gaze Corrector ===")
    print(f"  Strength: {args.strength:.0%}")
    print(f"  Preview:  {args.preview}")
    print(f"  VCam:     {not args.no_vcam}")
    print(f"  Warp:     {'piecewise' if args.piecewise else 'affine'}")
    print()

    pipeline = Pipeline(
        enable_vcam=not args.no_vcam,
        enable_preview=args.preview,
        correction_strength=args.strength,
        use_piecewise=args.piecewise,
        camera_index=args.camera,
    )

    # Graceful shutdown on Ctrl+C
    def handle_signal(sig, frame):
        print("\nShutting down...")
        pipeline.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline.start()

    # Menu bar tray (runs on main thread for macOS)
    use_tray = not args.no_tray and not args.preview
    if use_tray:
        try:
            from ui.tray import run_tray
            print("Menu bar icon active. Use it to toggle correction or quit.")
            run_tray(pipeline)  # Blocks until quit
        except ImportError:
            print("rumps not available, running without tray icon.")
            print("Press Ctrl+C to quit.")
            pipeline.wait()
        except Exception as e:
            print(f"Tray error: {e}. Running without tray.")
            pipeline.wait()
    else:
        if args.preview:
            print("Preview mode. Press 'q' in the preview window or Ctrl+C to quit.")
        else:
            print("Press Ctrl+C to quit.")
        pipeline.wait()

    pipeline.stop()
    pipeline.wait()
    print("Done.")


if __name__ == "__main__":
    main()
