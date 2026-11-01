"""Measure one swing video and write a 0.125x drawing plus a report.

Example:
    python -m swinglab path/to/swing.mp4 -o swing-out
"""

from __future__ import annotations

import argparse
import json

from swinglab.analyze import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a down-the-line golf swing video.")
    parser.add_argument("video", help="Path to the swing video")
    parser.add_argument("-o", "--out", default="swing-out", help="Directory for the video and report")
    parser.add_argument("--handed", choices=("right", "left"), default="right")
    parser.add_argument("--speed", type=float, default=0.125, help="Playback speed baked into the video")
    args = parser.parse_args()
    report = analyze_video(args.video, args.out, handed=args.handed, playback_speed=args.speed)
    print(json.dumps({key: report[key] for key in ("reading", "readingReason", "playbackSpeed", "videoPath") if key in report}, indent=2))


if __name__ == "__main__":
    main()
