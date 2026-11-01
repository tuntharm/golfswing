"""Redraw a marked swing with the overlay elements.

Shaft lines are kept in the elements file and left off the video.
The clubhead path stays on the video.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from swinglab.analyze import _grip_from_pose, detect_poses
from swinglab.render import draw_swing_fan, overlay_catalog, write_playback


def marked_samples(frames: list[np.ndarray], marks: dict, poses: list[dict | None]) -> list[dict]:
    clubhead = {int(frame): tuple(point) for frame, point in marks["clubhead"].items()}
    order = np.array(sorted(clubhead))
    xs = np.array([clubhead[frame][0] for frame in order], float)
    ys = np.array([clubhead[frame][1] for frame in order], float)
    first = int(marks["first"])
    last = int(marks["last"])
    top = int(marks["top"])
    samples = []
    for index, pose in enumerate(poses):
        grip = _grip_from_pose(pose) if pose else None
        if grip is None or not (first <= index <= last):
            samples.append({"grip": None, "head": None, "downswing": False})
            continue
        samples.append(
            {
                "grip": (float(grip[0]), float(grip[1])),
                "head": (float(np.interp(index, order, xs)), float(np.interp(index, order, ys))),
                "downswing": index > top,
            }
        )
    return samples


def render_marked_swing(
    frames: list[np.ndarray],
    marks: dict,
    video_path: Path,
    elements_path: Path,
    *,
    fps: float = 66.0,
    playback_speed: float = 0.25,
) -> dict:
    poses = detect_poses(frames, fps)
    samples = marked_samples(frames, marks, poses)
    first = int(marks["first"])
    last = int(marks["last"])
    catalog = overlay_catalog(samples[first : last + 1])
    clip = [draw_swing_fan(frames[index], samples, index) for index in range(first, last + 1)]
    clip += [clip[-1]] * 25
    write_playback(clip, video_path, fps, playback_speed)
    elements_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog
