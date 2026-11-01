"""Draw the plane, the shaft, and the clubhead trail, then save a slow video."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


def dashed_line(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 14,
    gap: int = 9,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = float(np.hypot(dx, dy))
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    traveled = 0.0
    while traveled < length:
        stop = min(traveled + dash, length)
        p1 = (int(round(start[0] + ux * traveled)), int(round(start[1] + uy * traveled)))
        p2 = (int(round(start[0] + ux * stop)), int(round(start[1] + uy * stop)))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        traveled += dash + gap


def extend_across_frame(
    a: tuple[float, float],
    b: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return None
    p1 = (int(round(a[0] - dx * 8)), int(round(a[1] - dy * 8)))
    p2 = (int(round(b[0] + dx * 8)), int(round(b[1] + dy * 8)))
    ok, c1, c2 = cv2.clipLine((0, 0, width, height), p1, p2)
    if not ok:
        return None
    return (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1]))


def _skeleton(image: np.ndarray, pose: dict | None) -> None:
    if not pose:
        return
    pairs = (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )
    color = (210, 205, 190)
    for left, right in pairs:
        a = pose.get(left)
        b = pose.get(right)
        if a is None or b is None:
            continue
        cv2.line(
            image,
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            color,
            2,
            cv2.LINE_AA,
        )
    nose = pose.get("nose")
    if nose is not None:
        cv2.circle(image, (int(nose[0]), int(nose[1])), 5, color, 1, cv2.LINE_AA)


# BGR. Blue is the way up. Orange is the way down.
UPSWING_COLOR = (255, 150, 20)
DOWNSWING_COLOR = (20, 140, 255)

# A shaft this close to vertical is perpendicular to the ground. The line stays
# in the drawing data. It is hidden until something turns that element on.
PERPENDICULAR_TILT_DEG = 75.0

DEFAULT_ELEMENTS = {
    "shaft_fan": False,
    "clubhead_curve": True,
    "clubhead": True,
    "perpendicular": False,
}


def shaft_tilt_deg(grip: tuple[float, float], head: tuple[float, float]) -> float:
    """0 is horizontal, 90 is vertical."""
    angle = abs(math.degrees(math.atan2(head[1] - grip[1], head[0] - grip[0]))) % 180
    if angle > 90:
        angle = 180 - angle
    return angle


def shaft_element(grip: tuple[float, float], head: tuple[float, float]) -> str:
    if shaft_tilt_deg(grip, head) >= PERPENDICULAR_TILT_DEG:
        return "perpendicular"
    return "shaft_fan"


def element_visible(elements: dict | None, name: str) -> bool:
    if elements is None or name not in elements:
        return DEFAULT_ELEMENTS[name]
    return bool(elements[name])


def overlay_catalog(samples: list[dict], elements: dict | None = None) -> dict:
    """Every drawn element, including the ones left off the picture."""
    flags = dict(DEFAULT_ELEMENTS)
    if elements:
        flags.update(elements)
    grouped: dict[str, list] = {"shaft_fan": [], "perpendicular": []}
    heads = []
    for sample in samples:
        grip = sample.get("grip")
        head = sample.get("head")
        if not grip or not head:
            continue
        grouped[shaft_element(grip, head)].append(
            {
                "grip": [round(float(grip[0]), 1), round(float(grip[1]), 1)],
                "head": [round(float(head[0]), 1), round(float(head[1]), 1)],
                "downswing": bool(sample.get("downswing")),
            }
        )
        heads.append(
            {
                "point": [round(float(head[0]), 1), round(float(head[1]), 1)],
                "downswing": bool(sample.get("downswing")),
            }
        )
    return {
        "shaft_fan": {"visible": flags["shaft_fan"], "lines": grouped["shaft_fan"]},
        "clubhead_curve": {"visible": flags["clubhead_curve"], "points": heads},
        "clubhead": {"visible": flags["clubhead"]},
        "perpendicular": {"visible": flags["perpendicular"], "lines": grouped["perpendicular"]},
    }


def _point_xy(point) -> tuple[float, float] | None:
    if point is None:
        return None
    return (float(point[0]), float(point[1]))


def _point_downswing(point) -> bool:
    return len(point) >= 3 and bool(point[2])


def _trail(image: np.ndarray, points: list) -> None:
    previous = None
    limit = max(image.shape[0], image.shape[1]) * 0.22
    for point in points:
        current = _point_xy(point)
        if current is None or previous is None:
            previous = point if current is not None else None
            continue
        prev_xy = _point_xy(previous)
        if math.hypot(current[0] - prev_xy[0], current[1] - prev_xy[1]) > limit:
            previous = point
            continue
        color = DOWNSWING_COLOR if _point_downswing(point) else UPSWING_COLOR
        cv2.line(
            image,
            (int(prev_xy[0]), int(prev_xy[1])),
            (int(current[0]), int(current[1])),
            color,
            7,
            cv2.LINE_AA,
        )
        previous = point


def _smooth_path(points: list[tuple[float, float]], window: int = 5) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    xs = np.array([point[0] for point in points], dtype=float)
    ys = np.array([point[1] for point in points], dtype=float)
    kernel = np.ones(window) / window
    if len(points) >= window:
        xs = np.convolve(xs, kernel, mode="same")
        ys = np.convolve(ys, kernel, mode="same")
        # The same-mode edges are biased. Keep the real ends.
        xs[0], ys[0] = points[0]
        xs[-1], ys[-1] = points[-1]
    return list(zip(xs.tolist(), ys.tolist()))


def _draw_shaft(image: np.ndarray, sample: dict, color: tuple[int, int, int], thickness: int, elements: dict | None) -> None:
    grip = sample["grip"]
    head = sample["head"]
    if not element_visible(elements, shaft_element(grip, head)):
        return
    cv2.line(
        image,
        (int(grip[0]), int(grip[1])),
        (int(head[0]), int(head[1])),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_swing_fan(
    frame: np.ndarray,
    samples: list[dict],
    up_to: int,
    *,
    show_legend: bool = True,
    elements: dict | None = None,
) -> np.ndarray:
    """Draw the club the way a launch-monitor screen does.

    Each sample is grip, head, and whether that moment is the downswing.
    The clubhead curve stays on screen. Shaft lines are kept in the element
    data and left off the picture unless ``elements['shaft_fan']`` is true.
    """
    image = frame.copy()
    overlay = image.copy()
    shown = [sample for sample in samples[: up_to + 1] if sample.get("head") and sample.get("grip")]
    step = 1 if len(shown) < 18 else max(1, len(shown) // 16)
    if element_visible(elements, "shaft_fan") or element_visible(elements, "perpendicular"):
        for sample in shown[::step]:
            color = DOWNSWING_COLOR if sample.get("downswing") else UPSWING_COLOR
            _draw_shaft(overlay, sample, color, 3, elements)
        cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    if element_visible(elements, "clubhead_curve"):
        up_heads = [sample["head"] for sample in shown if not sample.get("downswing")]
        down_heads = [sample["head"] for sample in shown if sample.get("downswing")]
        for heads, color in ((up_heads, UPSWING_COLOR), (down_heads, DOWNSWING_COLOR)):
            curve = _smooth_path(heads, 5)
            for start, end in zip(curve, curve[1:]):
                cv2.line(
                    image,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color,
                    8,
                    cv2.LINE_AA,
                )
            for head in curve[:: max(1, len(curve) // 12)]:
                cv2.circle(image, (int(head[0]), int(head[1])), 7, color, -1, cv2.LINE_AA)

    current = samples[up_to] if up_to < len(samples) else None
    if current and current.get("grip") and current.get("head"):
        color = DOWNSWING_COLOR if current.get("downswing") else UPSWING_COLOR
        if element_visible(elements, "shaft_fan") or element_visible(elements, "perpendicular"):
            _draw_shaft(image, current, color, 6, elements)
        if element_visible(elements, "clubhead"):
            cv2.circle(image, (int(current["head"][0]), int(current["head"][1])), 12, (0, 0, 255), -1, cv2.LINE_AA)
    if show_legend:
        cv2.rectangle(image, (12, 12), (430, 92), (10, 14, 18), -1)
        cv2.circle(image, (32, 58), 9, UPSWING_COLOR, -1)
        cv2.circle(image, (210, 58), 9, DOWNSWING_COLOR, -1)
        cv2.putText(image, "0.25x", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "up", (48, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "down", (226, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def paint_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw the readout after any mirroring, so the words stay readable."""
    if not lines:
        return image
    _hud(image, lines)
    return image


def _hud(image: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    height = 16 + 22 * len(lines)
    width = min(image.shape[1] - 16, 460)
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (12, 16, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 32 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def draw_analysis_frame(
    frame: np.ndarray,
    *,
    plane: tuple[tuple[float, float], tuple[float, float]] | None,
    grip: tuple[float, float] | None,
    head: tuple[float, float] | None,
    predicted: bool,
    above_plane_deg: float | None,
    trail: list[tuple[float, float] | None],
    pose: dict | None,
    hud_lines: list[str],
    after_top: bool,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    _trail(image, trail)
    if plane is not None:
        clipped = extend_across_frame(plane[0], plane[1], width, height)
        if clipped is not None:
            dashed_line(image, clipped[0], clipped[1], (0, 220, 255), 2)
    _skeleton(image, pose)
    if grip is not None and head is not None:
        steep = above_plane_deg is not None and above_plane_deg > 8
        if after_top and steep:
            color = (40, 70, 235)
        elif after_top and above_plane_deg is not None and above_plane_deg < -4:
            color = (80, 200, 70)
        else:
            color = (230, 230, 230)
        p1 = (int(round(grip[0])), int(round(grip[1])))
        p2 = (int(round(head[0])), int(round(head[1])))
        if predicted:
            dashed_line(image, p1, p2, color, 3, dash=8, gap=6)
        else:
            cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(image, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p1, 4, (255, 255, 255), -1, cv2.LINE_AA)
    _hud(image, hud_lines)
    return image


def write_playback(
    frames: list[np.ndarray],
    path: Path,
    fps: float,
    playback_speed: float,
) -> None:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    speed = min(1.0, max(0.1, float(playback_speed)))
    factor = 1.0 / speed
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-vf",
        f"setpts={factor:.4f}*PTS",
        "-r",
        f"{fps:.4f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}")
