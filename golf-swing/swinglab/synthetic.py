"""Cartoon swings with a known shaft, used to test the measurement."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from swinglab.analyze import above_plane_deg, elevation_deg


def _pose(grip: tuple[float, float]) -> dict:
    return {
        "nose": (430.0, 150.0),
        "left_shoulder": (390.0, 230.0),
        "right_shoulder": (470.0, 225.0),
        "left_elbow": (360.0, 290.0),
        "right_elbow": (500.0, 285.0),
        "left_wrist": (grip[0] - 8.0, grip[1]),
        "right_wrist": (grip[0] + 8.0, grip[1]),
        "left_hip": (400.0, 330.0),
        "right_hip": (465.0, 328.0),
        "left_knee": (395.0, 400.0),
        "right_knee": (470.0, 400.0),
        "left_ankle": (390.0, 450.0),
        "right_ankle": (475.0, 450.0),
        "visibility": 1.0,
    }


def _shift_to_offset(
    grip: tuple[float, float],
    head: tuple[float, float],
    plane_grip: tuple[float, float],
    plane_head: tuple[float, float],
    target_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Slide the shaft along the plane's sky normal until it hits target_deg."""
    from swinglab.analyze import sky_normal

    vx = plane_head[0] - plane_grip[0]
    vy = plane_head[1] - plane_grip[1]
    nx, ny = sky_normal(vx, vy)
    low, high = -500.0, 500.0
    for _ in range(48):
        mid = (low + high) / 2
        moved_grip = (grip[0] + mid * nx, grip[1] + mid * ny)
        moved_head = (head[0] + mid * nx, head[1] + mid * ny)
        measured = above_plane_deg(moved_grip, moved_head, plane_grip, plane_head)
        if measured < target_deg:
            low = mid
        else:
            high = mid
    offset = (low + high) / 2
    return (grip[0] + offset * nx, grip[1] + offset * ny), (
        head[0] + offset * nx,
        head[1] + offset * ny,
    )


def _shaft(angle_deg: float, length: float, grip: tuple[float, float]) -> tuple[float, float]:
    radians = np.deg2rad(angle_deg)
    return (
        grip[0] + length * float(np.cos(radians)),
        grip[1] + length * float(np.sin(radians)),
    )


def design_swing(kind: str) -> list[dict]:
    """Build a right-handed down-the-line swing.

    ``shallow`` drops the shaft under the address line by P6.
    ``steep`` leaves the shaft above that line.
    """
    if kind not in {"shallow", "steep"}:
        raise ValueError(kind)
    address_grip = (420.0, 210.0)
    address_head = (300.0, 390.0)
    plane = (address_grip, address_head)

    top_grip = (400.0, 150.0)
    top_head = _shaft(230.0, 170.0, top_grip)
    top_grip, top_head = _shift_to_offset(top_grip, top_head, *plane, 22.0)

    p6_grip = (390.0, 240.0)
    p6_head = _shaft(8.0, 160.0, p6_grip)
    target = -16.0 if kind == "shallow" else 20.0
    p6_grip, p6_head = _shift_to_offset(p6_grip, p6_head, *plane, target)

    def sample(grip: tuple[float, float], head: tuple[float, float]) -> dict:
        return {"grip": grip, "head": head, "pose": _pose(grip)}

    # Address is held long enough that the cartoon tempo is in a normal range.
    # The poses themselves are what the measurement is checked against.
    keys = [
        (0, 24, sample(address_grip, address_head)),
        (24, 30, sample(top_grip, top_head)),
        (30, 36, sample(p6_grip, p6_head)),
        (36, 48, sample(address_grip, address_head)),
    ]
    frames: list[dict] = []
    for start, stop, pose in keys:
        for _ in range(start, stop):
            frames.append(
                {
                    "grip": pose["grip"],
                    "head": pose["head"],
                    "pose": pose["pose"],
                    "confidence": 1.0,
                    "predicted": False,
                    "angle": None,
                }
            )
    # One-frame joins so the phase finder sees a direction change, without
    # spending long enough at an intermediate angle to steal P6.
    return frames


def paint_swing(samples: list[dict], size: tuple[int, int] = (640, 480)) -> list[np.ndarray]:
    width, height = size
    frames = []
    rng = np.random.default_rng(4)
    for sample in samples:
        image = np.zeros((height, width, 3), np.uint8)
        image[:220, :] = (180, 120, 70)
        image[220:, :] = (45, 130, 55)
        noise = rng.integers(0, 18, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        pose = sample["pose"]
        for joint in (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_elbow",
            "right_elbow",
        ):
            point = pose[joint]
            cv2.circle(image, (int(point[0]), int(point[1])), 16, (40, 50, 70), -1, cv2.LINE_AA)
        cv2.circle(image, (int(pose["nose"][0]), int(pose["nose"][1])), 18, (50, 60, 90), -1, cv2.LINE_AA)
        grip = sample["grip"]
        head = sample["head"]
        cv2.line(
            image,
            (int(grip[0]), int(grip[1])),
            (int(head[0]), int(head[1])),
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.circle(image, (int(head[0]), int(head[1])), 8, (10, 10, 10), -1, cv2.LINE_AA)
        frames.append(image)
    return frames


def write_pose_sidecar(samples: list[dict], path: Path) -> None:
    payload = []
    for sample in samples:
        pose = {}
        for key, value in sample["pose"].items():
            pose[key] = list(value) if isinstance(value, tuple) else value
        payload.append(pose)
    path.write_text(json.dumps(payload))


def write_raw_video(frames: list[np.ndarray], path: Path, fps: float = 30.0) -> None:
    from swinglab.render import write_playback

    write_playback(frames, path, fps, playback_speed=1.0)


def true_plane_readings(samples: list[dict]) -> dict[str, float]:
    plane_grip = samples[0]["grip"]
    plane_head = samples[0]["head"]
    readings = []
    for sample in samples:
        readings.append(
            {
                "above": above_plane_deg(sample["grip"], sample["head"], plane_grip, plane_head),
                "elevation": elevation_deg(sample["grip"], sample["head"]),
                "head_y": sample["head"][1],
            }
        )
    return {"frames": readings}
