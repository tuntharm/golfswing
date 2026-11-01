"""Measure a down-the-line swing from one camera.

The address shaft is frozen as a line on the picture. ``above_plane_deg``
is positive when the middle of the shaft sits on the sky side of that line
(upward in the image). That is the steep side. Shallowing is that number
falling between the top and P6, the downswing frame where the shaft is
closest to horizontal.

A single down-the-line camera cannot measure TrackMan club path, face
angle, or clubhead speed. Those stay out of the report on purpose.
"""

from __future__ import annotations

import json
import math
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from swinglab.render import draw_analysis_frame, paint_hud, write_playback

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MAX_FRAMES = 450
MAX_WIDTH = 960

# Positive = shaft midpoint is on the sky side of the address line.
SHALLOWED_P6_MAX = 6.0
SHALLOWED_DROP_MIN = 8.0
STEEP_P6_MIN = 12.0
STEEP_DROP_MAX = 12.0


def sky_normal(vx: float, vy: float) -> tuple[float, float]:
    """Unit normal of a shaft that points toward the top of the image."""
    length = math.hypot(vx, vy) or 1.0
    ux, uy = vx / length, vy / length
    left = (-uy, ux)
    right = (uy, -ux)
    if left[1] == right[1]:
        return left
    return left if left[1] < right[1] else right


def above_plane_deg(
    grip: tuple[float, float],
    head: tuple[float, float],
    plane_grip: tuple[float, float],
    plane_head: tuple[float, float],
) -> float:
    """Signed degrees of the shaft midpoint from the address shaft line."""
    vx = plane_head[0] - plane_grip[0]
    vy = plane_head[1] - plane_grip[1]
    reference = math.hypot(vx, vy) or 1.0
    nx, ny = sky_normal(vx, vy)
    mid_x = (grip[0] + head[0]) / 2
    mid_y = (grip[1] + head[1]) / 2
    distance = (mid_x - plane_grip[0]) * nx + (mid_y - plane_grip[1]) * ny
    return math.degrees(math.atan2(distance, reference))


def elevation_deg(grip: tuple[float, float], head: tuple[float, float]) -> float:
    """0 means the shaft is horizontal in the picture. 90 means vertical."""
    dx = head[0] - grip[0]
    dy = head[1] - grip[1]
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def shaft_angle_deg(grip: tuple[float, float], head: tuple[float, float]) -> float:
    return math.degrees(math.atan2(head[1] - grip[1], head[0] - grip[0])) % 360


def _angle_delta(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def classify_shallowing(
    above_top: float | None,
    above_p6: float | None,
    shaft_confidence: float,
    predicted_fraction: float,
) -> tuple[str, str]:
    if (
        above_top is None
        or above_p6 is None
        or shaft_confidence < 0.35
        or predicted_fraction > 0.45
    ):
        return "unclear", "shaft_track_is_not_solid"
    drop = above_top - above_p6
    if drop <= -8:
        return "steepened", "shaft_moved_further_above_the_plane"
    if above_p6 <= SHALLOWED_P6_MAX and drop >= SHALLOWED_DROP_MIN:
        return "shallowed", "shaft_dropped_toward_the_address_plane"
    if above_p6 >= STEEP_P6_MIN and drop < STEEP_DROP_MAX:
        return "not_shallowing", "shaft_stayed_above_the_plane"
    if above_p6 <= SHALLOWED_P6_MAX:
        return "shallowed", "delivery_is_on_or_under_the_plane"
    return "mixed", "delivery_only_partly_moved_toward_the_plane"


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values.astype(float), kernel, mode="same")


def find_phases(samples: list[dict], frame_height: int, fps: float = 30.0) -> dict | None:
    usable = [
        index
        for index, sample in enumerate(samples)
        if sample.get("grip") is not None and sample.get("head") is not None
    ]
    if len(usable) < 8:
        return None
    heads = np.full((len(samples), 2), np.nan)
    for index, sample in enumerate(samples):
        if sample.get("head") is not None and not sample.get("predicted"):
            heads[index] = sample["head"]
    ys = _smooth(np.nan_to_num(heads[:, 1], nan=np.nanmedian(heads[:, 1])))
    address_zone = max(3, len(samples) // 3)
    address_candidates = [
        index
        for index in range(address_zone)
        if samples[index].get("head") is not None
        and not samples[index].get("predicted")
        and samples[index].get("confidence", 0) >= 0.45
    ]
    if not address_candidates:
        address_candidates = list(range(address_zone))
    address = max(address_candidates, key=lambda index: ys[index])
    search_end = max(address + 4, int(len(samples) * 0.92))
    measured_top = [
        index
        for index in range(address + 1, search_end)
        if samples[index].get("head") is not None and not samples[index].get("predicted")
    ]
    if not measured_top:
        return None
    top = min(measured_top, key=lambda index: heads[index, 1])
    if top <= address:
        return None
    # A held top is one pose for a while. Impact is after the club leaves
    # that pose, not 0.9s after the first frame of the hold.
    peak = float(ys[top])
    tolerance = max(16.0, frame_height * 0.04)
    end_hold = top
    gap = 0
    for index in range(top, search_end):
        if ys[index] <= peak + tolerance:
            end_hold = index
            gap = 0
        else:
            gap += 1
            if gap > 4:
                break
    address_head = heads[address]
    rate = fps if fps and fps > 1 else 30.0
    window = max(6, int(round(1.1 * rate)))
    stop = min(len(samples), end_hold + window)
    distances = np.linalg.norm(heads[end_hold:stop] - address_head, axis=1)
    distances = np.nan_to_num(distances, nan=1e9)
    if len(distances) == 0:
        return None
    impact = end_hold + int(np.argmin(distances))
    if impact <= top:
        return None

    def closest_horizontal(start: int, stop: int) -> int | None:
        measured = [
            index
            for index in range(start, stop)
            if samples[index].get("head") is not None and not samples[index].get("predicted")
        ]
        pool = measured or [
            index for index in range(start, stop) if samples[index].get("head") is not None
        ]
        best_index = None
        best_elevation = 1e9
        for index in pool:
            sample = samples[index]
            elevation = elevation_deg(sample["grip"], sample["head"])
            if elevation < best_elevation:
                best_elevation = elevation
                best_index = index
        return best_index

    shaft_parallel = closest_horizontal(top + 1, impact)
    toe_up = closest_horizontal(address + 1, top)
    lead_arm = _lead_arm_parallel(samples, top, shaft_parallel or impact)
    finish = min(len(samples) - 1, impact + max(1, (impact - top) // 3))
    return {
        "address": address,
        "toe_up": toe_up if toe_up is not None else address,
        "top": top,
        "lead_arm_parallel": lead_arm,
        "shaft_parallel": shaft_parallel if shaft_parallel is not None else impact,
        "impact": impact,
        "finish": finish,
        "frame_height": frame_height,
    }


def _lead_arm_parallel(samples: list[dict], top: int, stop: int) -> int | None:
    best_index = None
    best_score = 1e9
    for index in range(top + 1, stop):
        pose = samples[index].get("pose") or {}
        shoulder = pose.get("left_shoulder")
        wrist = pose.get("left_wrist")
        if shoulder is None or wrist is None:
            continue
        score = elevation_deg(shoulder, wrist)
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def _enhance_shaft(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    responses = []
    for size in (9, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel))
        responses.append(cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel))
    return np.maximum.reduce(responses)


def _body_mask(shape: tuple[int, int], pose: dict | None, grip: tuple[float, float]) -> np.ndarray:
    """Hide the torso and arms so the search follows the club, not a limb."""
    mask = np.zeros(shape, np.uint8)
    if not pose:
        return mask
    shoulders = [pose.get("left_shoulder"), pose.get("right_shoulder")]
    hips = [pose.get("left_hip"), pose.get("right_hip")]
    points = [point for point in shoulders + hips if point is not None]
    if len(points) >= 2:
        center_x = int(sum(point[0] for point in points) / len(points))
        center_y = int(sum(point[1] for point in points) / len(points))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (max(18, int(0.55 * (max(xs) - min(xs) + 20))), max(24, int(0.55 * (max(ys) - min(ys) + 20)))),
            0,
            0,
            360,
            255,
            -1,
        )
    limbs = (
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    )
    for start_name, end_name in limbs:
        start = pose.get(start_name)
        end = pose.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(mask, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 255, 26, cv2.LINE_AA)
    cv2.circle(mask, (int(grip[0]), int(grip[1])), 34, 0, -1)
    return mask


def detect_shaft(
    gray: np.ndarray,
    grip: tuple[float, float],
    prev_angle: float | None = None,
    pose: dict | None = None,
    search_window: float | None = None,
) -> dict:
    height, width = gray.shape[:2]
    enhanced = _enhance_shaft(gray)
    mask = _body_mask(enhanced.shape, pose, grip)

    radius_min = max(36, int(height * 0.045))
    radius_max = max(radius_min + 30, int(min(width, height) * 0.78))
    radii = np.linspace(radius_min, radius_max, 100)
    angles = np.arange(0, 360, 2, dtype=float)
    radians = np.deg2rad(angles)
    xs = grip[0] + radii[None, :] * np.cos(radians)[:, None]
    ys = grip[1] + radii[None, :] * np.sin(radians)[:, None]
    valid = (xs >= 1) & (xs < width - 1) & (ys >= 1) & (ys < height - 1)
    xi = np.clip(np.rint(xs).astype(int), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, height - 1)
    values = np.where(valid, enhanced[yi, xi], 0).astype(np.float32)
    masked = mask[yi, xi] > 0

    scores = np.zeros(len(angles), dtype=np.float32)
    ends = np.zeros(len(angles), dtype=np.int32)
    for index, row in enumerate(values):
        smooth = np.convolve(row, np.ones(5) / 5, mode="same")
        row_mask = masked[index]
        visible = smooth.copy()
        visible[row_mask] = 0
        if float(visible.max()) < 8:
            continue
        threshold = max(10.0, 0.42 * float(np.percentile(visible, 92)))
        last = -1
        gap = 0
        started = False
        unmasked_seen = 0
        for step, hot in enumerate(smooth >= threshold):
            if row_mask[step]:
                continue
            unmasked_seen += 1
            if hot:
                if not started and unmasked_seen > 18:
                    break
                started = True
                last = step
                gap = 0
            elif started:
                gap += 1
                if gap > 4:
                    break
        if last < 6:
            continue
        length_px = float(radii[last] - radii[0])
        if length_px < 45:
            length_weight = 0.15
        elif length_px > min(width, height) * 0.7:
            length_weight = 0.35
        else:
            length_weight = 1.0
        if last >= len(radii) - 4:
            length_weight *= 0.35
        visible_segment = visible[: last + 1]
        visible_segment = visible_segment[visible_segment > 0]
        mean = float(visible_segment.mean()) if len(visible_segment) else 0.0
        scores[index] = mean * length_weight * math.sqrt(length_px / 70.0)
        ends[index] = last

    mid = int(np.argmin(np.abs(radii - (radius_min + 80))))
    perpendicular_x = -np.sin(radians)
    perpendicular_y = np.cos(radians)
    thin = np.zeros(len(angles), dtype=np.float32)
    center_x = grip[0] + radii[mid] * np.cos(radians)
    center_y = grip[1] + radii[mid] * np.sin(radians)
    for offset, weight in ((-16, -0.5), (-7, -0.15), (0, 1.0), (7, -0.15), (16, -0.5)):
        sample_x = np.clip(np.rint(center_x + offset * perpendicular_x).astype(int), 0, width - 1)
        sample_y = np.clip(np.rint(center_y + offset * perpendicular_y).astype(int), 0, height - 1)
        thin += weight * enhanced[sample_y, sample_x]
    thin_ratio = np.clip(thin / (np.abs(thin) + 12), 0, 1)
    scores *= 0.2 + thin_ratio

    if prev_angle is not None:
        delta = np.abs((angles - prev_angle + 180) % 360 - 180)
        if search_window is not None:
            wide = np.where(delta <= search_window, scores, -1)
            local = np.where(delta <= 8, scores, -1)
            # Stay on the current shaft unless a nearby line is clearly stronger.
            # This stops a bright background edge from walking the track off the club.
            if float(local.max()) > 8 and float(local.max()) >= 0.45 * float(wide.max()):
                scores = local
            else:
                scores = wide
        else:
            scores = scores + 12 * np.exp(-0.5 * (delta / 18) ** 2)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = next(
        (int(index) for index in order[1:] if abs(_angle_delta(angles[index], angles[best])) > 8),
        int(order[1]),
    )
    margin = float(scores[best] - scores[second])
    strength = float(scores[best])
    confidence = float(np.clip(margin / 18.0, 0, 1) * np.clip(strength / 16.0, 0, 1))
    if strength <= 0:
        confidence = 0.0
    if ends[best] < 6:
        confidence *= 0.2
    radius = float(radii[int(ends[best])]) if ends[best] else float(radii[0])
    angle = float(angles[best])
    head = (
        float(grip[0] + radius * math.cos(math.radians(angle))),
        float(grip[1] + radius * math.sin(math.radians(angle))),
    )
    return {
        "grip": (float(grip[0]), float(grip[1])),
        "head": head,
        "angle": angle,
        "confidence": confidence,
        "predicted": False,
    }


def _grip_from_pose(pose: dict | None) -> tuple[float, float] | None:
    if not pose:
        return None
    left = pose.get("left_wrist")
    right = pose.get("right_wrist")
    if left is not None and right is not None:
        return (0.45 * left[0] + 0.55 * right[0], 0.45 * left[1] + 0.55 * right[1])
    if left is not None:
        return (float(left[0]), float(left[1]))
    if right is not None:
        return (float(right[0]), float(right[1]))
    return None


def _empty_sample(pose: dict | None) -> dict:
    return {
        "grip": None,
        "head": None,
        "angle": None,
        "confidence": 0.0,
        "predicted": True,
        "pose": pose,
    }


def _continue_shaft(grip: tuple[float, float], angle: float, radius: float, pose: dict | None) -> dict:
    return {
        "grip": (float(grip[0]), float(grip[1])),
        "head": (
            float(grip[0] + radius * math.cos(math.radians(angle))),
            float(grip[1] + radius * math.sin(math.radians(angle))),
        ),
        "angle": float(angle % 360),
        "confidence": 0.15,
        "predicted": True,
        "pose": pose,
    }


def track_shafts(frames: list[np.ndarray], poses: list[dict | None]) -> list[dict]:
    """Follow the shaft, and refuse jumps to background lines.

    A golf shaft does not teleport. When the detector's best line is far
    from the last measured angle, that frame is marked predicted and is
    not allowed to decide shallowing.
    """
    samples = []
    previous: float | None = None
    velocity = 0.0
    radius = 140.0
    missed = 0
    locked = False
    for frame, pose in zip(frames, poses):
        grip = _grip_from_pose(pose)
        if grip is None:
            samples.append(_empty_sample(pose))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_search = (not locked) or missed >= 10
        detected = detect_shaft(
            gray,
            grip,
            None if full_search else previous,
            pose,
            None if full_search else 24,
        )
        detected["pose"] = pose
        angle = detected.get("angle")
        head = detected.get("head")
        grounded = head is not None and head[1] > grip[1] + 30
        if not locked:
            # The address club points down to the ball. An upward line here
            # is the background, and following it ruins the rest of the swing.
            if grounded and detected["confidence"] >= 0.45 and angle is not None:
                locked = True
                previous = angle
                radius = max(50.0, math.hypot(head[0] - grip[0], head[1] - grip[1]))
                missed = 0
                samples.append(detected)
            else:
                detected["predicted"] = True
                detected["confidence"] = min(float(detected["confidence"]), 0.2)
                samples.append(detected)
            continue
        jumped = (
            previous is not None
            and not full_search
            and angle is not None
            and abs(_angle_delta(angle, previous)) > 40
        )
        weak = detected["confidence"] < 0.3 or angle is None
        if weak or jumped:
            if previous is None:
                samples.append(_empty_sample(pose))
                missed += 1
                continue
            velocity *= 0.8
            angle = (previous + velocity) % 360
            samples.append(_continue_shaft(grip, angle, radius, pose))
            previous = angle
            missed += 1
            continue
        if angle is None:
            samples.append(_empty_sample(pose))
            missed += 1
            continue
        if previous is not None:
            velocity = 0.65 * velocity + 0.35 * _angle_delta(angle, previous)
        head = detected["head"]
        radius = max(50.0, math.hypot(head[0] - grip[0], head[1] - grip[1]))
        previous = angle
        missed = 0
        samples.append(detected)
    return _fill_gaps(samples)


def _fill_gaps(samples: list[dict]) -> list[dict]:
    good = [
        index
        for index, sample in enumerate(samples)
        if sample.get("angle") is not None and sample["confidence"] >= 0.45 and not sample["predicted"]
    ]
    if len(good) < 2:
        return samples
    for left, right in zip(good, good[1:]):
        span = right - left
        if span <= 1:
            continue
        start = samples[left]["angle"]
        change = _angle_delta(samples[right]["angle"], start)
        for step in range(1, span):
            current = samples[left + step]
            if current["confidence"] >= 0.45 and not current["predicted"]:
                continue
            if current.get("grip") is None:
                continue
            fraction = step / span
            angle = (start + change * fraction) % 360
            radius_left = math.hypot(
                samples[left]["head"][0] - samples[left]["grip"][0],
                samples[left]["head"][1] - samples[left]["grip"][1],
            )
            radius_right = math.hypot(
                samples[right]["head"][0] - samples[right]["grip"][0],
                samples[right]["head"][1] - samples[right]["grip"][1],
            )
            radius = radius_left + (radius_right - radius_left) * fraction
            grip = current["grip"]
            current["angle"] = angle
            current["head"] = (
                grip[0] + radius * math.cos(math.radians(angle)),
                grip[1] + radius * math.sin(math.radians(angle)),
            )
            current["predicted"] = True
            current["confidence"] = min(current["confidence"], 0.25)
    return samples


def _needs_target_flip(samples: list[dict]) -> bool:
    zone = max(1, len(samples) // 2)
    candidates = [
        sample
        for sample in samples[:zone]
        if sample.get("head") is not None and sample.get("grip") is not None and sample["confidence"] >= 0.3
    ]
    if not candidates:
        return False
    lowest = max(candidates, key=lambda sample: sample["head"][1])
    return lowest["head"][0] > lowest["grip"][0] + 12


def _flip_point(point: tuple[float, float] | None, width: int) -> tuple[float, float] | None:
    if point is None:
        return None
    return (width - 1 - point[0], point[1])


def _flip_pose(pose: dict | None, width: int) -> dict | None:
    if pose is None:
        return None
    flipped = {}
    for key, value in pose.items():
        if key == "visibility" or not isinstance(value, tuple):
            flipped[key] = value
        else:
            flipped[key] = _flip_point(value, width)
    return flipped


def _flip_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.flip(frame, 1) for frame in frames]


def _phase_label(index: int, phases: dict) -> str:
    if index <= phases["address"]:
        return "address"
    if index < phases["toe_up"]:
        return "takeaway"
    if index < phases["top"]:
        return "backswing"
    if index == phases["top"]:
        return "top"
    if index < phases["shaft_parallel"]:
        return "transition"
    if index < phases["impact"]:
        return "delivery"
    if index == phases["impact"]:
        return "impact"
    return "finish"


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def build_report(samples: list[dict], fps: float, frame_size: tuple[int, int]) -> dict:
    height = frame_size[1]
    phases = find_phases(samples, height, fps)
    warnings = [
        "Down-the-line picture. Not a TrackMan club path, face angle, or speed.",
    ]
    if phases is None:
        warnings.append("Could not find address, top, and impact in this clip.")
        return {
            "reading": "unclear",
            "readingReason": "could_not_segment_swing",
            "warnings": warnings,
            "phases": None,
            "shallowing": None,
            "confidence": {"shaft": 0, "predictedFraction": 1},
        }

    plane_sample = samples[phases["address"]]
    plane_grip = plane_sample["grip"]
    plane_head = plane_sample["head"]
    above = []
    for sample in samples:
        if sample.get("grip") is None or sample.get("head") is None or plane_grip is None:
            above.append(None)
        else:
            above.append(above_plane_deg(sample["grip"], sample["head"], plane_grip, plane_head))

    def at(name: str) -> float | None:
        index = phases.get(name)
        if index is None or index < 0 or index >= len(above):
            return None
        return above[index]

    window = range(phases["address"], phases["impact"] + 1)
    confidences = [samples[index]["confidence"] for index in window if samples[index].get("head") is not None]
    predicted = [samples[index]["predicted"] for index in window]
    shaft_confidence = float(np.median(confidences)) if confidences else 0.0
    predicted_fraction = float(np.mean(predicted)) if predicted else 1.0
    above_top = at("top")
    above_p6 = at("shaft_parallel")
    top_predicted = bool(samples[phases["top"]].get("predicted"))
    p6_predicted = bool(samples[phases["shaft_parallel"]].get("predicted"))
    reading, reason = classify_shallowing(above_top, above_p6, shaft_confidence, predicted_fraction)
    if top_predicted or p6_predicted:
        reading, reason = "unclear", "club_was_not_visible_at_the_checkpoint"
    impact_head = samples[phases["impact"]].get("head")
    address_head = samples[phases["address"]].get("head")
    if impact_head is None or address_head is None:
        reading, reason = "unclear", "impact_was_not_near_the_ball"
    else:
        miss = math.hypot(impact_head[0] - address_head[0], impact_head[1] - address_head[1])
        if miss > 0.18 * height:
            reading, reason = "unclear", "impact_was_not_near_the_ball"
    drop = None if above_top is None or above_p6 is None else above_top - above_p6

    address_time = phases["address"] / fps
    top_time = phases["top"] / fps
    impact_time = phases["impact"] / fps
    backswing = top_time - address_time
    downswing = impact_time - top_time
    tempo = backswing / downswing if downswing > 1e-3 else None

    descent = _image_descent(samples, phases["impact"])
    hips = _hip_motion(samples, phases["address"], phases["impact"])
    pose_frames = sum(1 for sample in samples if _grip_from_pose(sample.get("pose")) is not None)
    if shaft_confidence < 0.35:
        warnings.append("The shaft line is weak. Treat the degrees as unreliable.")
    if predicted_fraction > 0.45:
        warnings.append("The club was missing for too much of the swing.")

    return {
        "reading": reading,
        "readingReason": reason,
        "playbackSpeed": None,
        "warnings": warnings,
        "phases": {key: phases[key] for key in ("address", "toe_up", "top", "lead_arm_parallel", "shaft_parallel", "impact", "finish")},
        "shallowing": {
            "abovePlaneDeg": {
                "address": _round(at("address")),
                "top": _round(above_top),
                "p5": _round(at("lead_arm_parallel")),
                "p6": _round(above_p6),
                "impact": _round(at("impact")),
            },
            "shallowedDeg": _round(drop),
            "p6IsShaftNearestHorizontal": True,
        },
        "tempo": {
            "backswingSeconds": _round(backswing, 2),
            "downswingSeconds": _round(downswing, 2),
            "ratio": _round(tempo, 2),
            "guide": "A common full-swing guide is about 3:1. It is not a pass mark.",
        },
        "delivery": {
            "imageDescentDeg": _round(descent),
            "imageDescentMeaning": _descent_label(descent),
            "clubPathDegrees": None,
            "clubPath": "not_measurable_from_one_down_the_line_camera",
        },
        "body": hips,
        "confidence": {
            "shaft": _round(shaft_confidence, 2),
            "predictedFraction": _round(predicted_fraction, 2),
            "poseFrames": pose_frames,
            "frameCount": len(samples),
        },
        "_above": above,
        "_plane": (plane_grip, plane_head),
    }


def _image_descent(samples: list[dict], impact: int) -> float | None:
    start = max(0, impact - 3)
    first = samples[start]
    last = samples[impact]
    if first.get("head") is None or last.get("head") is None:
        return None
    dx = last["head"][0] - first["head"][0]
    dy = last["head"][1] - first["head"][1]
    toward_target = -dx
    if math.hypot(dx, dy) < 2:
        return None
    return math.degrees(math.atan2(dy, toward_target))


def _descent_label(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    if degrees > 8:
        return "moving_down_in_the_picture"
    if degrees < -8:
        return "moving_up_in_the_picture"
    return "level_in_the_picture"


def _hip_motion(samples: list[dict], address: int, impact: int) -> dict:
    def center(index: int) -> tuple[float, float] | None:
        pose = samples[index].get("pose") or {}
        left = pose.get("left_hip")
        right = pose.get("right_hip")
        if left is None or right is None:
            return None
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

    start = center(address)
    end = center(impact)
    if start is None or end is None:
        return {"hipsInImage": "unknown"}
    drop = end[1] - start[1]
    pose = samples[address].get("pose") or {}
    shoulder = pose.get("left_shoulder")
    hip = pose.get("left_hip")
    torso = 120.0
    if shoulder is not None and hip is not None:
        torso = max(40.0, math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]))
    if drop > 0.12 * torso:
        label = "toward_the_ball"
    elif drop < -0.12 * torso:
        label = "away_from_the_ball"
    else:
        label = "steady"
    return {"hipsInImage": label, "hipDropPx": _round(drop)}


def _hud(index: int, phases: dict, report: dict, speed: float) -> list[str]:
    label = _phase_label(index, phases)
    above = report["_above"][index]
    above_text = "n/a" if above is None else f"{above:+.0f} deg"
    reading = report["reading"].replace("_", " ")
    return [
        f"{speed:.2f}x   {label}",
        f"Above plane {above_text}",
        reading,
        "DTL estimate, not TrackMan",
    ]


def _render(frames: list[np.ndarray], samples: list[dict], report: dict, speed: float) -> list[np.ndarray]:
    phases = report["phases"]
    plane = report.get("_plane")
    drawn = []
    trail: list[tuple[float, float] | None] = []
    for index, (frame, sample) in enumerate(zip(frames, samples)):
        reliable = (
            sample.get("head") is not None
            and not sample.get("predicted")
            and sample.get("confidence", 0) >= 0.45
        )
        head = sample.get("head") if reliable else None
        trail.append(head)
        visible_trail = trail[-50:]
        image = draw_analysis_frame(
            frame,
            plane=plane,
            grip=sample.get("grip"),
            head=sample.get("head"),
            predicted=bool(sample.get("predicted")),
            above_plane_deg=report["_above"][index],
            trail=visible_trail,
            pose=sample.get("pose"),
            hud_lines=[],
            after_top=index >= phases["top"],
        )
        drawn.append(image)
    return drawn


def _add_hud(frames: list[np.ndarray], phases: dict, report: dict, speed: float) -> list[np.ndarray]:
    painted = []
    for index, frame in enumerate(frames):
        painted.append(paint_hud(frame, _hud(index, phases, report, speed)))
    return painted


def _public_report(report: dict, speed: float, extras: dict) -> dict:
    cleaned = {key: value for key, value in report.items() if not key.startswith("_")}
    cleaned["playbackSpeed"] = speed
    cleaned.update(extras)
    return cleaned


def analyze_frames(
    frames: list[np.ndarray],
    *,
    fps: float,
    poses: list[dict | None] | None = None,
    handed: str = "right",
    playback_speed: float = 0.25,
    out_dir: str | Path | None = None,
    mirror_output: bool = True,
) -> dict:
    if not frames:
        raise ValueError("video has no frames")
    speed = min(1.0, max(0.1, float(playback_speed)))
    working = [frame.copy() for frame in frames]
    working_poses = list(poses) if poses is not None else None
    flips = 0
    if handed == "left":
        working = _flip_frames(working)
        if working_poses is not None:
            width = working[0].shape[1]
            working_poses = [_flip_pose(pose, width) for pose in working_poses]
        flips += 1
    if working_poses is None:
        working_poses = detect_poses(working, fps)

    samples = track_shafts(working, working_poses)
    if _needs_target_flip(samples):
        width = working[0].shape[1]
        working = _flip_frames(working)
        working_poses = [_flip_pose(pose, width) for pose in working_poses]
        samples = track_shafts(working, working_poses)
        flips += 1

    report = build_report(samples, fps, working[0].shape[1::-1])
    drawn = frames
    if report.get("phases") is not None:
        drawn = _render(working, samples, report, speed)
        if mirror_output and flips % 2 == 1:
            drawn = _flip_frames(drawn)
        drawn = _add_hud(drawn, report["phases"], report, speed)

    output = {}
    if out_dir is not None and report.get("phases") is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        video_path = destination / "swing-analysis.mp4"
        write_playback(drawn, video_path, fps, speed)
        phases = report["phases"]
        stills = {
            "address.jpg": phases["address"],
            "top.jpg": phases["top"],
            "p6.jpg": phases["shaft_parallel"],
            "impact.jpg": phases["impact"],
        }
        for name, index in stills.items():
            cv2.imwrite(str(destination / name), drawn[index])
        output = {
            "videoPath": str(video_path),
            "addressStillPath": str(destination / "address.jpg"),
            "topStillPath": str(destination / "top.jpg"),
            "p6StillPath": str(destination / "p6.jpg"),
            "impactStillPath": str(destination / "impact.jpg"),
        }
    public = _public_report(
        report,
        speed,
        {
            "handed": handed,
            "orientedForTargetLeft": flips % 2 == 1,
            "fps": fps,
            "frameCount": len(frames),
            **output,
        },
    )
    if out_dir is not None:
        Path(out_dir, "report.json").write_text(json.dumps(public, indent=2))
    return public


def _load_pose_sidecar(path: Path) -> list[dict | None] | None:
    """Drawn examples ship the pose they were built with. Real videos do not."""
    sidecar = path.with_suffix(".poses.json")
    if not sidecar.is_file():
        return None
    raw = json.loads(sidecar.read_text())
    poses: list[dict | None] = []
    for frame in raw:
        if frame is None:
            poses.append(None)
            continue
        pose = {}
        for key, value in frame.items():
            if isinstance(value, list) and len(value) == 2:
                pose[key] = (float(value[0]), float(value[1]))
            else:
                pose[key] = value
        poses.append(pose)
    return poses


def analyze_video(
    path: str | Path,
    out_dir: str | Path,
    *,
    handed: str = "right",
    playback_speed: float = 0.25,
) -> dict:
    frames, fps, trimmed = load_video(path)
    poses = _load_pose_sidecar(Path(path))
    if poses is not None and len(poses) != len(frames):
        poses = None
        trimmed = True
    report = analyze_frames(
        frames,
        fps=fps,
        poses=poses,
        handed=handed,
        playback_speed=playback_speed,
        out_dir=out_dir,
    )
    if trimmed:
        report["warnings"] = list(report.get("warnings") or []) + [
            f"Only the first {len(frames)} frames were analyzed.",
        ]
        Path(out_dir, "report.json").write_text(json.dumps(report, indent=2))
    return report


def load_video(path: str | Path) -> tuple[list[np.ndarray], float, bool]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    temp_dir = Path("/tmp/golf-swing")
    temp_dir.mkdir(parents=True, exist_ok=True)
    normalized = temp_dir / f"{source.stem}-normalized.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"scale='min({MAX_WIDTH},iw)':-2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(normalized),
        ],
        check=True,
    )
    capture = cv2.VideoCapture(str(normalized))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    trimmed = False
    extra_ok, _extra = capture.read()
    if extra_ok:
        trimmed = True
    capture.release()
    if not frames:
        raise ValueError(f"could not read frames from {source}")
    return frames, fps, trimmed


def model_path() -> Path:
    cache = Path.home() / ".cache" / "golf-swing" / "pose_landmarker_lite.task"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, cache)
    return cache


def detect_poses(frames: list[np.ndarray], fps: float) -> list[dict | None]:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path())),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    names = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    poses: list[dict | None] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for index, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(index * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp)
            landmarks = result.pose_landmarks
            if not landmarks:
                poses.append(None)
                continue
            height, width = frame.shape[:2]
            pose: dict = {"visibility": 1.0}
            person = landmarks[0]
            seen = 0
            for name, landmark_index in names.items():
                landmark = person[landmark_index]
                visibility = float(getattr(landmark, "visibility", 1.0) or 0)
                if visibility < 0.35:
                    pose[name] = None
                    continue
                pose[name] = (landmark.x * width, landmark.y * height)
                seen += 1
            pose["visibility"] = seen / len(names)
            poses.append(pose if seen >= 4 else None)
    return poses
