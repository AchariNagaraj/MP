from __future__ import annotations

import os
import urllib.request
import math
import time
from datetime import datetime
from collections import deque
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"
NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

# Basic skeleton edges for drawing (BlazePose 33 landmarks indices).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
]


def ensure_model_exists() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading pose model (one-time setup)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_pose(frame, landmarks) -> None:
    height, width, _ = frame.shape
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (245, 117, 66), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (245, 66, 230), 2)


def landmark_to_pixel(landmark, frame_shape):
    height, width, _ = frame_shape
    return (int(landmark.x * width), int(landmark.y * height))


def midpoint(point_a, point_b):
    return (
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
    )


def angle_from_vertical(point_a, point_b) -> float:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def signed_angle_from_vertical(point_a, point_b) -> float:
    """
    Computes angle from vertical. Positive means point_b is to the right of point_a.
    Negative means point_b is to the left of point_a.
    """
    dx = point_b[0] - point_a[0]
    dy = -(point_b[1] - point_a[1]) # Y-axis increases downwards, so negate dy
    return math.degrees(math.atan2(dx, dy + 1e-6))


def extract_posture_keypoints(landmarks, frame_shape):
    return {
        "nose": landmark_to_pixel(landmarks[NOSE_IDX], frame_shape),
        "left_shoulder": landmark_to_pixel(landmarks[LEFT_SHOULDER_IDX], frame_shape),
        "right_shoulder": landmark_to_pixel(landmarks[RIGHT_SHOULDER_IDX], frame_shape),
        "left_hip": landmark_to_pixel(landmarks[LEFT_HIP_IDX], frame_shape),
        "right_hip": landmark_to_pixel(landmarks[RIGHT_HIP_IDX], frame_shape),
    }


def compute_posture_features(keypoints):
    shoulder_mid = midpoint(keypoints["left_shoulder"], keypoints["right_shoulder"])
    hip_mid = midpoint(keypoints["left_hip"], keypoints["right_hip"])
    shoulder_width = abs(
        keypoints["left_shoulder"][0] - keypoints["right_shoulder"][0]
    )

    neck_angle = angle_from_vertical(shoulder_mid, keypoints["nose"])
    shoulder_alignment = abs(
        keypoints["left_shoulder"][1] - keypoints["right_shoulder"][1]
    )
    
    side_lean_angle = signed_angle_from_vertical(hip_mid, shoulder_mid)
    spine_angle = abs(side_lean_angle) # absolute magnitude for classification
    
    if side_lean_angle > 5:
        lean_direction = "RIGHT LEAN"
    elif side_lean_angle < -5:
        lean_direction = "LEFT LEAN"
    else:
        lean_direction = "CENTERED"

    return {
        "neck_angle": neck_angle,
        "shoulder_alignment": shoulder_alignment,
        "spine_angle": spine_angle,
        "side_lean_angle": side_lean_angle,
        "lean_direction": lean_direction,
        "shoulder_width": shoulder_width,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }


def detect_camera_view(shoulder_width, frame_width):
    """
    Simple heuristic:
    - FRONT view when shoulder span is wide enough in image
    - SIDE view when shoulder span appears narrow
    """
    shoulder_ratio = shoulder_width / max(frame_width, 1)
    return "FRONT" if shoulder_ratio >= 0.14 else "SIDE"


def update_calibration(features, calib_state, current_time):
    """
    Collects baseline posture angles during the initial calibration phase.
    """
    if not calib_state["is_calibrating"]:
        return False
        
    if current_time - calib_state["start_time"] < calib_state["duration"]:
        calib_state["neck_samples"].append(features["neck_angle"])
        calib_state["spine_samples"].append(features["spine_angle"])
        return True
    else:
        if calib_state["neck_samples"]:
            calib_state["base_neck"] = sum(calib_state["neck_samples"]) / len(calib_state["neck_samples"])
        if calib_state["spine_samples"]:
            calib_state["base_spine"] = sum(calib_state["spine_samples"]) / len(calib_state["spine_samples"])
        calib_state["is_calibrating"] = False
        return False


def classify_posture_front(neck_angle, spine_angle, shoulder_diff, calib_state):
    """
    Front-view logic using dynamic thresholds and hysteresis.
    """
    margin = 0
    if calib_state["prev_status"] == "GOOD":
        margin = 3  # Allow slightly worse posture before switching to MODERATE/BAD
    elif calib_state["prev_status"] == "BAD":
        margin = -3 # Require significantly better posture to recover to GOOD/MODERATE

    base_neck = calib_state["base_neck"]
    base_spine = calib_state["base_spine"]
    base_shoulder = calib_state["base_shoulder"]

    mod_neck = base_neck + 5 + margin
    bad_neck = base_neck + 15 + margin
    
    mod_spine = base_spine + 5 + margin
    bad_spine = base_spine + 15 + margin
    
    mod_shoulder = base_shoulder + 10 + (margin * 2)
    bad_shoulder = base_shoulder + 30 + (margin * 2)

    if neck_angle > bad_neck or spine_angle > bad_spine or shoulder_diff > bad_shoulder:
        status = "BAD"
    elif neck_angle > mod_neck or spine_angle > mod_spine or shoulder_diff > mod_shoulder:
        status = "MODERATE"
    else:
        status = "GOOD"
        
    calib_state["prev_status"] = status
    return status


def classify_posture_side(spine_angle, calib_state):
    """
    Side-view logic using dynamic thresholds and hysteresis.
    """
    margin = 0
    if calib_state["prev_status"] == "GOOD":
        margin = 3
    elif calib_state["prev_status"] == "BAD":
        margin = -3
        
    base_spine = calib_state["base_spine"]
    mod_spine = base_spine + 5 + margin
    bad_spine = base_spine + 10 + margin
    
    if spine_angle > bad_spine:
        status = "BAD"
    elif spine_angle > mod_spine:
        status = "MODERATE"
    else:
        status = "GOOD"
        
    calib_state["prev_status"] = status
    return status


def classify_posture(view_mode, features, calib_state):
    if view_mode == "FRONT":
        return classify_posture_front(
            features["neck_angle"],
            features["spine_angle"],
            features["shoulder_alignment"],
            calib_state
        )
    return classify_posture_side(features["spine_angle"], calib_state)


def smooth_value(history: deque, value: float) -> float:
    history.append(value)
    return sum(history) / len(history)


# Angle-based severity: max of smoothed neck, spine, |side lean| (degrees).
SEVERITY_LOW_MAX = 12.0
SEVERITY_MEDIUM_MAX = 22.0


def update_bad_posture_duration(current_status, bad_start_time, current_time):
    """
    Returns:
    - bad_duration_sec: current continuous BAD duration
    - updated_bad_start_time
    """
    if current_status == "BAD":
        if bad_start_time is None:
            bad_start_time = current_time
        bad_duration_sec = current_time - bad_start_time
    elif current_status == "GOOD":
        bad_start_time = None
        bad_duration_sec = 0.0
    else:
        # For MODERATE, keep prior BAD timer state but do not force reset.
        bad_duration_sec = 0.0 if bad_start_time is None else current_time - bad_start_time
    return bad_duration_sec, bad_start_time


def compute_severity(features) -> Tuple[str, float]:
    """
    Severity score = max(smoothed neck, smoothed spine, |smoothed side lean|).
    Levels: LOW / MEDIUM / HIGH from fixed angle thresholds (degrees).
    """
    neck = features["neck_angle"]
    spine = features["spine_angle"]
    side_abs = abs(features["side_lean_angle"])
    severity_score = max(neck, spine, side_abs)

    if severity_score < SEVERITY_LOW_MAX:
        level = "LOW"
    elif severity_score < SEVERITY_MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return level, severity_score


def evaluate_risk_level(bad_duration_sec: float, severity_level: str) -> str:
    """
    Combines sustained BAD posture (seconds) with angle severity.
    Rules (simple, realistic):
    - No sustained bad → LOW
    - Long bad duration → HIGH regardless of severity band
    - Short duration + HIGH severity → escalate faster
    """
    if bad_duration_sec <= 0.0:
        return "LOW"

    if bad_duration_sec >= 25.0:
        return "HIGH"
    if bad_duration_sec >= 15.0:
        return "HIGH" if severity_level != "LOW" else "MEDIUM"

    if severity_level == "HIGH":
        if bad_duration_sec >= 3.0:
            return "HIGH"
        if bad_duration_sec >= 1.0:
            return "MEDIUM"
    elif severity_level == "MEDIUM":
        if bad_duration_sec >= 10.0:
            return "HIGH"
        if bad_duration_sec >= 4.0:
            return "MEDIUM"
    else:
        if bad_duration_sec >= 20.0:
            return "HIGH"
        if bad_duration_sec >= 10.0:
            return "MEDIUM"

    return "LOW"


def draw_severity_heatmap(
    frame,
    keypoints,
    features,
    posture_status: str,
    severity_level: str,
) -> None:
    """
    Simple circle-based risk visualization.
    """
    if posture_status == "GOOD":
        return

    shoulder_mid = (
        int(features["shoulder_mid"][0]),
        int(features["shoulder_mid"][1]),
    )
    hip_mid = (
        int(features["hip_mid"][0]),
        int(features["hip_mid"][1]),
    )
    nose = keypoints["nose"]
    neck_point = (
        int((shoulder_mid[0] + nose[0]) / 2),
        int((shoulder_mid[1] + nose[1]) / 2),
    )
    spine_point = (
        int((shoulder_mid[0] + hip_mid[0]) / 2),
        int((shoulder_mid[1] + hip_mid[1]) / 2),
    )

    # 1. Per-region severity (0–1)
    neck_s     = min(1.0, features["neck_angle"]         / 30.0)
    spine_s    = min(1.0, features["spine_angle"]        / 30.0)
    shoulder_s = min(1.0, features["shoulder_alignment"] / 50.0)

    # 2. Dynamic radii (capped at 80)
    BASE = 20
    neck_r     = min(80, int(BASE + neck_s     * 40))
    spine_r    = min(80, int(BASE + spine_s    * 50))
    shoulder_r = min(80, int(BASE + shoulder_s * 35))

    # 3. Per-region opacity
    neck_a     = 0.2 + neck_s     * 0.5
    spine_a    = 0.2 + spine_s    * 0.5
    shoulder_a = 0.2 + shoulder_s * 0.5

    # 4. Draw each region on its own overlay and blend
    def _blend_circle(dst, pt, r, col, a):
        ov = dst.copy()
        cv2.circle(ov, pt, r, col, -1)
        dst[:] = cv2.addWeighted(ov, a, dst, 1.0 - a, 0)

    _blend_circle(frame, neck_point,                    neck_r,     (0, 80,  255), neck_a)
    _blend_circle(frame, spine_point,                   spine_r,    (0, 165, 255), spine_a)
    _blend_circle(frame, keypoints["left_shoulder"],    shoulder_r, (0, 220, 220), shoulder_a)
    _blend_circle(frame, keypoints["right_shoulder"],   shoulder_r, (0, 220, 220), shoulder_a)


def initialize_heatmap_matrix(frame_shape) -> np.ndarray:
    """
    Stores cumulative region scores: [neck, spine, shoulders].
    """
    return np.zeros((3,), dtype=np.float32)


def create_torso_mask(frame_shape, keypoints: dict) -> np.ndarray:
    """
    Creates a torso-only binary mask using shoulders and hips.
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    torso_poly = np.array(
        [
            keypoints["left_shoulder"],
            keypoints["right_shoulder"],
            keypoints["right_hip"],
            keypoints["left_hip"],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [torso_poly], 255)
    return mask


def accumulate_strain_heatmap(
    heatmap_matrix: np.ndarray,
    keypoints: dict,
    features: dict,
    posture_status: str,
    severity_score: float,
    calib_state: dict,
    is_calibrating: bool,
    severity_threshold: float = 14.0,
    decay: float = 0.90,
    max_intensity: float = 65.0,
) -> None:
    """
    Accumulates simple region scores for report circles.
    """
    if heatmap_matrix is None or is_calibrating:
        return

    heatmap_matrix *= decay

    add_heat = posture_status != "GOOD" or severity_score >= severity_threshold
    if not add_heat:
        return

    neck_strain = max(0.0, features["neck_angle"] - calib_state["base_neck"])
    spine_strain = max(0.0, features["spine_angle"] - calib_state["base_spine"])
    shoulder_strain = max(
        0.0, features["shoulder_alignment"] - calib_state["base_shoulder"]
    )

    heatmap_matrix[0] += min(
        severity_score * (1.0 + neck_strain / 20.0),
        max_intensity,
    )
    heatmap_matrix[1] += min(
        severity_score * (1.0 + spine_strain / 20.0),
        max_intensity,
    )
    heatmap_matrix[2] += min(
        severity_score * (1.0 + shoulder_strain / 25.0),
        max_intensity,
    )


def apply_body_mask(image, torso_mask: np.ndarray):
    """
    Applies torso mask so heatmap is limited to body polygon.
    """
    return cv2.bitwise_and(image, image, mask=torso_mask)


def generate_report_overlay(
    base_frame,
    keypoints: dict,
    features: dict,
    severity_level: str,
):
    """
    Draws severity-scaled circles on the report frame — same logic as real-time view.
    neck between shoulder_mid and nose; spine at torso center; both shoulders.
    Color: LOW=yellow, MEDIUM=orange, HIGH=red (BGR).
    """
    if base_frame is None or keypoints is None or features is None:
        return base_frame

    # Points
    sm = (int(features["shoulder_mid"][0]), int(features["shoulder_mid"][1]))
    hm = (int(features["hip_mid"][0]),      int(features["hip_mid"][1]))
    nose = keypoints["nose"]
    neck_point  = (int((sm[0] + nose[0]) / 2), int((sm[1] + nose[1]) / 2))
    spine_point = (int((sm[0] + hm[0]) / 2),   int((sm[1] + hm[1]) / 2))

    # Per-region severity (0–1)
    neck_s     = min(1.0, features["neck_angle"]         / 30.0)
    spine_s    = min(1.0, features["spine_angle"]        / 30.0)
    shoulder_s = min(1.0, features["shoulder_alignment"] / 50.0)

    # Dynamic radii (capped at 80)
    BASE = 20
    neck_r     = min(80, int(BASE + neck_s     * 40))
    spine_r    = min(80, int(BASE + spine_s    * 50))
    shoulder_r = min(80, int(BASE + shoulder_s * 35))

    # Color by overall severity level (BGR)
    if severity_level == "HIGH":
        neck_col = shoulder_col = (0, 0, 255)
        spine_col = (0, 60, 255)
    elif severity_level == "MEDIUM":
        neck_col = shoulder_col = (0, 140, 255)
        spine_col = (0, 100, 255)
    else:  # LOW
        neck_col = shoulder_col = (0, 220, 255)
        spine_col = (0, 180, 255)

    overlay = base_frame.copy()
    cv2.circle(overlay, neck_point,                   neck_r,     neck_col,     -1)
    cv2.circle(overlay, spine_point,                  spine_r,    spine_col,    -1)
    cv2.circle(overlay, keypoints["left_shoulder"],   shoulder_r, shoulder_col, -1)
    cv2.circle(overlay, keypoints["right_shoulder"],  shoulder_r, shoulder_col, -1)

    return cv2.addWeighted(overlay, 0.4, base_frame, 0.6, 0)


def render_report(
    base_frame,
    heat_overlay,
    posture_status: str,
    severity_level: str,
    risk_level: str,
    duration_sec: float,
):
    """
    Renders a dashboard-like posture report layout.
    """
    if base_frame is None or heat_overlay is None:
        return base_frame

    h, w = heat_overlay.shape[:2]
    panel_w = 420
    canvas = np.zeros((h + 120, w + panel_w + 60, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    # Main image card (left)
    img_x, img_y = 30, 90
    cv2.rectangle(canvas, (img_x - 6, img_y - 6), (img_x + w + 6, img_y + h + 6), (45, 45, 55), -1)
    canvas[img_y:img_y + h, img_x:img_x + w] = heat_overlay

    # Right panel
    right_x = img_x + w + 24
    cv2.rectangle(canvas, (right_x, img_y), (right_x + panel_w, img_y + h), (28, 28, 36), -1)

    # Title and meta
    cv2.putText(
        canvas,
        "POSTURE ANALYSIS REPORT",
        (30, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        (240, 240, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        (30, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 185),
        1,
        cv2.LINE_AA,
    )

    # Summary block
    sy = img_y + 36
    cv2.putText(canvas, "Summary", (right_x + 20, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 235), 2, cv2.LINE_AA)
    severity_color = (0, 255, 255) if severity_level == "LOW" else (0, 180, 255) if severity_level == "MEDIUM" else (0, 0, 255)
    risk_color = (0, 255, 0) if risk_level == "LOW" else (0, 255, 255) if risk_level == "MEDIUM" else (0, 0, 255)
    posture_color = (0, 255, 0) if posture_status == "GOOD" else (0, 255, 255) if posture_status == "MODERATE" else (0, 0, 255)

    lines = [
        ("Posture", posture_status, posture_color),
        ("Severity", severity_level, severity_color),
        ("Risk", risk_level, risk_color),
        ("Duration", f"{duration_sec:.1f} s", (210, 210, 220)),
    ]
    for idx, (label, value, color) in enumerate(lines):
        y = sy + 36 + idx * 32
        cv2.putText(canvas, f"{label}:", (right_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 195), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (right_x + 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

    # Circle-size legend
    ly = sy + 190
    cv2.putText(canvas, "Circle Size = Strain Level", (right_x + 20, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 235), 1, cv2.LINE_AA)

    legend_items = [
        (12, (0, 220, 255), "Low"),
        (20, (0, 140, 255), "Medium"),
        (30, (0, 0, 255),   "High"),
    ]
    lx_start = right_x + 30
    lx = lx_start
    for r, col, label in legend_items:
        cy_leg = ly + 30 + r
        cv2.circle(canvas, (lx + r, cy_leg), r, col, -1)
        cv2.putText(canvas, label, (lx, cy_leg + r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 185), 1, cv2.LINE_AA)
        lx += r * 2 + 28


    return canvas


def draw_feedback_message(frame, posture_status: str, risk_level: str) -> None:
    """
    Real-time user feedback:
    - RED: high risk or bad posture
    - YELLOW: moderate posture
    """
    message = ""
    color = (255, 255, 255)

    if risk_level == "HIGH" or posture_status == "BAD":
        message = "Correct your posture!"
        color = (0, 0, 255)  # Red in BGR
    elif posture_status == "MODERATE":
        message = "Sit straight"
        color = (0, 255, 255)  # Yellow in BGR

    if message:
        cv2.putText(
            frame,
            message,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )


def create_and_save_posture_report(
    output_path: str,
    source_frame,
    keypoints,
    features,
    heatmap_matrix: np.ndarray,
    posture_status: str,
    severity_level: str,
    risk_level: str,
    duration_sec: float,
) -> bool:
    """
    Builds and saves a final report image with cumulative heatmap and summary labels.
    """
    if source_frame is None or keypoints is None or features is None:
        return False

    heat_overlay = generate_report_overlay(
        source_frame.copy(),
        keypoints,
        features,
        severity_level,
    )
    report_img = render_report(
        source_frame,
        heat_overlay,
        posture_status,
        severity_level,
        risk_level,
        duration_sec,
    )

    return cv2.imwrite(output_path, report_img)


def append_session_log(
    session_log: list,
    session_start: float,
    features: dict,
    posture: str,
    severity_level: str,
    severity_score: float,
    risk_level: str,
) -> None:
    session_log.append(
        {
            "time_sec": round(time.time() - session_start, 3),
            "neck_angle": round(features["neck_angle"], 3),
            "spine_angle": round(features["spine_angle"], 3),
            "side_lean_angle": round(features["side_lean_angle"], 3),
            "posture": posture,
            "severity": severity_level,
            "severity_score": round(severity_score, 3),
            "risk": risk_level,
        }
    )


def draw_posture_overlay(
    frame,
    keypoints,
    features,
    view_mode,
    status,
    bad_duration_sec,
    risk_level,
    severity_level,
    severity_score,
    calib_state,
) -> None:
    shoulder_mid = tuple(map(int, features["shoulder_mid"]))
    hip_mid = tuple(map(int, features["hip_mid"]))

    cv2.circle(frame, keypoints["nose"], 5, (0, 255, 255), -1)
    cv2.circle(frame, shoulder_mid, 5, (0, 255, 0), -1)
    cv2.circle(frame, hip_mid, 5, (255, 255, 0), -1)
    cv2.line(frame, shoulder_mid, keypoints["nose"], (0, 255, 255), 2)
    cv2.line(frame, shoulder_mid, hip_mid, (255, 255, 0), 2)

    if calib_state["is_calibrating"]:
        status_text = "CALIBRATING..."
        color = (0, 165, 255)  # Orange in BGR
    else:
        status_text = status
        if status == "GOOD":
            color = (0, 255, 0)  # Green
        elif status == "MODERATE":
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

    overlay_lines = [
        f"View: {view_mode}",
        f"Lean: {features['lean_direction']}",
        f"Shoulder width: {features['shoulder_width']:.1f} px",
        f"Neck (smoothed): {features['neck_angle']:.1f} deg",
        f"Spine (smoothed): {features['spine_angle']:.1f} deg",
        f"Side lean (smoothed): {features['side_lean_angle']:.1f} deg",
        f"Shoulder y-diff: {features['shoulder_alignment']:.1f} px",
    ]

    if calib_state["is_calibrating"]:
        overlay_lines.append(f"Status: {status_text}")
    else:
        overlay_lines.append(
            f"Baseline: neck {calib_state['base_neck']:.1f} | spine {calib_state['base_spine']:.1f}"
        )
        overlay_lines.append(
            f"Posture: {status_text}  |  Severity: {severity_level} ({severity_score:.1f} deg)"
        )
        overlay_lines.append(f"Bad posture time: {bad_duration_sec:.1f} s")
        overlay_lines.append(f"Risk: {risk_level}")

    highlight_start = len(overlay_lines) - 3 if not calib_state["is_calibrating"] else len(overlay_lines) - 1

    for idx, text in enumerate(overlay_lines):
        y = 30 + idx * 30
        text_color = color if idx >= highlight_start else (0, 255, 0)
            
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")
    frame_index = 0
    neck_angle_history = deque(maxlen=7)
    side_lean_history = deque(maxlen=7)
    bad_start_time = None
    session_start = time.time()
    session_log: list = []
    baseline_capture_done = False
    heatmap_matrix = None
    last_eval = {"status": "GOOD", "severity_level": "LOW", "risk_level": "LOW"}
    baseline_state = {
        "frame": None,
        "keypoints": None,
        "features": None,
        "status": "GOOD",
        "severity_level": "LOW",
        "risk_level": "LOW",
    }
    calib_state = {
        "is_calibrating": True,
        "start_time": time.time(),
        "duration": 8.0, # 8 seconds calibration period
        "neck_samples": [],
        "spine_samples": [],
        "base_neck": 10.0,
        "base_spine": 5.0,
        "base_shoulder": 10.0,
        "prev_status": "GOOD"
    }

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_index * 33
            frame_index += 1

            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks[0]
                draw_pose(frame, pose_landmarks)

                keypoints = extract_posture_keypoints(pose_landmarks, frame.shape)
                features = compute_posture_features(keypoints)
                if heatmap_matrix is None:
                    heatmap_matrix = initialize_heatmap_matrix(frame.shape)
                features["neck_angle"] = smooth_value(
                    neck_angle_history, features["neck_angle"]
                )
                features["side_lean_angle"] = smooth_value(
                    side_lean_history, features["side_lean_angle"]
                )
                features["spine_angle"] = abs(features["side_lean_angle"])
                # Recompute lean label from smoothed side angle for display consistency
                sl = features["side_lean_angle"]
                if sl > 5:
                    features["lean_direction"] = "RIGHT"
                elif sl < -5:
                    features["lean_direction"] = "LEFT"
                else:
                    features["lean_direction"] = "CENTERED"

                view_mode = detect_camera_view(features["shoulder_width"], frame.shape[1])
                current_time = time.time()

                is_calibrating = update_calibration(features, calib_state, current_time)
                if is_calibrating and (not baseline_capture_done):
                    baseline_capture_done = True
                    baseline_state["frame"] = frame.copy()
                    baseline_state["keypoints"] = dict(keypoints)
                    baseline_state["features"] = dict(features)

                if is_calibrating:
                    status = "CALIBRATING"
                    bad_duration_sec = 0.0
                    risk_level = "LOW"
                    severity_level = "LOW"
                    severity_score = 0.0
                else:
                    status = classify_posture(view_mode, features, calib_state)
                    severity_level, severity_score = compute_severity(features)
                    bad_duration_sec, bad_start_time = update_bad_posture_duration(
                        status, bad_start_time, current_time
                    )
                    risk_level = evaluate_risk_level(bad_duration_sec, severity_level)
                    draw_severity_heatmap(
                        frame,
                        keypoints,
                        features,
                        status,
                        severity_level,
                    )
                    append_session_log(
                        session_log,
                        session_start,
                        features,
                        status,
                        severity_level,
                        severity_score,
                        risk_level,
                    )
                    accumulate_strain_heatmap(
                        heatmap_matrix,
                        keypoints,
                        features,
                        status,
                        severity_score,
                        calib_state,
                        is_calibrating,
                    )
                    last_eval["status"] = status
                    last_eval["severity_level"] = severity_level
                    last_eval["risk_level"] = risk_level

                draw_posture_overlay(
                    frame,
                    keypoints,
                    features,
                    view_mode,
                    status,
                    bad_duration_sec,
                    risk_level,
                    severity_level,
                    severity_score,
                    calib_state,
                )
                if not is_calibrating:
                    draw_feedback_message(frame, status, risk_level)

                if not is_calibrating:
                    print(
                        f"View: {view_mode} | "
                        "Neck angle: "
                        f"{features['neck_angle']:.1f} deg | "
                        "Shoulder y-diff: "
                        f"{features['shoulder_alignment']:.1f} px | "
                        "Spine angle: "
                        f"{features['spine_angle']:.1f} deg | "
                        f"Lean: {features['lean_direction']} | "
                        f"Posture: {status} | "
                        f"Severity: {severity_level} ({severity_score:.1f}) | "
                        f"Bad Time: {bad_duration_sec:.1f} s | "
                        f"Risk: {risk_level}"
                    )

            cv2.imshow("Real-Time Posture Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    report_saved = create_and_save_posture_report(
        "posture_report.png",
        baseline_state["frame"],
        baseline_state["keypoints"],
        baseline_state["features"],
        heatmap_matrix,
        last_eval["status"],
        last_eval["severity_level"],
        last_eval["risk_level"],
        time.time() - session_start,
    )
    if report_saved:
        print("Saved report image: posture_report.png")
    else:
        print("Report image not saved (no valid posture frame captured).")

    print(f"Session log entries: {len(session_log)} (list stored in memory during run).")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
