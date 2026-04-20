import os
import urllib.request
import math

import cv2
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
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }


def classify_posture(neck_angle, spine_angle, shoulder_diff):
    """
    Classifies posture based on defined thresholds:
    - BAD if any angle/diff is severely out of alignment
    - MODERATE if any angle/diff is slightly out of alignment
    - GOOD if all are within healthy ranges
    """
    if neck_angle > 25 or spine_angle > 20 or shoulder_diff > 40:
        return "BAD"
    elif neck_angle > 15 or spine_angle > 10 or shoulder_diff > 20:
        return "MODERATE"
    else:
        return "GOOD"


def draw_posture_overlay(frame, keypoints, features) -> None:
    shoulder_mid = tuple(map(int, features["shoulder_mid"]))
    hip_mid = tuple(map(int, features["hip_mid"]))

    cv2.circle(frame, keypoints["nose"], 5, (0, 255, 255), -1)
    cv2.circle(frame, shoulder_mid, 5, (0, 255, 0), -1)
    cv2.circle(frame, hip_mid, 5, (255, 255, 0), -1)
    cv2.line(frame, shoulder_mid, keypoints["nose"], (0, 255, 255), 2)
    cv2.line(frame, shoulder_mid, hip_mid, (255, 255, 0), 2)

    status = classify_posture(
        features["neck_angle"],
        features["spine_angle"],
        features["shoulder_alignment"]
    )

    if status == "GOOD":
        color = (0, 255, 0)      # Green in BGR
    elif status == "MODERATE":
        color = (0, 255, 255)    # Yellow in BGR
    else:
        color = (0, 0, 255)      # Red in BGR

    overlay_lines = [
        f"Neck angle: {features['neck_angle']:.1f} deg",
        f"Shoulder y-diff: {features['shoulder_alignment']:.1f} px",
        f"Spine angle: {features['spine_angle']:.1f} deg",
        f"Lean direction: {features['lean_direction']} ({features['side_lean_angle']:.1f} deg)",
        f"Posture status: {status}",
    ]

    for idx, text in enumerate(overlay_lines):
        y = 30 + idx * 30
        
        # Only use the dynamic color for the last line (Posture status)
        text_color = color if idx == len(overlay_lines) - 1 else (0, 255, 0)
        
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
                draw_posture_overlay(frame, keypoints, features)

                print(
                    "Neck angle: "
                    f"{features['neck_angle']:.1f} deg | "
                    "Shoulder y-diff: "
                    f"{features['shoulder_alignment']:.1f} px | "
                    "Spine angle: "
                    f"{features['spine_angle']:.1f} deg | "
                    f"Lean: {features['lean_direction']} ({features['side_lean_angle']:.1f} deg)"
                )

            cv2.imshow("Real-Time Posture Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
