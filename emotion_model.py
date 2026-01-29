"""
emotion_model.py - Face Detection and Feature Extraction using MediaPipe FaceMesh
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
CONTEXT_LABELS = ["Happy", "Sad", "Angry", "Surprise", "Neutral"]


@dataclass
class FaceAnalysisResult:
    context_label: str
    intensity: float
    confidence: float
    features: Dict[str, float]
    face_bbox: Tuple[int, int, int, int]
    face_rgb: np.ndarray
    landmarks_xy: Dict[int, Tuple[float, float]]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def _bbox_from_landmarks(lm_xy, w, h, pad=0.18):
    xs = [p[0] for p in lm_xy]
    ys = [p[1] for p in lm_xy]
    x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    bw, bh = (x1 - x0), (y1 - y0)
    px, py = pad * bw, pad * bh
    x0 = int(max(0, x0 - px))
    y0 = int(max(0, y0 - py))
    x1 = int(min(w - 1, x1 + px))
    y1 = int(min(h - 1, y1 + py))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def analyze_face(rgb: np.ndarray) -> Optional[FaceAnalysisResult]:
    h, w = rgb.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        lm_xy_full = [(p.x * w, p.y * h) for p in lm]

        x, y, bw, bh = _bbox_from_landmarks(lm_xy_full, w, h, pad=0.20)
        face_rgb = rgb[y:y + bh, x:x + bw].copy()

        landmarks_xy = {}
        for idx, p in enumerate(lm):
            fx = (p.x * w) - x
            fy = (p.y * h) - y
            landmarks_xy[idx] = (fx, fy)

        def pt(idx):
            return landmarks_xy[idx]

        mouth_left, mouth_right = pt(61), pt(291)
        mouth_top, mouth_bottom = pt(13), pt(14)
        left_eye_top, left_eye_bottom = pt(159), pt(145)
        right_eye_top, right_eye_bottom = pt(386), pt(374)
        left_brow, right_brow = pt(70), pt(300)
        left_eye_center, right_eye_center = pt(33), pt(263)

        mouth_width = _dist(mouth_left, mouth_right)
        mouth_open = _dist(mouth_top, mouth_bottom)
        left_eye_open = _dist(left_eye_top, left_eye_bottom)
        right_eye_open = _dist(right_eye_top, right_eye_bottom)
        brow_raise = ((left_eye_center[1] - left_brow[1]) + (right_eye_center[1] - right_brow[1])) / 2.0

        mouth_open_n = _safe_div(mouth_open, mouth_width)
        eye_open_n = _safe_div((left_eye_open + right_eye_open) / 2.0, mouth_width)
        brow_raise_n = _safe_div(brow_raise, mouth_width)
        smile_n = _safe_div(((mouth_top[1] + mouth_bottom[1]) / 2.0 - (mouth_left[1] + mouth_right[1]) / 2.0), mouth_width)

        scores = {
            "Happy": 1.4 * max(smile_n, 0.0) + 0.5 * mouth_open_n,
            "Sad": 1.2 * max(-smile_n, 0.0) + 0.4 * max(0.02 - eye_open_n, 0.0),
            "Angry": 1.0 * max(0.02 - brow_raise_n, 0.0) + 0.4 * max(0.02 - mouth_open_n, 0.0),
            "Surprise": 1.2 * mouth_open_n + 1.0 * max(brow_raise_n, 0.0) + 0.4 * eye_open_n,
            "Neutral": 0.35,
        }

        context_label = max(scores.items(), key=lambda x: x[1])[0]
        best = float(scores[context_label])
        intensity = float(np.clip(best / 0.12, 0.0, 1.0))
        sorted_vals = sorted(scores.values(), reverse=True)
        margin = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else float(sorted_vals[0])
        confidence = float(np.clip(margin / 0.08, 0.0, 1.0))

        features = {
            "mouth_open_norm": float(mouth_open_n),
            "smile_norm": float(smile_n),
            "brow_raise_norm": float(brow_raise_n),
            "eye_open_norm": float(eye_open_n),
        }

        if confidence < 0.15:
            context_label = "Neutral"
            intensity = 0.35
            confidence = 0.2

        return FaceAnalysisResult(
            context_label=context_label,
            intensity=intensity,
            confidence=confidence,
            features=features,
            face_bbox=(x, y, bw, bh),
            face_rgb=face_rgb,
            landmarks_xy=landmarks_xy,
        )
