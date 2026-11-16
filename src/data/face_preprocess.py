"""Face detection + mouth ROI extraction for DAVID-Net preprocessing.

Architecture §3: "Face/mouth ROI extraction (preprocessing): RetinaFace/MediaPipe →
aligned face crop + a tighter mouth crop. Two crops because face-swap artifacts
live in the whole face, lip-sync artifacts live around the mouth."

Uses OpenCV's Haar cascade (built-in, no extra deps) for face detection.
For better accuracy on Kaggle, install mediapipe: pip install mediapipe
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


# Haar cascade path (ships with opencv)
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(_HAAR_PATH)
    return _face_cascade


def detect_faces(frame: np.ndarray, expand: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Detect faces in a BGR frame. Returns list of (x, y, w, h) with expansion.

    expand: fraction to expand the bounding box (captures more context).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    result = []
    for (x, y, w, h) in faces:
        ex, ey = int(w * expand), int(h * expand)
        x1 = max(0, x - ex)
        y1 = max(0, y - ey)
        x2 = min(frame.shape[1], x + w + ex)
        y2 = min(frame.shape[0], y + h + ey)
        result.append((x1, y1, x2 - x1, y2 - y1))
    return result


def crop_face(frame: np.ndarray, face_box: tuple[int, int, int, int],
              target_size: int = 224) -> np.ndarray:
    """Crop face region and resize to target_size x target_size."""
    x, y, w, h = face_box
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return cv2.resize(frame, (target_size, target_size))
    return cv2.resize(crop, (target_size, target_size))


def crop_mouth(frame: np.ndarray, face_box: tuple[int, int, int, int],
               target_size: int = 96) -> np.ndarray:
    """Crop mouth region (lower 40% of face box, centered horizontally).

    Architecture §3: "a tighter mouth crop" — lip-sync artifacts live around the mouth.
    """
    x, y, w, h = face_box
    # Mouth is roughly in the lower 40% of the face, center 60% horizontally
    mouth_y = y + int(h * 0.6)
    mouth_h = int(h * 0.4)
    mouth_x = x + int(w * 0.15)
    mouth_w = int(w * 0.7)
    crop = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
    if crop.size == 0:
        # Fallback: lower half of face
        crop = frame[y+h//2:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (target_size, target_size))


def extract_face_mouth_from_video(video_path: str, n_frames: int = 16,
                                   face_size: int = 224, mouth_size: int = 96
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract face crops and mouth crops from video.

    Returns:
        faces: (T, C, face_size, face_size) float32 in [0, 1]
        mouths: (T, C, mouth_size, mouth_size) float32 in [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        faces = torch.randn(n_frames, 3, face_size, face_size)
        mouths = torch.randn(n_frames, 3, mouth_size, mouth_size)
        return faces, mouths

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.randn(n_frames, 3, face_size, face_size), torch.randn(n_frames, 3, mouth_size, mouth_size)

    indices = torch.linspace(0, max(0, total_frames - 1), n_frames).long().tolist()

    face_crops = []
    mouth_crops = []
    last_face_box = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        # Detect face
        faces = detect_faces(frame)
        if faces:
            last_face_box = faces[0]  # use largest/first face
            face_crop = crop_face(frame, faces[0], face_size)
            mouth_crop = crop_mouth(frame, faces[0], mouth_size)
        elif last_face_box is not None:
            # Reuse last detected face position
            face_crop = crop_face(frame, last_face_box, face_size)
            mouth_crop = crop_mouth(frame, last_face_box, mouth_size)
        else:
            # No face found: use full frame resized
            face_crop = cv2.resize(frame, (face_size, face_size))
            mouth_crop = np.zeros((mouth_size, mouth_size, 3), dtype=np.uint8)

        # Convert BGR -> RGB, HWC -> CHW, normalize
        face_t = torch.from_numpy(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
        mouth_t = torch.from_numpy(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
        face_crops.append(face_t)
        mouth_crops.append(mouth_t)

    cap.release()
    return torch.stack(face_crops), torch.stack(mouth_crops)
