"""
모듈: processing/landmark_extractor.py
설명:
- 학습 시 사용된 coordinate_extractor의 특징 생성 로직을 재현하여 실시간 프레임으로부터
  프레임별 특징 벡터(147-dim)를 생성합니다.
- 주요 구성 요소:
  - RealtimeLandmarkExtractor: 단일 프레임에서 랜드마크를 추출하고 정규화된 특징 벡터를 반환
  - SequenceBuilder / extract_sequence_from_frames: 복수 프레임을 시퀀스로 빌드

구현 세부:
- Pose 상체 관절 6개, 좌/우 손의 MediaPipe 21포인트, 양 손목 거리 벡터 등을 이용해 총 49 포인트 (147 차원) 특징을 생성
- 정규화 규칙과 누락 프레임 처리(패딩/스킵)를 학습 스크립트와 호환되게 구현

주의: mediapipe, opencv, numpy 등이 없으면 안전하게 None을 반환하도록 방어 코드를 포함합니다.

since: 2025.10.17
author: 백승현
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any

# -------- Dependencies --------
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image
import io


POSE_INDICES = [11, 12, 13, 14, 15, 16]
NORMALIZATION_SCALE = 0.3
POSE_Z_DAMPING = 0.7
WRIST_Z_DAMPING = 0.7
FINGER_Z_DAMPING = 0.6
FRAME_FEATURE_DIM = 49 * 3  # 147
HAND_LANDMARK_COUNT = 21


def _decode_frame(frame_bytes: bytes):
    """JPEG/PNG 바이트를 RGB ndarray(uint8)로 디코딩합니다. 실패 시 None 반환."""
    # OpenCV 우선 시도. 실패하면 Pillow로 폴백.
    try:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        # ignore and try Pillow fallback below
        pass

    # Pillow fallback
    try:
        if Image is not None and io is not None:
            with Image.open(io.BytesIO(frame_bytes)) as im:
                return np.array(im.convert("RGB"))
    except Exception:
        return None

    return None


@dataclass
class RealtimeLandmarkExtractor:
    """실시간 랜드마크 추출기.

    역할/정의:
    - MediaPipe Holistic을 사용하여 포즈/양손 랜드마크를 추출하고 학습 방식과 동일한
      정규화/패딩 규칙으로 (147,) 크기의 특징 벡터를 반환합니다.

    반환값/예외:
    - available()가 False이거나 디코딩 실패 시 None을 반환합니다.

    since: 2025.10.17
    author: 백승현
    """
    static_image_mode: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    skip_missing: bool = False  # True 면 pose 미검출 프레임은 None 반환 (학습과 유사), False 면 0 벡터
    _holistic: Optional[Any] = field(init=False, default=None)
    _ok: bool = field(init=False, default=False)

    def __post_init__(self):
        try:
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                refine_face_landmarks=False,
                enable_segmentation=False,
            )
            self._ok = True
        except Exception:  # noqa: E722
            self._ok = False
            self._holistic = None

    def available(self) -> bool:
        return self._ok and self._holistic is not None

    def close(self):
        if self._holistic is not None:
            try:
                self._holistic.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: E722
                pass

    # ------------- Core -------------
    def extract(self, frame_bytes: bytes) -> Optional["np.ndarray"]:
        """단일 프레임 바이트 -> (147,) float32 또는 None 반환.

        None 반환 조건:
          - 의존성 없음
          - 디코딩 실패
          - pose 미검출 및 skip_missing=True
        """
        if not self.available():
            return None
        rgb = _decode_frame(frame_bytes)
        if rgb is None:
            return None
        try:
            results = self._holistic.process(rgb)  # type: ignore[union-attr]
        except Exception:  # noqa: E722
            return None
        if (not hasattr(results, "pose_landmarks")
                or results.pose_landmarks is None
                or not results.pose_landmarks.landmark):
            if self.skip_missing:
                return None
            # zero frame 채움
            return np.zeros((FRAME_FEATURE_DIM,), dtype=np.float32)

        pose_lm = results.pose_landmarks.landmark
        nose = pose_lm[0]
        left_shoulder = pose_lm[11]
        right_shoulder = pose_lm[12]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        stable_center_x = (shoulder_center_x + nose.x) / 2
        stable_center_y = (shoulder_center_y + nose.y) / 2
        stable_center_z = (shoulder_center_z + nose.z) / 2

        feats: List[float] = []

        def _rel(v, cx, cy, cz, scale, damp=1.0):
            return (
                (v.x - cx) / scale,
                (v.y - cy) / scale,
                ((v.z - cz) / scale) * damp,
            )

        # Pose 6 joints
        try:
            for idx in POSE_INDICES:
                v = pose_lm[idx]
                rx, ry, rz = _rel(v, stable_center_x, stable_center_y, stable_center_z, NORMALIZATION_SCALE, POSE_Z_DAMPING)
                feats.extend([rx, ry, rz])
        except Exception as e:
            print(f"Error processing pose landmarks: {e}")
            # Re-raise or handle as appropriate
            raise

        # Left hand
        left_wrist_pos = None
        try:
            if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks and results.left_hand_landmarks.landmark:
                lms = results.left_hand_landmarks.landmark
                wrist = lms[0]
                wx, wy, wz = _rel(wrist, stable_center_x, stable_center_y, stable_center_z, NORMALIZATION_SCALE, WRIST_Z_DAMPING)
                left_wrist_pos = (wx, wy, wz)
                feats.extend([wx, wy, wz])  # wrist relative to stable center
                for i in range(1, HAND_LANDMARK_COUNT):  # 1..20
                    f = lms[i]
                    fx = (f.x - wrist.x) / NORMALIZATION_SCALE
                    fy = (f.y - wrist.y) / NORMALIZATION_SCALE
                    fz = ((f.z - wrist.z) / NORMALIZATION_SCALE) * FINGER_Z_DAMPING
                    feats.extend([fx, fy, fz])
                # 만약 landmark 개수가 부족하면 패딩
                missing = HAND_LANDMARK_COUNT - len(lms)
                if missing > 0:
                    feats.extend([0.0] * (missing * 3))
            else:
                feats.extend([0.0] * (HAND_LANDMARK_COUNT * 3))
        except Exception as e:
            print(f"Error processing left hand landmarks: {e}")
            raise

        # Right hand
        right_wrist_pos = None
        try:
            if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks and results.right_hand_landmarks.landmark:
                lms = results.right_hand_landmarks.landmark
                wrist = lms[0]
                wx, wy, wz = _rel(wrist, stable_center_x, stable_center_y, stable_center_z, NORMALIZATION_SCALE, WRIST_Z_DAMPING)
                right_wrist_pos = (wx, wy, wz)
                feats.extend([wx, wy, wz])
                for i in range(1, HAND_LANDMARK_COUNT):
                    f = lms[i]
                    fx = (f.x - wrist.x) / NORMALIZATION_SCALE
                    fy = (f.y - wrist.y) / NORMALIZATION_SCALE
                    fz = ((f.z - wrist.z) / NORMALIZATION_SCALE) * FINGER_Z_DAMPING
                    feats.extend([fx, fy, fz])
                missing = HAND_LANDMARK_COUNT - len(lms)
                if missing > 0:
                    feats.extend([0.0] * (missing * 3))
            else:
                feats.extend([0.0] * (HAND_LANDMARK_COUNT * 3))
        except Exception as e:
            print(f"Error processing right hand landmarks: {e}")
            raise

        # Hand distance vector
        if left_wrist_pos and right_wrist_pos:
            dx = right_wrist_pos[0] - left_wrist_pos[0]
            dy = right_wrist_pos[1] - left_wrist_pos[1]
            dz = right_wrist_pos[2] - left_wrist_pos[2]
            feats.extend([dx, dy, dz])
        else:
            feats.extend([0.0, 0.0, 0.0])

        arr = np.asarray(feats, dtype=np.float32)
        if arr.shape[0] != FRAME_FEATURE_DIM:
            # 안전 실패: 차원 불일치 → 패딩/절단
            if arr.shape[0] < FRAME_FEATURE_DIM:
                pad = np.zeros((FRAME_FEATURE_DIM - arr.shape[0],), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:FRAME_FEATURE_DIM]
        return arr


@dataclass
class SequenceBuilder:
    extractor: RealtimeLandmarkExtractor
    frames: List["np.ndarray"] = field(default_factory=list)

    def add_frame(self, frame_bytes: bytes):
        feat = self.extractor.extract(frame_bytes)
        if feat is not None:
            self.frames.append(feat)

    def build(self, target_len: Optional[int] = None, pad_value: float = 0.0):
        if np is None or not self.frames:
            return None
        seq = np.stack(self.frames, axis=0)  # (T, 147)
        if target_len is not None:
            T, F = seq.shape
            if T > target_len:
                seq = seq[:target_len]
            elif T < target_len:
                pad = np.full((target_len - T, F), pad_value, dtype=seq.dtype)
                seq = np.concatenate([seq, pad], axis=0)
        return seq  # (target_len, 147)


def extract_sequence_from_frames(frames: List[bytes], target_len: Optional[int] = None, skip_missing: bool = False):
    """프레임 바이트 리스트 -> (T|target_len, 147) numpy or None.
    mediapipe / numpy 없으면 None.
    """
    extractor = RealtimeLandmarkExtractor(skip_missing=skip_missing)
    if not extractor.available():
        return None
    builder = SequenceBuilder(extractor)
    for fb in frames:
        builder.add_frame(fb)
    seq = builder.build(target_len=target_len)
    extractor.close()
    return seq


__all__ = [
    "RealtimeLandmarkExtractor",
    "SequenceBuilder",
    "extract_sequence_from_frames",
    "FRAME_FEATURE_DIM",
]
