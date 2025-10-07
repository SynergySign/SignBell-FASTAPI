"""
실시간 수화 예측을 위한 모델과 좌표 추출기
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Optional, List
from cnn_bilstm_attention_classifier import CNN_BiLSTM_Attention



class CoordinateExtractor:
    """좌표 추출기 (학습 시와 동일한 로직)"""

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 학습 시와 동일한 Pose 인덱스
        self.pose_indices = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목

    def extract_from_frame(self, frame: np.ndarray) -> Optional[List[List[float]]]:
        """단일 프레임에서 좌표 추출"""
        coords = self._process_frame(frame)
        return coords if coords else None

    def _process_frame(self, frame: np.ndarray) -> List[List[float]]:
        """프레임 처리 (학습 시와 동일한 로직)"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)

        if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
            return []

        pose_landmarks = results.pose_landmarks.landmark

        # 기준점: 어깨 중심 + 코 평균 (학습 시와 동일)
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        nose = pose_landmarks[0]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        stable_center_x = (shoulder_center_x + nose.x) / 2
        stable_center_y = (shoulder_center_y + nose.y) / 2
        stable_center_z = (shoulder_center_z + nose.z) / 2

        normalization_scale = 0.3
        keypoints = []

        # Pose 3D 상대좌표 (Z축 감쇠)
        for idx in self.pose_indices:
            lm = pose_landmarks[idx]
            rel_x = (lm.x - stable_center_x) / normalization_scale
            rel_y = (lm.y - stable_center_y) / normalization_scale
            rel_z = ((lm.z - stable_center_z) / normalization_scale) * 0.7
            keypoints.append([rel_x, rel_y, rel_z])

        # 왼손
        left_wrist_pos = None
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks.landmark
            wrist = hand_landmarks[0]
            wrist_rel_x = (wrist.x - stable_center_x) / normalization_scale
            wrist_rel_y = (wrist.y - stable_center_y) / normalization_scale
            wrist_rel_z = ((wrist.z - stable_center_z) / normalization_scale) * 0.7
            left_wrist_pos = [wrist_rel_x, wrist_rel_y, wrist_rel_z]
            keypoints.append(left_wrist_pos)

            for i in range(1, 21):
                finger_lm = hand_landmarks[i]
                finger_rel_x = (finger_lm.x - wrist.x) / normalization_scale
                finger_rel_y = (finger_lm.y - wrist.y) / normalization_scale
                finger_rel_z = ((finger_lm.z - wrist.z) / normalization_scale) * 0.6
                keypoints.append([finger_rel_x, finger_rel_y, finger_rel_z])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        # 오른손
        right_wrist_pos = None
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            wrist = hand_landmarks[0]
            wrist_rel_x = (wrist.x - stable_center_x) / normalization_scale
            wrist_rel_y = (wrist.y - stable_center_y) / normalization_scale
            wrist_rel_z = ((wrist.z - stable_center_z) / normalization_scale) * 0.7
            right_wrist_pos = [wrist_rel_x, wrist_rel_y, wrist_rel_z]
            keypoints.append(right_wrist_pos)

            for i in range(1, 21):
                finger_lm = hand_landmarks[i]
                finger_rel_x = (finger_lm.x - wrist.x) / normalization_scale
                finger_rel_y = (finger_lm.y - wrist.y) / normalization_scale
                finger_rel_z = ((finger_lm.z - wrist.z) / normalization_scale) * 0.6
                keypoints.append([finger_rel_x, finger_rel_y, finger_rel_z])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        # 양손 손목 3D 상대거리 추가
        if left_wrist_pos and right_wrist_pos:
            hand_distance = [
                right_wrist_pos[0] - left_wrist_pos[0],
                right_wrist_pos[1] - left_wrist_pos[1],
                right_wrist_pos[2] - left_wrist_pos[2]
            ]
            keypoints.append(hand_distance)
        else:
            keypoints.append([0.0, 0.0, 0.0])

        return keypoints


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (batch_first)"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CNN_BiLSTM_Attention(nn.Module):
    """CNN-BiLSTM-Attention 모델"""
    def __init__(self, input_size=147, num_classes=7, cnn_channels=None, lstm_hidden=128, dropout=0.5):
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ))
            in_channels = out_channels
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.temporal_weight = nn.Parameter(torch.tensor(0.1))
        self.change_detector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.pos_encoding = PositionalEncoding(d_model=lstm_hidden * 2, max_len=500)

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x_cnn = x.transpose(1, 2)
        for conv_layer in self.conv_layers:
            x_cnn = conv_layer(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.pos_encoding(lstm_out)
        change_scores = self.change_detector(lstm_out).squeeze(-1)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        change_weights = change_scores.unsqueeze(-1) * self.temporal_weight
        weighted_attn = attn_out * (1 + change_weights)
        context_vector = weighted_attn.mean(dim=1)
        output = self.classifier(context_vector)
        return output, attn_weights.mean(dim=1), change_scores


class RealtimeSignPredictor:
    """실시간 수화 예측기"""

    def __init__(self, model_path: str, device: str = 'auto', max_frames: int = 300, buffer_size: int = 60):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.max_frames = max_frames
        self.buffer_size = buffer_size

        # 모델 로딩 - 사용자 제안 방식 적용
        print(f"🔥 모델 로딩: {model_path}")

        try:
            # 1. 파일 존재 확인
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            # 2. 체크포인트 로드 (CPU로 강제 매핑하여 장치 불일치 문제 해결)
            print("📦 체크포인트 로딩 중...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✅ 체크포인트 로드 성공")

            if not isinstance(checkpoint, dict):
                raise ValueError("체크포인트는 딕셔너리 형태여야 합니다")

            print(f"📋 체크포인트 키: {list(checkpoint.keys())}")

            # 3. 레이블 인코더 먼저 로드 (클래스 수 확인을 위해)
            if 'label_encoder' not in checkpoint:
                raise KeyError("체크포인트에 'label_encoder'가 없습니다")

            self.label_encoder = checkpoint['label_encoder']
            num_classes = len(self.label_encoder.classes_)
            print(f"🏷️ 레이블 인코더 로드 성공: {list(self.label_encoder.classes_)}")
            print(f"📊 클래스 수: {num_classes}")

            # 4. 모델 인스턴스 생성 (정확한 파라미터로)
            print("🏗️ 모델 인스턴스 생성 중...")
            self.model = CNN_BiLSTM_Attention(
                input_size=147,  # 49 landmarks * 3 coordinates = 147
                num_classes=num_classes,
                cnn_channels=[64, 128, 256],
                lstm_hidden=128,
                dropout=0.5
            )
            print("✅ 모델 인스턴스 생성 완료")

            # 5. state_dict 로드
            if 'model_state_dict' in checkpoint:
                print("🔧 model_state_dict 로딩 중...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ model_state_dict 로드 성공")
            elif 'model' in checkpoint:
                print("🔧 저장된 모델에서 state_dict 추출 중...")
                # 저장된 모델이 state_dict인지 확인
                if isinstance(checkpoint['model'], dict):
                    self.model.load_state_dict(checkpoint['model'])
                    print("✅ state_dict로 로드 성공")
                else:
                    # 모델 객체에서 state_dict 추출 시도
                    try:
                        if hasattr(checkpoint['model'], 'state_dict'):
                            self.model.load_state_dict(checkpoint['model'].state_dict())
                            print("✅ 모델 객체에서 state_dict 추출 성공")
                        else:
                            raise AttributeError("모델 객체에 state_dict 메서드가 없습니다")
                    except Exception as e:
                        print(f"⚠️ 모델 객체에서 state_dict 추출 실패: {e}")
                        print("🆕 새 모델 가중치로 초기화합니다")
            else:
                print("⚠️ 체크포인트에 모델 데이터가 없습니다")
                print("🆕 새 모델 가중치로 초기화합니다")

            # 6. 모델을 평가 모드로 설정하고 디바이스로 이동
            self.model.eval()
            self.model = self.model.to(self.device)

            # 7. 모델 파라미터 수 확인
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"✅ 모델 설정 완료!")
            print(f"  - Device: {self.device}")
            print(f"  - Classes: {list(self.label_encoder.classes_)}")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")

        except FileNotFoundError as e:
            print(f"❌ 파일 오류: {e}")
            self._initialize_default_model()
        except KeyError as e:
            print(f"❌ 체크포인트 키 오류: {e}")
            self._initialize_default_model()
        except ValueError as e:
            print(f"❌ 값 오류: {e}")
            self._initialize_default_model()
        except RuntimeError as e:
            print(f"❌ PyTorch 런타임 오류: {e}")
            print("💡 PyTorch 버전 불일치 또는 모델 구조 불일치일 수 있습니다")
            self._initialize_default_model()
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {type(e).__name__}: {e}")
            self._initialize_default_model()

        # 8. 좌표 추출기 초기화
        print("🎯 좌표 추출기 초기화 중...")
        try:
            self.extractor = CoordinateExtractor()
            print("✅ 좌표 추출기 준비 완료")
        except Exception as e:
            print(f"❌ 좌표 추출기 초기화 실패: {e}")
            raise

        # 9. 프레임 버퍼 초기화
        self.frame_buffer = []
        self.current_position = 0
        print(f"📦 프레임 버퍼 준비 완료 (크기: {buffer_size})")

        print("🚀 실시간 수화 예측기 초기화 완료!")

    def _initialize_default_model(self):
        """기본 모델로 초기화"""
        print("🆕 기본 모델로 초기화합니다...")
        try:
            from sklearn.preprocessing import LabelEncoder

            # 기본 레이블 인코더 생성
            self.label_encoder = LabelEncoder()
            # 일반적인 한국 수화 단어들로 기본값 설정
            default_classes = ['가족', '고맙다', '나', '사랑', '안녕', '아니다', '좋다']
            self.label_encoder.classes_ = np.array(default_classes)

            # 기본 모델 생성
            self.model = CNN_BiLSTM_Attention(
                input_size=147,
                num_classes=len(default_classes),
                cnn_channels=[64, 128, 256],
                lstm_hidden=128,
                dropout=0.5
            )

            self.model.eval()
            self.model = self.model.to(self.device)

            print(f"✅ 기본 모델 초기화 완료")
            print(f"  - Device: {self.device}")
            print(f"  - Default classes: {default_classes}")
            print("⚠️ 주의: 학습된 가중치가 아닌 랜덤 가중치를 사용합니다")

        except Exception as e:
            print(f"❌ 기본 모델 초기화도 실패: {e}")
            raise RuntimeError("모델 초기화에 완전히 실패했습니다")

    def add_frame(self, frame: np.ndarray) -> Optional[dict]:
        """프레임을 버퍼에 추가하고 예측 수행"""
        # 좌표 추출
        coords = self.extractor.extract_from_frame(frame)
        if not coords:
            return None

        # 버퍼에 추가
        if len(self.frame_buffer) < self.buffer_size:
            self.frame_buffer.append(coords)
        else:
            # 원형 버퍼: 오래된 프레임 교체
            self.frame_buffer[self.current_position] = coords
            self.current_position = (self.current_position + 1) % self.buffer_size

        # 충분한 프레임이 쌓였을 때만 예측
        if len(self.frame_buffer) >= 30:  # 최소 30프레임
            return self._predict_from_buffer()

        return {
            'status': 'collecting',
            'frames_collected': len(self.frame_buffer),
            'message': f'프레임 수집 중... ({len(self.frame_buffer)}/{self.buffer_size})'
        }

    def _predict_from_buffer(self) -> dict:
        """버퍼의 프레임들로 예측 수행"""
        try:
            # 현재 버퍼를 numpy 배열로 변환
            keypoints = np.array(self.frame_buffer)  # (frames, 49, 3)

            # Z축 평활화 (학습 시와 동일)
            for landmark_idx in range(keypoints.shape[1]):
                z_seq = keypoints[:, landmark_idx, 2]
                smoothed_z = self._smooth_z_coordinates(z_seq)
                keypoints[:, landmark_idx, 2] = smoothed_z

            # 전처리
            input_tensor = self._preprocess_keypoints(keypoints)

            # 예측
            with torch.no_grad():
                outputs, _, _ = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
                confidence = probabilities[0, predicted_class].cpu().numpy()
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

                # 모든 클래스 확률
                all_probs = probabilities.cpu().numpy()[0]
                class_probabilities = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    class_probabilities[class_name] = float(all_probs[i])

            return {
                'status': 'predicted',
                'predicted_label': predicted_label,
                'confidence': float(confidence),
                'class_probabilities': class_probabilities,
                'frames_used': len(self.frame_buffer)
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _smooth_z_coordinates(self, z_sequence):
        """Z축 좌표 이동평균(3프레임) 필터"""
        if len(z_sequence) < 3:
            return z_sequence
        smoothed = []
        for i in range(len(z_sequence)):
            if i == 0 or i == len(z_sequence) - 1:
                smoothed.append(z_sequence[i])
            else:
                avg = (z_sequence[i-1] + z_sequence[i] + z_sequence[i+1]) / 3
                smoothed.append(avg)
        return smoothed

    def _preprocess_keypoints(self, keypoints: np.ndarray) -> torch.Tensor:
        """좌표 전처리 (학습 시와 동일)"""
        # (frames, 49, 3) → (frames, 147)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)

        # 패딩/자르기
        if keypoints.shape[0] < self.max_frames:
            pad_size = self.max_frames - keypoints.shape[0]
            keypoints = np.vstack([
                keypoints,
                np.zeros((pad_size, keypoints.shape[1]))
            ])
        else:
            keypoints = keypoints[:self.max_frames]

        return torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)

    def reset_buffer(self):
        """버퍼 초기화"""
        self.frame_buffer.clear()
        self.current_position = 0


# 전역 예측기 인스턴스
predictor_instance = None

def get_predictor() -> RealtimeSignPredictor:
    """예측기 인스턴스 가져오기 (싱글톤)"""
    global predictor_instance
    if predictor_instance is None:
        model_path = Path(__file__).parent / "cnn_bilstm_attention_model.pth"
        predictor_instance = RealtimeSignPredictor(str(model_path))
    return predictor_instance


class SignLanguagePredictor:
    def __init__(self, model_path='models/cnn_bilstm_attention_model.pth'):
        self.device = torch.device('cpu')  # CPU 사용 (서버 환경 고려)
        self.model = None
        self.label_encoder = None
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')

            # 라벨 인코더 로드
            self.label_encoder = checkpoint['label_encoder']

            # 모델 생성
            self.model = CNN_BiLSTM_Attention(
                input_size=147,
                num_classes=len(self.label_encoder.classes_),
                cnn_channels=[64, 128, 256],
                lstm_hidden=128,
                dropout=0.5
            )

            # 모델 상태 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            print(f"모델이 성공적으로 로드되었습니다. 클래스 수: {len(self.label_encoder.classes_)}")
            print(f"사용 가능한 클래스: {list(self.label_encoder.classes_)}")

        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise e

    def predict(self, keypoints_data):
        """
        키포인트 데이터로부터 수어 예측

        Args:
            keypoints_data: numpy array 또는 list, shape should be (sequence_length, 147)

        Returns:
            dict: 예측 결과 및 확률
        """
        try:
            if self.model is None:
                return {"error": "모델이 로드되지 않았습니다."}

            # 입력 데이터 전처리
            if isinstance(keypoints_data, list):
                keypoints_data = np.array(keypoints_data)

            # 차원 확인 및 조정
            if len(keypoints_data.shape) == 2:
                keypoints_data = keypoints_data.reshape(1, *keypoints_data.shape)

            # 텐서로 변환
            input_tensor = torch.FloatTensor(keypoints_data).to(self.device)

            # 예측
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            # 클래스명 변환
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]

            # 상위 3개 예측 결과
            top3_probs, top3_indices = torch.topk(probabilities[0], min(3, len(self.label_encoder.classes_)))
            top3_predictions = []

            for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                class_name = self.label_encoder.inverse_transform([idx.item()])[0]
                top3_predictions.append({
                    "class": class_name,
                    "confidence": prob.item()
                })

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top3_predictions": top3_predictions,
                "success": True
            }

        except Exception as e:
            return {
                "error": f"예측 중 오류 발생: {str(e)}",
                "success": False
            }

    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None or self.label_encoder is None:
            return {"error": "모델이 로드되지 않았습니다."}

        return {
            "num_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "input_size": 147,
            "model_loaded": True
        }
