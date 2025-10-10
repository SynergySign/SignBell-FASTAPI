import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math

# --- 모델 클래스 정의 ---
# 학습 스크립트(cnn_bilstm_attention_classifier.py)와 동일한 모델 구조로 교체합니다.

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (batch_first)"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CNN_BiLSTM_Attention(nn.Module):
    """학습 스크립트와 동일한 모델 아키텍처"""

    def __init__(self, input_size=147, num_classes=7,
                 cnn_channels=[64, 128, 256],
                 lstm_hidden=128,
                 dropout=0.5):
        super().__init__()

        # 1D-CNN Layers
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

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)

        # 동작 변화점 가중치 학습
        self.temporal_weight = nn.Parameter(torch.tensor(0.1))
        self.change_detector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Positional Encoding
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

        # 추론 시에는 output만 반환하도록 수정
        return output

# --- Predictor 모듈 ---
class Predictor:
    def __init__(self, model_path, sequence_length=300, input_size=147):
        """
        모델을 로드하고 추론을 준비합니다.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Predictor가 사용할 디바이스: {self.device}")

        # 1. 체크포인트를 먼저 로드하여 모델의 설정을 확인합니다.
        checkpoint = torch.load(model_path, map_location=self.device)

        # 2. 체크포인트에서 클래스 개수와 레이블 정보를 가져옵니다.
        if 'label_encoder' in checkpoint:
            self.label_encoder = checkpoint['label_encoder']
            num_classes = len(self.label_encoder.classes_)
            self.class_labels = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            print(f"✅ 체크포인트에서 {num_classes}개의 클래스 정보를 로드했습니다.")
        else:
            # label_encoder가 없는 경우, state_dict에서 직접 추론합니다.
            try:
                num_classes = checkpoint['model_state_dict']['classifier.6.bias'].shape[0]
                self.class_labels = {i: f'Sign_{i}' for i in range(num_classes)}
                print(f"⚠️ 체크포인트에 레이블 정보가 없어, 모델 가중치에서 {num_classes}개 클래스를 추론했습니다.")
            except KeyError:
                raise ValueError("모델 체크포인트에서 클래스 개수를 확인할 수 없습니다.")

        # 3. 확인된 클래스 개수로 모델 구조를 정의합니다.
        self.model = CNN_BiLSTM_Attention(
            input_size=input_size,
            num_classes=num_classes, # 동적으로 설정된 클래스 개수 사용
            cnn_channels=[64, 128, 256],
            lstm_hidden=128,
            dropout=0.5
        ).to(self.device)

        # 4. 모델 가중치를 로드합니다.
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
            state_dict = checkpoint['model'].state_dict()
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()  # 추론 모드로 설정

        print(f"✅ Predictor 초기화 완료. 모델: {model_path}")

    def predict(self, landmark_sequence: np.ndarray) -> tuple[str, float]:
        """
        랜드마크 시퀀스를 받아 예측을 수행하고, 레이블과 신뢰도 점수를 반환합니다.
        :param landmark_sequence: 랜드마크 데이터의 numpy 배열
        :return: (예측된 클래스 레이블, 신뢰도 점수) 튜플
        """
        # NumPy 배열의 유효성을 올바르게 확인합니다.
        if landmark_sequence is None or landmark_sequence.size == 0:
            return "랜드마크 데이터가 없습니다.", 0.0

        # landmark_sequence가 이미 numpy 배열이라고 가정하고 타입 변환
        landmarks_np = landmark_sequence.astype(np.float32)

        # 입력 데이터 전처리 (정규화, 크기 조절 등)
        # TODO: 학습 시 사용했던 전처리 과정을 여기에 동일하게 적용해야 합니다.
        # 예: landmarks_np = (landmarks_np - mean) / std
        
        # 입력 데이터 크기 조절 (패딩/자르기)
        num_frames = landmarks_np.shape[0]
        max_frames = 300 # 학습 시 사용한 max_frames와 동일해야 함
        if num_frames < max_frames:
            pad_size = max_frames - num_frames
            # 패딩 배열의 데이터 타입도 float32로 명시하여 타입 불일치 문제를 방지합니다.
            landmarks_np = np.vstack([
                landmarks_np,
                np.zeros((pad_size, landmarks_np.shape[1]), dtype=np.float32)
            ])
        elif num_frames > max_frames:
            landmarks_np = landmarks_np[:max_frames]

        # 모델 입력 형태에 맞게 텐서로 변환 (batch_size=1)
        input_tensor = torch.from_numpy(landmarks_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Softmax를 적용하여 확률 계산
            probabilities = torch.softmax(outputs, dim=1)
            # 가장 높은 확률(점수)과 해당 인덱스(예측)를 가져옴
            score, predicted_idx = torch.max(probabilities, 1)

        predicted_label = self.class_labels.get(predicted_idx.item(), "알 수 없는 기호")
        
        return predicted_label, score.item()

def get_predictor():
    """Predictor 인스턴스를 생성하여 반환하는 팩토리 함수"""
    
    # 프로젝트 루트 디렉토리를 기준으로 모델 경로 설정
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "cnn_bilstm_attention_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # Predictor 생성 시 num_classes 인자를 제거합니다.
    # 클래스 개수는 체크포인트에서 직접 읽어오기 때문입니다.
    predictor = Predictor(
        model_path=str(model_path),
        sequence_length=300,
        input_size=147
    )
    return predictor

if __name__ == '__main__':
    # 모듈 단독 테스트용 코드
    try:
        print("Predictor 모듈 테스트 시작...")
        predictor = get_predictor()
        
        # 더미 데이터 생성 (batch=1, sequence_length=30, input_size=147)
        dummy_input = np.random.rand(30, 147)

        prediction, score = predictor.predict(dummy_input)
        print(f"🚀 더미 데이터 예측 결과: {prediction} (Score: {score:.4f})")

        print("Predictor 모듈 테스트 성공.")
    except Exception as e:
        print(f"❌ Predictor 모듈 테스트 실패: {e}")
