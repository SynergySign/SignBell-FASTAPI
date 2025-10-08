import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math

# --- 모델 클래스 정의 ---

class PositionalEncoding(nn.Module):
    """
    Transformer 모델에서 사용하는 Positional Encoding을 구현합니다.
    모델이 저장될 때 이 클래스도 함께 저장되었을 가능성이 높습니다.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 참고: 이 클래스는 'models/cnn_bilstm_attention_model.pth' 파일이 저장될 때 사용된
# 모델의 아키텍처와 정확히 일치해야 합니다.
# 만약 실제 모델과 구조가 다를 경우, 여기서 수정해야 합니다.
class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(CNN_BiLSTM_Attention, self).__init__()
        # 이 부분은 모델의 실제 구조에 따라 변경되어야 합니다.
        # 아래는 파일명을 기반으로 한 기본적인 추정입니다.
        self.feature_embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)

        # 1. Feature Embedding & Positional Encoding
        x = self.feature_embedding(x) # (batch_size, sequence_length, hidden_size)
        x = self.pos_encoder(x)       # (batch_size, sequence_length, hidden_size)

        # 2. CNN
        x = x.permute(0, 2, 1) # (batch_size, hidden_size, sequence_length)
        x = self.cnn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1) # (batch_size, sequence_length, hidden_size)
        
        # 3. BiLSTM
        lstm_out, _ = self.lstm(x) # (batch_size, sequence_length, hidden_size * 2)
        
        # 4. Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # (batch_size, hidden_size * 2)
        
        out = self.fc(context_vector)
        return out

# --- Predictor 모듈 ---
class Predictor:
    def __init__(self, model_path, num_classes=10, sequence_length=30, input_size=147):
        """
        모델을 로드하고 추론을 준비합니다.
        :param model_path: .pth 모델 파일의 경로
        :param num_classes: 모델의 출력 클래스 수
        :param sequence_length: 모델이 기대하는 입력 시퀀스 길이
        :param input_size: 각 프레임의 랜드마크 특성 수 (e.g., 49 * 3 = 147)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Predictor가 사용할 디바이스: {self.device}")

        # 모델 구조 정의
        # 참고: hidden_size, num_layers는 저장된 모델과 동일한 값이어야 합니다.
        self.model = CNN_BiLSTM_Attention(
            input_size=input_size,
            hidden_size=256,
            num_layers=2,
            num_classes=num_classes,
            sequence_length=sequence_length
        ).to(self.device)

        # 모델 가중치 로드
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 추론 모드로 설정
        
        # TODO: 클래스 레이블 맵을 실제 학습 데이터에 맞게 수정해야 합니다.
        self.class_labels = {i: f'Sign_{i}' for i in range(num_classes)}
        print(f"✅ Predictor 초기화 완료. 모델: {model_path}")

    def predict(self, landmark_sequence: list) -> str:
        """
        랜드마크 시퀀스를 받아 예측을 수행합니다.
        :param landmark_sequence: 랜드마크 데이터의 리스트 (길이는 sequence_length와 유사해야 함)
        :return: 예측된 클래스 레이블 (문자열)
        """
        if not landmark_sequence:
            return "랜드마크 데이터가 없습니다."

        # 리스트를 numpy 배열로 변환
        landmarks_np = np.array(landmark_sequence, dtype=np.float32)

        # 입력 데이터 전처리 (정규화, 크기 조절 등)
        # TODO: 학습 시 사용했던 전처리 과정을 여기에 동일하게 적용해야 합니다.
        # 예: landmarks_np = (landmarks_np - mean) / std
        
        # 모델 입력 형태에 맞게 텐서로 변환 (batch_size=1)
        input_tensor = torch.from_numpy(landmarks_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        predicted_label = self.class_labels.get(predicted_idx.item(), "알 수 없는 기호")
        
        return predicted_label

def get_predictor():
    """Predictor 인스턴스를 생성하여 반환하는 팩토리 함수"""
    
    # 프로젝트 루트 디렉토리를 기준으로 모델 경로 설정
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "cnn_bilstm_attention_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # TODO: num_classes, sequence_length, input_size를 실제 모델에 맞게 조정
    predictor = Predictor(
        model_path=str(model_path),
        num_classes=10,      # 예시: 10개의 수어 단어
        sequence_length=30,  # 예시: 30프레임
        input_size=147       # 랜드마크 추출기(147D)에 맞게 수정
    )
    return predictor

if __name__ == '__main__':
    # 모듈 단독 테스트용 코드
    try:
        print("Predictor 모듈 테스트 시작...")
        predictor = get_predictor()
        
        # 더미 데이터 생성 (batch=1, sequence_length=30, input_size=147)
        dummy_input = np.random.rand(30, 147).tolist()

        prediction = predictor.predict(dummy_input)
        print(f"🚀 더미 데이터 예측 결과: {prediction}")
        
        print("Predictor 모듈 테스트 성공.")
    except Exception as e:
        print(f"❌ Predictor 모듈 테스트 실패: {e}")
