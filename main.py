"""
SignSense Server - 실시간 수화 인식 FastAPI 서버

=== 서버 목적 ===
이 서버는 학습된 CNN-BiLSTM-Attention 모델을 사용하여 웹캠을 통한 실시간 수화 인식을 제공합니다.
웹소켓을 통해 클라이언트와 실시간으로 통신하며, 브라우저에서 바로 수화 인식을 체험할 수 있습니다.

=== 서버 실행 방법 ===
1. 터미널에서 이 폴더로 이동:
   cd "D:\Sirosho\signTranslator\SignSense_Server\SignSense_Server"

2. 서버 시작:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

   또는 개발용 (자동 재시작):
   uvicorn main:app --reload

3. 서버 실행 확인:
   브라우저에서 http://localhost:8000 접속

=== 서버 종료 방법 ===
- 터미널에서 Ctrl+C 키를 눌러 서버 종료

=== 주요 엔드포인트 ===
- GET  /              : 서버 상태 확인
- GET  /demo          : 실시간 수화 인식 데모 페이지 (웹 인터페이스)
- GET  /model/info    : 로드된 모델 정보 조회
- POST /predict/image : 단일 이미지 파일로 수화 예측
- WebSocket /ws/realtime : 실시간 웹캠 수화 예측
- GET  /health        : 서버 헬스체크

=== 사용 방법 ===
1. 브라우저 데모 (추천):
   http://localhost:8000/demo
   - 웹캠을 통한 실시간 수화 인식
   - PC와 모바일 모두 지원

2. 모바일/다른 기기에서 접속:
   http://[컴퓨터IP주소]:8000/demo
   예: http://192.168.1.100:8000/demo
   (같은 Wi-Fi 네트워크에 연결된 기기)

3. API 테스트:
   curl http://localhost:8000/model/info

=== 필요한 파일들 ===
- model.pth : 학습된 모델 파일 (자동으로 로드됨)
- predictor.py : 실시간 예측 엔진
- main.py : 이 FastAPI 서버

=== 지원 기능 ===
- 실시간 웹캠 수화 인식 (WebSocket)
- 모바일 최적화 (반응형 UI, 터치 제스처, 진동 피드백)
- 신뢰도 표시 및 모든 클래스 확률 분포
- 자동 재연결 및 오류 처리
- 원형 버퍼를 통한 메모리 효율적 프레임 관리

=== 트러블슈팅 ===
- 모델 로딩 실패: model.pth 파일이 같은 폴더에 있는지 확인
- 카메라 접근 실패: 브라우저에서 카메라 권한 허용 필요
- 모바일 연결 실패: 같은 Wi-Fi 네트워크 연결 및 방화벽 확인
- WebSocket 연결 실패: 서버가 정상 실행 중인지 확인

Created by: SignSense Team
Last Updated: 2025-01-10
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image
import json
import logging
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from cnn_bilstm_attention_classifier import CNN_BiLSTM_Attention

# 모델 로딩을 위해 필요한 클래스들을 여기에도 정의
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 생성
app = Flask(__name__)
CORS(app)  # 크로스 오리진 요청 허용 (아이패드에서 접근 가능)

# 전역 예측기 인스턴스
predictor = None

class SignLanguagePredictor:
    def __init__(self, model_path='models/cnn_bilstm_attention_model.pth'):
        self.device = torch.device('cpu')  # CPU 사용 (서버 환경 고려)
        self.model = None
        self.label_encoder = None
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            print(f"모델 파일 로드 시도: {model_path}")

            # 모델 파일 존재 확인
            import os
            if not os.path.exists(model_path):
                print(f"⚠️ 모델 파일이 존재하지 않습니다: {model_path}")
                print("기본 모델로 초기화합니다...")
                self._initialize_default_model()
                return

            # 체크포인트 로드 - weights_only=False로 설정하여 객체도 로드
            print("체크포인트 로드 중...")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"체크포인트 키들: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'dict가 아님'}")

            # 라벨 인코더 로드
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
                print(f"라벨 인코더 로드 성공: {list(self.label_encoder.classes_)}")
            else:
                print("⚠️ 라벨 인코더가 체크포인트에 없습니다.")
                self._initialize_default_model()
                return

            # 모델 생성 - 새로운 구조로
            print("모델 인스턴스 생성 중...")
            self.model = CNN_BiLSTM_Attention(
                input_size=147,
                num_classes=len(self.label_encoder.classes_),
                cnn_channels=[64, 128, 256],
                lstm_hidden=128,
                dropout=0.5
            )

            # 모델 상태 로드
            if 'model_state_dict' in checkpoint:
                print("model_state_dict 로드 중...")
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ model_state_dict 로드 성공")
                except RuntimeError as e:
                    print(f"⚠️ state_dict 로드 실패: {e}")
                    print("가중치 매칭 문제로 기본 모델로 전환합니다.")
                    self._initialize_default_model()
                    return
            elif 'model' in checkpoint:
                print("모델 객체에서 state_dict 추출 중...")
                try:
                    if hasattr(checkpoint['model'], 'state_dict'):
                        self.model.load_state_dict(checkpoint['model'].state_dict())
                    elif isinstance(checkpoint['model'], dict):
                        self.model.load_state_dict(checkpoint['model'])
                    else:
                        print("⚠️ 모델 state_dict를 찾을 수 없습니다.")
                        self._initialize_default_model()
                        return
                except RuntimeError as e:
                    print(f"⚠️ state_dict 로드 실패: {e}")
                    print("가중치 매칭 문제로 기본 모델로 전환합니다.")
                    self._initialize_default_model()
                    return
            else:
                print("⚠️ 모델 가중치를 찾을 수 없습니다.")
                self._initialize_default_model()
                return

            self.model.eval()
            print(f"✅ 모델이 성공적으로 로드되었습니다!")
            print(f"   클래스 수: {len(self.label_encoder.classes_)}")
            print(f"   사용 가능한 클래스: {list(self.label_encoder.classes_)}")

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print(f"오류 타입: {type(e).__name__}")
            print("기본 모델로 초기화합니다...")
            self._initialize_default_model()

    def _initialize_default_model(self):
        """기본 모델로 초기화 (모델 파일이 없을 때)"""
        try:
            print("기본 모델 초기화 중...")

            # 기본 레이블 인코더 생성
            self.label_encoder = LabelEncoder()
            default_classes = ['가족', '고맙다', '나', '사랑', '안녕', '아니다', '좋다']
            self.label_encoder.classes_ = np.array(default_classes)

            # 기본 모델 생성 - 새로운 구조로
            self.model = CNN_BiLSTM_Attention(
                input_size=147,
                num_classes=len(default_classes),
                cnn_channels=[64, 128, 256],
                lstm_hidden=128,
                dropout=0.5
            )

            self.model.eval()
            print(f"✅ 기본 모델 초기화 완료")
            print(f"   기본 클래스: {default_classes}")
            print("⚠️ 주의: 학습된 가중치가 아닌 랜덤 가중치를 사용합니다")

        except Exception as e:
            print(f"❌ 기본 모델 초기화도 실패: {e}")
            raise RuntimeError("모델 초기화에 완전히 실패했습니다")

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
                return {"error": "모델이 로드되지 않았습니다.", "success": False}

            # 입력 데이터 전처리
            if isinstance(keypoints_data, list):
                keypoints_data = np.array(keypoints_data)

            # 차원 확인 및 조정
            if len(keypoints_data.shape) == 2:
                keypoints_data = keypoints_data.reshape(1, *keypoints_data.shape)

            # 텐서로 변환
            input_tensor = torch.FloatTensor(keypoints_data).to(self.device)

            # 예측 - 새로운 forward 출력 형식 처리
            with torch.no_grad():
                model_output = self.model(input_tensor)

                # 모델 출력이 튜플인지 확인 (output, attention_weights, change_scores)
                if isinstance(model_output, tuple):
                    outputs = model_output[0]  # 첫 번째 요소가 실제 출력
                else:
                    outputs = model_output

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

def init_predictor():
    """예측기 초기화"""
    global predictor
    try:
        predictor = SignLanguagePredictor('models/cnn_bilstm_attention_model.pth')
        logger.info("예측기 초기화 성공")
        return True
    except Exception as e:
        logger.error(f"예측기 초기화 실패: {e}")
        return False

# HTML 템플릿 (아이패드용 웹 인터페이스)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignSense - 수어 인식</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .camera-container {
            position: relative;
            margin-bottom: 30px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        #videoElement {
            width: 100%;
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .btn {
            flex: 1;
            min-width: 120px;
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #2196F3;
            color: white;
        }
        
        .btn-success {
            background: #4CAF50;
            color: white;
        }
        
        .btn-danger {
            background: #f44336;
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .result-item {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .result-item:last-child {
            margin-bottom: 0;
        }
        
        .prediction {
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 1.1em;
            color: #666;
            margin-bottom: 15px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        
        .top3-list {
            margin-top: 15px;
        }
        
        .top3-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .top3-item:last-child {
            border-bottom: none;
        }
        
        .status {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .status.loading {
            background: #fff3cd;
            color: #856404;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .model-info {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .model-info h3 {
            color: #1976d2;
            margin-bottom: 10px;
        }
        
        .classes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .class-tag {
            background: #2196F3;
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            text-align: center;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
        }
        
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤟 SignSense</h1>
            <p>AI 기반 실시간 수어 인식 시스템</p>
        </div>
        
        <div class="content">
            <div class="camera-container">
                <video id="videoElement" autoplay muted playsinline></video>
            </div>
            
            <div class="controls">
                <button id="startBtn" class="btn btn-primary">카메라 시작</button>
                <button id="stopBtn" class="btn btn-danger" disabled>카메라 중지</button>
                <button id="predictBtn" class="btn btn-success" disabled>수어 인식</button>
            </div>
            
            <div id="status" class="status loading hidden">
                시스템 준비 중...
            </div>
            
            <div id="results" class="results hidden">
                <h3>🎯 인식 결과</h3>
                <div id="resultContent"></div>
            </div>
            
            <div id="modelInfo" class="model-info">
                <h3>📊 모델 정보</h3>
                <div id="modelContent">모델 정보를 불러오는 중...</div>
            </div>
        </div>
    </div>

    <script>
        let videoElement = document.getElementById('videoElement');
        let startBtn = document.getElementById('startBtn');
        let stopBtn = document.getElementById('stopBtn');
        let predictBtn = document.getElementById('predictBtn');
        let statusDiv = document.getElementById('status');
        let resultsDiv = document.getElementById('results');
        let resultContent = document.getElementById('resultContent');
        let modelInfo = document.getElementById('modelInfo');
        let modelContent = document.getElementById('modelContent');
        
        let stream = null;
        let isRecording = false;
        let frameBuffer = [];
        let maxFrames = 60; // 2초 정도의 프레임 (30fps 기준)
        
        // 상태 업데이트 함수
        function updateStatus(message, type = 'loading') {
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.classList.remove('hidden');
        }
        
        // 결과 표시 함수
        function displayResults(data) {
            if (data.error) {
                updateStatus(`오류: ${data.error}`, 'error');
                return;
            }
            
            if (!data.success) {
                updateStatus('예측에 실패했습니다.', 'error');
                return;
            }
            
            let html = `
                <div class="result-item">
                    <div class="prediction">🤟 ${data.predicted_class}</div>
                    <div class="confidence">신뢰도: ${(data.confidence * 100).toFixed(1)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    ${data.top3_predictions ? `
                        <div class="top3-list">
                            <strong>상위 3개 예측:</strong>
                            ${data.top3_predictions.map(pred => `
                                <div class="top3-item">
                                    <span>${pred.class}</span>
                                    <span>${(pred.confidence * 100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
            
            resultContent.innerHTML = html;
            resultsDiv.classList.remove('hidden');
            updateStatus('예측 완료!', 'success');
        }
        
        // 모델 정보 로드
        async function loadModelInfo() {
            try {
                const response = await fetch('/api/model-info');
                const data = await response.json();
                
                if (data.error) {
                    modelContent.innerHTML = `<p style="color: red;">오류: ${data.error}</p>`;
                    return;
                }
                
                let html = `
                    <p><strong>클래스 수:</strong> ${data.num_classes}</p>
                    <p><strong>입력 크기:</strong> ${data.input_size}</p>
                    <p><strong>모델 상태:</strong> ${data.model_loaded ? '✅ 로드됨' : '❌ 로드 안됨'}</p>
                    <div class="classes-grid">
                        ${data.classes.map(cls => `<div class="class-tag">${cls}</div>`).join('')}
                    </div>
                `;
                
                modelContent.innerHTML = html;
            } catch (error) {
                modelContent.innerHTML = `<p style="color: red;">모델 정보 로드 실패: ${error.message}</p>`;
            }
        }
        
        // 카메라 시작
        async function startCamera() {
            try {
                updateStatus('카메라를 시작하는 중...', 'loading');
                
                const constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 },
                        facingMode: 'user' // 전면 카메라 사용
                    }
                };
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                videoElement.srcObject = stream;
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                predictBtn.disabled = false;
                
                updateStatus('카메라가 시작되었습니다. 수어를 준비해주세요!', 'success');
                
            } catch (error) {
                updateStatus(`카메라 시작 실패: ${error.message}`, 'error');
                console.error('카메라 오류:', error);
            }
        }
        
        // 카메라 중지
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                videoElement.srcObject = null;
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            predictBtn.disabled = true;
            
            updateStatus('카메라가 중지되었습니다.', 'loading');
        }
        
        // 프레임 캡처 및 예측
        async function predictSign() {
            if (!stream) {
                updateStatus('카메라가 시작되지 않았습니다.', 'error');
                return;
            }
            
            try {
                updateStatus('수어를 인식하는 중...', 'loading');
                predictBtn.disabled = true;
                
                // 캔버스에 현재 프레임 캡처
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                ctx.drawImage(videoElement, 0, 0);
                
                // Base64로 변환
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // 서버로 전송
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: imageData,
                        timestamp: Date.now()
                    })
                });
                
                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                updateStatus(`예측 오류: ${error.message}`, 'error');
                console.error('예측 오류:', error);
            } finally {
                predictBtn.disabled = false;
            }
        }
        
        // 이벤트 리스너
        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);
        predictBtn.addEventListener('click', predictSign);
        
        // 페이지 로드 시 모델 정보 로드
        window.addEventListener('load', loadModelInfo);
        
        // 페이지 종료 시 카메라 정리
        window.addEventListener('beforeunload', stopCamera);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """모델 정보 반환"""
    try:
        if predictor is None:
            return jsonify({"error": "예측기가 초기화되지 않았습니다."})

        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"모델 정보 조회 오류: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """수어 예측 API"""
    try:
        if predictor is None:
            return jsonify({"error": "예측기가 초기화되지 않았습니다.", "success": False})

        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "이미지 데이터가 필요합니다.", "success": False})

        # Base64 이미지 디코딩
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # 이미지 변환
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # MediaPipe를 사용한 키포인트 추출
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        # RGB로 변환하여 처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        if not results.pose_landmarks:
            return jsonify({
                "error": "포즈를 감지할 수 없습니다. 전신이 화면에 나오도록 해주세요.",
                "success": False
            })

        # 키포인트 추출 (간단한 버전)
        keypoints = []

        # 포즈 랜드마크 (상체 주요 포인트)
        pose_indices = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
        for idx in pose_indices:
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0.0, 0.0, 0.0])

        # 손 랜드마크 (왼손)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)  # 21개 랜드마크 * 3좌표

        # 손 랜드마크 (오른손)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)  # 21개 랜드마크 * 3좌표

        # 양손 거리 (3차원)
        keypoints.extend([0.0, 0.0, 0.0])

        # 147차원으로 맞추기
        while len(keypoints) < 147:
            keypoints.append(0.0)
        keypoints = keypoints[:147]

        # 시퀀스 형태로 변환 (30프레임으로 가정)
        sequence_length = 30
        keypoints_sequence = np.tile(np.array(keypoints), (sequence_length, 1))

        # 예측 수행
        result = predictor.predict(keypoints_sequence)

        holistic.close();
        return jsonify(result)

    except Exception as e:
        logger.error(f"예측 API 오류: {e}")
        return jsonify({
            "error": f"예측 중 오류가 발생했습니다: {str(e)}",
            "success": False
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "timestamp": "2024-10-07"
    })

if __name__ == '__main__':
    print("🚀 SignSense 서버를 시작합니다...")

    # 예측기 초기화
    if init_predictor():
        print("✅ 예측기 초기화 성공")
    else:
        print("⚠️ 예측기 초기화 실패 - 기본 모드로 실행")

    print("🌐 웹 서버 시작...")
    print("📱 아이패드에서 접속하려면:")
    print("   1. 컴퓨터와 아이패드가 같은 WiFi에 연결되어 있는지 확인")
    print("   2. 컴퓨터의 IP 주소 확인 (예: 192.168.1.100)")
    print("   3. 아이패드 Safari/Chrome에서 http://[컴퓨터IP]:5000 접속")
    print("   4. 로컬에서는 http://localhost:5000 또는 http://127.0.0.1:5000")

    # Flask 서버 실행
    app.run(
        host='0.0.0.0',  # 모든 IP에서 접근 가능
        port=5000,
        debug=True,
        threaded=True
    )
