# React 클라이언트 → FastAPI: WebSocket 핸드셰이크 및 WebRTC 시그널링 가이드

이 문서는 이 레포지토리(서버)의 WebSocket/ WebRTC 시그널링 규약을 바탕으로, React 프론트엔드에서 어떻게 핸드셰이크를 수행하고 실시간 영상(DataChannel) 전송을 시작할지 단계별 예제와 권장 방식을 설명합니다.

간단 요약
- 인증: Authorization 헤더(`Authorization: Bearer <token>`) 전송을 권장합니다. 서버는 Authorization 헤더를 우선으로 검사하도록 구성되어 있습니다. 브라우저 클라이언트의 경우는 쿠키 기반 인증 또는 BFF 패턴을 사용하세요.
+인증: 본 프로젝트의 서버(`main.py`)는 WebSocket 핸드셰이크 시 토큰을 다음 우선순위로 검사합니다:
+1) HTTP 쿠키(named by `configs.settings.COOKIE_ACCESS_TOKEN_NAME`, 기본값 `ACCESS_TOKEN`) (브라우저 클라이언트 권장)
+2) Authorization 헤더: `Authorization: Bearer <token>` (서버사이드/스크립트 클라이언트 권장)
+3) 쿼리 파라미터: `?token=<token>` (fallback)
+
+따라서 외부(서버사이드) 클라이언트는 Authorization 헤더 방식으로 연결하세요. 브라우저는 로그인 응답에서 `Set-Cookie`로 토큰을 내려주거나 개발용 `GET /debug/set-cookie?token=...`를 사용해 쿠키를 설정하면 됩니다.
- 메타전달: 연결 수립 후 `meta` 타입의 JSON으로 단어 정보(word_pk, word_name)를 전송합니다.
- 시그널링: SDP Offer/Answer와 ICE 후보는 웹소켓(text JSON)으로 교환합니다.
- 미디어/프레임: 실제 프레임 바이트는 WebRTC DataChannel(또는 미디어 트랙)을 통해 전송합니다.
- 추론 실행: 프레임 전송 후 `flush` 시그널을 보내면 서버가 추론을 실행하고 결과를 WebSocket으로 푸시합니다.

사전조건
- 브라우저 환경(HTTPS 또는 localhost), getUserMedia 권한 허용
- 로그인으로부터 획득한 유효한 JWT 토큰
- 서버의 시그널링 엔드포인트(예: `wss://<host>/ws/signaling`) 사용

체크리스트
- 1. JWT를 쿼리 파라미터로 웹소켓 연결에 포함
+체크리스트
+- 1. JWT를 쿠키에 저장하여 웹소켓 연결 시 브라우저가 자동으로 쿠키를 포함하도록 합니다.
2. 연결 수립 후 `meta` 메시지 전송
3. RTCPeerConnection / DataChannel 생성 및 Offer 전송
4. 서버로부터 Answer 수신 및 원격설정
5. DataChannel을 통해 프레임 전송 (또는 미디어 트랙 사용)
6. 추론을 원할 때 `flush` 시그널 전송

1) WebSocket 핸드셰이크 (권장: Authorization 헤더)
- 서버는 Authorization 헤더(`Authorization: Bearer <token>`)를 우선적으로 검사하도록 변경되었습니다. 외부(서버사이드) 클라이언트에서 WebSocket 업그레이드 요청을 보낼 때는 이 헤더를 설정해 주시기 바랍니다.

브라우저 클라이언트 제약
- 브라우저의 WebSocket API는 임의의 헤더를 설정할 수 없으므로, 브라우저에서 직접 `Authorization` 헤더를 넣을 수 없습니다. 브라우저에서 WebSocket으로 연결하려면 아래 중 하나를 사용하세요:
  - 서버가 로그인 응답에서 `Set-Cookie`로 토큰을 내려주어 브라우저가 쿠키를 자동 포함하도록 한다 (현재 프로젝트의 기존 방식과 호환).
  - BFF(Backend-for-Frontend)를 두어 브라우저는 BFF에 인증 정보를 보내고, BFF가 서버로 Authorization 헤더를 포함해 WebSocket 연결을 맺고 프록시한다.

외부(서버사이드) 클라이언트 예시 — Authorization 헤더 사용(권장)
```javascript
// Node.js (ws 라이브러리)
const WebSocket = require('ws');
const token = 'REPLACE_WITH_TOKEN';
const ws = new WebSocket('wss://your.server/ws/session-id', {
  headers: { Authorization: `Bearer ${token}` }
});
ws.on('open', () => console.log('open'));
ws.on('message', (m) => console.log('msg', m.toString()));
```

```python
# Python (websockets) - server-side client example
import asyncio
import websockets

async def run():
    token = "REPLACE_WITH_TOKEN"
    uri = "wss://your.server/ws/session-id"
    # `extra_headers`는 ('Header-Name', 'value') 튜플 리스트를 받습니다.
    headers = [("Authorization", f"Bearer {token}")]
    async with websockets.connect(uri, extra_headers=headers) as ws:
        # 예: 연결 직후 meta 전송
        await ws.send('{"type":"meta","word_pk":1, "word_name":"사랑"}')
        print(await ws.recv())

asyncio.run(run())
```

브라우저에서 사용(권장)
- 1) 로그인 또는 `/debug/set-cookie?token=...`를 통해 서버가 `Set-Cookie`로 토큰을 발급하면 브라우저는 WS 핸드셰이크에 쿠키를 자동으로 포함합니다 (프로젝트 기본 방식).
- 2) 또는 BFF(Backend-For-Frontend) 패턴을 사용해 브라우저가 BFF에 연결하고 BFF가 서버로 Authorization 헤더를 포함해 WebSocket을 연결/프록시하도록 구현할 수 있습니다.

개발자 편의: 서버에는 개발용 `GET /debug/set-cookie?token=<token>` 엔드포인트가 있어 브라우저 테스트 시 쉽게 토큰 쿠키를 설정할 수 있습니다 (실서비스에서는 사용 금지 또는 적절히 보호하세요).

2) 시그널링/제어 메시지 (프로젝트 구현 기준)
`main.py`의 `/ws/{session_id}` 엔드포인트는 텍스트(JSON) 메시지를 받아 아래 타입을 처리합니다:
   - `meta` : 단어 메타데이터 저장 (서버는 `meta_ack`로 응답)
   - `save_learning` : 수집된 frames를 학습 저장 스케줄링 (응답: `learning_ack`)
   - `flush` : 수집된 frames로 추론을 실행하고 퀴즈 저장을 스케줄링 (응답: `inference_result`)
   - 기타 메시지는 `noop`로 응답됩니다.

- 주의: 현재 저장소의 기본 `/ws/{session_id}` 핸들러는 텍스트 제어용으로 Offer/Answer/Candidate를 직접 처리하는 로직을 포함하지 않습니다. aiortc 기반의 시그널링(Offer/Answer/Candidate)을 사용하려면 서버의 별도 시그널링 경로(또는 aiortc 관련 구현)를 확인하거나 `main.py`를 확장해야 합니다. 다만 프로젝트에는 aiortc import 시도를 하는 코드가 있으므로, aiortc 관련 기능이 활성화된 환경에서는 확장된 시그널링 경로를 추가할 수 있습니다.

React 예제: 카메라를 이용한 Offer 생성 및 DataChannel 사용
```js
async function startConnection(ws, token) {
  // NOTE: 프로젝트의 현재 WS 핸들러는 DataChannel 시그널링을 처리하지 않습니다.
  // 만약 Media/DataChannel을 사용하려면 서버 측 aiortc 관련 엔드포인트(별도 구현)를 확인해야 합니다.
  // 이 파일의 기본 예제는 제어용 WebSocket으로 프레임 전송은 WebRTC DataChannel이 아닌
  // 브라우저에서 WebSocket(또는 별도 업로드 경로)로도 가능하다는 점을 고려해 참고만 하세요.
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  });

  // DataChannel - 프레임을 바이너리로 보낼 때 사용
  const dc = pc.createDataChannel('frames');
  dc.binaryType = 'arraybuffer';
  dc.onopen = () => console.log('DataChannel open');
  dc.onmessage = (ev) => console.log('DC msg', ev.data);

  // ICE 후보 발생 시 웹소켓으로 전송
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      ws.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
    }
  };

  // 로컬 카메라 트랙 추가(선택) - 서버에서 미디어 트랙을 사용하지 않고 DataChannel만 사용하는 경우 생략 가능
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach((t) => pc.addTrack(t, stream));
    // 뷰에 로컬 미리뷰 연결
    const localVideo = document.getElementById('localVideo');
    if (localVideo) localVideo.srcObject = stream;
  } catch (e) {
    console.error('getUserMedia 실패', e);
  }

  // Offer 생성
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Offer를 웹소켓으로 전송 (서버가 aiortc로 Answer 생성)
  ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp, sdpType: offer.type }));

  // 웹소켓에서 answer / candidate 메시지를 받는 핸들러가 필요
  ws.addEventListener('message', async (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    if (msg.type === 'answer') {
      await pc.setRemoteDescription({ type: 'answer', sdp: msg.sdp });
      console.log('Remote description set (answer)');
    } else if (msg.type === 'candidate') {
      try { await pc.addIceCandidate(msg.candidate); } catch (e) { console.warn('addIceCandidate failed', e); }
    }
  });

  return { pc, dc };
}
```

3) 프레임을 DataChannel로 전송하는 방법
- 캔버스에 현재 비디오 프레임을 그려서 Blob/ArrayBuffer로 변환해 전송합니다.
- 서버 측에서는 DataChannel으로 받은 ArrayBuffer를 프레임으로 취급해 SequenceCollector에 추가합니다.

예제: 캔버스 -> ArrayBuffer -> dataChannel.send
```js
async function sendFrame(dc, videoEl) {
  // 비디오를 캔버스에 그리기
  const cvs = document.createElement('canvas');
  cvs.width = videoEl.videoWidth || 640;
  cvs.height = videoEl.videoHeight || 480;
  const ctx = cvs.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, cvs.width, cvs.height);

  // Blob으로 변환 후 ArrayBuffer로
  const blob = await new Promise((res) => cvs.toBlob(res, 'image/jpeg', 0.7));
  const ab = await blob.arrayBuffer();

  // binary로 전송
  if (dc && dc.readyState === 'open') {
    dc.send(ab);
  }
}
```
- 전송 빈도는 서버/네트워크 부담을 고려해 적절히(예: 5~15 fps) 조절하세요.

4) 추론(좌표추출 + save-leaning) 요청 방법
- 서버는 클라이언트가 "flush" 시그널을 보내면 `run_inference`를 호출하도록 설계되어 있습니다.
- 어디로 보내는지는 서버 구현에 따라 달라질 수 있지만(문서 기준), WebSocket 제어 채널에 `{"type":"flush"}` 형태로 보내는 것이 안전합니다.

예시: flush 전송
```js
// inference 요청
ws.send(JSON.stringify({ type: 'flush' }));
```
- 서버는 추론이 끝나면 웹소켓으로 `{"type":"inference_result", "predicted": ..., "score": ...}` 형태로 결과를 전송합니다.

5) 메타데이터(단어 정보) 전송
- 연결 직후 `meta` 메시지로 전송하세요. 서버는 이 값을 세션 상태에 저장합니다.
```js
ws.send(JSON.stringify({ type: 'meta', word_pk: 42, word_name: '사랑' }));
```

6) 예외/운영 이슈 & 권장사항
- Token 만료: 연결 시 토큰 검증 실패할 수 있으니, 실패 톤을 받으면 재로그인/토큰 갱신 후 재연결 처리 필요
- 네트워크 장애: WebSocket 재연결 로직과 시그널링 재시도(Offer 재생성 등)를 구현하세요.
- 보안: 반드시 HTTPS/WSS 사용. STUN/TURN 서버 구성은 NAT 환경에서 안정적 연결을 위해 권장.
- 프레임 크기/빈도 제한: 서버의 SequenceCollector/메모리 제한 고려.
- binary 형태: DataChannel에서 binary(ArrayBuffer/Blob)를 사용하면 서버측 aiortc에서 더 쉽게 처리됩니다.

7) 간단 플로우 다이어그램
1) React: JWT 포함하여 WebSocket 연결
2) React -> WS: meta 메시지
3) React: RTCPeerConnection + DataChannel 생성, Offer 생성
4) React -> WS: offer 전달
5) Server (aiortc) -> WS: answer 전달
6) React: setRemoteDescription(answer)
7) React -> DataChannel: 프레임 전송
8) React -> WS: flush (추론 실행 요청)
9) Server -> WS: inference_result 전송

마무리
- 위 예제는 레포지토리의 `docs/클라이언트-FASTAPI연동.md`에 정리된 규약을 바탕으로 작성했습니다. 필요하면 샘플 React 컴포넌트 형태로 더 상세한 헬퍼(오토 재연결, 프레임 큐, 전송 스케줄러 등)를 추가해 드리겠습니다.

---

프로젝트에 맞춘 React 예제 컴포넌트

아래 예제는 이 저장소(`main.py`, `inference_pipeline.py`, `configs/settings.py`)의 구현을 기준으로 동작하도록 작성되었습니다.
- 쿠키 이름(기본): `ACCESS_TOKEN` (`configs/settings.py`의 `COOKIE_ACCESS_TOKEN_NAME` 값을 사용)
- WebSocket 엔드포인트: `wss://<host>/ws/{session_id}`
- 처리 가능한 WS 메시지 타입: `meta`, `save_learning`, `flush` (서버는 `meta_ack`, `learning_ack`, `inference_result` 등으로 응답합니다)

이 컴포넌트는 두 가지 전송 모드(우선순위 순서)를 제공합니다:
- 모드 A (권장, 서버가 aiortc로 DataChannel 시그널링을 제공할 때): WebRTC DataChannel로 바이너리 프레임 전송
- 모드 B (서버가 DataChannel 시그널링을 제공하지 않는 경우): WebSocket 제어 채널로 base64-encoded frame 전송 (서버에 `frame` 메시지 핸들러가 필요함 — 아래에 서버 스니펫 포함)

```jsx
// React 컴포넌트: SignSenseClient.jsx
import React, { useEffect, useRef, useState } from 'react';

// 설정 (필요시 변경)
const COOKIE_NAME = 'ACCESS_TOKEN'; // configs/settings.py 기본값: ACCESS_TOKEN

function setCookie(name, value, days = 1) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${value}; expires=${expires}; path=/; Secure; SameSite=Strict`;
}

function SignSenseClient({ sessionId }) {
  const wsRef = useRef(null);
  const pcRef = useRef(null);
  const dcRef = useRef(null);
  const videoRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [inferenceResult, setInferenceResult] = useState(null);

  useEffect(() => {
    return () => {
      // cleanup on unmount
      if (wsRef.current) wsRef.current.close();
      if (pcRef.current) pcRef.current.close();
    };
  }, []);

  // WS 제어 채널 열기 (쿠키 기반 인증 — 브라우저가 자동으로 쿠키를 보냄)
  function openControlWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/${encodeURIComponent(sessionId)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Control WS open');
      setConnected(true);
      // meta 전송: 서버는 meta_ack로 응답합니다.
      ws.send(JSON.stringify({ type: 'meta', word_pk: 42, word_name: '사랑' }));
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        console.log('WS msg', msg);
        if (msg.type === 'meta_ack') {
          console.log('meta acknowledged by server');
        } else if (msg.type === 'learning_ack') {
          console.log('learning accepted', msg);
        } else if (msg.type === 'inference_result') {
          setInferenceResult(msg.result);
        }
      } catch (e) {
        console.warn('non-json ws message', ev.data);
      }
    };

    ws.onclose = () => {
      console.log('Control WS closed');
      setConnected(false);
    };

    ws.onerror = (e) => console.error('WS error', e);
  }

  // 모드 A: WebRTC DataChannel 사용 (서버가 SDP 시그널링을 처리하는 경우에만 동작)
  async function startWebRTCWithSignaling() {
    if (!wsRef.current) {
      console.warn('Control WS not open — opening now');
      openControlWebSocket();
      // 짧은 대기 후 재시도
      await new Promise((r) => setTimeout(r, 200));
    }

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });
    pcRef.current = pc;

    const dc = pc.createDataChannel('frames');
    dc.binaryType = 'arraybuffer';
    dc.onopen = () => console.log('DataChannel open');
    dc.onmessage = (ev) => console.log('DC message', ev.data);
    dcRef.current = dc;

    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
      }
    };

    // 로컬 카메라 트랙(미리보기)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (e) {
      console.error('getUserMedia 실패', e);
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // 서버가 시그널링 경로를 제공하면 offer를 전송합니다.
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'offer', sdp: offer.sdp, sdpType: offer.type }));
    } else {
      console.warn('Control WS not ready — cannot send offer');
    }

    // Control WS에서 answer/candidate를 받도록 핸들러 추가
    const onWsMessage = async (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'answer') {
          await pc.setRemoteDescription({ type: 'answer', sdp: msg.sdp });
          console.log('Remote description set (answer)');
        } else if (msg.type === 'candidate') {
          try {
            await pc.addIceCandidate(msg.candidate);
          } catch (err) {
            console.warn('addIceCandidate failed', err);
          }
        }
      } catch (_) {
        // ignore non-json messages
      }
    };

    wsRef.current.addEventListener('message', onWsMessage);

    return { pc, dc };
  }

  // DataChannel로 캔버스 프레임 전송
  async function sendFrameViaDataChannel(videoEl) {
    const dc = dcRef.current;
    if (!dc || dc.readyState !== 'open') return;

    const cvs = document.createElement('canvas');
    cvs.width = videoEl.videoWidth || 640;
    cvs.height = videoEl.videoHeight || 480;
    const ctx = cvs.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, cvs.width, cvs.height);

    const blob = await new Promise((res) => cvs.toBlob(res, 'image/jpeg', 0.7));
    const ab = await blob.arrayBuffer();
    dc.send(ab);
  }

  // 모드 B: 제어 WebSocket으로 base64-encoded 프레임 전송 (서버에 'frame' 메시지 핸들러 필요)
  async function sendFrameViaControlWS(videoEl) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const cvs = document.createElement('canvas');
    cvs.width = videoEl.videoWidth || 640;
    cvs.height = videoEl.videoHeight || 480;
    const ctx = cvs.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, cvs.width, cvs.height);

    const blob = await new Promise((res) => cvs.toBlob(res, 'image/jpeg', 0.7));
    const b64 = await new Promise((res) => {
      const reader = new FileReader();
      reader.onload = () => res(reader.result.split(',')[1]);
      reader.readAsDataURL(blob);
    });

    // 'frame' 메시지 예시: { type: 'frame', data: '<base64 jpeg>' }
    ws.send(JSON.stringify({ type: 'frame', data: b64 }));
  }

  // 서버에 flush 요청 보내기
  function requestInference() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'flush' }));
  }

  // 서버에 save_learning 요청 보내기
  function requestSaveLearning() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'save_learning' }));
  }

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline muted style={{ width: 320, height: 240 }} />
      <div>
        <button onClick={() => { /* 예시: 로그인 후 토큰을 쿠키로 설정 */ setCookie(COOKIE_NAME, 'REPLACE_WITH_TOKEN'); }}>Set Token Cookie (dev)</button>
        <button onClick={() => openControlWebSocket()}>Open Control WS</button>
        <button onClick={() => startWebRTCWithSignaling()}>Start WebRTC (if supported)</button>
        <button onClick={() => { if (videoRef.current) sendFrameViaControlWS(videoRef.current); }}>Send Frame via WS</button>
        <button onClick={() => { if (videoRef.current) sendFrameViaDataChannel(videoRef.current); }}>Send Frame via DataChannel</button>
        <button onClick={() => requestInference()}>Flush → Inference</button>
        <button onClick={() => requestSaveLearning()}>Save Learning</button>
      </div>
      <pre>{inferenceResult ? JSON.stringify(inferenceResult, null, 2) : 'No result'}</pre>
    </div>
  );
}

export default SignSenseClient;
```

# main.py websocket 루프 내에서 (예시 - 서버측 변경 안내)
import base64

# 아래는 예시 핸들러 스니펫입니다. 실제로는 websocket receive 루프 내부의 msg 파싱 분기에서
# 아래 흐름과 유사하게 처리하시면 됩니다.
# (예시 - 마크다운 내에서는 일반 텍스트로 표기하여 정적 검사 오류를 방지합니다)

# 예시 분기:
# if mtype == 'frame':
#     b64 = msg.get('data')
#     if b64:
#         try:
#             frame_bytes = base64.b64decode(b64)
#             collector.add_frame(frame_bytes)
#         except Exception:
#             # 무시 또는 로깅
#             pass
#     # 선택적으로 ack
#     await websocket.send_text(json.dumps({'type': 'frame_ack'}))
