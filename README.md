# 2048 A3C 강화학습 에이전트

A3C (Asynchronous Advantage Actor-Critic) 알고리즘을 사용하여 2048 게임을 학습하는 강화학습 프로젝트입니다.

## ⚠️ 중요: 프레임워크 변경 사항

**이 프로젝트는 원래 Keras/TensorFlow로 작성되었으나, 현재는 PyTorch로 완전히 변환되었습니다.**

- ✅ **PyTorch 기반 구현**: 모든 신경망이 PyTorch로 재작성됨
- ✅ **GPU 지원**: CUDA가 사용 가능한 경우 자동으로 GPU 사용
- ✅ **테스트 완료**: 코드가 정상 작동함을 확인함

## 프로젝트 개요

이 프로젝트는 A3C 강화학습 알고리즘을 구현하여 2048 게임을 자동으로 플레이하도록 학습시킵니다. Actor-Critic 구조를 사용하여 정책 네트워크(Actor)와 가치 네트워크(Critic)를 동시에 학습합니다.

## 주요 기능

- **A3C 알고리즘**: Actor-Critic 구조를 사용한 강화학습 구현
- **2048 게임 환경**: 커스텀 2048 게임 보드 구현
- **자동 학습**: 에이전트가 게임을 반복하며 자동으로 학습
- **모델 저장/로드**: 학습된 모델을 저장하고 불러오기
- **학습 결과 시각화**: 
  - Matplotlib을 사용한 고해상도 이미지 그래프 저장
  - Plotly를 사용한 인터랙티브 HTML 그래프
  - CSV 파일로 모든 에피소드 데이터 기록
- **GPU 가속**: CUDA 지원으로 빠른 학습

## 설치 방법

### 필수 요구사항

- Python 3.6 이상
- PyTorch 1.9 이상 (CUDA 지원 선택사항)

### 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- `numpy>=1.19.0`
- `torch>=1.9.0`
- `plotly>=4.0.0`
- `matplotlib>=3.3.0`

### PyTorch 설치 (CUDA 지원)

CUDA가 설치된 시스템의 경우:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU만 사용하는 경우:
```bash
pip install torch torchvision torchaudio
```

## 사용 방법

### 학습 시작

**전체 학습 (권장):**
```bash
python test_full_training.py
```

또는 Windows에서:
```bash
py test_full_training.py
```

**기본 학습 스크립트:**
```bash
python main.py
```

또는 Windows에서:
```bash
py main.py
```

학습이 시작되면:
- 에이전트가 2048 게임을 반복적으로 플레이합니다
- 매 1000 스텝마다 모델이 자동으로 저장됩니다 (`save_model/` 디렉토리)
- 게임이 종료될 때마다 점수와 최대 타일 번호가 출력됩니다
- 매 100 에피소드마다 학습 결과 그래프가 `results/` 디렉토리에 저장됩니다
- 모든 에피소드 데이터가 CSV 파일로 기록됩니다

### 모델 로드

이미 학습된 모델이 `save_model/py2048_a3c_actor.pth`와 `save_model/py2048_a3c_critic.pth`에 있으면 자동으로 로드됩니다.

## 프로젝트 구조

```
2048-A3C-agent/
├── main.py                  # 메인 학습 스크립트
├── test_full_training.py    # 전체 학습 테스트 스크립트 (권장)
├── ddqn.py                  # A3C 에이전트 구현 (PyTorch 기반 Actor-Critic 네트워크)
├── py2048.py                # 2048 게임 보드 구현
├── requirements.txt         # Python 패키지 의존성
├── README.md               # 프로젝트 문서
├── save_model/              # 학습된 모델 저장 디렉토리
│   ├── py2048_a3c_actor.pth
│   └── py2048_a3c_critic.pth
└── results/                 # 학습 결과 저장 디렉토리
    ├── training_results.csv # 모든 에피소드 데이터 (CSV)
    ├── learning_curve.png   # 학습 곡선 이미지 (고해상도)
    └── learning.html        # 인터랙티브 Plotly 그래프
```

## 코드 설명

### main.py
- 게임 보드 초기화 및 학습 루프 관리
- **개선된 상태 표현**: 기본 보드 상태 + 빈 칸 비율 + 최대 타일 위치 + monotonicity 점수
- **전략적 보상 함수**: Monotonicity, Smoothness, 전략적 배치 고려
- 보상 계산 및 에피소드 종료 처리
- 학습 통계 수집 및 출력

### ddqn.py
- **ActorCritic 클래스**: 개선된 PyTorch nn.Module 기반 Actor-Critic 네트워크
  - 더 깊고 넓은 네트워크 구조 (128→512→512→512)
  - Dropout을 통한 정규화 (0.1)
  - 공통 feature extractor와 분리된 Actor/Critic 헤드
- **DQNAgent 클래스**: A3C 에이전트 구현
  - `train_episode()`: 에피소드별 학습 수행 (Policy Gradient + Value Function)
  - `get_action()`: 정책에 따라 행동 선택
  - `discount_rewards()`: 보상 할인 계산
  - `save_model()` / `load_model()`: PyTorch 모델 저장/로드
  - 학습률 스케줄링: StepLR로 점진적 학습률 감소

### py2048.py
- **GameBoard 클래스**: 2048 게임 보드 구현
  - `performAction()`: 방향키 입력 처리 (0: 왼쪽, 1: 위, 2: 오른쪽, 3: 아래)
  - `newTile()`: 새로운 타일 생성 (90% 확률로 2, 10% 확률로 4)
  - `reset()`: 게임 보드 초기화

## 학습 파라미터

주요 하이퍼파라미터는 `main.py`와 `ddqn.py`에서 설정할 수 있습니다:

- `discount_factor`: 0.99 (할인 계수, 장기 보상 고려)
- `actor_lr`: 0.0005 (초기 학습률, 스케줄링으로 점진적 감소)
- `t_max`: 100 (에피소드당 최대 스텝 수, 장기 전략 학습)
- `state_size`: 21 (16 기본 + 5 추가 특징: 빈 칸 비율, 최대 타일 위치, monotonicity)
- `action_size`: 4 (상하좌우 4가지 행동)
- `entropy_coef`: 0.1 (엔트로피 정규화 계수, 탐험 증가)

## 보상 함수

**대폭 개선된 전략적 보상 함수** (2048 달성 및 4096 이상 목표):

1. **빈 칸 증가**: 타일 병합 시 큰 보상 (×1.0)
2. **점수 증가**: 타일 값 합의 증가에 대한 보상 (×0.0001)
3. **최대 타일 증가**: 지수적 보상 (1024 이상에서 큰 보상)
4. **Monotonicity (단조성)**: 큰 타일이 모서리/가장자리에 있을수록 보상
5. **Smoothness (부드러움)**: 인접 타일의 차이가 작을수록 보상
6. **게임 종료 페널티**: -50.0
7. **위험 상태 페널티**: 빈 칸이 2개 이하일 때 -1.0

이 보상 함수는 2048 게임의 핵심 전략(큰 타일을 모서리에 배치, 타일 병합 최적화)을 학습하도록 설계되었습니다.

## 학습 결과

학습 중 다음 정보가 출력됩니다:
- **score**: 평균 점수 (최근 1500 게임)
- **expScore**: 지수 변환된 평균 점수
- **MaxNumber**: 달성한 최대 타일 번호 (2의 거듭제곱)
- **board**: 게임 종료 시 보드 상태

### 결과 저장

학습 결과는 `results/` 디렉토리에 자동으로 저장됩니다:

1. **CSV 파일** (`training_results.csv`): 모든 에피소드의 상세 데이터
   - Episode, Score, MaxNumber, MaxProb, Timestamp

2. **이미지 파일** (`learning_curve.png`): 학습 곡선 시각화 (고해상도)
   - Score 그래프
   - MaxNumber 그래프
   - MaxProbability 그래프
   - Score 이동 평균 그래프

3. **HTML 파일** (`learning.html`): 인터랙티브 Plotly 그래프
   - 브라우저에서 열어서 확대/축소 가능

매 100 에피소드마다 자동으로 그래프가 업데이트됩니다.

### 결과 확인 방법

1. **이미지 그래프 확인**: `results/learning_curve.png` 파일을 열어서 학습 진행 상황을 시각적으로 확인
2. **CSV 데이터 분석**: `results/training_results.csv`를 Excel이나 Python pandas로 열어서 상세 데이터 분석
3. **인터랙티브 그래프**: `results/learning.html`을 브라우저에서 열어서 확대/축소하며 확인

### 결과 파일 예시

```
results/
├── training_results.csv      # Episode별 상세 데이터
├── learning_curve.png        # 4개 서브플롯 (Score, MaxNumber, MaxProb, Moving Avg)
└── learning.html             # Plotly 인터랙티브 그래프
```

## 테스트 결과

코드는 PyTorch로 변환 후 다음 테스트를 통과했습니다:

### 테스트 환경
- **Python**: 3.10.11
- **PyTorch**: 최신 버전
- **Device**: CUDA (GPU 가속 사용)

### 테스트 결과
✅ **Import 및 초기화**: 성공  
✅ **행동 선택**: 정상 작동  
✅ **학습 루프**: 정상 작동  
✅ **모델 저장/로드**: 정상 작동  
✅ **에피소드 학습**: 정상 작동  
✅ **그래프 이미지 저장**: 정상 작동  
✅ **CSV 데이터 기록**: 정상 작동  

**샘플 실행 결과:**
```
Using device: cuda
Starting training test...
Game 1 - Score: 35.0, MaxNumber: 32.0
Game 2 - Score: 39.0, MaxNumber: 64.0
Game 3 - Score: 50.0, MaxNumber: 64.0
...
Training test completed successfully!
```

**생성된 결과 파일:**
- `results/learning_curve.png`: 학습 곡선 이미지 (고해상도)
- `results/training_results.csv`: 모든 에피소드 데이터
- `results/learning.html`: 인터랙티브 Plotly 그래프

## 문제 해결

### 모델 로드 실패
모델 파일이 없거나 경로가 잘못된 경우 "FAILED TO LOAD NETWORK" 메시지가 출력되지만, 학습은 계속 진행됩니다. 첫 실행 시에는 새로운 모델이 생성됩니다.

### GPU 사용 확인
코드는 자동으로 CUDA 사용 가능 여부를 확인합니다. GPU를 사용하려면 PyTorch가 CUDA를 지원하는 버전으로 설치되어 있어야 합니다.

### 메모리 부족
GPU 메모리가 부족한 경우, `ddqn.py`의 `device` 설정을 `"cpu"`로 변경할 수 있습니다.

## Keras에서 PyTorch로의 변환 사항

### 주요 변경점
1. **신경망 구조**: Keras Model → PyTorch nn.Module
2. **최적화**: Keras Optimizer → PyTorch optim.Adam
3. **손실 함수**: Keras backend → PyTorch nn 모듈
4. **모델 저장**: `.h5` → `.pth` (PyTorch state_dict)
5. **텐서 연산**: NumPy 배열을 PyTorch Tensor로 변환

### 성능 개선
- GPU 가속 지원으로 학습 속도 향상
- PyTorch의 동적 계산 그래프 활용
- 더 유연한 모델 구조 수정 가능

## 참고 자료

- 2048 게임 참고: https://github.com/mckeown12/2048-DQN-in-python
- A3C 알고리즘 참고: https://github.com/wuyx/rlcode
- PyTorch 공식 문서: https://pytorch.org/docs/

## 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.
