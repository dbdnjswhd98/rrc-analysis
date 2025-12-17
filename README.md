# RRC State Prediction

RRC (Radio Resource Control) 상태 예측을 위한 머신러닝 프로젝트입니다. Markov Chain, LSTM, 1D-CNN, Random Forest 모델을 사용하여 모바일 네트워크의 RRC 상태 전환을 예측합니다.

## 📋 프로젝트 개요

이 프로젝트는 모바일 네트워크에서 **RRC 상태**(Idle/Connected)를 예측하기 위해 다양한 머신러닝 및 딥러닝 모델을 비교 평가합니다.

### 주요 특징
- **4가지 모델**: Markov Chain (baseline), LSTM, 1D-CNN, Random Forest
- **2가지 입력 방식**: 
  - Traffic-only (트래픽 데이터만)
  - RRC+Traffic (RRC 상태 + 트래픽 데이터)
- **전환 이벤트 분석**: 상태 전환이 발생하는 시점에 대한 별도 평가
- **체계적인 파이프라인**: 전처리 → 데이터셋 생성 → 학습 → 평가

## 📁 프로젝트 구조

```
rrc_analysis_project/
├── src/                                    # 소스 코드
│   ├── 1_preprocess_all.py                # 데이터 전처리
│   ├── 2_build_datasets.py                # RRC+Traffic 데이터셋 생성
│   ├── 2_build_datasets_traffic.py        # Traffic-only 데이터셋 생성
│   ├── lstm.py                            # LSTM 모델 (RRC+Traffic)
│   ├── lstm_traffic.py                    # LSTM 모델 (Traffic-only)
│   ├── cnn1d.py                           # 1D-CNN 모델 (RRC+Traffic)
│   ├── cnn1d_traffic.py                   # 1D-CNN 모델 (Traffic-only)
│   ├── rf.py                              # Random Forest (RRC+Traffic)
│   ├── rf_traffic.py                      # Random Forest (Traffic-only)
│   └── eval_all_models.py                 # 전체 모델 평가 (CLI)
│
├── notebooks/                              # Jupyter 노트북
│   └── 4_eval_all_models.ipynb            # 모델 평가 및 시각화
│
├── data/                                   # 데이터 (gitignore)
│   ├── raw/                               # 원본 데이터
│   │   ├── enb_s1ap/                      # S1AP 데이터
│   │   └── ue_pcap/                       # UE PCAP 데이터
│   └── processed/                         # 전처리된 데이터
│       ├── processed_testbed/             # 전처리된 CSV
│       ├── seq_dataset.npz                # RRC+Traffic 시퀀스 데이터
│       └── traffic_only_seq_dataset.npz   # Traffic-only 시퀀스 데이터
│
├── artifacts/                              # 학습 결과물
│   ├── models/                            # 학습된 모델 (gitignore)
│   │   ├── lstm_best.keras
│   │   ├── lstm_traffic_best.keras
│   │   ├── cnn1d_best.keras
│   │   ├── cnn1d_traffic_best.keras
│   │   ├── rf_model.joblib
│   │   └── rf_traffic_model.joblib
│   └── results/                           # 평가 결과 (gitignore)
│       ├── models_comparison.png
│       ├── models_comparison_summary.csv
│       ├── eval_transition_only_summary.csv
│       └── cm_transition_*.png
│
├── .gitignore                              # Git 제외 파일 목록
└── README.md                               # 프로젝트 설명서
```

## 🚀 시작하기

### 사전 요구사항

```bash
Python 3.8+
TensorFlow 2.x / Keras 3.x
scikit-learn
pandas
numpy
matplotlib
joblib
```

### 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/rrc_analysis_project.git
cd rrc_analysis_project

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

## 📊 사용 방법

### 1. 데이터 전처리

```bash
python src/1_preprocess_all.py
```

S1AP 데이터와 UE PCAP 데이터를 병합하여 1초 단위 시계열 데이터로 전처리합니다.

### 2. 데이터셋 생성

```bash
# RRC+Traffic 데이터셋
python src/2_build_datasets.py

# Traffic-only 데이터셋
python src/2_build_datasets_traffic.py
```

60초 슬라이딩 윈도우로 시퀀스 데이터셋을 생성합니다.

### 3. 모델 학습

```bash
# LSTM 모델
python src/lstm.py
python src/lstm_traffic.py

# 1D-CNN 모델
python src/cnn1d.py
python src/cnn1d_traffic.py

# Random Forest 모델
python src/rf.py
python src/rf_traffic.py
```

### 4. 모델 평가

#### 4-1. CLI 평가
```bash
python src/eval_all_models.py
```

#### 4-2. Jupyter Notebook (권장)
```bash
jupyter notebook notebooks/4_eval_all_models.ipynb
```

Jupyter 노트북에서는 다음 기능을 제공합니다:
- 📊 대화형 시각화
- 🏆 Top-3 모델 성능 순위
- 📈 전체 모델 비교 그래프
- 🔀 Transition-only 평가
- 📝 상세 에러 로그

## 📈 평가 지표

- **Accuracy**: 전체 예측 정확도
- **Macro F1-score**: 클래스 불균형을 고려한 F1 점수
- **Confusion Matrix**: 예측 분포 시각화
- **Transition-only Metrics**: 상태 전환 시점만의 성능

## 🎯 주요 결과

프로젝트 실행 후 `artifacts/results/` 폴더에서 다음 결과를 확인할 수 있습니다:

- `models_comparison.png`: 전체 모델 성능 비교 그래프
- `models_comparison_summary.csv`: 성능 요약 테이블
- `models_comparison_transition_only.png`: 전환 이벤트 성능 비교
- `cm_transition_*.png`: 각 모델의 Confusion Matrix

## 🔧 설정 커스터마이징

`notebooks/4_eval_all_models.ipynb`의 **설정 상수 섹션**에서 다음 파라미터를 조정할 수 있습니다:

```python
TRAIN_TIME_SPLIT = 3600  # Train/Val 분할 시간 (초)
HEATMAP_COLORMAP = 'YlOrRd'  # 히트맵 컬러맵
FIGURE_DPI = 300  # 그래프 저장 해상도
```

## 📝 모델 설명

### Markov Chain (Baseline)
- RRC 상태 전이 확률만 사용
- Laplace smoothing 적용

### LSTM (Long Short-Term Memory)
- 시계열 패턴 학습
- 60초 윈도우 입력

### 1D-CNN (1D Convolutional Neural Network)
- 로컬 패턴 추출
- Max pooling으로 중요 특징 선택

### Random Forest
- 앙상블 학습
- 시퀀스를 flatten하여 입력

## 🤝 기여

이슈와 PR을 환영합니다!

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 저자

- 경희대학교 프로젝트

## 🙏 감사의 말

이 프로젝트는 모바일 네트워크 최적화 연구의 일환으로 진행되었습니다.
