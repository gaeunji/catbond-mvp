# Catastrophe Bond Risk Modeling MVP

## 📋 프로젝트 개요

이 프로젝트는 강수량 시계열 데이터를 기반으로 재해 발생 가능성과 손실 규모를 예측하는 통합 모델을 개발합니다. 예측 결과를 바탕으로 CAT Bond의 기대 손실과 수익률을 계산해 가격을 결정하는 시스템을 구현합니다.

## 🎯 주요 기능

### 1. **Two-Part 모델 구조**

- **Part 1**: Trigger 발생 확률 모델 (LightGBM 분류)
- **Part 2**: 조건부 손실 규모 모델 (LightGBM 회귀)

### 2. **지역별 위험 모델링**

- 159개 지역의 강수 임계값 기반 트리거 모델
- 충주 지역 특화 분석 및 가중치 적용

### 3. **Expected Loss 계산**

- 연간/3년 만기 Expected Loss 계산

## 🏗️ 프로젝트 구조

```
catbond-mvp/
├── src/
│   ├── pipeline.py              # 메인 파이프라인 (수정됨)
│   ├── data.py                  # 데이터 로더 (수정됨)
│   ├── model.py                 # 모델 클래스 (수정됨)
│   ├── features.py              # 피처 엔지니어링
│   └── utils.py                 # 유틸리티 함수
├── data/
│   └── processed/               # 처리된 데이터
├── models/                      # 학습된 모델
├── config.yaml                  # 설정 파일
├── train.py                     # 실행 스크립트
└── requirements.txt             # 의존성 패키지
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv catbond-env
source catbond-env/bin/activate  # Windows: catbond-env\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
# 새로운 파이프라인 실행 (권장)
python train.py --config config.yaml

# 또는 직접 파이프라인 실행
python -c "
import sys
import os
sys.path.append('src')
from src.pipeline import train_catbond_pipeline
results = train_catbond_pipeline('config.yaml')
print('파이프라인 실행 완료!')
"
```

### 3. 결과 확인

- 트리거 모델: `trigger_probability_model.pkl`
- 손실 모델: `conditional_severity_model.txt`
- 지역 정보: `region_info_mtl.pkl`

## 📊 주요 결과 (최신)

### 충주 지역 Expected Loss (3년)

- **충주 지역**: 충청북도 충주시
- **임계값**: 227.0mm
- **연간 기대 사건 수 (λ)**: 0.030건
- **사건당 손실률 (L)**: 9.88%
- **액면가 (F)**: 1,000억원
- **기대 손실률**: 0.89%
- **기대손실 (EL)**: 8.9억원
- **3년간 최소 1회 발생 확률**: 8.61%
- **추정 보험료**: 148만원

### 모델 성능

- **트리거 모델**: LightGBM 분류 (ROC-AUC: 0.996)
- **손실 모델**: LightGBM 회귀 (MAE 최적화)

## 📋 필수 데이터 파일

```
data/processed/
├── merged_flood_data.csv           # 메인 손실 데이터 (5.2MB)
├── rain_nation_daily_glm.parquet   # 강수 데이터 (1.5MB)
├── features.parquet                # 피처 데이터 (2.1MB)
├── climate_indices.parquet         # 기후지수 데이터 (0.8MB)
└── land_use_merged.parquet         # 토지이용 데이터 (1.2MB)
```

## ⚙️ 설정 파일 (config.yaml)

```yaml
model:
  face_value: 100000000000 # 1,000억원 (액면가)
  quantile_level: 0.999918 # 99.9918% 분위수
  random_state: 42

expected_loss:
  face_value: 100000000000 # 1,000억원
  years: 3 # 3년 만기
  loading_factor: 0.05 # 5% 수준
```

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT
