# Modeling Scripts

모델 학습, 앙상블, 최종 솔루션 스크립트

## 스크립트 목록

### baseline_models.py
**주요 목적**: 베이스라인 모델 학습

**모델 유형**:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- CatBoost

**출력**:
- 베이스라인 성능 지표
- 교차 검증 결과
- 모델 파일 (선택적)

**사용법**:
```bash
python scripts/modeling/baseline_models.py
```

### advanced_ensemble.py
**주요 목적**: 고급 앙상블 모델

**앙상블 기법**:
- Voting Classifier
- Stacking
- Blending
- Weighted Average

**출력**:
- 앙상블 성능 지표
- 최적 가중치
- 앙상블 모델

**사용법**:
```bash
python scripts/modeling/advanced_ensemble.py
```

### final_solution.py
**주요 목적**: 최종 솔루션 및 제출 파일 생성

**기능**:
- 최종 모델 학습
- 테스트 데이터 예측
- 제출 파일 생성 (`outputs/submissions/`)

**출력**:
- `outputs/submissions/final_submission.csv`
- 최종 모델 성능 리포트

**사용법**:
```bash
python scripts/modeling/final_solution.py
```

## 모델링 워크플로우

1. **베이스라인**: `baseline_models.py`로 기본 성능 확인
2. **앙상블**: `advanced_ensemble.py`로 모델 결합
3. **최종화**: `final_solution.py`로 제출 파일 생성

## 성능 노트

- 모든 모델은 5-fold 계층화 교차 검증 사용
- 성능 지표: Accuracy, F1-Score, ROC-AUC
- 하이퍼파라미터는 베이지안 최적화 또는 그리드 서치 사용

## 의존성

- 전처리된 데이터: `models/processed_data_*.pkl`
- `src/utils/` - 모델링 유틸리티

## 헌법 준수

이 스크립트들은 **헌법 원칙 III & IV: 성능 우선, 연구 주도**를 따릅니다:
- 모델 정확도가 주요 성공 기준
- 최신 알고리즘 및 기법 활용
- 체계적인 하이퍼파라미터 튜닝