# Tracking Scripts

실험 추적 및 설정 유틸리티

## 스크립트 목록

### setup_experiment_tracking.py
**주요 목적**: 실험 추적 시스템 설정

**기능**:
- MLflow 설정 및 초기화
- WandB 설정 (선택적)
- 추적 디렉토리 생성 (`experiments/`)
- 로깅 설정

**사용법**:
```bash
python scripts/tracking/setup_experiment_tracking.py
```

**출력**:
- `experiments/mlruns/` - MLflow 추적 데이터
- `experiments/wandb/` - WandB 추적 데이터 (선택적)
- 설정 확인 메시지

### view_experiments.py
**주요 목적**: 실험 결과 조회 및 비교

**기능**:
- 모든 실험 리스트 표시
- 성능 지표 비교
- 최고 성능 실험 식별
- 시각화 및 리포트 생성

**사용법**:
```bash
python scripts/tracking/view_experiments.py
```

**출력**:
- 실험 비교 표
- 성능 시각화
- 최적 하이퍼파라미터

### test_libraries.py
**주요 목적**: 라이브러리 및 의존성 테스트

**기능**:
- 필수 라이브러리 설치 확인
- 버전 호환성 검증
- 간단한 동작 테스트

**사용법**:
```bash
python scripts/tracking/test_libraries.py
```

**출력**:
- 라이브러리 버전 리스트
- 테스트 결과 (통과/실패)

## 실험 추적 워크플로우

1. **설정**: `setup_experiment_tracking.py`로 추적 시스템 초기화
2. **실험**: 모델링 스크립트 실행 (자동으로 추적)
3. **조회**: `view_experiments.py`로 결과 비교

## MLflow 사용

```python
import mlflow

# 실험 시작
with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

## 의존성

- MLflow
- WandB (선택적)
- Pandas (결과 조회)

## 헌법 준수

이 스크립트들은 **헌법 원칙 V: 반복적 분석과 성찰**을 따릅니다:
- 모든 실험 추적 및 비교
- 성능 개선 모니터링
- 배운 교훈 문서화