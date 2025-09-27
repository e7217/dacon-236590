# Submission Scripts

경쟁 제출 파일 생성 스크립트

## 스크립트 목록

### simple_submission.py
**주요 목적**: 간단한 베이스라인 제출 파일 생성

**기능**:
- 단순 모델 학습 (예: Logistic Regression)
- 빠른 제출 파일 생성
- 베이스라인 성능 확인

**출력**:
- `outputs/submissions/submission_baseline.csv`

**사용법**:
```bash
python scripts/submissions/simple_submission.py
```

### quick_baseline.py
**주요 목적**: 빠른 베이스라인 제출

**기능**:
- 최소한의 전처리
- 기본 모델 (Random Forest 또는 LightGBM)
- 신속한 결과 확인

**출력**:
- `outputs/submissions/quick_baseline.csv`

**사용법**:
```bash
python scripts/submissions/quick_baseline.py
```

## CSV 형식 요구사항

제출 파일은 다음 형식을 따라야 합니다:
```csv
ID,target
0,predicted_class
1,predicted_class
...
```

- **ID**: 테스트 데이터 샘플 ID
- **target**: 예측된 클래스 레이블

## 제출 워크플로우

1. **빠른 테스트**: `quick_baseline.py`로 즉시 제출 가능한 파일 생성
2. **베이스라인**: `simple_submission.py`로 간단한 모델 제출
3. **최종 제출**: `scripts/modeling/final_solution.py`로 최고 성능 모델 제출

## 버전 관리

제출 파일은 버전 또는 타임스탬프를 포함해야 합니다:
```
outputs/submissions/
├── submission_baseline.csv      # 베이스라인
├── quick_baseline.csv           # 빠른 제출
├── final_submission.csv         # 최종 제출
├── ensemble_v2_20250927.csv     # 버전 관리 예시
```

## 의존성

- 전처리된 데이터: `models/processed_data_*.pkl`
- 테스트 데이터: `data/open/` 또는 `data/raw/`

## 헌법 준수

이 스크립트들은 **헌법 원칙 III: 성능 우선 최적화**를 따릅니다:
- 제출 성능 최대화
- 여러 전략 실험
- 리더보드 피드백 반영