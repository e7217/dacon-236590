# Preprocessing Scripts

데이터 전처리 및 변환 스크립트

## 스크립트 목록

### preprocessing.py
**주요 목적**: 데이터 전처리 파이프라인

**기능**:
- 결측치 처리
- 이상치 제거 또는 변환
- 스케일링 및 정규화
- 피처 엔지니어링
- 데이터 인코딩

**전처리 전략**:
- **Basic**: 기본 전처리 (preprocessor_basic.pkl)
- **Aggressive**: 적극적 전처리 (preprocessor_aggressive.pkl)
- **Feature Selected**: 피처 선택 포함 (preprocessor_feature_selected.pkl)

**출력**:
- 전처리된 데이터: `models/processed_data_*.pkl`
- 전처리기 객체: `models/preprocessor_*.pkl`

**사용법**:
```bash
python scripts/preprocessing/preprocessing.py
```

## 전처리 워크플로우

1. **데이터 로드**: `data/open/` 또는 `data/raw/`에서 원본 데이터 읽기
2. **전처리 적용**: 결측치 처리, 스케일링, 인코딩
3. **저장**: 전처리된 데이터와 전처리기를 `models/`에 저장
4. **검증**: 전처리 결과 검증 및 통계 확인

## 의존성

- `src/utils/` - 전처리 유틸리티
- `data/open/` 또는 `data/raw/` - 원본 데이터

## 헌법 준수

이 스크립트는 **헌법 원칙 III: 성능 우선 최적화**를 따릅니다:
- 모델 성능 향상에 초점
- 여러 전처리 전략 실험
- 교차 검증으로 성능 검증