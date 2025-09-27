# Analysis Scripts

탐색적 데이터 분석 (EDA) 및 데이터 탐색 스크립트

## 스크립트 목록

### eda.py
**주요 목적**: 탐색적 데이터 분석 (EDA)

**기능**:
- 데이터 구조 및 분포 분석
- 결측치 및 이상치 탐지
- 기본 통계량 계산
- 시각화 생성 (class_distribution, target_distribution 등)

**사용법**:
```bash
python scripts/analysis/eda.py
```

### advanced_analysis.py
**주요 목적**: 고급 데이터 분석

**기능**:
- 상관관계 분석 (correlation_heatmap)
- 차원 축소 분석 (dimensionality_analysis)
- 피처 중요도 분석 (feature_importance)
- 통계적 가설 검정

**사용법**:
```bash
python scripts/analysis/advanced_analysis.py
```

## 출력 위치

모든 시각화 출력은 `outputs/figures/` 디렉토리에 저장됩니다.

## 의존성

분석 스크립트는 다음 모듈을 사용할 수 있습니다:
- `src/analysis/` - 분석 유틸리티
- `src/utils/` - 일반 유틸리티

## 헌법 준수

이 스크립트들은 **헌법 원칙 II: 교육적 명확성**을 따릅니다:
- 코드 주석으로 분석 로직 설명
- 결과 해석 포함
- 10세 어린이도 이해할 수 있는 설명