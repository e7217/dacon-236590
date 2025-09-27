# Directory Structure Documentation

**프로젝트**: Dacon 스마트 제조 장비 이상 감지 AI 경진대회
**재구성 일자**: 2025-09-27
**버전**: 1.0

## 개요

이 문서는 프로젝트 파일 재구성의 전체 구조와 근거를 설명합니다. 재구성은 헌법 원칙을 따르며, 데이터 분석 워크플로우 (EDA → 전처리 → 모델링 → 평가)에 맞춰 파일을 논리적으로 구성합니다.

## 완전한 디렉토리 트리

```
dacon-236590/
├── notebooks/                              # Jupyter 노트북 (단계별 분석)
│   ├── 01_main_analysis.ipynb             # 주요 분석 노트북
│   └── README.md                          # 실행 순서 및 네이밍 컨벤션
│
├── scripts/                                # 실행 가능한 Python 스크립트
│   ├── analysis/                          # 탐색적 데이터 분석
│   │   ├── eda.py                         # 기본 EDA
│   │   ├── advanced_analysis.py           # 고급 분석
│   │   └── README.md                      # 분석 스크립트 설명
│   │
│   ├── preprocessing/                     # 데이터 전처리
│   │   ├── preprocessing.py               # 전처리 파이프라인
│   │   └── README.md                      # 전처리 워크플로우 설명
│   │
│   ├── modeling/                          # 모델 학습
│   │   ├── baseline_models.py             # 베이스라인 모델
│   │   ├── advanced_ensemble.py           # 앙상블 모델
│   │   ├── final_solution.py              # 최종 솔루션
│   │   └── README.md                      # 모델링 전략 설명
│   │
│   ├── submissions/                       # 제출 파일 생성
│   │   ├── simple_submission.py           # 간단한 제출
│   │   ├── quick_baseline.py              # 빠른 베이스라인
│   │   └── README.md                      # 제출 형식 설명
│   │
│   └── tracking/                          # 실험 추적
│       ├── setup_experiment_tracking.py   # 추적 시스템 설정
│       ├── view_experiments.py            # 실험 결과 조회
│       ├── test_libraries.py              # 라이브러리 테스트
│       └── README.md                      # 추적 도구 설명
│
├── src/                                    # 재사용 가능한 모듈 (변경 없음)
│   ├── analysis/                          # 분석 유틸리티
│   ├── diagnosis/                         # 디버깅 도구
│   ├── tracking/                          # 추적 헬퍼
│   └── utils/                             # 일반 유틸리티
│
├── data/                                   # 데이터 디렉토리
│   ├── open/                              # 원본 경쟁 데이터 (기존)
│   └── processed/                         # 전처리된 데이터 (gitignored)
│
├── models/                                 # 학습된 모델 및 전처리기 (gitignored)
│   ├── preprocessor_basic.pkl
│   ├── preprocessor_aggressive.pkl
│   ├── preprocessor_feature_selected.pkl
│   ├── processed_data_basic.pkl
│   ├── processed_data_aggressive.pkl
│   └── processed_data_feature_selected.pkl
│
├── outputs/                                # 생성된 출력물
│   ├── figures/                           # 시각화 (PNG 파일)
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── dimensionality_analysis.png
│   │   ├── feature_importance.png
│   │   └── target_distribution.png
│   │
│   └── submissions/                       # 경쟁 제출 파일
│       ├── submission_baseline.csv
│       └── final_submission.csv
│
├── docs/                                   # 모든 문서
│   ├── README.md                          # 주요 프로젝트 문서
│   ├── PROJECT_SUMMARY.md                 # 프로젝트 개요
│   ├── DIRECTORY_STRUCTURE.md             # 이 파일
│   ├── file_inventory_before_reorganization.txt  # 재구성 전 백업
│   └── analysis_reports/                  # 분석 리포트
│       ├── chapter_00_current_codebase_analysis.md
│       ├── chapter3_phase1_task_T007_results.md
│       ├── master_plan_0_90_achievement.md
│       └── tasks_0_90_achievement.md
│
├── experiments/                            # 실험 추적 (변경 없음)
│   ├── mlruns/                            # MLflow 추적 데이터
│   ├── wandb/                             # WandB 추적 데이터
│   └── plots/                             # 실험 시각화
│
├── tests/                                  # 테스트 파일 (변경 없음)
├── config/                                 # 설정 파일 (변경 없음)
├── .specify/                               # Specify 프레임워크 (변경 없음)
├── specs/                                  # 기능 명세 (변경 없음)
│
├── pyproject.toml                          # 프로젝트 의존성
├── .gitignore                             # Git 제외 패턴 (업데이트됨)
├── .python-version                        # Python 버전
└── README.md                              # 프로젝트 메인 README

```

## 각 디렉토리의 목적

### notebooks/
**목적**: Jupyter 노트북 기반 분석
**내용**: 단계별로 구성된 노트북 (01_, 02_, 03_ 접두사)
**헌법 정렬**: 원칙 I - Jupyter 노트북 우선 개발
**네이밍**: `{단계번호}_{단계이름}.ipynb`

### scripts/
**목적**: 실행 가능한 Python 스크립트
**하위 디렉토리**:
- `analysis/` - 탐색적 데이터 분석 및 시각화
- `preprocessing/` - 데이터 변환 파이프라인
- `modeling/` - 모델 학습 및 앙상블
- `submissions/` - 경쟁 제출 파일 생성
- `tracking/` - 실험 추적 유틸리티

**scripts vs src 구분**:
- `scripts/`: `if __name__ == "__main__"` 블록이 있는 실행 가능한 워크플로우
- `src/`: 다른 스크립트와 노트북에서 import되는 재사용 가능한 모듈

### outputs/
**목적**: 생성된 모든 출력물
**하위 디렉토리**:
- `figures/` - 시각화 (PNG, SVG 등)
- `submissions/` - 경쟁 제출 CSV 파일

**버전 관리**: 제출 파일은 버전 또는 타임스탬프 포함 권장

### models/
**목적**: 학습된 모델 및 전처리기
**Gitignore**: 모든 .pkl 파일은 gitignore됨 (재생성 가능)
**내용**:
- `preprocessor_*.pkl` - 전처리기 객체
- `processed_data_*.pkl` - 전처리된 데이터셋
- 학습된 모델 파일 (선택적)

### docs/
**목적**: 모든 프로젝트 문서 중앙화
**내용**:
- 메인 README 및 프로젝트 요약
- 분석 리포트 (analysis_reports/)
- 구조 문서 (이 파일)

**README 처리**: 메인 README는 루트에 유지 (GitHub 가시성), docs/에 복사됨

### data/
**목적**: 원본 및 전처리된 데이터
**하위 디렉토리**:
- `open/` 또는 `raw/` - 원본 경쟁 데이터
- `processed/` - 전처리된 데이터셋 (gitignored)

## 파일 이동 결정

### Python 스크립트 분류 (11개 파일)

| 원본 파일 | 목적 | 대상 위치 |
|---------|------|----------|
| eda.py | 분석 | scripts/analysis/ |
| advanced_analysis.py | 분석 | scripts/analysis/ |
| preprocessing.py | 전처리 | scripts/preprocessing/ |
| baseline_models.py | 모델링 | scripts/modeling/ |
| advanced_ensemble.py | 모델링 | scripts/modeling/ |
| final_solution.py | 모델링 | scripts/modeling/ |
| simple_submission.py | 제출 | scripts/submissions/ |
| quick_baseline.py | 제출 | scripts/submissions/ |
| setup_experiment_tracking.py | 추적 | scripts/tracking/ |
| view_experiments.py | 추적 | scripts/tracking/ |
| test_libraries.py | 추적 | scripts/tracking/ |

**분류 기준**: 파일명 및 주요 기능 분석

### Jupyter 노트북 (1개 파일)

| 원본 파일 | 새 파일명 | 근거 |
|---------|---------|------|
| main.ipynb | notebooks/01_main_analysis.ipynb | 단계별 네이밍 컨벤션 |

**향후 개선**: 단일 통합 노트북을 01_eda, 02_preprocessing 등으로 분리 가능

### 시각화 (5개 파일)

모든 PNG 파일은 `outputs/figures/`로 이동:
- class_distribution.png
- correlation_heatmap.png
- dimensionality_analysis.png
- feature_importance.png
- target_distribution.png

### 모델 아티팩트 (6개 파일)

모든 .pkl 파일은 `models/`로 이동 및 gitignored:
- preprocessor_basic.pkl
- preprocessor_aggressive.pkl
- preprocessor_feature_selected.pkl
- processed_data_basic.pkl
- processed_data_aggressive.pkl
- processed_data_feature_selected.pkl

### 제출 파일 (2개 파일)

| 원본 파일 | 새 파일명 | 근거 |
|---------|---------|------|
| submission.csv | outputs/submissions/submission_baseline.csv | 버전 식별자 추가 |
| final_submission.csv | outputs/submissions/final_submission.csv | 유지 |

### 문서 (3+ 파일)

| 원본 파일 | 대상 위치 |
|---------|----------|
| PROJECT_SUMMARY.md | docs/PROJECT_SUMMARY.md |
| README.md | docs/README.md (루트에 복사 유지) |
| answers/*.md | docs/analysis_reports/ |

## 다목적 파일 처리

**감지된 다목적 파일**: 없음

**처리 전략** (헌법에서 명확히 함):
- 주요 목적 디렉토리에 배치
- 부차적 목적은 디렉토리 README에 문서화

**예시**:
```
# 만약 analysis_and_submission.py가 있었다면:
# 배치: scripts/analysis/analysis_and_submission.py
# 문서화: scripts/analysis/README.md에 제출 기능 언급
```

## Gitignore 패턴 설명

재구성 중 .gitignore에 추가된 패턴:

```gitignore
# 모델 아티팩트 및 대용량 바이너리 파일
*.pkl
**/processed_data*.pkl
**/preprocessor*.pkl
models/*.pkl
data/processed/*.pkl
```

**근거**:
- 저장소 비대화 방지
- 재생성 가능한 아티팩트 (스크립트에서 생성 가능)
- 버전 관리는 코드와 설정에 집중

## 헌법 정렬 노트

### 원칙 I: Jupyter 노트북 우선 개발
✅ `notebooks/` 디렉토리가 단계별 네이밍으로 생성됨
✅ README가 실행 순서를 설명함
✅ 향후 분리 전략이 문서화됨

### 원칙 II: 교육적 명확성
✅ 모든 디렉토리에 README 포함
✅ 명확하고 설명적인 디렉토리 이름
✅ 파일 이동 근거 문서화됨

### 원칙 III: 성능 우선 최적화
✅ 모델링 워크플로우가 명확하게 구성됨
✅ 실험 추적 도구 지원
✅ 재구성이 모델 성능에 영향 없음

### 원칙 IV: 연구 주도 혁신
✅ `experiments/` 디렉토리 보존됨
✅ 추적 스크립트가 실험 관리 지원
✅ 여러 전처리 및 모델링 전략 지원

### 원칙 V: 반복적 분석과 성찰
✅ `docs/analysis_reports/` 디렉토리가 과거 분석 보존
✅ 추적 도구가 실험 비교 가능
✅ README가 개선 영역 문서화

## Git 히스토리 보존

모든 파일 이동은 `git mv` 명령을 사용하여 히스토리를 보존했습니다:

```bash
git mv eda.py scripts/analysis/eda.py
git mv main.ipynb notebooks/01_main_analysis.ipynb
# ... 등등
```

**검증**:
```bash
git log --follow scripts/analysis/eda.py
```

## 롤백 절차

재구성을 되돌려야 하는 경우:

```bash
# 1. 재구성 전 커밋 해시 확인
git log --oneline | head -5

# 2. 해당 커밋으로 체크아웃
git checkout <commit-hash>

# 3. 또는 브랜치 삭제 (브랜치를 사용한 경우)
git checkout main
git branch -D 001-
```

**백업 참조**: `docs/file_inventory_before_reorganization.txt`

## 검증 체크리스트

재구성 후 검증:

- [x] 디렉토리 구조 생성됨
- [x] 11개 Python 스크립트 이동됨
- [x] 1개 노트북 이동 및 이름 변경됨
- [x] 5개 시각화 이동됨
- [x] 6개 모델 아티팩트 이동됨
- [x] 2개 제출 파일 이동됨
- [x] 문서 통합됨
- [x] .gitignore 업데이트됨
- [x] 모든 디렉토리에 README 생성됨
- [x] Git 히스토리 보존됨

## 변경되지 않은 디렉토리

다음 디렉토리는 **변경 없이** 유지됨:
- `src/` - 재사용 가능한 모듈
- `experiments/` - MLflow, WandB 추적
- `tests/` - 테스트 파일
- `config/` - 설정 파일
- `.specify/` - Specify 프레임워크
- `specs/` - 기능 명세

## 추가 노트

### main.py 처리
`main.py` 파일은 백업 인벤토리에 나타났지만 아직 분류되지 않았습니다. 향후 파일 내용을 검토하여 적절한 위치로 이동해야 합니다.

### 향후 개선 사항
1. 단일 노트북을 여러 단계별 노트북으로 분리
2. 모델 디렉토리에 하위 구조 추가 (baseline/, ensemble/, final/)
3. 시각화를 카테고리별로 정리 (eda/, model_performance/)
4. 자동화된 테스트 스크립트 추가
5. CI/CD 파이프라인 설정

---

**재구성 완료**: 2025-09-27
**헌법 버전**: 1.0.1
**명세 참조**: specs/001-/spec.md