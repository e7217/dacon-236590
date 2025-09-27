# 🎯 Tasks: Macro F1-Score 0.90+ 달성

**Input**: 마스터 플랜 (`answers/master_plan_0_90_achievement.md`) 및 현재 코드베이스 분석 (`answers/chapter_00_current_codebase_analysis.md`)
**Prerequisites**: 기존 EDA, 전처리, 베이스라인 모델 완료
**Goal**: 현재 67.6%에서 90%+로 성능 향상

## Execution Flow (main)
```
1. Load 마스터 플랜과 현재 코드 분석 결과
   → 성능 격차 22.4% 해결 필요 확인
2. Phase별 태스크 생성:
   → Phase 1: 성능 진단 및 분석 (7개 태스크)
   → Phase 2: 고급 특성 엔지니어링 (8개 태스크)
   → Phase 3: 최신 모델 적용 (10개 태스크)
   → Phase 4: 앙상블 최적화 (8개 태스크)
   → Phase 5: 검증 및 제출 (7개 태스크)
3. 태스크 의존성 설정:
   → 테스트 → 구현 → 검증 순서
   → 병렬 처리 가능한 태스크 [P] 표시
4. 실행 가이드 제공
5. Return: 40개 구체적이고 실행 가능한 태스크
```

## Format: `[ID] [P?] Description`
- **[P]**: 병렬 실행 가능 (서로 다른 파일, 의존성 없음)
- 정확한 파일 경로와 구체적 실행 방법 포함

## Path Conventions
- **데이터 분석 프로젝트**: `src/`, `tests/`, `answers/` at repository root
- **스크립트**: 분석 및 모델링 스크립트들
- **문서**: `/answers` 폴더에 챕터별 학습 문서
- **모델**: pickle 파일로 저장된 모델들과 전처리기

---

## Phase 1: 성능 진단 및 분석 (1-2일)

### Phase 1.1: 환경 및 도구 설정
- [ ] T001 [P] 고급 ML 라이브러리 설치 (pyproject.toml 업데이트)
- [ ] T002 [P] 실험 추적 환경 설정 (MLflow/WandB 설정)
- [ ] T003 [P] 성능 진단 디렉토리 구조 생성 (src/diagnosis/, tests/diagnosis/)

### Phase 1.2: 성능 격차 분석 ⚠️ MUST COMPLETE BEFORE Phase 2
**CRITICAL: CV vs 실제 점수 격차(9.36%) 원인 규명 필수**
- [ ] T004 [P] Adversarial Validation 구현 in src/diagnosis/adversarial_validation.py
- [ ] T005 [P] 교차검증 전략 비교 분석 in src/diagnosis/cv_analysis.py
- [ ] T006 [P] 과적합 진단 시스템 구현 in src/diagnosis/overfitting_analysis.py
- [ ] T007 클래스별 성능 분석 (21개 클래스 개별 F1-score) in src/diagnosis/class_performance.py

### Phase 1.3: 모델 해석성 강화
- [ ] T008 [P] SHAP 분석 구현 in src/interpretation/shap_analysis.py
- [ ] T009 [P] Permutation Importance 계산 in src/interpretation/permutation_importance.py
- [ ] T010 혼동행렬 및 오분류 패턴 분석 in src/diagnosis/confusion_analysis.py

---

## Phase 2: 고급 특성 엔지니어링 (2-3일)

### Phase 2.1: 도메인 기반 특성 생성 (Phase 1 완료 후)
- [ ] T011 [P] 센서 그룹 통계 특성 생성기 in src/features/sensor_group_features.py
- [ ] T012 [P] 센서 간 상호작용 특성 생성기 in src/features/interaction_features.py
- [ ] T013 [P] 주요 센서 파생 특성 생성기 in src/features/derived_features.py
- [ ] T014 특성 생성 통합 파이프라인 in src/features/advanced_feature_engineering.py

### Phase 2.2: 차원 축소 및 특성 선택
- [ ] T015 [P] PCA + 원본 특성 조합기 in src/features/pca_features.py
- [ ] T016 [P] UMAP 임베딩 특성 생성기 in src/features/umap_features.py
- [ ] T017 [P] 고급 특성 선택기 (RFE, SelectKBest) in src/features/feature_selection.py
- [ ] T018 특성 엔지니어링 성능 검증 in tests/features/test_feature_engineering.py

---

## Phase 3: 최신 모델 적용 (3-4일)

### Phase 3.1: Gradient Boosting 모델들 (Phase 2 완료 후)
- [ ] T019 [P] XGBoost 모델 및 하이퍼파라미터 튜닝 in src/models/xgboost_model.py
- [ ] T020 [P] LightGBM 모델 및 하이퍼파라미터 튜닝 in src/models/lightgbm_model.py
- [ ] T021 [P] CatBoost 모델 및 하이퍼파라미터 튜닝 in src/models/catboost_model.py
- [ ] T022 Gradient Boosting 모델 비교 분석 in src/evaluation/gb_models_comparison.py

### Phase 3.2: 딥러닝 모델들
- [ ] T023 [P] TabNet 모델 구현 in src/models/tabnet_model.py
- [ ] T024 [P] Neural Network 모델 개선 in src/models/neural_network.py
- [ ] T025 [P] Wide & Deep Learning 모델 in src/models/wide_deep_model.py
- [ ] T026 딥러닝 모델 성능 검증 in tests/models/test_deep_models.py

### Phase 3.3: AutoML 및 고급 기법
- [ ] T027 [P] AutoGluon 실험 in src/models/autogluon_model.py
- [ ] T028 모델 성능 벤치마킹 시스템 in src/evaluation/model_benchmarking.py

---

## Phase 4: 앙상블 최적화 (2-3일)

### Phase 4.1: 고급 앙상블 전략 (Phase 3 완료 후)
- [ ] T029 [P] Stacking Ensemble 구현 in src/ensemble/stacking_ensemble.py
- [ ] T030 [P] Blending 최적 가중치 탐색 in src/ensemble/blending_ensemble.py
- [ ] T031 [P] Voting Ensemble 고도화 in src/ensemble/voting_ensemble.py
- [ ] T032 앙상블 다양성 분석 및 최적화 in src/ensemble/ensemble_diversity.py

### Phase 4.2: 하이퍼파라미터 대규모 최적화
- [ ] T033 [P] Optuna 기반 개별 모델 최적화 in src/optimization/hyperparameter_tuning.py
- [ ] T034 [P] 앙상블 가중치 베이지안 최적화 in src/optimization/ensemble_optimization.py
- [ ] T035 Multi-objective 최적화 (성능+속도) in src/optimization/multi_objective_optimization.py
- [ ] T036 최적화 결과 분석 및 검증 in src/evaluation/optimization_analysis.py

---

## Phase 5: 검증 및 최종 제출 (1-2일)

### Phase 5.1: 강건성 검증 (Phase 4 완료 후)
- [ ] T037 [P] Multiple Random Seed 실험 in src/validation/robustness_test.py
- [ ] T038 [P] 교차검증 전략 다양화 검증 in src/validation/cv_strategies.py
- [ ] T039 [P] 리더보드 과적합 방지 검증 in src/validation/leaderboard_validation.py
- [ ] T040 최종 모델 선택 및 제출 파일 생성 in src/submission/final_submission.py

### Phase 5.2: 문서화 및 학습 정리
- [ ] T041 [P] Phase 1 결과 문서화 in answers/chapter_01_diagnosis.md
- [ ] T042 [P] Phase 2 특성 엔지니어링 문서화 in answers/chapter_02_features.md
- [ ] T043 [P] Phase 3 모델링 결과 문서화 in answers/chapter_03_models.md
- [ ] T044 [P] Phase 4 앙상블 전략 문서화 in answers/chapter_04_ensemble.md
- [ ] T045 [P] Phase 5 최종 검증 결과 문서화 in answers/chapter_05_results.md
- [ ] T046 전체 프로젝트 학습 요약 in answers/final_learnings_summary.md

---

## Dependencies

### Phase Dependencies
- Phase 1 (T001-T010) → Phase 2 (T011-T018) → Phase 3 (T019-T028) → Phase 4 (T029-T036) → Phase 5 (T037-T046)
- 문서화 태스크 (T041-T046)는 각 Phase 완료 후 병렬 실행 가능

### Task Dependencies
- T001-T003 (Setup) before all other tasks
- T004-T007 (Performance Analysis) before T011 (Feature Engineering)
- T019-T021 (GB Models) before T022 (GB Comparison)
- T029-T031 (Ensemble Methods) before T032 (Ensemble Analysis)
- T037-T039 (Validation) before T040 (Final Submission)

### Critical Path
- T004 (Adversarial Validation) → T011 (Feature Engineering) → T019 (XGBoost) → T029 (Stacking) → T040 (Final Submission)

---

## Parallel Execution Examples

### Phase 1 Parallel Tasks:
```
# Launch T004-T006 together:
Task: "Adversarial Validation 구현 in src/diagnosis/adversarial_validation.py"
Task: "교차검증 전략 비교 분석 in src/diagnosis/cv_analysis.py"
Task: "과적합 진단 시스템 구현 in src/diagnosis/overfitting_analysis.py"
```

### Phase 2 Parallel Tasks:
```
# Launch T011-T013 together:
Task: "센서 그룹 통계 특성 생성기 in src/features/sensor_group_features.py"
Task: "센서 간 상호작용 특성 생성기 in src/features/interaction_features.py"
Task: "주요 센서 파생 특성 생성기 in src/features/derived_features.py"
```

### Phase 3 Parallel Tasks:
```
# Launch T019-T021 together:
Task: "XGBoost 모델 및 하이퍼파라미터 튜닝 in src/models/xgboost_model.py"
Task: "LightGBM 모델 및 하이퍼파라미터 튜닝 in src/models/lightgbm_model.py"
Task: "CatBoost 모델 및 하이퍼파라미터 튜닝 in src/models/catboost_model.py"
```

### Phase 4 Parallel Tasks:
```
# Launch T029-T031 together:
Task: "Stacking Ensemble 구현 in src/ensemble/stacking_ensemble.py"
Task: "Blending 최적 가중치 탐색 in src/ensemble/blending_ensemble.py"
Task: "Voting Ensemble 고도화 in src/ensemble/voting_ensemble.py"
```

---

## Notes
- [P] tasks = 서로 다른 파일, 의존성 없음 → 병렬 실행 가능
- 각 태스크는 구체적인 파일 경로와 함께 실행 가능한 수준으로 정의
- 모든 결과는 `answers/` 폴더에 챕터별로 문서화
- Phase별 성능 목표: 0.70+ → 0.75+ → 0.85+ → 0.90+ → 안정화
- 실험 추적과 재현성을 위한 random seed 일관성 유지

## Task Generation Rules
*Applied during execution*

1. **From Performance Diagnosis**:
   - 성능 격차 원인 → 진단 태스크 [P]
   - 각 진단 영역 → 별도 분석 태스크

2. **From Feature Engineering**:
   - 센서 그룹 → 그룹별 특성 생성 태스크 [P]
   - 차원 축소 기법 → 개별 구현 태스크 [P]

3. **From Model Development**:
   - 각 모델 타입 → 개별 구현 및 튜닝 태스크 [P]
   - 앙상블 방법 → 별도 구현 태스크 [P]

4. **Ordering**:
   - Setup → Diagnosis → Features → Models → Ensemble → Validation
   - 의존성이 있는 태스크는 순차 실행

## Validation Checklist
*GATE: Checked before proceeding to next phase*

- [ ] 모든 진단 태스크가 성능 격차 원인을 명확히 규명
- [ ] 모든 특성 엔지니어링이 검증 데이터에서 개선 효과 확인
- [ ] 모든 모델이 베이스라인 대비 통계적 유의한 개선 달성
- [ ] 모든 앙상블이 개별 모델 대비 성능 향상 확인
- [ ] 병렬 태스크들이 실제로 독립적이고 의존성 없음
- [ ] 각 태스크가 정확한 파일 경로와 함께 실행 가능한 수준으로 정의됨
- [ ] 문서화 태스크가 학습 목적에 맞게 상세하고 이해하기 쉽게 구성됨

---

**Total Tasks**: 46개
**Estimated Duration**: 9-14일
**Success Criteria**: Macro F1-Score 0.90+ 달성
**Documentation**: 각 Phase별 학습 내용을 `/answers` 폴더에 체계적으로 기록