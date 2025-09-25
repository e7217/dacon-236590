# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools
- [ ] T004 [P] Setup documentation structure in `/answers` folder per constitution
- [ ] T005 [P] Initialize experiment tracking and reproducibility framework

## Phase 3.2: 데이터 분석 검증 (Data Analysis Validation) ⚠️ MUST COMPLETE BEFORE 3.3
**중요: 데이터 분석 테스트는 구현 전에 작성되고 실패해야 함**
- [ ] T006 [P] 데이터 로딩 및 전처리 테스트 in tests/test_data_loading.py
- [ ] T007 [P] 특성 엔지니어링 검증 테스트 in tests/test_feature_engineering.py
- [ ] T008 [P] 모델 학습 파이프라인 테스트 in tests/test_model_training.py
- [ ] T009 [P] 성능 평가 및 검증 테스트 in tests/test_evaluation.py

## Phase 3.3: 핵심 데이터 분석 구현 (데이터 분석 테스트 실패 후에만)
- [ ] T010 [P] 데이터 로딩 및 전처리 모듈 in src/data/preprocessing.py
- [ ] T011 [P] 특성 엔지니어링 파이프라인 in src/features/feature_engineering.py
- [ ] T012 [P] 베이스라인 모델 구현 in src/models/baseline_model.py
- [ ] T013 고급 모델 구현 in src/models/advanced_models.py
- [ ] T014 교차검증 및 평가 in src/evaluation/cv_evaluation.py
- [ ] T015 모델 성능 비교 및 선택 in src/evaluation/model_selection.py
- [ ] T016 실험 로깅 및 추적 in src/utils/experiment_tracking.py

## Phase 3.4: 분석 통합 및 최적화
- [ ] T017 하이퍼파라미터 튜닝 파이프라인 in src/optimization/hyperparameter_tuning.py
- [ ] T018 앙상블 모델 구현 in src/models/ensemble_models.py
- [ ] T019 결과 시각화 및 해석 in src/visualization/results_viz.py
- [ ] T020 모델 해석 가능성 분석 in src/interpretation/model_interpretation.py

## Phase 3.5: 문서화 및 마무리
- [ ] T021 [P] 분석 결과 문서화 in answers/chapter_XX_results.md
- [ ] T022 모델 성능 벤치마킹 및 검증
- [ ] T023 [P] 학습 인사이트 문서화 in answers/chapter_XX_learnings.md
- [ ] T024 코드 정리 및 주석 개선
- [ ] T025 재현성 검증 및 최종 제출

## Dependencies
- Tests (T006-T009) before implementation (T010-T016)
- T010 blocks T011, T017
- T018 blocks T020
- Implementation before polish (T021-T025)

## Parallel Example
```
# Launch T006-T009 together:
Task: "데이터 로딩 및 전처리 테스트 in tests/test_data_loading.py"
Task: "특성 엔지니어링 검증 테스트 in tests/test_feature_engineering.py"
Task: "모델 학습 파이프라인 테스트 in tests/test_model_training.py"
Task: "성능 평가 및 검증 테스트 in tests/test_evaluation.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each endpoint → implementation task
   
2. **From Data Model**:
   - Each entity → model creation task [P]
   - Relationships → service layer tasks
   
3. **From User Stories**:
   - Each story → integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All contracts have corresponding tests
- [ ] All entities have model tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task