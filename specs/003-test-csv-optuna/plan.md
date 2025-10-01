# Implementation Plan: Model Hyperparameter Optimization with Optuna

**Branch**: `003-test-csv-optuna` | **Date**: 2025-10-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/e7217/projects/dacon-236590/specs/003-test-csv-optuna/spec.md`

## Execution Flow (/plan command scope)
```
✅ 1. Load feature spec from Input path
✅ 2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project Type: Data Analysis (Jupyter notebooks)
   → Structure Decision: Notebooks-first with educational documentation
✅ 3. Fill the Constitution Check section
✅ 4. Evaluate Constitution Check section
   → No violations: Educational Jupyter workflow aligns with constitution
   → Update Progress Tracking: Initial Constitution Check PASS
✅ 5. Execute Phase 0 → research.md COMPLETE
   → All NEEDS CLARIFICATION resolved (optimization time: ~3 hours total)
✅ 6. Execute Phase 1 → contracts/, data-model.md, quickstart.md COMPLETE
✅ 7. Re-evaluate Constitution Check section
   → Design maintains constitutional compliance
   → Update Progress Tracking: Post-Design Constitution Check PASS
✅ 8. Plan Phase 2 → Task generation approach documented below
✅ 9. STOP - Ready for /tasks command
```

**STATUS**: Plan phase complete. Ready for `/tasks` command to generate tasks.md.

## Summary

This feature implements systematic hyperparameter optimization for 4 machine learning models (QDA, SVC, RandomForest, DecisionTree) using Optuna's Bayesian optimization with trial pruning. The workflow includes feature engineering, ensemble model creation, and generation of competition submission files. All work is implemented in educational Jupyter notebooks with GPU acceleration where available.

**Primary Requirement**: Optimize model hyperparameters to improve F1-macro scores over baseline performance from notebook 06_model_comparison.ipynb.

**Technical Approach** (from user requirements):
- Hyperparameter optimization using Optuna
- Feature engineering to boost performance
- Generate individual model predictions
- Create ensemble models (voting, stacking)
- Document concepts and reasoning in markdown (educational focus)
- Use GPU acceleration where possible (cuML, PyTorch)
- Create competition submission files for all models

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**:
- Core ML: scikit-learn 1.7.2, pandas 2.3.2, numpy 1.24+
- Optimization: optuna 3.4.0+, optuna-integration 4.5.0+
- GPU Acceleration: cuml-cu12 24.0.0+ (RAPIDS), torch 2.0.0+
- Gradient Boosting: lightgbm 4.1.0+, xgboost 2.0.0+, catboost 1.2.0+
- Visualization: matplotlib 3.7.0+, seaborn 0.12.0+, plotly 5.17.0+

**Storage**:
- Input: CSV files (train.csv 11.6MB, test.csv 8.0MB)
- Output: Jupyter notebooks (.ipynb), trained models (joblib), submission CSVs
- Models: Persisted to `models/` directory (~500MB total after optimization)

**Testing**:
- Validation: 5-fold Stratified Cross-Validation (constitutional requirement)
- Metrics: F1-macro (primary), Accuracy, Precision, Recall
- Format Validation: Submission CSV format checking

**Target Platform**: Linux (Ubuntu 20.04+), GPU-enabled (NVIDIA RTX 3090, 24GB VRAM)

**Project Type**: Data Analysis (Jupyter-first workflow)

**Performance Goals**:
- QDA: F1 ≥ 0.880 (improve/maintain 0.8782 baseline)
- RandomForest: F1 ≥ 0.760 (improve 0.7349 baseline by 2.5%)
- DecisionTree: F1 ≥ 0.735 (improve 0.7105 baseline by 2.5%)
- SVC: F1 ≥ 0.450 (improve 0.3277 baseline by 12%+)
- Ensemble: F1 > 0.890 (beat best individual model)

**Constraints**:
- Optimization Time: ~3 hours total for 4 models × 50 trials (resolved NEEDS CLARIFICATION)
- Memory: Polynomial features create ~1,300 features (must fit in RAM)
- GPU Memory: 24GB VRAM (adequate for all models)
- Reproducibility: All random operations use seed=42

**Scale/Scope**:
- Training Data: 21,693 samples, 52 features, 21 classes (perfectly balanced)
- Test Data: 15,004 samples
- Models: 4 individual + 3 ensemble = 7 total models
- Submissions: 7 CSV files for competition
- Notebooks: 3 new notebooks (07, 08, 09)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitutional Principles Compliance:**
- [x] **Jupyter Notebook-First**: All work implemented in numbered .ipynb files (07, 08, 09)
- [x] **Educational Clarity**: Each notebook includes markdown cells explaining concepts, analogies for complex topics (Bayesian optimization, pruning, ensemble methods)
- [x] **Performance-First**: Primary goal is F1-macro improvement through systematic optimization, all decisions justified by expected performance impact
- [x] **Research-Driven**: Incorporates Optuna (state-of-the-art optimization), references existing notebook patterns (03_optuna_optimization.ipynb), uses GPU acceleration (cuML/RAPIDS)
- [x] **Iterative Reflection**: Each notebook section ends with reflection on learnings, improvements, limitations, and next steps

**Design Validation**:
- ✅ Notebooks organized by logical stages (optimization → engineering → ensemble)
- ✅ Markdown-heavy documentation with 10-year-old explanations
- ✅ Cross-validation rigor maintained (5-fold stratified, constitutional minimum)
- ✅ Comprehensive docstrings and code comments
- ✅ Performance tracking and comparison tables

*Note: For data analysis projects, traditional software architecture principles may not apply. Focus on analysis workflow, documentation quality, and performance optimization.*

## Project Structure

### Documentation (this feature)
```
specs/003-test-csv-optuna/
├── plan.md              # This file (/plan command output) ✅
├── research.md          # Phase 0 output (/plan command) ✅
├── data-model.md        # Phase 1 output (/plan command) ✅
├── quickstart.md        # Phase 1 output (/plan command) ✅
├── contracts/           # Phase 1 output (/plan command) ✅
│   ├── submission_format.md        # Competition CSV format contract
│   └── optimization_results.md     # Optimization output format contract
└── tasks.md             # Phase 2 output (/tasks command - NOT YET CREATED)
```

### Source Code (repository root)
```
notebooks/
├── 07_hyperparameter_optimization.ipynb  # Main optimization notebook
├── 08_feature_engineering.ipynb          # Feature creation and selection
└── 09_final_ensemble.ipynb               # Ensemble models and predictions

models/
├── optuna_study_qda.pkl          # Optuna study for QDA
├── optuna_study_svc.pkl          # Optuna study for SVC
├── optuna_study_rf.pkl           # Optuna study for RandomForest
├── optuna_study_dt.pkl           # Optuna study for DecisionTree
├── qda_optimized.pkl             # Trained QDA with best params
├── svc_optimized.pkl             # Trained SVC with best params
├── rf_optimized.pkl              # Trained RandomForest with best params
├── dt_optimized.pkl              # Trained DecisionTree with best params
├── ensemble_voting_hard.pkl      # Hard voting ensemble
├── ensemble_voting_soft.pkl      # Soft voting ensemble (weighted)
├── ensemble_stacking.pkl         # Stacking with QDA meta-learner
└── scaler_optimized.pkl          # Feature scaler (MinMaxScaler)

outputs/
├── optimization_results.csv      # Summary of all optimization results
├── ensemble_results.csv          # Ensemble performance comparison
├── final_comparison.csv          # Overall results summary
└── submissions/
    ├── submission_qda_optimized.csv
    ├── submission_svc_optimized.csv
    ├── submission_rf_optimized.csv
    ├── submission_dt_optimized.csv
    ├── submission_ensemble_voting_hard.csv
    ├── submission_ensemble_voting_soft.csv
    └── submission_ensemble_stacking.csv

data/open/  # Existing data (not created by this feature)
├── train.csv            # Training data (21,693 samples)
├── test.csv             # Test data (15,004 samples)
└── sample_submission.csv # Competition format example
```

**Structure Decision**:
This is a **data analysis project** using Jupyter notebooks as the primary development environment following the constitutional requirement for numbered, stage-based notebooks.

## Phase 0: Outline & Research ✅

**Status**: COMPLETE (see research.md)

**Key Decisions**: Optuna framework, GPU acceleration strategy, 50 trials per model (~3 hours total), multi-technique feature engineering, multi-level ensembles.

See [research.md](./research.md) for full details.

## Phase 1: Design & Contracts ✅

**Status**: COMPLETE

**Artifacts Generated**:
- data-model.md: 6 core entities (OptimizationTrial, OptimizationStudy, OptimizedModel, FeatureSet, EnsembleModel, PredictionOutput)
- contracts/submission_format.md: Competition CSV requirements with validation
- contracts/optimization_results.md: Optimization output format standards
- quickstart.md: Complete execution guide with troubleshooting

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

The `/tasks` command will generate ~35 tasks organized by notebook and workflow stage.

**Task Categories**:

### Notebook 07: Hyperparameter Optimization (~15 tasks)
- Create notebook structure, GPU setup, data loading
- Define and run optimization for QDA, SVC, RF, DT (50 trials each)
- Compare results, save models, create summary

### Notebook 08: Feature Engineering (~8 tasks)
- Create features (polynomial, statistical, interactions)
- Perform feature selection
- Re-train and compare performance

### Notebook 09: Ensemble and Predictions (~10 tasks)
- Create ensembles (voting, stacking)
- Generate predictions for all models
- Create 7 submission CSVs

### Validation (~2 tasks)
- Validate submission formats
- Generate final performance summary

**Ordering**: Sequential by notebook (07 → 08 → 09), [P] markers for parallel-ready tasks

**Estimated Output**: 35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the `/tasks` command, NOT by `/plan`

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (create and execute all 3 notebooks)
**Phase 5**: Validation (verify outputs, check performance criteria)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

**No violations detected.** Design aligns with all constitutional principles.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command - NEXT STEP)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (optimization time = ~3 hours)
- [x] Complexity deviations documented (none)

**Artifacts Generated**:
- [x] research.md (Phase 0)
- [x] data-model.md (Phase 1)
- [x] contracts/submission_format.md (Phase 1)
- [x] contracts/optimization_results.md (Phase 1)
- [x] quickstart.md (Phase 1)
- [ ] tasks.md (Phase 2 - awaiting /tasks command)

**Next Action**: User should run `/tasks` command to generate tasks.md from this plan.

---
*Based on Constitution v1.0.1 - See `.specify/memory/constitution.md`*
*Plan generated via `/plan` command on 2025-10-01*