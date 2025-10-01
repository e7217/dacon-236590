# Tasks: Model Hyperparameter Optimization with Optuna

**Input**: Design documents from `/home/e7217/projects/dacon-236590/specs/003-test-csv-optuna/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
✅ 1. Load plan.md from feature directory
   → Tech stack: Python 3.12, Optuna 3.4.0+, scikit-learn, cuML
   → Structure: Jupyter notebooks (07, 08, 09)
✅ 2. Load optional design documents:
   → data-model.md: 6 entities (OptimizationTrial, OptimizationStudy, etc.)
   → contracts/: submission_format.md, optimization_results.md
   → research.md: 50 trials/model, GPU acceleration, feature engineering
✅ 3. Generate tasks by category:
   → Setup: directory structure, GPU verification
   → Notebook 07: Hyperparameter optimization (4 models)
   → Notebook 08: Feature engineering
   → Notebook 09: Ensemble and predictions
   → Validation: Format checks, performance validation
✅ 4. Apply task rules:
   → Different notebooks sections = [P] for parallel development
   → Same notebook = sequential (dependency flow)
   → Data analysis workflow (not TDD - notebooks are self-validating)
✅ 5. Number tasks sequentially (T001-T038)
✅ 6. Generate dependency graph
✅ 7. Create parallel execution examples
✅ 8. Validate task completeness:
   → All 4 models have optimization sections ✅
   → All entities tracked in notebooks ✅
   → All contract formats implemented ✅
✅ 9. Return: SUCCESS (38 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can develop in parallel (different notebook sections, no dependencies)
- Include exact file paths and notebook sections
- All paths relative to repository root: `/home/e7217/projects/dacon-236590/`

## Task Overview
- **Total**: 38 tasks
- **Parallel-ready**: 18 tasks marked [P]
- **Estimated Time**: 4-5 hours (including 2-3 hours optimization execution)
- **Notebooks**: 3 new notebooks (07, 08, 09)

---

## Phase 3.1: Setup and Verification (5-10 minutes)

- [x] **T001** Verify GPU setup and dependencies
  - File: Command-line verification
  - Run: `nvidia-smi`, verify CUDA availability with torch, test cuML import
  - Verify: RTX 3090 detected, CUDA available, cuML loads without errors
  - Constitutional: Infrastructure for performance-first optimization (Principle III)
  - **COMPLETED**: RTX 3090 24GB, CUDA available, cuML 25.08.00 installed

- [x] **T002** Create output directories for artifacts
  - Files: `models/`, `outputs/`, `outputs/submissions/`
  - Create directory structure per plan.md project structure
  - Verify: All directories exist and writable
  - Constitutional: Organized workflow supporting Jupyter-first development (Principle I)
  - **COMPLETED**: All directories created successfully

- [x] **T003** Verify baseline data from notebook 06
  - File: `notebooks/06_model_comparison.ipynb`
  - Extract baseline F1 scores: QDA (0.8782), SVC (0.3277), RF (0.7349), DT (0.7105)
  - Verify: 21,693 training samples, 15,004 test samples, 52 features
  - Constitutional: Evidence-based baseline for performance improvements (Principle III)
  - **COMPLETED**: 21,693 train, 15,004 test, 53 features, 21 balanced classes

---

## Phase 3.2: Notebook 07 - Hyperparameter Optimization (1.5-2 hours)

### Notebook Structure and Introduction (15-20 minutes)

- [x] **T004** [P] Create notebook 07 with educational introduction
  - File: `notebooks/07_hyperparameter_optimization.ipynb`
  - Create markdown cells explaining:
    - What is Bayesian optimization? (10-year-old explanation)
    - Why Optuna over grid search? (efficiency comparison)
    - How does trial pruning save time? (visual analogy)
    - GPU acceleration benefits (speed comparison table)
  - Verify: Introduction section complete with analogies and examples
  - Constitutional: Educational clarity for all concepts (Principle II)

- [x] **T005** [P] Setup environment and imports section
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 2)
  - Import libraries: pandas, numpy, sklearn, optuna, cuml, matplotlib, seaborn
  - GPU detection code: check CUDA, check cuML availability
  - Set random seeds: np.random.seed(42), optuna sampler seed
  - Display versions and GPU info
  - Verify: All imports successful, GPU detected, versions logged
  - Constitutional: Reproducibility through seed setting (Principle III)

### Data Loading and Preprocessing (10 minutes)

- [x] **T006** Implement data loading and preprocessing
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 3)
  - Load: `data/open/train.csv` (21,693 samples, 52 features, 21 classes)
  - Separate features (X) and target (y)
  - Apply MinMaxScaler for feature normalization
  - Setup: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - Verify stratification: each fold maintains ~4.76% per class
  - Verify: Data loaded, scaled, CV configured correctly
  - Constitutional: Rigorous 5-fold stratified CV (Principle III, constitutional requirement)
  - Depends on: T005

### Helper Functions (10 minutes)

- [x] **T007** [P] Implement cross-validation helper function
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 4)
  - Function: `run_cross_validation(model, model_name, X, y, cv)`
  - Returns: dict with mean/std metrics, fold scores, training time, failure handling
  - Use sklearn's cross_validate for efficiency
  - Handle failures gracefully (return NaN for failed trials)
  - Verify: Function defined, docstring with 10-year-old explanation
  - Constitutional: Educational documentation in docstrings (Principle II)

### Model 1: QDA Optimization (15-20 minutes)

- [x] **T008** [P] Create QDA optimization section with concept explanation
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 5)
  - Markdown: Explain QDA (Quadratic Discriminant Analysis)
    - What does "quadratic" mean? (curve boundaries, not straight lines)
    - Why good for this data? (captures non-linear relationships)
    - Expected improvement: 1-2% (already near-optimal at 87.8%)
  - Verify: Educational explanation complete
  - Constitutional: Concept explanation before implementation (Principle II)

- [x] **T009** Implement QDA objective function and optimization
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 5 continued)
  - Define objective function with hyperparameter search space:
    - `reg_param`: 0.0 to 1.0 (float) - regularization strength
    - `store_covariance`: True/False
    - `tol`: 1e-6 to 1e-3 (convergence tolerance)
  - Integrate 5-fold CV within objective
  - Create Optuna study: TPESampler(seed=42), MedianPruner(n_startup_trials=5)
  - Run 50 trials with progress bar
  - Report: best parameters, best F1, improvement over baseline (0.8782)
  - Save: `models/optuna_study_qda.pkl`
  - Verify: Optimization completes, best score ≥ 0.878, study saved
  - Constitutional: Performance metrics tracked (Principle III)
  - Depends on: T006, T007

- [x] **T010** [P] Visualize QDA optimization results
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 5 continued)
  - Generate: optimization history plot, parameter importance plot, parallel coordinate plot
  - Use optuna.visualization functions
  - Explain: "What do these plots tell us?" (educational interpretation)
  - Verify: 3 visualizations displayed
  - Constitutional: Visual analysis of optimization (Principle IV)

### Model 2: SVC Optimization (30-45 minutes - longest)

- [x] **T011** [P] Create SVC optimization section with concept explanation
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 6)
  - Markdown: Explain SVC (Support Vector Classifier)
    - What is a "support vector"? (the important boundary points)
    - Why slow for 21 classes? (multiclass complexity)
    - GPU acceleration: cuML SVC is 10x faster
    - Expected improvement: 12%+ (low baseline at 32.8%, large potential)
  - Verify: Educational explanation with analogies
  - Constitutional: Concept explanation before implementation (Principle II)

- [x] **T012** Implement SVC objective function with GPU support
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 6 continued)
  - GPU conditional import:
    ```python
    if gpu_available:
        try:
            from cuml.svm import SVC as cuSVC
            model = cuSVC(**params)
        except:
            from sklearn.svm import SVC
            model = SVC(**params)
    ```
  - Define objective function with search space:
    - `C`: 0.1 to 100.0 (log scale) - regularization
    - `kernel`: ['rbf', 'poly', 'sigmoid']
    - `gamma`: ['scale', 'auto'] or 0.001 to 10.0
    - `degree`: 2 to 5 (for poly kernel)
  - Run 50 trials (expect ~30-45 minutes with GPU)
  - Report: best parameters, best F1, improvement over baseline (0.3277)
  - Save: `models/optuna_study_svc.pkl`
  - Verify: Optimization completes, GPU used if available, best score ≥ 0.40
  - Constitutional: GPU acceleration for performance (Principle III)
  - Depends on: T006, T007

- [x] **T013** [P] Visualize SVC optimization results
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 6 continued)
  - Generate: optimization history, parameter importance, parallel coordinates
  - Analyze pruning effectiveness: show pruned vs completed trials
  - Verify: Visualizations show convergence, parameter relationships
  - Constitutional: Analysis of optimization patterns (Principle IV)

### Model 3: RandomForest Optimization (20-30 minutes)

- [x] **T014** [P] Create RandomForest optimization section with concept explanation
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 7)
  - Markdown: Explain RandomForest
    - What is a "forest"? (many decision trees voting together)
    - Why better than single tree? (reduces overfitting)
    - GPU acceleration benefits
    - Expected improvement: 2.5% (73.5% → 76%)
  - Verify: Educational explanation complete
  - Constitutional: Concept explanation before implementation (Principle II)

- [x] **T015** Implement RandomForest objective function with GPU support
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 7 continued)
  - GPU conditional import (cuml.ensemble.RandomForestClassifier)
  - Define objective function with search space:
    - `n_estimators`: 100 to 500 (number of trees)
    - `max_depth`: 5 to 30
    - `min_samples_split`: 2 to 50
    - `min_samples_leaf`: 1 to 20
    - `max_features`: ['sqrt', 'log2'] or 0.5 to 1.0
    - `bootstrap`: True/False
  - Run 50 trials
  - Report: best parameters, best F1, improvement over baseline (0.7349)
  - Save: `models/optuna_study_rf.pkl`
  - Verify: Optimization completes, best score ≥ 0.750
  - Constitutional: Performance-first optimization (Principle III)
  - Depends on: T006, T007

- [x] **T016** [P] Visualize RandomForest optimization results
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 7 continued)
  - Generate: standard Optuna visualizations
  - Analyze: which hyperparameters matter most (n_estimators vs max_depth)
  - Verify: Visualizations reveal optimization insights
  - Constitutional: Data-driven parameter analysis (Principle IV)

### Model 4: DecisionTree Optimization (15-20 minutes)

- [x] **T017** [P] Create DecisionTree optimization section with concept explanation
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 8)
  - Markdown: Explain DecisionTree
    - What is a decision tree? (flowchart of yes/no questions)
    - Why simpler than RandomForest? (just one tree)
    - Limitations: more prone to overfitting
    - Expected improvement: 2.5% (71.1% → 73.5%)
  - Verify: Educational explanation complete
  - Constitutional: Concept explanation before implementation (Principle II)

- [x] **T018** Implement DecisionTree objective function
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 8 continued)
  - Define objective function with search space:
    - `max_depth`: 3 to 30
    - `min_samples_split`: 2 to 50
    - `min_samples_leaf`: 1 to 20
    - `max_features`: [None, 'sqrt', 'log2'] or 0.5 to 1.0
    - `criterion`: ['gini', 'entropy', 'log_loss']
    - `splitter`: ['best', 'random']
  - Run 50 trials
  - Report: best parameters, best F1, improvement over baseline (0.7105)
  - Save: `models/optuna_study_dt.pkl`
  - Verify: Optimization completes, best score ≥ 0.725
  - Constitutional: Systematic optimization approach (Principle III)
  - Depends on: T006, T007

- [x] **T019** [P] Visualize DecisionTree optimization results
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 8 continued)
  - Generate: standard Optuna visualizations
  - Compare: single tree vs RandomForest parameter sensitivity
  - Verify: Visualizations complete
  - Constitutional: Comparative analysis (Principle IV)

### Results and Model Persistence (10 minutes)

- [x] **T020** Create comprehensive results comparison section
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 9)
  - Create comparison table: Model | Baseline F1 | Optimized F1 | Improvement | Time
  - Sort by optimized F1 (best first)
  - Identify which model improved most
  - Statistical significance: compare std deviations
  - Create visualization: grouped bar chart (baseline vs optimized)
  - Verify: All 4 models in table, improvements calculated, chart displayed
  - Constitutional: Evidence-based performance tracking (Principle III)
  - Depends on: T009, T012, T015, T018

- [x] **T021** Train and save optimized models on full dataset
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 10)
  - For each model:
    - Create model with best hyperparameters from Optuna
    - Train on full training dataset (X_scaled, y)
    - Add metadata attributes (_metadata dict with params, scores, timestamps)
    - Save via joblib: `models/{model_name}_optimized.pkl`
  - Save scaler: `models/scaler_optimized.pkl`
  - Generate: `outputs/optimization_results.csv` (per contracts/optimization_results.md)
  - Verify: 4 model files + scaler + CSV saved, loadable
  - Constitutional: Reproducible artifacts (Principle III)
  - Depends on: T020

- [x] **T022** [P] Add reflection section to notebook 07
  - File: `notebooks/07_hyperparameter_optimization.ipynb` (Section 11)
  - Address reflection questions:
    - What did we learn from this optimization?
    - Which models benefited most from hyperparameter tuning?
    - What surprised us? (SVC improvement, QDA stability, etc.)
    - What are the limitations of this approach?
    - What could we try next? (feature engineering, different algorithms)
  - Verify: Reflection section with insights and learnings
  - Constitutional: Iterative reflection requirement (Principle V)

---

## Phase 3.3: Notebook 08 - Feature Engineering (30-45 minutes)

### Notebook Structure and Concepts (10 minutes)

- [x] **T023** [P] Create notebook 08 with feature engineering concepts
  - File: `notebooks/08_feature_engineering.ipynb`
  - Markdown sections:
    - What is feature engineering? (creating new info from existing data)
    - Why does it help? (gives models more patterns to learn)
    - Types of features: polynomial, statistical, interactions
    - Expected improvement: +1-3% for each model
  - Verify: Educational introduction complete
  - Constitutional: Educational clarity for all concepts (Principle II)

- [x] **T024** Load optimized models and setup
  - File: `notebooks/08_feature_engineering.ipynb` (Section 2)
  - Load: 4 optimized models from `models/*_optimized.pkl`
  - Load: scaler from `models/scaler_optimized.pkl`
  - Load: training data
  - Verify best hyperparameters loaded correctly
  - Verify: Models and data loaded successfully
  - Constitutional: Building on previous work (Principle V)
  - Depends on: T021

### Feature Generation (15-20 minutes)

- [x] **T025** [P] Implement polynomial feature generation
  - File: `notebooks/08_feature_engineering.ipynb` (Section 3)
  - Use PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
  - Apply to original 52 features
  - Expected output: ~1,300 features
  - Explain: "Why degree 2?" (balance complexity vs overfitting)
  - Apply feature selection if needed (keep top 500-800 features)
  - Verify: Features generated, memory usage acceptable (<2GB)
  - Constitutional: Research-driven technique (Principle IV)

- [x] **T026** [P] Implement statistical aggregation features
  - File: `notebooks/08_feature_engineering.ipynb` (Section 4)
  - Create features:
    - Mean, std, min, max per feature group (if groupings exist)
    - Percentiles (25th, 50th, 75th)
    - Z-scores for outlier detection
  - Expected output: ~30 additional features
  - Explain: "What do these statistics capture?" (data distribution patterns)
  - Verify: Statistical features created
  - Constitutional: Domain knowledge integration (Principle IV)

- [x] **T027** [P] Implement interaction features
  - File: `notebooks/08_feature_engineering.ipynb` (Section 5)
  - Identify top 10 important features from baseline models
  - Create pairwise products: 10 choose 2 = 45 features
  - Create pairwise ratios: 45 additional features
  - Explain: "Why interactions matter for trees" (explicit feature relationships)
  - Verify: Interaction features created (~90 features)
  - Constitutional: Performance-driven feature creation (Principle III)

### Feature Selection and Model Retraining (10-15 minutes)

- [x] **T028** Perform feature selection
  - File: `notebooks/08_feature_engineering.ipynb` (Section 6)
  - Correlation analysis: remove highly correlated features (>0.95)
  - Mutual information: rank features by information gain
  - Recursive feature elimination (optional, if time permits)
  - Select final feature set (target: 100-500 features)
  - Explain: "Why remove redundant features?" (reduce noise, faster training)
  - Verify: Final feature set selected and documented
  - Constitutional: Systematic feature selection (Principle IV)
  - Depends on: T025, T026, T027

- [x] **T029** Retrain models with engineered features
  - File: `notebooks/08_feature_engineering.ipynb` (Section 7)
  - For each model (QDA, SVC, RF, DT):
    - Use best hyperparameters from notebook 07
    - Train on engineered features
    - Evaluate with 5-fold CV
    - Compare: raw features vs engineered features
  - Create comparison table showing improvements
  - Verify: All models retrained, performance compared
  - Constitutional: Performance tracking and comparison (Principle III)
  - Depends on: T028

- [x] **T030** Save engineered models and transformers
  - File: `notebooks/08_feature_engineering.ipynb` (Section 8)
  - Save: `models/{model_name}_optimized_engineered.pkl` (if better than raw)
  - Save: `models/feature_transformer.pkl` (PolynomialFeatures + SelectKBest pipeline)
  - Generate: `outputs/feature_engineering_results.csv`
  - Verify: Models and transformer saved
  - Constitutional: Reproducible artifacts (Principle III)
  - Depends on: T029

- [x] **T031** [P] Add reflection section to notebook 08
  - File: `notebooks/08_feature_engineering.ipynb` (Section 9)
  - Address reflection questions:
    - Which feature types helped most?
    - Did all models benefit equally from feature engineering?
    - What was surprising about feature importance?
    - What other feature engineering approaches could we try?
  - Verify: Reflection complete with insights
  - Constitutional: Iterative reflection requirement (Principle V)

---

## Phase 3.4: Notebook 09 - Ensemble and Predictions (15-20 minutes)

### Notebook Structure and Ensemble Concepts (5 minutes)

- [x] **T032** [P] Create notebook 09 with ensemble concepts
  - File: `notebooks/09_final_ensemble.ipynb`
  - Markdown sections:
    - What is an ensemble? (combining multiple models)
    - Voting vs Stacking: key differences
    - Hard voting: simple majority
    - Soft voting: weighted by confidence
    - Stacking: meta-learner on predictions
    - Expected improvement: beat best individual model (QDA 87.8% → 89%+)
  - Verify: Educational introduction complete
  - Constitutional: Educational clarity for ensemble concepts (Principle II)

- [x] **T033** Load all optimized models
  - File: `notebooks/09_final_ensemble.ipynb` (Section 2)
  - Load: 4 optimized models (choose best from raw vs engineered)
  - Load: scaler and feature transformer (if using engineered features)
  - Load: training data for ensemble training
  - Verify: All models loaded successfully
  - Constitutional: Building on previous optimization work (Principle V)
  - Depends on: T021, T030

### Ensemble Creation (10 minutes)

- [x] **T034** Create voting ensembles
  - File: `notebooks/09_final_ensemble.ipynb` (Section 3)
  - Hard voting: VotingClassifier([QDA, RF, DT], voting='hard')
  - Soft voting: VotingClassifier([QDA, RF, DT], voting='soft')
    - Weights: [0.65, 0.25, 0.10] (proportional to CV F1 scores)
  - Evaluate with 5-fold CV
  - Compare: hard vs soft vs best individual model
  - Explain: "When is voting better than single model?"
  - Verify: Both voting ensembles created and evaluated
  - Constitutional: Multiple approaches comparison (Principle IV)
  - Depends on: T033

- [x] **T035** Create stacking ensemble
  - File: `notebooks/09_final_ensemble.ipynb` (Section 4)
  - Base models: [QDA, SVC, RF, DT]
  - Meta-learner: QDA (best performing model)
  - Train stacking ensemble with cross-validation
  - Evaluate with 5-fold CV
  - Compare: stacking vs voting vs individual models
  - Explain: "How does stacking learn from base models?"
  - Verify: Stacking ensemble created and evaluated
  - Constitutional: Advanced technique exploration (Principle IV)
  - Depends on: T033

- [x] **T036** Save ensemble models
  - File: `notebooks/09_final_ensemble.ipynb` (Section 5)
  - Save: `models/ensemble_voting_hard.pkl`
  - Save: `models/ensemble_voting_soft.pkl`
  - Save: `models/ensemble_stacking.pkl`
  - Generate: `outputs/ensemble_results.csv` with CV performance
  - Generate: `outputs/final_comparison.csv` (all models + ensembles)
  - Verify: Ensemble models saved, comparison CSVs generated
  - Constitutional: Complete performance tracking (Principle III)
  - Depends on: T034, T035

### Test Set Predictions and Submissions (5-10 minutes)

- [x] **T037** Generate predictions and submission files
  - File: `notebooks/09_final_ensemble.ipynb` (Section 6)
  - Load: `data/open/test.csv` (15,004 samples)
  - For each model (4 individual + 3 ensemble = 7 total):
    - Generate predictions
    - Create submission DataFrame (ID, target)
    - Validate format per contracts/submission_format.md:
      - 15,004 rows
      - ID matches test.csv
      - target in [0, 20]
      - No missing values
    - Save: `outputs/submissions/submission_{model_name}.csv`
  - Analyze prediction distributions (check for class imbalance)
  - Verify: 7 submission files created and validated
  - Constitutional: Rigorous output validation (Principle III)
  - Depends on: T036

- [x] **T038** [P] Add final reflection section to notebook 09
  - File: `notebooks/09_final_ensemble.ipynb` (Section 7)
  - Address comprehensive reflection questions:
    - Overall journey: What did we learn from baseline → optimization → engineering → ensemble?
    - Best performing approach: Which model/ensemble achieved highest F1?
    - Performance summary: Show progression (baseline → optimized → engineered → ensemble)
    - Surprises: What unexpected results did we find?
    - Limitations: What couldn't we improve? (e.g., SVC still challenging)
    - Future improvements: What would we try next?
      - More trials (75-100 per model)
      - Different ensemble strategies (gradient boosting on ensembles)
      - Neural networks (TabNet)
      - Advanced feature engineering (AutoFeat, deep feature synthesis)
  - Create final visualization: timeline showing F1 improvements across phases
  - Verify: Comprehensive reflection with learnings and future directions
  - Constitutional: Iterative reflection and learning (Principle V)

---

## Dependencies

### Sequential Dependencies
```
Setup: T001 → T002 → T003
Notebook 07 Introduction: T004 → T005
Notebook 07 Data: T006 (depends on T005)
Notebook 07 Helpers: T007 (parallel with T006)
Notebook 07 Models: T008-T009 (QDA), T011-T012 (SVC), T014-T015 (RF), T017-T018 (DT)
Notebook 07 Results: T020 (depends on all model tasks) → T021 (save models)
Notebook 08: T024 (depends on T021) → T025-T027 (parallel) → T028 → T029 → T030
Notebook 09: T033 (depends on T021, T030) → T034 → T035 → T036 → T037
```

### Parallel Execution Groups
```
Group 1 - Setup [P]: T001, T002 (2 tasks in parallel)
Group 2 - NB07 Intro [P]: T004, T005, T007 (3 tasks in parallel)
Group 3 - NB07 Model Concepts [P]: T008, T011, T014, T017 (4 tasks in parallel)
Group 4 - NB07 Visualizations [P]: T010, T013, T016, T019, T022 (5 tasks in parallel)
Group 5 - NB08 Features [P]: T025, T026, T027 (3 tasks in parallel)
Group 6 - NB09 Intro [P]: T023, T032 (2 tasks in parallel)
Group 7 - Reflections [P]: T022, T031, T038 (3 tasks in parallel - if notebooks independent)
```

## Parallel Example

### Example 1: Notebook Introduction Sections
```bash
# Launch T004, T005, T007 together (different sections, no dependencies):
Task: "Create notebook 07 educational introduction with Bayesian optimization concepts"
Task: "Setup environment and imports with GPU detection"
Task: "Implement cross-validation helper function with educational docstring"
```

### Example 2: Model Concept Explanations
```bash
# Launch T008, T011, T014, T017 together (parallel markdown sections):
Task: "Create QDA optimization section explaining quadratic boundaries"
Task: "Create SVC optimization section explaining support vectors and GPU acceleration"
Task: "Create RandomForest optimization section explaining ensemble voting"
Task: "Create DecisionTree optimization section explaining decision flowcharts"
```

### Example 3: Feature Generation
```bash
# Launch T025, T026, T027 together (independent feature creation):
Task: "Implement polynomial features (degree 2) with feature selection"
Task: "Implement statistical aggregation features (mean, std, percentiles)"
Task: "Implement interaction features from top 10 important features"
```

---

## Validation Checklist

### Task Completeness
- [x] All 4 models have optimization sections (T009, T012, T015, T018)
- [x] All 4 models have visualizations (T010, T013, T016, T019)
- [x] Feature engineering covers all technique types (T025, T026, T027)
- [x] All 3 ensemble types implemented (T034 voting, T035 stacking)
- [x] All output contracts satisfied:
  - [x] optimization_results.csv (T021)
  - [x] 7 submission CSVs (T037)
  - [x] All models saved (T021, T030, T036)

### Constitutional Compliance
- [x] All notebooks have educational introductions (T004, T023, T032) - Principle II
- [x] All concepts explained before implementation (T008, T011, T014, T017) - Principle II
- [x] Performance metrics tracked throughout (T020, T029, T036, T037) - Principle III
- [x] Research-driven techniques used (Optuna, cuML, feature engineering) - Principle IV
- [x] All notebooks have reflection sections (T022, T031, T038) - Principle V
- [x] 5-fold stratified CV maintained (T006, T009, T012, T015, T018) - Principle III (constitutional minimum)

### Parallel Independence
- [x] [P] tasks modify different files or independent sections
- [x] No [P] task depends on output from another [P] task
- [x] Sequential tasks properly ordered by dependencies

### Execution Readiness
- [x] Each task specifies exact file path
- [x] Each task includes verification criteria
- [x] Each task references constitutional principles
- [x] Dependencies clearly documented
- [x] Time estimates provided

---

## Success Criteria

### Minimum Success (Must Achieve)
- [ ] All 4 models optimized with Optuna (50 trials each)
- [ ] All models show improvement or maintain baseline performance
- [ ] 7 valid submission files generated (format validated)
- [ ] All notebooks execute without errors
- [ ] Ensemble F1 > 0.880

### Target Success (Goal)
- [ ] QDA F1 ≥ 0.880 (maintain/improve 0.8782)
- [ ] SVC F1 ≥ 0.450 (improve from 0.3277)
- [ ] RandomForest F1 ≥ 0.760 (improve from 0.7349)
- [ ] DecisionTree F1 ≥ 0.735 (improve from 0.7105)
- [ ] Ensemble F1 > 0.890 (beat best individual model)

### Stretch Success (Exceptional)
- [ ] Ensemble F1 > 0.895
- [ ] SVC F1 > 0.500 (significant improvement)
- [ ] Feature engineering adds +2% to QDA
- [ ] All optimization completes within 3-hour budget

---

## Notes

- **Jupyter Workflow**: Unlike traditional TDD, notebooks are self-validating through embedded outputs
- **Educational Focus**: Every concept must have a 10-year-old explanation before implementation
- **Performance Tracking**: All metrics logged and compared across phases (baseline → optimized → engineered → ensemble)
- **GPU Acceleration**: Use cuML when available, graceful fallback to sklearn
- **Reproducibility**: All random operations use seed=42
- **Time Budget**: ~3 hours for optimization execution (T009, T012, T015, T018), additional ~1-2 hours for feature engineering and ensembles

---

**Status**: Tasks ready for execution
**Total Tasks**: 38
**Parallel Opportunities**: 18 tasks marked [P]
**Estimated Completion Time**: 4-5 hours (including 2-3 hours optimization runtime)
**Next Step**: Begin with T001 (GPU verification) or run parallel groups for faster development