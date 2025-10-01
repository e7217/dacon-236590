# Research: Model Hyperparameter Optimization with Optuna

**Feature**: 003-test-csv-optuna
**Date**: 2025-10-01
**Purpose**: Document technical decisions and research for hyperparameter optimization implementation

## Overview

This feature extends the baseline model comparison (notebook 06) by implementing systematic hyperparameter optimization using Optuna's Bayesian optimization approach. The goal is to improve F1-macro scores through intelligent hyperparameter search with GPU acceleration.

## Technical Stack Decisions

### Decision 1: Optimization Framework - Optuna
**Chosen**: Optuna 3.4.0+
**Rationale**:
- Bayesian optimization (TPE sampler) more efficient than grid/random search
- Built-in pruning reduces wasted computation on unpromising trials
- Excellent integration with scikit-learn, LightGBM, XGBoost via callbacks
- Visualization tools for parameter importance and optimization history
- Native support for distributed optimization if needed in future

**Alternatives Considered**:
- **Hyperopt**: Older, less maintained, fewer integrations
- **Scikit-optimize**: Good but less feature-rich than Optuna
- **Ray Tune**: Overkill for this scale, better for deep learning

**Reference**: Based on patterns from notebooks/03_optuna_optimization.ipynb (LightGBM + XGBoost optimization examples)

### Decision 2: GPU Acceleration Strategy
**Chosen**: Conditional GPU usage with cuML fallback to sklearn
**Rationale**:
- cuML provides GPU-accelerated versions of sklearn algorithms
- Available hardware: NVIDIA GeForce RTX 3090 (confirmed from notebook 06)
- Significant speedups for SVC and RandomForest (10x for SVC, 6x for RF based on notebook analysis)
- Graceful fallback to CPU implementations if GPU unavailable

**Implementation Pattern**:
```python
if gpu_available:
    try:
        from cuml.svm import SVC as cuSVC
        model = cuSVC(...)
    except:
        from sklearn.svm import SVC
        model = SVC(...)
```

**Alternatives Considered**:
- **CPU-only**: Too slow for iterative optimization (227s for LogReg, 55s for SVC)
- **GPU-only**: Breaks portability, not all models have GPU versions

### Decision 3: Cross-Validation Strategy
**Chosen**: 5-fold Stratified K-Fold
**Rationale**:
- Constitutional requirement (minimum 5-fold stratified CV)
- Dataset is perfectly balanced (21 classes, 1033 samples each = 21,693 total)
- Stratification maintains class balance across folds
- 5 folds provides good bias-variance tradeoff

**Configuration**:
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Evidence**: From notebook 06, all folds maintain ~4.76% per class distribution

### Decision 4: Feature Engineering Approach
**Chosen**: Comprehensive feature engineering pipeline
**Rationale**:
- Baseline uses only 52 raw features
- Feature engineering can provide 2-4% improvement (constitutional expectation)
- Polynomial features, interactions, statistical aggregations

**Planned Techniques**:
1. **Polynomial Features**: Degree 2 for QDA (captures non-linear relationships)
2. **Statistical Features**: Rolling statistics, percentiles, z-scores
3. **Interaction Features**: Product of important feature pairs
4. **Domain-Specific**: Manufacturing sensor patterns (if applicable)

**Alternatives Considered**:
- **Deep feature learning**: TabNet - overkill for this dataset size
- **AutoML feature**: Too opaque, reduces educational value

**Reference**: notebooks/04_feature_engineering.ipynb and v2 show existing patterns

### Decision 5: Ensemble Strategy
**Chosen**: Multi-level ensemble approach
**Rationale**:
- Baseline already shows ensemble benefit (notebook 06 created voting classifiers)
- Combine diverse models for robustness
- Stack predictions using best-performing model as meta-learner

**Planned Ensembles**:
1. **Hard Voting**: QDA + RandomForest + DecisionTree (simple majority vote)
2. **Soft Voting**: Weighted by CV scores (better for calibrated probabilities)
3. **Stacking**: Use QDA as meta-learner (best baseline F1: 0.8782)

**Evidence**: From notebook 06, ensemble approaches were explored but need optimization

## Model-Specific Research

### Model 1: QuadraticDiscriminantAnalysis (QDA)
**Baseline F1**: 0.8782 ± 0.0029 (best performer)
**Training Time**: 0.33s (fastest)
**GPU Support**: No (CPU-only algorithm)

**Hyperparameters to Optimize**:
- `reg_param` (regularization): 0.0 to 1.0 (float)
  - Adds shrinkage to covariance estimates
  - Prevents singularity with small samples
- `store_covariance`: True/False
  - Memory vs computation tradeoff
- `tol`: 1e-6 to 1e-3 (convergence tolerance)

**Expected Improvement**: 1-2% (87.8% → 88.5-89.0%)
**Rationale**: Already near-optimal, small tuning gains expected

### Model 2: Support Vector Classifier (SVC)
**Baseline F1**: 0.3277 ± 0.0161 (worst performer)
**Training Time**: 32.90s (with GPU)
**GPU Support**: Yes (cuML SVC available)

**Hyperparameters to Optimize**:
- `C` (regularization): 0.1 to 100.0 (log scale)
  - Controls margin vs misclassification tradeoff
- `kernel`: 'rbf', 'poly', 'sigmoid'
  - RBF most flexible for non-linear boundaries
- `gamma`: 'scale', 'auto', or 0.001 to 10.0
  - Kernel coefficient (higher = more complex)
- `degree`: 2 to 5 (for poly kernel)
- `class_weight`: None, 'balanced'

**Expected Improvement**: 10-20% (32.8% → 42-52%)
**Rationale**: Low baseline suggests poor default hyperparameters, large optimization potential

**Challenge**: 21-class multiclass problem → One-vs-Rest or One-vs-One strategy impacts performance

### Model 3: RandomForestClassifier
**Baseline F1**: 0.7349 ± 0.0014 (second best)
**Training Time**: 5.92s (with GPU)
**GPU Support**: Yes (cuML RandomForest available)

**Hyperparameters to Optimize**:
- `n_estimators`: 100 to 500 (number of trees)
- `max_depth`: 5 to 30 (tree depth limit)
- `min_samples_split`: 2 to 50
- `min_samples_leaf`: 1 to 20
- `max_features`: 'sqrt', 'log2', or 0.5 to 1.0 (fraction)
- `bootstrap`: True/False
- `class_weight`: None, 'balanced', 'balanced_subsample'

**Expected Improvement**: 3-5% (73.5% → 76-78%)
**Rationale**: Moderate baseline, good optimization potential

**Reference**: notebook 03_optuna_optimization.ipynb shows RandomForest optimization patterns

### Model 4: DecisionTreeClassifier
**Baseline F1**: 0.7105 ± 0.0060 (third best)
**Training Time**: 8.38s
**GPU Support**: No (CPU-only algorithm)

**Hyperparameters to Optimize**:
- `max_depth`: 3 to 30
- `min_samples_split`: 2 to 50
- `min_samples_leaf`: 1 to 20
- `max_features`: None, 'sqrt', 'log2', or 0.5 to 1.0
- `criterion`: 'gini', 'entropy', 'log_loss'
- `splitter`: 'best', 'random'
- `class_weight`: None, 'balanced'

**Expected Improvement**: 2-4% (71.1% → 73-75%)
**Rationale**: Single tree limits, but good hyperparameter sensitivity

## Optimization Configuration

### Trial Budget
**Decision**: 50 trials per model (conservative)
**Rationale**:
- Based on notebook 03: LightGBM (50 trials, 54 min), XGBoost (50 trials, 39 min)
- With 4 models: ~2-3 hours total optimization time
- Resolves NEEDS CLARIFICATION from spec (reasonable = ~3 hours total)

**Pruning Strategy**:
- MedianPruner with n_startup_trials=5, n_warmup_steps=2
- Expected 20-30% trial reduction (time savings)

### Search Space Strategy
**Decision**: Model-specific tailored search spaces
**Rationale**:
- QDA: Small space (few parameters) → dense sampling
- SVC: Large space (kernel, C, gamma) → smart exploration critical
- RF/DT: Medium space → balance breadth and depth

### Random State Management
**Decision**: Fixed seed (42) for reproducibility
**Configuration**:
```python
np.random.seed(42)
optuna.samplers.TPESampler(seed=42)
StratifiedKFold(..., random_state=42)
model(..., random_state=42)
```

**Rationale**: Constitutional requirement for reproducibility

## Feature Engineering Research

### Technique 1: Polynomial Features
**Applicability**: QDA, DecisionTree
**Configuration**:
- Degree: 2 (avoid curse of dimensionality)
- Interaction only: True (reduce feature explosion)
- Expected: 52 → ~1,300 features

**Rationale**: Captures non-linear relationships QDA needs

### Technique 2: Statistical Aggregations
**Applicability**: All models
**Features**:
- Rolling statistics (mean, std, min, max) over feature groups
- Percentile features (25th, 50th, 75th)
- Z-score normalization
- Expected: +20-30 features

**Rationale**: Manufacturing sensor data likely has temporal/group patterns

### Technique 3: Feature Interactions
**Applicability**: Tree-based models (RF, DT)
**Configuration**:
- Top 10 important features from baseline models
- Pairwise products: 10 choose 2 = 45 features
- Ratios: 45 additional features

**Rationale**: Tree models benefit from explicit interaction features

### Technique 4: Dimensionality Reduction (Optional)
**Applicability**: If feature explosion occurs
**Options**:
- PCA (linear): Preserve 95% variance
- UMAP (non-linear): 2-10 components for visualization
- Feature selection: SelectKBest, mutual information

**Trigger**: If features > 2000 and performance degrades

## Evaluation Metrics

### Primary Metric: F1-Macro
**Definition**: Average F1 across all 21 classes
**Rationale**: Competition metric, handles class imbalance (even though perfectly balanced)

### Secondary Metrics
- **Accuracy**: Overall correctness
- **Precision** (macro): Positive predictive value per class
- **Recall** (macro): Sensitivity per class
- **Training Time**: Track optimization cost

### Comparison Baseline
From notebook 06_model_comparison.ipynb:
```
QuadraticDiscriminantAnalysis: 0.8782 ± 0.0029
RandomForestClassifier:        0.7349 ± 0.0014
DecisionTreeClassifier:        0.7105 ± 0.0060
SVC:                           0.3277 ± 0.0161
LogisticRegression:            0.4987 ± 0.0044 (excluded)
```

### Success Criteria
- QDA: ≥ 0.880 (maintain or improve by 0.2%)
- RF: ≥ 0.760 (improve by 2.5%)
- DT: ≥ 0.735 (improve by 2.5%)
- SVC: ≥ 0.450 (improve by 12% - major gains needed)
- Ensemble: > 0.890 (beat best individual model)

## Notebook Structure

### Notebook 07: Hyperparameter Optimization with Optuna
**Purpose**: Systematic optimization of all 4 models
**Sections**:
1. Introduction and Concepts (교육적 설명)
   - What is Bayesian optimization?
   - Why Optuna over grid search?
   - How does trial pruning work?

2. Environment Setup
   - GPU detection and configuration
   - Library imports and versions
   - Random seed setting

3. Data Loading and Preprocessing
   - Load train.csv (21,693 samples, 52 features)
   - Feature scaling (MinMaxScaler for consistency)
   - Cross-validation setup

4. Model 1: QDA Optimization
   - Search space definition
   - Objective function with 5-fold CV
   - Run 50 trials with TPE sampler
   - Visualize optimization history
   - Report best parameters and improvement

5. Model 2: SVC Optimization
   - GPU-accelerated with cuML fallback
   - Kernel selection (rbf/poly/sigmoid)
   - C and gamma optimization
   - Extended trial budget if needed (75-100 trials)

6. Model 3: RandomForest Optimization
   - GPU-accelerated with cuML fallback
   - Tree count and depth optimization
   - Sample and feature sampling rates

7. Model 4: DecisionTree Optimization
   - Depth and split criteria
   - Regularization parameters

8. Results Comparison
   - Baseline vs Optimized table
   - Improvement metrics
   - Training time analysis

9. Best Models Persistence
   - Save optimized models (joblib)
   - Save Optuna studies (for resume)
   - Save scaler

10. Reflection Section (헌법 요구사항)
    - What did we learn?
    - Which models improved most?
    - What are the limitations?
    - Next steps for improvement

### Notebook 08: Feature Engineering
**Purpose**: Create engineered features to boost performance
**Sections**:
1. Feature Engineering Concepts
   - Why feature engineering matters
   - Types of features (polynomial, statistical, interactions)

2. Polynomial Features
   - QDA-focused (degree 2 interactions)
   - Feature selection to avoid explosion

3. Statistical Features
   - Rolling windows
   - Aggregations by feature groups

4. Interaction Features
   - Important feature pairs
   - Ratios and products

5. Feature Selection
   - Correlation analysis
   - Mutual information
   - Recursive feature elimination

6. Model Re-training
   - Apply optimized hyperparameters from notebook 07
   - Train on engineered features
   - Compare performance

7. Reflection
   - Feature importance analysis
   - Best feature combinations
   - Performance gains achieved

### Notebook 09: Final Ensemble and Predictions
**Purpose**: Create ensemble models and generate submissions
**Sections**:
1. Ensemble Concepts
   - Voting vs Stacking
   - When to use each approach

2. Load Optimized Models
   - From notebooks 07 and 08

3. Ensemble Creation
   - Hard voting (QDA + RF + DT)
   - Soft voting (weighted by CV scores)
   - Stacking with QDA meta-learner

4. Cross-Validation Evaluation
   - Compare ensemble vs individual models
   - Statistical significance testing

5. Test Set Predictions
   - Load test.csv (15,004 samples)
   - Generate predictions for each model
   - Generate ensemble predictions

6. Submission File Generation
   - Format: ID, target (competition format)
   - Create multiple submissions:
     * submission_qda_optimized.csv
     * submission_svc_optimized.csv
     * submission_rf_optimized.csv
     * submission_dt_optimized.csv
     * submission_ensemble_voting.csv
     * submission_ensemble_stacking.csv

7. Final Reflection
   - Overall journey and learnings
   - Performance summary
   - Future improvement ideas

## Implementation Constraints

### Memory Constraints
- Polynomial features can explode memory
- **Mitigation**: Use sparse matrices, PolynomialFeatures with interaction_only
- **Monitoring**: Track memory usage during feature engineering

### Time Constraints
- 50 trials × 4 models × 5 folds = 1000 model fits
- **Estimated Time**: 2-3 hours total (with GPU)
- **Mitigation**: Use trial pruning, GPU acceleration

### Reproducibility
- All random operations must use seed=42
- Document all library versions
- Save environment snapshot (uv export)

## Risk Mitigation

### Risk 1: SVC May Not Improve Significantly
**Likelihood**: Medium
**Impact**: Low (other models compensate)
**Mitigation**: Allocate more trials (75-100), try different kernels aggressively

### Risk 2: Feature Engineering May Hurt Performance
**Likelihood**: Low
**Impact**: Medium
**Mitigation**: Compare engineered vs raw features, use feature selection

### Risk 3: GPU Out of Memory
**Likelihood**: Low (RTX 3090 has 24GB)
**Impact**: Medium (falls back to CPU, slower)
**Mitigation**: Batch processing, graceful CPU fallback

### Risk 4: Optimization Time Exceeds Budget
**Likelihood**: Low
**Impact**: Low
**Mitigation**: Trial pruning, can stop early and use best so far

## References

1. **Existing Notebooks**:
   - 03_optuna_optimization.ipynb: Optuna patterns for LightGBM/XGBoost
   - 04_feature_engineering.ipynb: Feature engineering techniques
   - 06_model_comparison.ipynb: Baseline model results

2. **Optuna Documentation**:
   - TPESampler: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
   - Pruning: https://optuna.readthedocs.io/en/stable/reference/pruners.html

3. **cuML Documentation**:
   - GPU SVC: https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines
   - GPU RandomForest: https://docs.rapids.ai/api/cuml/stable/api.html#random-forest

4. **Scikit-learn Guides**:
   - QDA: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
   - Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html

## Next Steps

After this research phase, proceed to Phase 1 (Design & Contracts) to define:
1. Data model for optimization results
2. Contract specifications for prediction outputs
3. Quickstart guide for running optimization notebooks
4. Integration with existing notebook workflow

---
**Status**: Research Complete
**Next Phase**: Phase 1 - Design & Contracts