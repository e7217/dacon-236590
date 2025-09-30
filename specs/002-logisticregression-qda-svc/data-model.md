# Data Model: Multiple Model Evaluation

**Feature**: 002-logisticregression-qda-svc
**Date**: 2025-09-30
**Phase**: 1 - Design & Contracts

## Entity Definitions

### 1. PerformanceMetrics

**Purpose**: Represents standard classification performance metrics for a single evaluation

**Attributes**:
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| accuracy | float | Proportion of correct predictions | 0.0 ≤ value ≤ 1.0 |
| precision | float | Weighted average precision across classes | 0.0 ≤ value ≤ 1.0 |
| recall | float | Weighted average recall across classes | 0.0 ≤ value ≤ 1.0 |
| f1_score | float | Weighted average F1-score across classes | 0.0 ≤ value ≤ 1.0 |

**Lifecycle**:
- Created: After each fold evaluation or aggregated from fold scores
- Updated: Never (immutable once calculated)
- Deleted: After comparison report generation

**Business Rules**:
- All metrics must use 'weighted' averaging for multi-class support
- NaN values allowed only if model training failed
- F1-score is primary ranking metric for imbalanced classification

---

### 2. ModelEvaluationResult

**Purpose**: Complete evaluation results for a single model across all cross-validation folds

**Attributes**:
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| model_name | str | Name of the model (e.g., "LogisticRegression") | Non-empty, one of: LogisticRegression, QuadraticDiscriminantAnalysis, SVC, RandomForestClassifier, DecisionTreeClassifier |
| model_instance | sklearn estimator | Fitted model object | Must implement sklearn estimator interface |
| mean_metrics | PerformanceMetrics | Average metrics across all folds | All fields must be present |
| std_metrics | PerformanceMetrics | Standard deviation of metrics | All fields ≥ 0.0 |
| fold_scores | List[PerformanceMetrics] | Per-fold metric results | Length must equal n_folds |
| training_time_seconds | float | Total training time across folds | value > 0.0 |
| training_failed | bool | Whether model training encountered errors | Default: False |
| failure_message | str \| None | Error message if training failed | Non-null only if training_failed=True |

**Relationships**:
- Has-many: fold_scores → PerformanceMetrics (one per fold)
- Has-one: mean_metrics → PerformanceMetrics (aggregated)
- Has-one: std_metrics → PerformanceMetrics (aggregated)

**Lifecycle**:
- Created: After completing cross-validation for one model
- Updated: Never (results are immutable)
- Deleted: After inclusion in ComparisonReport

**Business Rules**:
- If training_failed=True, all metrics should be NaN
- training_time includes all folds' training time
- fold_scores length must match cv strategy n_folds
- Random state must be consistent across all folds (42)

---

### 3. CrossValidationRun

**Purpose**: Configuration and execution context for stratified k-fold cross-validation

**Attributes**:
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| n_folds | int | Number of cross-validation folds | value ≥ 5 (constitutional requirement) |
| stratification_strategy | str | How to maintain class distribution | Must be "stratified" |
| random_state | int | Seed for reproducibility | Must be 42 (project standard) |
| shuffle | bool | Whether to shuffle data before splitting | Must be True |
| dataset_shape | tuple[int, int] | (n_samples, n_features) of input data | Both values > 0 |
| target_classes | List[int \| str] | Unique class labels in target variable | Non-empty list |
| class_distribution | dict | Count of samples per class | Keys match target_classes |

**Lifecycle**:
- Created: Once at start of evaluation pipeline
- Updated: Never (immutable configuration)
- Used by: All ModelEvaluationResult instances

**Business Rules**:
- Must verify dataset has sufficient samples per class (min 5 per class for 5-fold)
- Stratification ensures each fold has proportional class representation
- random_state=42 ensures reproducibility across runs

---

### 4. ComparisonReport

**Purpose**: Aggregated comparison of all model evaluation results with rankings and visualizations

**Attributes**:
| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| all_results | List[ModelEvaluationResult] | Complete results for all evaluated models | Length = 5 (one per model type) |
| rankings_by_f1 | List[tuple[str, float]] | Models ranked by mean F1-score (desc) | Sorted list of (model_name, f1_score) |
| rankings_by_accuracy | List[tuple[str, float]] | Models ranked by mean accuracy (desc) | Sorted list of (model_name, accuracy) |
| best_model_name | str | Name of top-performing model (by F1) | Must be in all_results |
| comparison_table | pd.DataFrame | All metrics in tabular format | Rows=models, Columns=metrics |
| visualization_paths | dict[str, str] | Paths to generated plot files | Keys: 'bar_chart', 'box_plot' |
| generated_at | datetime | Timestamp of report generation | ISO 8601 format |

**Relationships**:
- Has-many: all_results → ModelEvaluationResult
- Aggregates: rankings from all_results
- References: best_model_name → one of all_results

**Lifecycle**:
- Created: After all models evaluated
- Updated: Never (immutable report)
- Persisted: In notebook output and optionally to disk

**Business Rules**:
- rankings_by_f1 is primary ranking (constitutional: performance-first)
- comparison_table must include mean ± std for all metrics
- Must handle NaN values for failed models gracefully
- Visualizations must be clear and publication-quality

---

## Data Flow Diagram

```
[Input Dataset]
    ↓
[CrossValidationRun Configuration]
    ↓
[For each model type:]
    ↓
[StratifiedKFold Split] → [Train/Test Folds]
    ↓
[For each fold:]
    ↓
[Model Training] → [Predictions] → [PerformanceMetrics]
    ↓
[Aggregate fold results] → [ModelEvaluationResult]
    ↓
[All models complete]
    ↓
[ComparisonReport] → [DataFrame + Visualizations]
```

## Validation Rules Summary

**Cross-Validation Configuration**:
- n_folds ≥ 5 (constitutional requirement)
- random_state = 42 (project standard)
- stratification = True (always for classification)

**Metrics Calculation**:
- All metrics use 'weighted' averaging for multi-class
- Values in range [0.0, 1.0] or NaN (if training failed)
- F1-score is primary ranking metric

**Result Integrity**:
- Immutable results (no post-calculation modification)
- Training time > 0 for successful runs
- fold_scores.length == n_folds

**Comparison Report**:
- Must include exactly 5 models (per spec)
- Ranked by F1-score (performance-first principle)
- Handles NaN gracefully (failed models shown but not ranked)

## Educational Notes

**For 10-Year-Old Understanding**:

- **PerformanceMetrics**: "Like a report card with 4 different grades (accuracy, precision, recall, F1) showing how well the model did"

- **ModelEvaluationResult**: "A complete test report for one student (model), including their average grade, how consistent they were, and how long they studied"

- **CrossValidationRun**: "The rules for the test: divide the class into 5 groups, make sure each group has a mix of all types of students, use the same mixing method every time (random_state=42)"

- **ComparisonReport**: "A comparison poster showing all 5 students' grades side-by-side, with a trophy for the best performer and graphs to see differences easily"

---

**Status**: ✅ Complete - All entities defined with validation rules and relationships