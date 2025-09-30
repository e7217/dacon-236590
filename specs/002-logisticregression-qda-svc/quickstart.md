# Quickstart: Multiple Model Evaluation

**Feature**: 002-logisticregression-qda-svc
**Purpose**: Step-by-step instructions to validate the model comparison feature
**Estimated Time**: 10-15 minutes

## Prerequisites

- Python 3.12 environment active
- Project dependencies installed (`pyproject.toml`)
- Competition dataset in `data/` directory
- Jupyter notebook server running

## Setup Steps

### 1. Verify Dependencies
```bash
# From project root
cd /home/e7217/projects/dacon-236590

# Check sklearn version (should be >= 1.3.0)
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"

# Verify all required models are available
python -c "from sklearn.linear_model import LogisticRegression; from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis; from sklearn.svm import SVC; from sklearn.ensemble import RandomForestClassifier; from sklearn.tree import DecisionTreeClassifier; print('All models available')"
```

### 2. Verify Dataset Availability
```bash
# Check data files exist
ls -lh data/*.csv

# Expected files:
# - train.csv (training data)
# - test.csv (test data for submission)
```

### 3. Open Evaluation Notebook
```bash
# Start Jupyter if not running
jupyter notebook

# Navigate to:
# notebooks/06_model_comparison.ipynb
```

## Validation Scenarios

### Scenario 1: Train All 5 Models Successfully

**User Story**: As a data scientist, I want to train multiple classification models on the same dataset

**Steps**:
1. Open `06_model_comparison.ipynb`
2. Run "Setup and Imports" cell
3. Run "Load Dataset" cell
4. Run "Configure Cross-Validation" cell (verify 5-fold stratified CV)
5. Run "Model Evaluation" cells for each model:
   - LogisticRegression
   - QuadraticDiscriminantAnalysis
   - SVC
   - RandomForestClassifier
   - DecisionTreeClassifier
6. Verify each model completes without errors

**Expected Outcome**:
- All 5 models train successfully
- Each model displays: "✓ {ModelName} evaluation complete"
- training_failed=False for all models

**Acceptance Criteria**:
- ✓ No exceptions raised during training
- ✓ Each model returns ModelEvaluationResult
- ✓ Training time recorded for each model

---

### Scenario 2: Evaluate with Stratified K-Fold Cross-Validation

**User Story**: As a data scientist, I want to use stratified k-fold CV to handle class imbalance

**Steps**:
1. In "Configure Cross-Validation" cell, verify:
   ```python
   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```
2. After training first model, check fold_scores length:
   ```python
   assert len(result.fold_scores) == 5
   ```
3. Verify class distribution maintained across folds:
   ```python
   # Cell should display class distribution per fold
   # Verify proportions are similar across all folds
   ```

**Expected Outcome**:
- Cross-validation uses exactly 5 folds
- Each fold maintains proportional class distribution
- All folds complete successfully

**Acceptance Criteria**:
- ✓ n_folds = 5 (constitutional requirement met)
- ✓ stratification='stratified'
- ✓ random_state = 42 (reproducibility)
- ✓ Class proportions similar across folds (±5%)

---

### Scenario 3: Display Performance Comparison Clearly

**User Story**: As a data scientist, I want to clearly identify which model performs best

**Steps**:
1. Run "Generate Comparison Report" cell
2. Verify comparison table displays all metrics:
   ```
   Model                         | Accuracy | Precision | Recall | F1-Score | Training Time
   ------------------------------|----------|-----------|--------|----------|---------------
   RandomForestClassifier        | 0.85±0.02| 0.84±0.03 | ...    | ...      | 12.3s
   SVC                           | 0.83±0.03| 0.82±0.02 | ...    | ...      | 8.7s
   ...
   ```
3. Verify visualizations are generated:
   - Grouped bar chart (all metrics)
   - Box plot (F1-score distribution)
4. Check best model identification:
   ```python
   print(f"Best model: {report.best_model_name}")
   print(f"Best F1-score: {report.rankings_by_f1[0][1]:.3f}")
   ```

**Expected Outcome**:
- Comparison table shows all 5 models with mean ± std
- Models ranked by F1-score (primary ranking)
- Visualizations clearly show performance differences
- Best model identified explicitly

**Acceptance Criteria**:
- ✓ All 4 metrics displayed for each model
- ✓ Mean and std shown (e.g., "0.85±0.02")
- ✓ Models sorted by F1-score descending
- ✓ Bar chart and box plot generated
- ✓ Best model name clearly stated

---

### Scenario 4: Handle Model Training Failures

**User Story**: As a data scientist, I want the system to handle training failures gracefully

**Steps**:
1. Simulate QDA failure (if dataset causes it):
   - QDA may fail on rank-deficient data
2. Verify error handling:
   - training_failed=True for failed model
   - failure_message contains exception text
   - All metrics are NaN for failed model
3. Verify other models continue evaluation
4. Check comparison report handles failed model:
   - Failed model shown in table with "Failed" status
   - Failed model excluded from rankings
   - Visualizations skip failed model

**Expected Outcome**:
- Failed model recorded with training_failed=True
- Other models complete successfully
- Comparison report generated despite failure

**Acceptance Criteria**:
- ✓ No crash when model fails
- ✓ failure_message populated
- ✓ Failed model metrics = NaN
- ✓ Other models unaffected
- ✓ Comparison report generated

---

## Verification Checklist

After running all scenarios:

### Functional Requirements Verification
- [ ] FR-001: LogisticRegression model trained ✓
- [ ] FR-002: QDA model trained ✓
- [ ] FR-003: SVC model trained ✓
- [ ] FR-004: RandomForest model trained ✓
- [ ] FR-004b: DecisionTree model trained ✓
- [ ] FR-005: All 4 metrics calculated (accuracy, precision, recall, F1) ✓
- [ ] FR-006: Performance comparison displayed ✓
- [ ] FR-007: Existing project dataset used ✓
- [ ] FR-008: Classification task confirmed ✓
- [ ] FR-009: Training time recorded ✓
- [ ] FR-010: Training failures handled gracefully ✓
- [ ] FR-011: Stratified k-fold CV (5-fold minimum) ✓
- [ ] FR-012: Results presented as table + visualizations ✓

### Constitutional Principles Verification
- [ ] Jupyter notebook-first implementation ✓
- [ ] Educational explanations included (10-year-old level) ✓
- [ ] Performance metrics prioritized ✓
- [ ] Best practices applied (sklearn patterns) ✓
- [ ] Reflection sections after each major step ✓

### Data Model Verification
- [ ] PerformanceMetrics: All 4 metrics calculated ✓
- [ ] ModelEvaluationResult: Complete results per model ✓
- [ ] CrossValidationRun: 5-fold stratified configuration ✓
- [ ] ComparisonReport: Rankings and visualizations ✓

## Troubleshooting

### Issue: QDA Training Fails
**Symptoms**: QuadraticDiscriminantAnalysis raises LinAlgError
**Cause**: Rank-deficient covariance matrix (common with QDA)
**Resolution**: This is expected behavior; verify:
- training_failed=True
- failure_message populated
- Other models continue normally

### Issue: SVC Takes Too Long
**Symptoms**: SVC training exceeds 5 minutes
**Cause**: Large dataset or expensive kernel
**Resolution**:
- Consider subsetting data for validation
- Or skip SVC temporarily and document in notebook

### Issue: Cross-Validation Folds Unbalanced
**Symptoms**: fold_scores vary widely (std > 0.10)
**Cause**: May indicate actual model instability, not necessarily error
**Resolution**: Review in reflection section; consider ensemble methods

## Next Steps After Validation

1. **Review Results**: Analyze which model performed best and why
2. **Reflection**: Document learnings in notebook reflection cells
3. **Hyperparameter Tuning**: Consider tuning best-performing models
4. **Ensemble Methods**: Explore combining top models
5. **Feature Engineering**: Use insights to improve features

---

**Validation Complete**: If all scenarios pass, feature is ready for production use

**Estimated Validation Time**: 10-15 minutes (longer if models are slow to train)