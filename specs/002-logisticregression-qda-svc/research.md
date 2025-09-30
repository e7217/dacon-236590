# Research: Multiple Model Evaluation

**Feature**: 002-logisticregression-qda-svc
**Date**: 2025-09-30
**Phase**: 0 - Outline & Research

## Research Questions

### 1. Output Format Preference (FR-12 Resolution)

**Question**: How should model comparison results be presented?

**Decision**: Combined approach with DataFrame table + visualizations
- DataFrame with all metrics for quick scanning
- Bar plots for metric comparison across models
- Box plots for cross-validation score distributions
- Heatmap for confusion matrices

**Rationale**:
- DataFrame provides precise numerical values for documentation
- Visualizations enable quick pattern recognition
- Multiple visualizations address different questions (overall performance vs. variability)
- Standard in scikit-learn documentation and Kaggle competitions

**Alternatives Considered**:
- Table only: Less intuitive for comparison
- Visualization only: Lacks precise values for reporting
- HTML report: Overkill for notebook environment

### 2. Hyperparameter Tuning Approach

**Question**: Should models use default parameters or optimized hyperparameters?

**Decision**: Use default parameters for initial comparison
- Fair baseline comparison across all models
- Simpler interpretation of inherent model strengths
- Faster execution for exploratory phase
- Document hyperparameter tuning as future improvement

**Rationale**:
- Purpose is model type comparison, not optimal performance
- Default parameters provide standardized starting point
- Existing notebooks (02, 03) already cover hyperparameter optimization
- Can iterate with tuning after identifying best model types

**Alternatives Considered**:
- Grid search for all: Too time-consuming, defeats comparison purpose
- Pre-optimized params: Introduces bias, unfair comparison
- Random search: Still time-intensive, adds variance

### 3. Cross-Validation Strategy

**Question**: How many folds for stratified k-fold cross-validation?

**Decision**: 5-fold stratified cross-validation
- Constitutional requirement: minimum 5-fold
- Balances bias-variance tradeoff
- Computationally manageable for 5 models
- Standard practice in scikit-learn documentation

**Rationale**:
- 5-fold provides reliable performance estimates
- Stratification maintains class distribution in each fold
- Sufficient for detecting significant performance differences
- Aligns with constitutional performance-first principle

**Alternatives Considered**:
- 10-fold: More robust but 2x computation time
- 3-fold: Faster but less reliable estimates
- Leave-one-out: Prohibitively expensive

### 4. Best Practices for Multi-Model Comparison

**Research Findings**:

**Metric Selection**:
- Accuracy: Overall correctness, good for balanced datasets
- Precision: Important when false positives are costly
- Recall: Important when false negatives are costly
- F1-score: Harmonic mean, good for imbalanced classes
- **Decision**: Use all four standard metrics for comprehensive view

**Statistical Comparison**:
- Mean + standard deviation across folds shows performance and stability
- Rank models by mean F1-score (best for imbalanced data)
- Report training time to assess computational cost
- **Decision**: Present mean ± std for each metric

**Visualization Best Practices**:
- Grouped bar chart: Compare metrics across models
- Box plot: Show cross-validation score distribution
- Radar chart: Multi-metric comparison per model
- **Decision**: Use grouped bar + box plots for clarity

**Reference Sources**:
- scikit-learn User Guide: Cross-validation section
- Kaggle Learn: Model Evaluation course
- "Python Data Science Handbook" by Jake VanderPlas
- "Hands-On Machine Learning" by Aurélien Géron

### 5. Handling Model Training Failures

**Research Findings**:

**Common Failure Modes**:
- QDA: May fail if classes don't have full rank covariance matrices
- SVC: Can be slow or memory-intensive on large datasets
- DecisionTree: Rarely fails but may overfit
- RandomForest: Memory-intensive, may fail on resource constraints

**Handling Strategy**:
```python
try:
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X, y, cv=cv_strategy)
except Exception as e:
    print(f"Model {model_name} failed: {e}")
    scores = np.array([np.nan] * n_folds)
    # Continue with other models
```

**Decision**:
- Wrap each model evaluation in try-except
- Record failure as NaN in results DataFrame
- Log exception message for debugging
- Continue evaluating remaining models
- Reflect on failures in notebook markdown

**Rationale**:
- Graceful degradation maintains workflow
- NaN handling in pandas preserves comparison structure
- Educational: shows real-world model limitations
- Performance-first: doesn't block other model evaluations

### 6. Implementation Patterns

**StratifiedKFold Implementation**:
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Metrics Calculation Pattern**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
```

**Cross-Validation Execution Pattern**:
```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                            return_train_score=False, error_score='raise')
```

**Decision**:
- Use sklearn's built-in cross_validate for efficiency
- Use 'weighted' averaging for multi-class metrics
- Set random_state=42 for reproducibility (constitutional requirement)
- Return train scores only if diagnosing overfitting

## Technology Stack Decisions

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Cross-validation | `sklearn.model_selection.StratifiedKFold` | Built-in, reliable, maintains class distribution |
| Metrics | `sklearn.metrics` classification functions | Standard, well-tested, supports multi-class |
| Visualization | `matplotlib` + `seaborn` | Already in dependencies, flexible, publication-quality |
| Results storage | `pandas.DataFrame` | Easy aggregation, sorting, export |
| Model implementations | `sklearn` (LogisticRegression, QDA, SVC, RandomForestClassifier, DecisionTreeClassifier) | All available in existing dependencies |

## Resolved Unknowns

| Unknown (from Technical Context) | Resolution |
|----------------------------------|------------|
| Output format preference (FR-12) | DataFrame table + bar plots + box plots |
| Hyperparameter tuning approach | Default parameters for fair comparison |
| Cross-validation fold count | 5-fold (constitutional minimum) |
| Metric averaging strategy | 'weighted' for multi-class support |
| Random seed value | 42 (project standard for reproducibility) |

## Educational Concepts to Explain

Per constitutional principle II (10-year-old explanation level):

1. **Stratified K-Fold Cross-Validation**:
   - Analogy: "Shuffling a deck of cards and dealing 5 piles, making sure each pile has the same mix of colors"
   - Purpose: Test model on data it hasn't seen, while keeping class balance

2. **Classification Metrics**:
   - Accuracy: "How many guesses were correct out of total guesses"
   - Precision: "When the model says YES, how often is it actually YES?"
   - Recall: "Out of all the actual YESes, how many did the model catch?"
   - F1-Score: "A balanced score between precision and recall"

3. **Model Types**:
   - LogisticRegression: "Draws a straight line to separate groups"
   - QDA: "Draws curved boundaries that fit each group's shape"
   - SVC: "Finds the widest road between different groups"
   - RandomForest: "Asks many simple questions and votes on the answer"
   - DecisionTree: "Follows a flowchart of yes/no questions"

## Performance Optimization Notes

**Computational Efficiency**:
- Use `n_jobs=-1` in models that support it (RandomForest, SVC)
- Cache cross-validation splits for consistency
- Measure and report training time per model
- Consider reducing dataset size for prototyping (document in notebook)

**Memory Efficiency**:
- Don't store all fold predictions unless needed for analysis
- Clear large objects after use in notebook cells
- Use pandas categorical dtype for target variable if appropriate

## Next Phase: Design

With all unknowns resolved, Phase 1 will create:
1. **data-model.md**: Entity definitions for evaluation results
2. **contracts/**: YAML specifications for evaluation interfaces
3. **quickstart.md**: Step-by-step validation instructions
4. **CLAUDE.md**: Updated agent context with evaluation framework

---
**Status**: ✅ Complete - All research questions resolved, ready for Phase 1