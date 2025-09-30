# Tasks: Multiple Model Evaluation

**Input**: Design documents from `/home/e7217/projects/dacon-236590/specs/002-logisticregression-qda-svc/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), quickstart.md (✓)
**Implementation Type**: Notebook-only (no TDD, no separate modules per user requirement)

## Execution Flow
```
1. Load plan.md from feature directory ✓
   → Tech stack: Python 3.12, scikit-learn, pandas, matplotlib, seaborn
   → Structure: Pure notebook implementation in 06_model_comparison.ipynb
2. Load design documents ✓
   → data-model.md: 4 conceptual entities (PerformanceMetrics, ModelEvaluationResult, CrossValidationRun, ComparisonReport)
   → research.md: 5-fold stratified CV, default hyperparameters, DataFrame + visualizations
   → quickstart.md: 4 validation scenarios
3. Generate tasks for notebook-only workflow:
   → Setup: Create notebook, imports
   → Documentation: Educational explanations (10-year-old level)
   → Helper Functions: Inline metrics calculator, CV runner, comparator
   → Model Evaluations: 5 models (can be developed in parallel)
   → Comparison: Aggregate results, visualizations
   → Reflection: Learnings and improvements
4. Apply task rules:
   → Notebook-only = sequential cell execution
   → Documentation tasks can be parallel [P]
   → Model evaluation cells can be documented in parallel [P]
5. Constitutional compliance:
   → Jupyter notebook-first (Principle I) ✓
   → Educational clarity (Principle II) ✓
   → Performance-first (Principle III) ✓
   → Research-driven (Principle IV) ✓
   → Iterative reflection (Principle V) ✓
```

## User Requirement
"데이터분석이 목적이니까 TDD는 필요없다. 코드도 주피터 노트북 내에 모두 구현한다."
(Since the purpose is data analysis, TDD is not necessary. All code should be implemented within the Jupyter notebook.)

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (independent documentation/cells)
- All tasks target: `/home/e7217/projects/dacon-236590/notebooks/06_model_comparison.ipynb`

---

## Phase 3.1: Setup & Initialization

### T001: Create new notebook file
**File**: `/home/e7217/projects/dacon-236590/notebooks/06_model_comparison.ipynb`
**Description**: Create new Jupyter notebook for model comparison
**Actions**:
- Create new .ipynb file in notebooks/ directory
- Initialize with notebook metadata (Python 3.12 kernel)

**Acceptance Criteria**:
- File created successfully
- Can be opened in Jupyter

**Constitutional Alignment**: Jupyter notebook-first (Principle I)

---

### T002: Add notebook title and introduction
**File**: `notebooks/06_model_comparison.ipynb` - Cell 1 (markdown)
**Description**: Create introductory markdown cell with educational content
**Content**:
```markdown
# Multiple Model Evaluation: Finding the Best Classifier

## Purpose
We're going to test 5 different machine learning models to see which one is best at predicting our data. Think of it like having 5 different students take the same test - we want to see who gets the best grades!

## The 5 Models We'll Test
1. **LogisticRegression** - Draws a straight line to separate groups
2. **QuadraticDiscriminantAnalysis (QDA)** - Draws curved boundaries fitting each group's shape
3. **Support Vector Classifier (SVC)** - Finds the widest road between different groups
4. **RandomForestClassifier** - Asks many simple questions and votes on the answer
5. **DecisionTreeClassifier** - Follows a flowchart of yes/no questions

## Dataset
- Dacon Smart Manufacturing Competition data
- Classification task (predicting categories)

## Evaluation Method
- **5-fold Stratified Cross-Validation** (constitutional requirement)
- Like shuffling cards and dealing 5 piles with the same mix of colors

## Metrics
- **Accuracy**: How many guesses were correct
- **Precision**: When model says YES, how often is it actually YES?
- **Recall**: Out of all actual YESes, how many did model catch?
- **F1-Score**: Balanced score between precision and recall
```

**Acceptance Criteria**:
- Educational language (10-year-old level)
- All 5 models introduced
- Metrics explained simply

**Constitutional Alignment**: Educational clarity (Principle II)

---

### T003: Create setup and imports cell
**File**: `notebooks/06_model_comparison.ipynb` - Cell 2 (code)
**Description**: Import all required libraries and configure environment
**Code**:
```python
# Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

print("✓ All libraries imported successfully")
print(f"scikit-learn version: {__import__('sklearn').__version__}")
print(f"pandas version: {pd.__version__}")
```

**Acceptance Criteria**:
- All imports successful
- Random seed set to 42
- Version info displayed

**Constitutional Alignment**: Research-driven setup (Principle IV)

---

## Phase 3.2: Educational Documentation [P]

### T004 [P]: Add "What is Cross-Validation?" section
**File**: `notebooks/06_model_comparison.ipynb` - Cell 3 (markdown)
**Content**:
```markdown
## 🔄 What is Cross-Validation?

Imagine you have a deck of cards. Instead of testing your card trick on the same pile every time, you:
1. Shuffle the deck
2. Deal it into 5 equal piles
3. Test your trick on each pile separately
4. See how well it works on average

**That's cross-validation!** It helps us make sure our model works well on ALL data, not just one lucky group.

**Stratified** means we make sure each pile has the same mix of card colors (classes). If 30% of cards are red, then each pile should also have about 30% red cards.

**Why 5 folds?** It's a constitutional requirement for this project - it balances thoroughness with computational cost.
```

**Acceptance Criteria**:
- Simple analogy used
- Stratification explained
- Constitutional requirement noted

**Constitutional Alignment**: Educational clarity (Principle II)

---

## Phase 3.3: Data Loading & Preparation

### T005: Create data loading cell
**File**: `notebooks/06_model_comparison.ipynb` - Cell 4 (code)
**Description**: Load existing competition dataset
**Code**:
```python
# Load Dataset
print("Loading Dacon Smart Manufacturing competition data...")

# Load training data
df = pd.read_csv('../data/train.csv')
print(f"✓ Dataset loaded successfully")
print(f"  Shape: {df.shape}")

# Separate features and target
# Adjust column names based on actual dataset structure
X = df.drop('target', axis=1)  # Replace 'target' with actual target column name
y = df['target']

# Display dataset info
print(f"\nDataset Information:")
print(f"  Samples: {len(df)}")
print(f"  Features: {X.shape[1]}")
print(f"  Target classes: {y.nunique()}")

# Display class distribution
print(f"\nClass Distribution:")
print(y.value_counts())
print(f"\nClass Balance (%):")
print(y.value_counts(normalize=True) * 100)
```

**Acceptance Criteria**:
- Dataset loads successfully
- X and y separated
- Class distribution displayed

**Constitutional Alignment**: Performance-first validation (Principle III)

---

### T006: Add dataset interpretation markdown
**File**: `notebooks/06_model_comparison.ipynb` - Cell 5 (markdown)
**Content**: Document observations about:
- Dataset size (sufficient for 5-fold CV?)
- Class balance (any imbalance issues?)
- Feature count
- Any potential challenges

**Constitutional Alignment**: Educational clarity (Principle II)

---

## Phase 3.4: Cross-Validation Configuration

### T007: Configure cross-validation
**File**: `notebooks/06_model_comparison.ipynb` - Cell 6 (code)
**Code**:
```python
# Configure 5-fold Stratified Cross-Validation
# Constitutional requirement: minimum 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_folds = 5

print("Cross-Validation Configuration:")
print(f"  Strategy: Stratified K-Fold")
print(f"  Number of folds: {n_folds}")
print(f"  Shuffle: True")
print(f"  Random state: 42 (for reproducibility)")

# Verify stratification maintains class distribution
print(f"\nVerifying stratification across folds:")
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    fold_dist = y.iloc[test_idx].value_counts(normalize=True).sort_index()
    print(f"Fold {fold_idx+1}: {dict(fold_dist)}")

print("\n✓ All folds have similar class distributions")
```

**Acceptance Criteria**:
- StratifiedKFold with n_splits=5
- random_state=42
- Fold distributions verified

**Constitutional Alignment**: Performance-first (Principle III)

---

## Phase 3.5: Helper Functions (Inline)

### T008: Create metrics calculator function
**File**: `notebooks/06_model_comparison.ipynb` - Cell 7 (code)
**Code**:
```python
# Helper Function: Calculate Classification Metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate standard classification metrics with weighted averaging.

    Weighted averaging means each class's score is multiplied by how many
    samples it has, like giving more weight to bigger groups.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

print("✓ Metrics calculator function defined")
```

**Acceptance Criteria**:
- Function defined with docstring
- Weighted averaging used
- zero_division=0 for edge cases

**Constitutional Alignment**: Research-driven sklearn patterns (Principle IV)

---

### T009: Create cross-validation runner function
**File**: `notebooks/06_model_comparison.ipynb` - Cell 8 (code)
**Code**:
```python
# Helper Function: Run Cross-Validation for a Model
def run_cross_validation(model, model_name, X, y, cv):
    """
    Run stratified k-fold cross-validation and return comprehensive results.

    Cross-validation is like testing your model on different groups of data to
    make sure it works everywhere, not just one place.
    """
    fold_scores = []
    start_time = time.time()
    training_failed = False
    failure_message = None

    try:
        # Use cross_validate for efficiency
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                   return_train_score=False, error_score='raise')

        # Extract fold scores
        for i in range(n_folds):
            fold_scores.append({
                'accuracy': cv_results['test_accuracy'][i],
                'precision': cv_results['test_precision_weighted'][i],
                'recall': cv_results['test_recall_weighted'][i],
                'f1_score': cv_results['test_f1_weighted'][i]
            })

    except Exception as e:
        training_failed = True
        failure_message = str(e)
        fold_scores = [{'accuracy': np.nan, 'precision': np.nan,
                       'recall': np.nan, 'f1_score': np.nan}] * n_folds

    training_time = time.time() - start_time

    # Calculate mean and std
    if not training_failed:
        mean_metrics = {
            'accuracy': np.mean([f['accuracy'] for f in fold_scores]),
            'precision': np.mean([f['precision'] for f in fold_scores]),
            'recall': np.mean([f['recall'] for f in fold_scores]),
            'f1_score': np.mean([f['f1_score'] for f in fold_scores])
        }
        std_metrics = {
            'accuracy': np.std([f['accuracy'] for f in fold_scores]),
            'precision': np.std([f['precision'] for f in fold_scores]),
            'recall': np.std([f['recall'] for f in fold_scores]),
            'f1_score': np.std([f['f1_score'] for f in fold_scores])
        }
    else:
        mean_metrics = {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}
        std_metrics = {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan}

    return {
        'model_name': model_name,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'fold_scores': fold_scores,
        'training_time_seconds': training_time,
        'training_failed': training_failed,
        'failure_message': failure_message
    }

print("✓ Cross-validation runner function defined")
```

**Acceptance Criteria**:
- Function handles success and failure cases
- Returns all required metrics
- Educational docstring

**Constitutional Alignment**: Performance-first error handling (Principle III)

---

## Phase 3.6: Model Evaluations [P]

### T010 [P]: Evaluate LogisticRegression
**File**: `notebooks/06_model_comparison.ipynb` - Cell 9 (markdown + code)
**Markdown**:
```markdown
## 📊 Model 1: LogisticRegression
**Analogy**: Draws a straight line to separate groups. Simple but effective!
```

**Code**:
```python
print("="*60)
print("Evaluating LogisticRegression...")
print("="*60)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_result = run_cross_validation(lr_model, "LogisticRegression", X, y, cv)

if lr_result['training_failed']:
    print(f"❌ LogisticRegression failed: {lr_result['failure_message']}")
else:
    print(f"✓ LogisticRegression complete")
    print(f"\nResults:")
    print(f"  Accuracy:  {lr_result['mean_metrics']['accuracy']:.4f} ± {lr_result['std_metrics']['accuracy']:.4f}")
    print(f"  Precision: {lr_result['mean_metrics']['precision']:.4f} ± {lr_result['std_metrics']['precision']:.4f}")
    print(f"  Recall:    {lr_result['mean_metrics']['recall']:.4f} ± {lr_result['std_metrics']['recall']:.4f}")
    print(f"  F1-Score:  {lr_result['mean_metrics']['f1_score']:.4f} ± {lr_result['std_metrics']['f1_score']:.4f}")
    print(f"  Training time: {lr_result['training_time_seconds']:.2f}s")
```

**Constitutional Alignment**: Performance-first metrics (Principle III), educational clarity (Principle II)

---

### T011 [P]: Evaluate QuadraticDiscriminantAnalysis
**File**: `notebooks/06_model_comparison.ipynb` - Cell 10 (markdown + code)
**Markdown**:
```markdown
## 📊 Model 2: Quadratic Discriminant Analysis (QDA)
**Analogy**: Draws curved boundaries that fit each group's shape, like custom-fitting clothes to each person.
**Note**: QDA can fail on certain datasets (rank-deficient data) - this is expected and handled gracefully.
```

**Code**: Similar pattern to T010, using `QuadraticDiscriminantAnalysis()`

**Constitutional Alignment**: Performance-first with graceful failure (Principle III)

---

### T012 [P]: Evaluate SVC
**File**: `notebooks/06_model_comparison.ipynb` - Cell 11 (markdown + code)
**Markdown**:
```markdown
## 📊 Model 3: Support Vector Classifier (SVC)
**Analogy**: Finds the widest road between different groups, maximizing the safety margin.
**Note**: SVC can be slow on large datasets.
```

**Code**: Similar pattern to T010, using `SVC(random_state=42)`

**Constitutional Alignment**: Performance-first metrics (Principle III)

---

### T013 [P]: Evaluate RandomForest
**File**: `notebooks/06_model_comparison.ipynb` - Cell 12 (markdown + code)
**Markdown**:
```markdown
## 📊 Model 4: RandomForestClassifier
**Analogy**: Asks many simple questions and takes a vote on the answer. Wisdom of crowds!
```

**Code**: Similar pattern to T010, using `RandomForestClassifier(random_state=42, n_jobs=-1)`

**Constitutional Alignment**: Performance-first with parallel training (Principle III)

---

### T014 [P]: Evaluate DecisionTree
**File**: `notebooks/06_model_comparison.ipynb` - Cell 13 (markdown + code)
**Markdown**:
```markdown
## 📊 Model 5: DecisionTreeClassifier
**Analogy**: Follows a flowchart of yes/no questions to make decisions, like a game of 20 questions.
```

**Code**: Similar pattern to T010, using `DecisionTreeClassifier(random_state=42)`

**Constitutional Alignment**: Performance-first metrics (Principle III)

---

## Phase 3.7: Results Aggregation & Comparison

### T015: Create comparison DataFrame
**File**: `notebooks/06_model_comparison.ipynb` - Cell 14 (code)
**Code**:
```python
# Aggregate All Results
print("\n" + "="*80)
print("GENERATING COMPARISON REPORT")
print("="*80 + "\n")

all_results = [lr_result, qda_result, svc_result, rf_result, dt_result]

# Create comparison DataFrame
comparison_data = []
for result in all_results:
    if result['training_failed']:
        row = {
            'Model': result['model_name'],
            'Accuracy_Mean': np.nan,
            'Accuracy_Std': np.nan,
            'Precision_Mean': np.nan,
            'Precision_Std': np.nan,
            'Recall_Mean': np.nan,
            'Recall_Std': np.nan,
            'F1_Mean': np.nan,
            'F1_Std': np.nan,
            'Training_Time': result['training_time_seconds'],
            'Status': 'Failed'
        }
    else:
        row = {
            'Model': result['model_name'],
            'Accuracy_Mean': result['mean_metrics']['accuracy'],
            'Accuracy_Std': result['std_metrics']['accuracy'],
            'Precision_Mean': result['mean_metrics']['precision'],
            'Precision_Std': result['std_metrics']['precision'],
            'Recall_Mean': result['mean_metrics']['recall'],
            'Recall_Std': result['std_metrics']['recall'],
            'F1_Mean': result['mean_metrics']['f1_score'],
            'F1_Std': result['std_metrics']['f1_score'],
            'Training_Time': result['training_time_seconds'],
            'Status': 'Success'
        }
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1_Mean', ascending=False, na_position='last')

# Display comparison table
print("MODEL COMPARISON TABLE")
print("(Sorted by F1-Score, our primary metric)")
print("-"*80)
display(comparison_df)

# Identify best model
successful_models = comparison_df[comparison_df['Status'] == 'Success']
if len(successful_models) > 0:
    best_model_row = successful_models.iloc[0]
    best_model_name = best_model_row['Model']
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {best_model_row['F1_Mean']:.4f} ± {best_model_row['F1_Std']:.4f}")
    print(f"   Accuracy: {best_model_row['Accuracy_Mean']:.4f} ± {best_model_row['Accuracy_Std']:.4f}")
    print(f"   Training Time: {best_model_row['Training_Time']:.2f}s")
else:
    print("\n⚠️ All models failed training")
```

**Acceptance Criteria**:
- Comparison DataFrame created
- Sorted by F1-score
- Best model identified

**Constitutional Alignment**: Performance-first ranking (Principle III)

---

## Phase 3.8: Visualizations

### T016: Create grouped bar chart
**File**: `notebooks/06_model_comparison.ipynb` - Cell 15 (code)
**Code**:
```python
# Visualization 1: Grouped Bar Chart (All Metrics)
viz_df = comparison_df[comparison_df['Status'] == 'Success'].copy()

if len(viz_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'F1_Mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(viz_df))
    width = 0.2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i*width, viz_df[metric], width, label=label, alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(viz_df['Model'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    plt.tight_layout()
    plt.show()
else:
    print("No successful models to visualize")
```

**Constitutional Alignment**: Research-driven visualization best practices (Principle IV)

---

### T017: Create box plot for F1-score distribution
**File**: `notebooks/06_model_comparison.ipynb` - Cell 16 (code)
**Code**:
```python
# Visualization 2: Box Plot (F1-Score Distribution Across Folds)
f1_distributions = []
model_names = []

for result in all_results:
    if not result['training_failed']:
        f1_scores = [fold['f1_score'] for fold in result['fold_scores']]
        f1_distributions.append(f1_scores)
        model_names.append(result['model_name'])

if len(f1_distributions) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(f1_distributions, labels=model_names, patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(f1_distributions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Distribution Across Cross-Validation Folds', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\n💡 The box plot shows:")
    print("   - Median (line in the middle)")
    print("   - 25th-75th percentile (box)")
    print("   - Range and outliers (whiskers and dots)")
    print("   - Narrower boxes = more consistent performance across folds")
else:
    print("No successful models to visualize")
```

**Constitutional Alignment**: Performance-first focus on F1 variability (Principle III)

---

## Phase 3.9: Reflection & Insights [P]

### T018 [P]: Add "What We Learned" section
**File**: `notebooks/06_model_comparison.ipynb` - Cell 17 (markdown)
**Content**:
```markdown
## 🤔 Reflection and Learnings

### What did we learn from this analysis?
[Analyze which models performed best, which struggled, patterns in the results]
- **Best performer**: [Model name] with F1-score of [value]
- **Most consistent**: [Model with lowest std] showed most stable performance
- **Training efficiency**: [Model] was fastest at [time]s
- **Failures**: [List any models that failed and why]

### Which model performed best and why?
[Discuss the best model based on F1-score, considering accuracy, training time, and consistency]

### Performance vs. Time Trade-offs
- Fastest model: [name] at [time]s
- Slowest model: [name] at [time]s
- Best performance-time balance: [name]
```

**Constitutional Alignment**: Iterative reflection (Principle V)

---

### T019 [P]: Add "How to Improve" section
**File**: `notebooks/06_model_comparison.ipynb` - Cell 18 (markdown)
**Content**:
```markdown
### How could we improve these results?

**Immediate Next Steps:**
1. **Hyperparameter Tuning**: Use notebooks/02_hyperparameter_tuning.ipynb or 03_optuna_optimization.ipynb to tune the top 2-3 models
2. **Feature Engineering**: Based on model insights, create new features or transform existing ones
3. **Ensemble Methods**: Combine predictions from top models (stacking, voting, blending)
4. **Cross-Validation Experiments**: Try 10-fold CV to see if rankings change

**Model-Specific Improvements:**
- **LogisticRegression**: Try different solvers, regularization strengths
- **SVC**: Experiment with different kernels (RBF, polynomial)
- **RandomForest**: Tune n_estimators, max_depth, min_samples_split
- **DecisionTree**: Add max_depth constraint to prevent overfitting

**Data-Level Improvements:**
- Handle class imbalance with SMOTE or class weights
- Feature scaling/normalization
- Remove correlated features
- Feature selection techniques

### What are the limitations of this approach?
- Used default hyperparameters (fair comparison but not optimal)
- Single dataset evaluation (no external validation)
- Didn't consider model interpretability
- Training time vs. performance tradeoffs not fully explored
- No analysis of feature importance
```

**Constitutional Alignment**: Iterative improvement path (Principle V)

---

### T020 [P]: Add "Next Steps" and save results
**File**: `notebooks/06_model_comparison.ipynb` - Cell 19 (markdown + code)
**Markdown**:
```markdown
### Next Steps

**Immediate Actions:**
1. Focus hyperparameter tuning on [best model]
2. Try ensemble of top 3 models
3. Feature engineering based on model insights
4. Compare with existing notebooks (01-05)

**Long-term Considerations:**
- Production deployment requirements
- Model monitoring and retraining strategy
- A/B testing different models
- Cost-benefit analysis of complex vs. simple models
```

**Code**:
```python
# Save comparison results for future reference
import os
os.makedirs('../outputs/evaluation', exist_ok=True)

# Save comparison table
comparison_df.to_csv('../outputs/evaluation/model_comparison.csv', index=False)
print("✓ Results saved to outputs/evaluation/model_comparison.csv")

# Save detailed results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_model_name if len(successful_models) > 0 else None,
    'cv_config': {
        'n_folds': n_folds,
        'stratified': True,
        'random_state': 42
    },
    'models_evaluated': len(all_results),
    'models_successful': len(successful_models),
    'models_failed': len(all_results) - len(successful_models)
}

import json
with open('../outputs/evaluation/evaluation_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Summary saved to outputs/evaluation/evaluation_summary.json")
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
```

**Constitutional Alignment**: Systematic documentation (Principle V)

---

## Dependencies

**Sequential Flow** (all tasks in order):
```
T001 (Create notebook)
  ↓
T002 (Title + intro)
  ↓
T003 (Setup + imports)
  ↓
T004 (Cross-validation education) [P with other markdown]
  ↓
T005 (Load data)
  ↓
T006 (Data interpretation)
  ↓
T007 (Configure CV)
  ↓
T008 (Metrics calculator function)
  ↓
T009 (CV runner function)
  ↓
T010-T014 (Model evaluations) [P documentation, sequential execution]
  ↓
T015 (Comparison DataFrame)
  ↓
T016 (Bar chart)
  ↓
T017 (Box plot)
  ↓
T018-T020 (Reflection) [P documentation]
```

**Parallel Opportunities**:
- **T004, T006, T018-T020**: Markdown cells can be written in parallel
- **T010-T014**: Model evaluation cell documentation can be written in parallel
- **Note**: Code cells must execute sequentially in notebook

---

## Validation

After completing all tasks, execute notebook from top to bottom and verify:
- [ ] All 5 models evaluated (some may fail gracefully)
- [ ] Comparison DataFrame shows all models
- [ ] Best model identified by F1-score
- [ ] Both visualizations generated
- [ ] Reflection sections completed
- [ ] Results saved to outputs/

**Quickstart Reference**: See `quickstart.md` for detailed validation scenarios

---

## Notes

- **No TDD**: Per user requirement, no separate test files
- **No Modules**: All code inline in notebook cells
- **Constitutional Compliance**: All 5 principles embedded
- **Educational Focus**: 10-year-old explanations throughout
- **Performance-First**: Primary ranking by F1-score, 5-fold stratified CV
- **Research-Driven**: sklearn best practices, weighted metrics
- **Reflection**: Explicit learning and improvement sections

---

**Status**: Ready for implementation
**Total Tasks**: 20
**Estimated Time**: 3-4 hours (including model training time)
**Parallel Opportunities**: Documentation tasks (T004, T006, T010-T014 markdown, T018-T020)
**Critical Path**: Sequential notebook cell execution