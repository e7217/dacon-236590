# Quickstart Guide: Hyperparameter Optimization Workflow

**Feature**: 003-test-csv-optuna
**Date**: 2025-10-01
**Purpose**: Step-by-step guide to execute optimization and generate submissions

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3090 recommended, 24GB VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 5GB free space for models and outputs
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Dependencies
Already configured in `pyproject.toml`. Install with:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Verify GPU Setup
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cuml; print('cuML imported successfully')"
```

Expected output:
```
CUDA available: True
cuML imported successfully
```

## Quick Start (5 Minutes)

### Option 1: Run All Notebooks Sequentially
```bash
cd notebooks

# 1. Optimization (creates optimized models)
jupyter nbconvert --execute --to notebook --inplace 07_hyperparameter_optimization.ipynb

# 2. Feature Engineering (optional, improves performance)
jupyter nbconvert --execute --to notebook --inplace 08_feature_engineering.ipynb

# 3. Ensemble and Predictions (generates submissions)
jupyter nbconvert --execute --to notebook --inplace 09_final_ensemble.ipynb
```

**Time Estimate**: 2-3 hours total
- Notebook 07: ~1.5-2 hours (optimization)
- Notebook 08: ~30-45 minutes (feature engineering)
- Notebook 09: ~10-15 minutes (ensemble + predictions)

### Option 2: Interactive Jupyter
```bash
jupyter lab
```
Navigate to:
1. `notebooks/07_hyperparameter_optimization.ipynb`
2. Run all cells (Cell → Run All)
3. Repeat for notebooks 08 and 09

## Workflow Overview

```
┌─────────────────────────────────────────┐
│ Notebook 07: Hyperparameter Optimization│
│ - Load train.csv (21,693 samples)       │
│ - Optimize QDA, SVC, RF, DT             │
│ - Save best models & Optuna studies     │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Notebook 08: Feature Engineering        │
│ - Create polynomial features            │
│ - Create statistical features           │
│ - Create interaction features           │
│ - Re-train with engineered features     │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Notebook 09: Ensemble & Predictions     │
│ - Load optimized models                 │
│ - Create voting & stacking ensembles    │
│ - Load test.csv (15,004 samples)        │
│ - Generate 7 submission files           │
└─────────────────────────────────────────┘
```

## Step-by-Step Guide

### Step 1: Prepare Data
Verify data files exist:
```bash
ls -lh data/open/
```
Expected files:
- `train.csv` (11.6 MB, 21,693 rows)
- `test.csv` (8.0 MB, 15,004 rows)
- `sample_submission.csv` (195 KB, 15,004 rows)

### Step 2: Run Hyperparameter Optimization

#### Open Notebook 07
```bash
jupyter lab notebooks/07_hyperparameter_optimization.ipynb
```

#### Key Sections to Understand

**Section 1: Introduction and Concepts**
- Read to understand Bayesian optimization
- Learn about trial pruning

**Section 2: Environment Setup**
- Verifies GPU availability
- Sets random seeds for reproducibility

**Section 3-6: Model Optimization**
- QDA: ~10-15 minutes (50 trials, CPU-only)
- SVC: ~30-45 minutes (50 trials, GPU-accelerated)
- RandomForest: ~20-30 minutes (50 trials, GPU-accelerated)
- DecisionTree: ~15-20 minutes (50 trials, CPU-only)

**Section 7: Results Comparison**
- Compare baseline vs optimized performance
- Identify which models improved most

**Section 8: Model Persistence**
- Saves optimized models to `models/`
- Saves Optuna studies for analysis

#### Expected Outputs
```
models/
├── optuna_study_qda.pkl
├── optuna_study_svc.pkl
├── optuna_study_rf.pkl
├── optuna_study_dt.pkl
├── qda_optimized.pkl
├── svc_optimized.pkl
├── rf_optimized.pkl
├── dt_optimized.pkl
└── scaler_optimized.pkl

outputs/
└── optimization_results.csv
```

#### Verify Success
Check final cell output for summary table:
```
Model           Baseline F1  Optimized F1  Improvement
QDA             0.8782       0.8856        +0.74%
SVC             0.3277       0.4521        +12.44%
RandomForest    0.7349       0.7601        +2.52%
DecisionTree    0.7105       0.7334        +2.29%
```

### Step 3: Run Feature Engineering (Optional)

**When to skip**: If time-constrained or baseline optimization already achieved target performance.

#### Open Notebook 08
```bash
jupyter lab notebooks/08_feature_engineering.ipynb
```

#### Key Sections

**Section 1: Feature Engineering Concepts**
- Understand polynomial, statistical, and interaction features

**Section 2-4: Feature Creation**
- Polynomial features: Creates ~1,300 features for QDA
- Statistical features: Creates ~30 aggregation features
- Interaction features: Creates ~90 feature pairs

**Section 5: Feature Selection**
- Removes redundant/low-importance features
- Reduces dimensionality if needed

**Section 6: Model Re-training**
- Trains optimized models on engineered features
- Compares raw vs engineered performance

#### Expected Additional Improvement
- QDA: +1-2% (88.5% → 89.5-90.0%)
- Tree models: +0.5-1.5%

#### Expected Outputs
```
models/
├── qda_optimized_engineered.pkl
├── rf_optimized_engineered.pkl
├── dt_optimized_engineered.pkl
└── feature_transformer.pkl

outputs/
└── feature_engineering_results.csv
```

### Step 4: Generate Ensemble and Predictions

#### Open Notebook 09
```bash
jupyter lab notebooks/09_final_ensemble.ipynb
```

#### Key Sections

**Section 1: Ensemble Concepts**
- Voting (hard/soft) vs Stacking
- When to use each approach

**Section 2: Load Optimized Models**
- Loads all models from notebook 07 (or 08 if available)

**Section 3: Create Ensembles**
- Hard voting: QDA + RF + DT (simple majority)
- Soft voting: Weighted by CV F1 scores
- Stacking: QDA as meta-learner on base predictions

**Section 4: Cross-Validation Evaluation**
- Compares ensemble performance to individual models
- Expected: Ensemble F1 > 0.890 (beats QDA's 0.8856)

**Section 5: Test Set Predictions**
- Loads test.csv (15,004 samples)
- Generates predictions for all models + ensembles

**Section 6: Submission File Generation**
- Creates 7 CSV files in competition format

#### Expected Outputs
```
outputs/submissions/
├── submission_qda_optimized.csv
├── submission_svc_optimized.csv
├── submission_rf_optimized.csv
├── submission_dt_optimized.csv
├── submission_ensemble_voting_hard.csv
├── submission_ensemble_voting_soft.csv
└── submission_ensemble_stacking.csv

models/
├── ensemble_voting_hard.pkl
├── ensemble_voting_soft.pkl
└── ensemble_stacking.pkl

outputs/
├── ensemble_results.csv
└── final_comparison.csv
```

#### Verify Submissions
```bash
# Check files exist
ls -lh outputs/submissions/

# Validate format
python -c "
import pandas as pd
for model in ['qda', 'svc', 'rf', 'dt', 'ensemble_voting_hard', 'ensemble_voting_soft', 'ensemble_stacking']:
    sub = pd.read_csv(f'outputs/submissions/submission_{model}_optimized.csv' if model in ['qda','svc','rf','dt'] else f'outputs/submissions/submission_{model}.csv')
    print(f'{model}: {len(sub)} rows, classes {sub[\"target\"].min()}-{sub[\"target\"].max()}')
    assert len(sub) == 15004, 'Wrong row count'
    assert sub['target'].between(0, 20).all(), 'Out of range predictions'
print('✅ All submissions valid!')
"
```

## Troubleshooting

### Issue 1: GPU Out of Memory
**Symptom**: `CUDA out of memory` error during SVC or RandomForest optimization

**Solution**:
```python
# In notebook, reduce batch size or use CPU fallback
# Option 1: Use CPU for problematic model
from sklearn.svm import SVC  # CPU version
model = SVC(random_state=42)

# Option 2: Reduce trial count
n_trials = 30  # instead of 50
```

### Issue 2: Optimization Takes Too Long
**Symptom**: Notebook 07 exceeds 3-hour estimate

**Solution**:
```python
# Reduce trial budget
n_trials = 30  # Faster, slightly less optimal

# Increase pruning aggressiveness
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,  # Start pruning earlier (was 5)
    n_warmup_steps=1     # Prune after 1 fold (was 2)
)
```

### Issue 3: Notebook Kernel Dies
**Symptom**: Kernel crashes during feature engineering

**Solution**:
```python
# Reduce polynomial degree
poly = PolynomialFeatures(degree=2, interaction_only=True)  # Keep interaction_only
# Don't use degree=3 (too many features)

# Or use sparse matrices
from scipy.sparse import csr_matrix
X_poly = csr_matrix(X_poly)  # Saves memory
```

### Issue 4: SVC Performance Still Poor
**Symptom**: SVC F1 < 0.45 after optimization

**Expected**: SVC is challenging for 21-class problems. Improvement to ~45% is acceptable.

**Optional Enhancement**:
```python
# Try One-vs-One instead of One-vs-Rest
from sklearn.multiclass import OneVsOneClassifier
svc_ovo = OneVsOneClassifier(SVC(**best_params))

# Or increase trial budget significantly
n_trials_svc = 100  # More exploration for difficult model
```

### Issue 5: Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'optuna'`

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Reinstall dependencies
uv sync --all-extras
```

## Performance Monitoring

### During Optimization
```python
# In notebook, monitor trial progress
print(f"Trial {trial.number}: F1 = {trial.value:.4f}")

# Check elapsed time periodically
import time
if time.time() - start_time > 7200:  # 2 hours
    study.stop()  # Manual stop if time limit reached
```

### After Completion
```bash
# Check optimization results
cat outputs/optimization_results.csv

# Check submission file sizes
ls -lh outputs/submissions/

# View prediction distributions
python -c "
import pandas as pd
for file in glob.glob('outputs/submissions/*.csv'):
    sub = pd.read_csv(file)
    print(f'{file}: {sub[\"target\"].value_counts().sort_index().to_dict()}')
"
```

## Next Steps

### 1. Submit to Competition
1. Choose best submission (likely `submission_ensemble_voting_soft.csv` or `submission_qda_optimized.csv`)
2. Upload to Dacon competition page
3. Check leaderboard score

### 2. Iterate if Needed
If leaderboard score < expected:
- Run notebook 08 (feature engineering) if skipped
- Try different ensemble combinations
- Increase optimization trials (75-100 per model)
- Experiment with advanced techniques (stacking with different meta-learners)

### 3. Document Results
Update README with:
- Final F1 scores (CV and leaderboard)
- Best performing model/ensemble
- Key learnings and insights

## Expected Timeline

| Phase | Time | Description |
|-------|------|-------------|
| Setup | 5 min | Install dependencies, verify GPU |
| Notebook 07 | 2 hours | Hyperparameter optimization (4 models × 50 trials) |
| Notebook 08 | 45 min | Feature engineering (optional) |
| Notebook 09 | 15 min | Ensemble creation and predictions |
| **Total** | **3 hours** | End-to-end workflow |

## Success Criteria

✅ **Minimum Success**:
- All 4 models optimized (QDA, SVC, RF, DT)
- All models show improvement or maintain baseline
- 7 valid submission files generated
- Ensemble F1 > 0.88

✅ **Target Success**:
- QDA F1 > 0.880
- RF F1 > 0.760
- DT F1 > 0.735
- Ensemble F1 > 0.890

✅ **Stretch Success**:
- Ensemble F1 > 0.895
- SVC F1 > 0.500 (significant improvement)
- Feature engineering adds +1-2% to QDA

## Validation Checklist

Before submission, verify:
- [ ] All notebooks executed without errors
- [ ] 7 submission CSV files exist in `outputs/submissions/`
- [ ] Each submission has exactly 15,004 rows
- [ ] All predictions in range [0, 20]
- [ ] No missing values (NaN) in submissions
- [ ] `optimization_results.csv` shows improvements
- [ ] Models saved in `models/` directory
- [ ] Ensemble F1 > best individual model F1

---
**Status**: Quickstart Guide Complete
**Time to Execute**: ~3 hours (including all notebooks)
**Primary Output**: 7 competition-ready submission files