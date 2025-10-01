# Contract: Optimization Results Format

**Feature**: 003-test-csv-optuna
**Date**: 2025-10-01
**Purpose**: Define standard format for documenting optimization results

## Optimization Summary CSV

### Purpose
Aggregate all model optimization results in a single comparable format.

### File Location
```
outputs/optimization_results.csv
```

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| model_name | string | Model identifier | "QDA", "SVC", "RandomForest", "DecisionTree" |
| baseline_f1_mean | float | F1-macro before optimization | 0.8782 |
| baseline_f1_std | float | Baseline standard deviation | 0.0029 |
| optimized_f1_mean | float | F1-macro after optimization | 0.8856 |
| optimized_f1_std | float | Optimized standard deviation | 0.0021 |
| improvement_pct | float | Percentage point improvement | 0.0074 (0.74%) |
| best_params | string | JSON string of best hyperparameters | '{"reg_param": 0.32}' |
| n_trials | int | Total optimization trials | 50 |
| n_completed | int | Successfully completed trials | 38 |
| n_pruned | int | Pruned trials | 10 |
| n_failed | int | Failed trials | 2 |
| optimization_time_sec | float | Total optimization time | 485.67 |
| best_trial_number | int | Trial number of best result | 23 |

### Example CSV
```csv
model_name,baseline_f1_mean,baseline_f1_std,optimized_f1_mean,optimized_f1_std,improvement_pct,best_params,n_trials,n_completed,n_pruned,n_failed,optimization_time_sec,best_trial_number
QDA,0.8782,0.0029,0.8856,0.0021,0.0074,"{""reg_param"": 0.32}",50,47,2,1,12.45,31
SVC,0.3277,0.0161,0.4521,0.0134,0.1244,"{""C"": 15.3, ""gamma"": 0.05, ""kernel"": ""rbf""}",50,38,10,2,485.67,23
RandomForest,0.7349,0.0014,0.7601,0.0018,0.0252,"{""n_estimators"": 347, ""max_depth"": 18}",50,45,4,1,312.89,42
DecisionTree,0.7105,0.0060,0.7334,0.0051,0.0229,"{""max_depth"": 12, ""min_samples_split"": 8}",50,48,1,1,89.12,19
```

### Generation Code
```python
def create_optimization_summary(studies_dict, baseline_results):
    """
    Create optimization results summary CSV.

    Parameters:
    -----------
    studies_dict : dict
        {model_name: optuna.Study object}
    baseline_results : dict
        {model_name: {'f1_mean': float, 'f1_std': float}}

    Returns:
    --------
    pd.DataFrame : Summary table
    ```"""
    import pandas as pd
    import json

    rows = []
    for model_name, study in studies_dict.items():
        # Get best trial
        best_trial = study.best_trial

        # Calculate statistics
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAILED]

        # Get baseline
        baseline = baseline_results[model_name]

        # Create row
        row = {
            'model_name': model_name,
            'baseline_f1_mean': baseline['f1_mean'],
            'baseline_f1_std': baseline['f1_std'],
            'optimized_f1_mean': study.best_value,
            'optimized_f1_std': best_trial.user_attrs.get('f1_std', 0.0),  # If stored
            'improvement_pct': study.best_value - baseline['f1_mean'],
            'best_params': json.dumps(study.best_params),
            'n_trials': len(study.trials),
            'n_completed': len(completed),
            'n_pruned': len(pruned),
            'n_failed': len(failed),
            'optimization_time_sec': sum(
                (t.datetime_complete - t.datetime_start).total_seconds()
                for t in completed if t.datetime_complete
            ),
            'best_trial_number': best_trial.number
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('optimized_f1_mean', ascending=False)  # Best first
    return df

# Usage
summary = create_optimization_summary(
    studies_dict={
        'QDA': qda_study,
        'SVC': svc_study,
        'RandomForest': rf_study,
        'DecisionTree': dt_study
    },
    baseline_results={
        'QDA': {'f1_mean': 0.8782, 'f1_std': 0.0029},
        'SVC': {'f1_mean': 0.3277, 'f1_std': 0.0161},
        'RandomForest': {'f1_mean': 0.7349, 'f1_std': 0.0014},
        'DecisionTree': {'f1_mean': 0.7105, 'f1_std': 0.0060}
    }
)

summary.to_csv('outputs/optimization_results.csv', index=False)
print(summary.to_string(index=False))
```

## Per-Model Detailed Results

### Purpose
Store complete trial history for each model for deep analysis.

### File Location
```
outputs/optimization_details_{model_name}.csv
```

### Schema

| Column | Type | Description |
|--------|------|-------------|
| trial_number | int | Trial identifier (0-indexed) |
| params_{param_name} | float/int/str | Each hyperparameter as separate column |
| fold_0_f1 | float | F1-macro for fold 0 |
| fold_1_f1 | float | F1-macro for fold 1 |
| fold_2_f1 | float | F1-macro for fold 2 |
| fold_3_f1 | float | F1-macro for fold 3 |
| fold_4_f1 | float | F1-macro for fold 4 |
| mean_f1 | float | Average across folds |
| std_f1 | float | Standard deviation across folds |
| training_time_sec | float | Total training time |
| state | string | "COMPLETE", "PRUNED", "FAILED" |
| prune_step | int | Fold where pruned (if pruned) |

### Example CSV (RandomForest)
```csv
trial_number,params_n_estimators,params_max_depth,params_min_samples_split,fold_0_f1,fold_1_f1,fold_2_f1,fold_3_f1,fold_4_f1,mean_f1,std_f1,training_time_sec,state,prune_step
0,100,10,2,0.7234,0.7198,0.7256,0.7189,0.7223,0.7220,0.0024,5.67,COMPLETE,
1,200,15,5,0.7345,0.7312,0.7367,0.7289,0.7334,0.7329,0.0028,8.92,COMPLETE,
2,150,8,3,0.7156,0.7123,,,,,0.0015,2.34,PRUNED,2
...
```

### Generation Code
```python
def export_trial_details(study, output_path):
    """
    Export all trial details to CSV.
    """
    import pandas as pd

    rows = []
    for trial in study.trials:
        row = {'trial_number': trial.number}

        # Add parameters
        for param_name, param_value in trial.params.items():
            row[f'params_{param_name}'] = param_value

        # Add fold scores if available
        if 'fold_scores' in trial.user_attrs:
            fold_scores = trial.user_attrs['fold_scores']
            for i, score in enumerate(fold_scores):
                row[f'fold_{i}_f1'] = score
            row['mean_f1'] = trial.value if trial.value else None
            row['std_f1'] = trial.user_attrs.get('f1_std')
        else:
            row['mean_f1'] = trial.value if trial.value else None

        # Add metadata
        row['training_time_sec'] = (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_complete else None
        )
        row['state'] = trial.state.name
        row['prune_step'] = trial.user_attrs.get('prune_step')

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

# Usage
export_trial_details(
    rf_study,
    'outputs/optimization_details_RandomForest.csv'
)
```

## Visualization Contract

### Required Visualizations
All optimization notebooks must generate these standard plots:

#### 1. Optimization History
**Purpose**: Show convergence over trials
**Format**: Line plot with trials on x-axis, F1-macro on y-axis
**Code**:
```python
optuna.visualization.plot_optimization_history(study)
```

#### 2. Parameter Importance
**Purpose**: Identify most impactful hyperparameters
**Format**: Horizontal bar chart
**Code**:
```python
optuna.visualization.plot_param_importances(study)
```

#### 3. Parallel Coordinate Plot
**Purpose**: Visualize high-dimensional parameter relationships
**Format**: Parallel coordinates with color by objective value
**Code**:
```python
optuna.visualization.plot_parallel_coordinate(study)
```

#### 4. Baseline vs Optimized Comparison
**Purpose**: Show improvement magnitude
**Format**: Grouped bar chart
**Code**:
```python
import matplotlib.pyplot as plt

models = list(summary['model_name'])
baseline = summary['baseline_f1_mean']
optimized = summary['optimized_f1_mean']

x = range(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width/2 for i in x], baseline, width, label='Baseline', alpha=0.8)
ax.bar([i + width/2 for i in x], optimized, width, label='Optimized', alpha=0.8)

ax.set_ylabel('F1-Macro Score')
ax.set_title('Model Performance: Baseline vs Optimized')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

## Persistence Requirements

### Required Files After Optimization
1. `optimization_results.csv` - Summary table
2. `optimization_details_{model}.csv` - Per-model trial history (4 files)
3. `optuna_study_{model}.pkl` - Serialized Optuna studies (4 files)
4. `{model}_optimized.pkl` - Trained models with best params (4 files)
5. `scaler_optimized.pkl` - Feature scaler

### Metadata Requirements
Each saved model must include metadata as attributes:
```python
import joblib

# Save model with metadata
model._metadata = {
    'model_name': 'RandomForest',
    'best_params': study.best_params,
    'cv_f1_mean': study.best_value,
    'cv_f1_std': 0.0018,
    'n_features': X.shape[1],
    'feature_names': list(X.columns),
    'optimization_trials': len(study.trials),
    'optimization_time': 312.89,
    'trained_on': datetime.now().isoformat(),
    'random_state': 42
}

joblib.dump(model, 'models/rf_optimized.pkl')

# Load with metadata
loaded_model = joblib.load('models/rf_optimized.pkl')
print(loaded_model._metadata)
```

---
**Status**: Optimization Results Contract Complete
**Used By**: Notebook 07 for tracking and reporting optimization outcomes