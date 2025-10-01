# Contract: Competition Submission Format

**Feature**: 003-test-csv-optuna
**Date**: 2025-10-01
**Purpose**: Define output format for Dacon competition submissions

## Submission File Format

### File Structure
- **Format**: CSV (comma-separated values)
- **Encoding**: UTF-8
- **Line Ending**: Unix-style (LF)
- **Header**: Required (ID, target)

### Schema

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| ID | string | Test sample identifier | Must match test.csv IDs exactly |
| target | integer | Predicted class label | Range: [0, 20] (21 classes) |

### Example
```csv
ID,target
TEST_00000,4
TEST_00001,5
TEST_00002,0
TEST_00003,9
TEST_00004,9
...
TEST_15003,6
```

## Validation Rules

### Required Checks
1. **Row Count**: Exactly 15,004 rows (excluding header)
2. **ID Completeness**: All IDs from test.csv must be present
3. **ID Uniqueness**: No duplicate IDs
4. **ID Format**: Must match pattern `TEST_\d{5}` (TEST_ followed by 5 digits)
5. **Target Range**: All predictions must be integers in [0, 20]
6. **No Missing Values**: No null/NaN in either column
7. **Column Order**: ID must be first column, target must be second
8. **Column Names**: Exact spelling "ID" and "target" (case-sensitive)

### Python Validation Function
```python
def validate_submission(submission_path, test_ids_path):
    """
    Validate submission file against competition requirements.

    Parameters:
    -----------
    submission_path : str
        Path to submission CSV file
    test_ids_path : str
        Path to test.csv for ID validation

    Returns:
    --------
    dict : Validation result with status and messages
    """
    import pandas as pd

    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Load submission
    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load submission: {e}")
        return results

    # Check columns
    if list(sub.columns) != ["ID", "target"]:
        results["valid"] = False
        results["errors"].append(
            f"Invalid columns: expected ['ID', 'target'], got {list(sub.columns)}"
        )

    # Check row count
    if len(sub) != 15004:
        results["valid"] = False
        results["errors"].append(
            f"Invalid row count: expected 15004, got {len(sub)}"
        )

    # Check for missing values
    if sub.isnull().any().any():
        results["valid"] = False
        results["errors"].append("Submission contains missing values (NaN)")

    # Check ID format
    invalid_ids = sub[~sub['ID'].str.match(r'^TEST_\d{5}$')]
    if len(invalid_ids) > 0:
        results["valid"] = False
        results["errors"].append(
            f"Invalid ID format: {len(invalid_ids)} IDs don't match TEST_XXXXX pattern"
        )

    # Check ID uniqueness
    if sub['ID'].duplicated().any():
        dup_count = sub['ID'].duplicated().sum()
        results["valid"] = False
        results["errors"].append(f"Duplicate IDs found: {dup_count} duplicates")

    # Check target range
    if not sub['target'].between(0, 20).all():
        out_of_range = sub[~sub['target'].between(0, 20)]
        results["valid"] = False
        results["errors"].append(
            f"Target values out of range: {len(out_of_range)} predictions not in [0, 20]"
        )

    # Check target dtype (should be integer)
    if not pd.api.types.is_integer_dtype(sub['target']):
        results["warnings"].append(
            "Target column is not integer type (will be cast during submission)"
        )

    # Validate against test IDs
    try:
        test_ids = pd.read_csv(test_ids_path)['ID']
        missing_ids = set(test_ids) - set(sub['ID'])
        extra_ids = set(sub['ID']) - set(test_ids)

        if missing_ids:
            results["valid"] = False
            results["errors"].append(
                f"Missing IDs from test set: {len(missing_ids)} IDs not in submission"
            )

        if extra_ids:
            results["valid"] = False
            results["errors"].append(
                f"Extra IDs not in test set: {len(extra_ids)} unknown IDs"
            )
    except Exception as e:
        results["warnings"].append(f"Could not validate against test IDs: {e}")

    return results

# Usage example
result = validate_submission(
    "outputs/submissions/submission_qda.csv",
    "data/open/test.csv"
)

if result["valid"]:
    print("✅ Submission is valid!")
else:
    print("❌ Submission has errors:")
    for error in result["errors"]:
        print(f"  - {error}")

if result["warnings"]:
    print("⚠️  Warnings:")
    for warning in result["warnings"]:
        print(f"  - {warning}")
```

## Output File Naming Convention

### Standard Format
```
submission_{model_identifier}.csv
```

### Examples
- `submission_qda_optimized.csv` - QDA with optimized hyperparameters
- `submission_svc_optimized.csv` - SVC with optimized hyperparameters
- `submission_rf_optimized.csv` - RandomForest with optimized hyperparameters
- `submission_dt_optimized.csv` - DecisionTree with optimized hyperparameters
- `submission_ensemble_voting_hard.csv` - Hard voting ensemble
- `submission_ensemble_voting_soft.csv` - Soft voting ensemble
- `submission_ensemble_stacking.csv` - Stacking ensemble

### Versioning (if needed)
```
submission_{model}_{version}_{date}.csv
```
Example: `submission_qda_v2_20251001.csv`

## Generation Code Template

```python
def generate_submission(model, X_test, test_ids, output_path):
    """
    Generate competition submission file.

    Parameters:
    -----------
    model : trained model
        Fitted scikit-learn compatible model
    X_test : array-like, shape (15004, n_features)
        Test feature matrix
    test_ids : array-like, shape (15004,)
        Test sample IDs from test.csv
    output_path : str
        Where to save submission file

    Returns:
    --------
    pd.DataFrame : The submission dataframe
    """
    import pandas as pd
    import time

    start_time = time.time()

    # Generate predictions
    predictions = model.predict(X_test)

    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': test_ids,
        'target': predictions.astype(int)  # Ensure integer type
    })

    # Validate before saving
    assert len(submission) == 15004, "Wrong number of predictions"
    assert submission['target'].between(0, 20).all(), "Predictions out of range"
    assert not submission.isnull().any().any(), "Missing values in submission"

    # Save to CSV
    submission.to_csv(output_path, index=False)

    prediction_time = time.time() - start_time

    print(f"✅ Submission saved to {output_path}")
    print(f"   Predictions: {len(submission)}")
    print(f"   Time: {prediction_time:.2f}s")
    print(f"   Class distribution:")
    print(submission['target'].value_counts().sort_index())

    return submission

# Usage example
submission = generate_submission(
    model=trained_model,
    X_test=X_test_scaled,
    test_ids=test_df['ID'],
    output_path="outputs/submissions/submission_qda_optimized.csv"
)
```

## Class Distribution Analysis

### Purpose
Analyze prediction distribution to detect potential issues

### Expected Behavior
- **Balanced predictions**: Each class should have roughly 500-800 predictions (15004 / 21 ≈ 715)
- **Warning signs**:
  - Any class with 0 predictions (model too conservative)
  - Any class with >2000 predictions (model biased)
  - Extreme imbalance (one class >50% of predictions)

### Analysis Code
```python
def analyze_submission_distribution(submission_path, test_df_path=None):
    """
    Analyze prediction distribution and compare to training distribution.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load submission
    sub = pd.read_csv(submission_path)

    # Calculate distribution
    dist = sub['target'].value_counts().sort_index()

    print(f"Submission: {submission_path}")
    print(f"Total predictions: {len(sub)}")
    print(f"\nPrediction distribution:")
    print(dist)
    print(f"\nStatistics:")
    print(f"  Mean: {dist.mean():.1f}")
    print(f"  Std: {dist.std():.1f}")
    print(f"  Min: {dist.min()} (class {dist.idxmin()})")
    print(f"  Max: {dist.max()} (class {dist.idxmax()})")

    # Check for missing classes
    missing_classes = set(range(21)) - set(dist.index)
    if missing_classes:
        print(f"\n⚠️  Warning: No predictions for classes {missing_classes}")

    # Check for extreme imbalance
    if dist.max() > 2000:
        print(f"\n⚠️  Warning: Class {dist.idxmax()} has {dist.max()} predictions (>2000)")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(dist.index, dist.values, alpha=0.7, label='Predictions')
    plt.axhline(y=len(sub)/21, color='r', linestyle='--', label='Uniform (715 per class)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Prediction Distribution: {submission_path}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # If training data available, compare distributions
    if test_df_path:
        train = pd.read_csv(test_df_path.replace('test.csv', 'train.csv'))
        train_dist = train['target'].value_counts().sort_index()

        print(f"\n📊 Comparison with training distribution:")
        print(f"Training: {train_dist.mean():.1f} ± {train_dist.std():.1f}")
        print(f"Test predictions: {dist.mean():.1f} ± {dist.std():.1f}")

# Usage
analyze_submission_distribution(
    "outputs/submissions/submission_qda_optimized.csv",
    "data/open/test.csv"
)
```

## Error Handling

### Common Issues and Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| Wrong row count | `len(submission) != 15004` | Check test data loading, ensure no filtering |
| Missing IDs | Compare IDs with test.csv | Merge predictions with original test IDs |
| Out-of-range predictions | `predictions < 0 or > 20` | Check model output, clip to [0, 20] if needed |
| Wrong data type | `target not integer` | Use `.astype(int)` before saving |
| Duplicate IDs | `submission['ID'].duplicated().any()` | Drop duplicates, investigate cause |
| Encoding issues | Special characters in IDs | Use `encoding='utf-8'` in pd.read_csv/to_csv |

### Defensive Submission Generation
```python
def safe_generate_submission(model, X_test, test_ids, output_path):
    """
    Generate submission with defensive checks and corrections.
    """
    import pandas as pd
    import numpy as np

    # Generate predictions
    predictions = model.predict(X_test)

    # Defensive corrections
    predictions = np.clip(predictions, 0, 20)  # Clip to valid range
    predictions = predictions.astype(int)       # Ensure integer type

    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids.astype(str),  # Ensure string type
        'target': predictions
    })

    # Remove any duplicates (keep first)
    if submission['ID'].duplicated().any():
        print(f"⚠️  Warning: Removed {submission['ID'].duplicated().sum()} duplicate IDs")
        submission = submission.drop_duplicates(subset=['ID'], keep='first')

    # Verify row count
    assert len(submission) == 15004, f"Expected 15004 rows, got {len(submission)}"

    # Save
    submission.to_csv(output_path, index=False, encoding='utf-8')

    # Run validation
    validation = validate_submission(output_path, "data/open/test.csv")
    if not validation["valid"]:
        print("❌ Validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        raise ValueError("Submission validation failed")

    print(f"✅ Valid submission saved to {output_path}")
    return submission
```

---
**Status**: Contract Specification Complete
**Used By**: Notebooks 07, 08, 09 for generating competition submissions