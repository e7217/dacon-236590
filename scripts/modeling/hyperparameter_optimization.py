"""
Hyperparameter Optimization for XGBoost and LightGBM
Expected improvement: 0.800 → 0.815-0.830 F1-macro
Implementation time: ~2-3 hours
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

def optimize_models():
    """Hyperparameter optimization for both XGBoost and LightGBM"""

    # Load and prepare data
    print("=== Loading Data ===")
    train_df = pd.read_csv('./data/open/train.csv')

    feature_cols = [col for col in train_df.columns if col.startswith('X_')]
    X = train_df[feature_cols]
    y = train_df['target']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"Data shape: {X_scaled.shape}")
    print(f"Classes: {y.nunique()}")

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')

    # XGBoost hyperparameter search space
    xgb_param_space = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2.0, 2.5],
        'min_child_weight': [1, 3, 5, 7]
    }

    # LightGBM hyperparameter search space
    lgb_param_space = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [4, 6, 8, 10, 12, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2.0, 2.5],
        'min_child_samples': [5, 10, 20, 30],
        'num_leaves': [31, 50, 70, 100]
    }

    results = {}

    # Optimize XGBoost
    print("\n=== Optimizing XGBoost ===")
    start_time = time.time()

    xgb_model = xgb.XGBClassifier(
        tree_method='hist',
        objective='multi:softmax',
        num_class=len(y.unique()),
        random_state=42,
        n_jobs=-1
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_space,
        n_iter=50,  # 50 random combinations
        cv=cv,
        scoring=f1_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    xgb_search.fit(X_scaled, y)
    xgb_time = time.time() - start_time

    print(f"XGBoost optimization completed in {xgb_time:.2f}s")
    print(f"Best XGBoost Score: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost Params: {xgb_search.best_params_}")

    results['xgb'] = {
        'best_score': xgb_search.best_score_,
        'best_params': xgb_search.best_params_,
        'best_model': xgb_search.best_estimator_,
        'time': xgb_time
    }

    # Optimize LightGBM
    print("\n=== Optimizing LightGBM ===")
    start_time = time.time()

    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=lgb_param_space,
        n_iter=50,  # 50 random combinations
        cv=cv,
        scoring=f1_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    lgb_search.fit(X_scaled, y)
    lgb_time = time.time() - start_time

    print(f"LightGBM optimization completed in {lgb_time:.2f}s")
    print(f"Best LightGBM Score: {lgb_search.best_score_:.4f}")
    print(f"Best LightGBM Params: {lgb_search.best_params_}")

    results['lgb'] = {
        'best_score': lgb_search.best_score_,
        'best_params': lgb_search.best_params_,
        'best_model': lgb_search.best_estimator_,
        'time': lgb_time
    }

    # Compare results
    print("\n=== OPTIMIZATION RESULTS ===")
    baseline_xgb = 0.7996  # From notebook
    baseline_lgb = 0.7984  # From notebook

    xgb_improvement = results['xgb']['best_score'] - baseline_xgb
    lgb_improvement = results['lgb']['best_score'] - baseline_lgb

    print(f"XGBoost: {baseline_xgb:.4f} → {results['xgb']['best_score']:.4f} (+{xgb_improvement:.4f})")
    print(f"LightGBM: {baseline_lgb:.4f} → {results['lgb']['best_score']:.4f} (+{lgb_improvement:.4f})")

    # Determine best model
    best_model_name = 'xgb' if results['xgb']['best_score'] > results['lgb']['best_score'] else 'lgb'
    best_result = results[best_model_name]

    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"📈 Score: {best_result['best_score']:.4f}")
    print(f"⏱️  Training Time: {best_result['time']:.2f}s")
    print(f"🎯 Improvement: +{best_result['best_score'] - (baseline_xgb if best_model_name == 'xgb' else baseline_lgb):.4f}")

    # Save optimized models for future use
    import joblib
    joblib.dump(results['xgb']['best_model'], 'models/optimized_xgb.pkl')
    joblib.dump(results['lgb']['best_model'], 'models/optimized_lgb.pkl')
    joblib.dump(scaler, 'models/scaler_optimized.pkl')

    print(f"\n✅ Models saved to models/ directory")

    return results

def create_optimized_submission(results):
    """Create submission with optimized model"""
    print("\n=== Creating Optimized Submission ===")

    # Load test data
    test_df = pd.read_csv('./data/open/test.csv')
    feature_cols = [col for col in test_df.columns if col.startswith('X_')]
    X_test = test_df[feature_cols]

    # Load scaler and apply
    import joblib
    scaler = joblib.load('models/scaler_optimized.pkl')
    X_test_scaled = scaler.transform(X_test)

    # Determine best model
    best_model_name = 'xgb' if results['xgb']['best_score'] > results['lgb']['best_score'] else 'lgb'
    best_model = results[best_model_name]['best_model']

    # Make predictions
    predictions = best_model.predict(X_test_scaled)

    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })

    submission.to_csv('outputs/submissions/optimized_submission.csv', index=False)

    print(f"✅ Optimized submission saved: optimized_submission.csv")
    print(f"📊 Model used: {best_model_name.upper()}")
    print(f"🎯 Expected CV Score: {results[best_model_name]['best_score']:.4f}")

    return submission

if __name__ == "__main__":
    # Run optimization
    results = optimize_models()

    # Create submission with best model
    submission = create_optimized_submission(results)

    print("\n🎉 HYPERPARAMETER OPTIMIZATION COMPLETE!")