import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
import time
import joblib

def load_and_preprocess_data():
    print("=== 데이터 로딩 및 전처리 ===")
    train_df = pd.read_csv('./data/open/train.csv')

    X = train_df.drop(columns=['ID', 'target'])
    y = train_df['target']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"데이터 크기: {X_scaled.shape}")
    print(f"클래스 분포:\n{y.value_counts().sort_index()}")

    return X_scaled, y, scaler

def tune_lightgbm(X, y, n_iter=50):
    print("\n=== LightGBM 하이퍼파라미터 튜닝 시작 ===")

    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [4, 6, 8, 10, 12, -1],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'num_leaves': [31, 50, 70, 100, 150],
        'min_child_samples': [20, 30, 40, 50],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }

    lgbm_base = lgb.LGBMClassifier(
        device='cpu',
        random_state=42,
        verbose=-1
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=lgbm_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring=f1_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    start_time = time.time()
    random_search.fit(X, y)
    elapsed_time = time.time() - start_time

    print(f"\n튜닝 완료 시간: {elapsed_time/60:.2f}분")
    print(f"최고 F1-macro 점수: {random_search.best_score_:.6f}")
    print(f"\n최적 하이퍼파라미터:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def tune_xgboost(X, y, n_iter=50):
    print("\n=== XGBoost 하이퍼파라미터 튜닝 시작 ===")

    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }

    xgb_base = xgb.XGBClassifier(
        tree_method='hist',
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        random_state=42,
        verbosity=0
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring=f1_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    start_time = time.time()
    random_search.fit(X, y)
    elapsed_time = time.time() - start_time

    print(f"\n튜닝 완료 시간: {elapsed_time/60:.2f}분")
    print(f"최고 F1-macro 점수: {random_search.best_score_:.6f}")
    print(f"\n최적 하이퍼파라미터:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def save_results(lgbm_model, lgbm_params, lgbm_score,
                 xgb_model, xgb_params, xgb_score, scaler):
    print("\n=== 모델 및 결과 저장 ===")

    joblib.dump(lgbm_model, './models/lgbm_tuned.pkl')
    joblib.dump(xgb_model, './models/xgb_tuned.pkl')
    joblib.dump(scaler, './models/scaler.pkl')

    results = {
        'lightgbm': {
            'best_score': lgbm_score,
            'best_params': lgbm_params
        },
        'xgboost': {
            'best_score': xgb_score,
            'best_params': xgb_params
        }
    }

    import json
    with open('./models/tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("저장 완료:")
    print("  - models/lgbm_tuned.pkl")
    print("  - models/xgb_tuned.pkl")
    print("  - models/scaler.pkl")
    print("  - models/tuning_results.json")

def main():
    print("=" * 60)
    print("하이퍼파라미터 튜닝 자동화 스크립트")
    print("=" * 60)

    X_scaled, y, scaler = load_and_preprocess_data()

    lgbm_model, lgbm_params, lgbm_score = tune_lightgbm(X_scaled, y, n_iter=50)

    xgb_model, xgb_params, xgb_score = tune_xgboost(X_scaled, y, n_iter=50)

    print("\n" + "=" * 60)
    print("=== 최종 결과 비교 ===")
    print("=" * 60)
    print(f"LightGBM 최고 점수: {lgbm_score:.6f}")
    print(f"XGBoost 최고 점수:  {xgb_score:.6f}")

    if lgbm_score > xgb_score:
        print(f"\n🏆 LightGBM이 {lgbm_score - xgb_score:.6f}만큼 더 우수합니다!")
    else:
        print(f"\n🏆 XGBoost가 {xgb_score - lgbm_score:.6f}만큼 더 우수합니다!")

    save_results(lgbm_model, lgbm_params, lgbm_score,
                 xgb_model, xgb_params, xgb_score, scaler)

    print("\n" + "=" * 60)
    print("튜닝 완료! 최적화된 모델을 사용하여 예측을 진행하세요.")
    print("=" * 60)

if __name__ == "__main__":
    main()