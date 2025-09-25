import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def create_final_ensemble():
    """최종 앙상블 모델 생성 및 제출"""
    print("=== Final Smart Manufacturing Equipment Status Classification ===")

    # 데이터 로드
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Classes: {y_train.nunique()}")

    # 전처리
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 최적화된 모델들
    models = {
        'rf1': RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'rf2': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=43,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_split=3,
            random_state=44,
            n_jobs=-1
        )
    }

    # 개별 성능 확인
    print("\nModel Performance:")
    model_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
        model_scores[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # 모든 모델 훈련
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    # 앙상블 예측 (소프트 보팅)
    print("\nCreating ensemble predictions...")
    all_predictions = []

    for name, model in trained_models.items():
        pred = model.predict(X_test_scaled)
        all_predictions.append(pred)

    # 다수결 투표
    ensemble_predictions = []
    for i in range(len(X_test_scaled)):
        votes = [pred[i] for pred in all_predictions]
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_predictions.append(ensemble_pred)

    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': ensemble_predictions
    })

    submission.to_csv('final_submission.csv', index=False)

    # 결과 요약
    print(f"\n🎯 Final Solution Summary:")
    print(f"✓ Ensemble of 3 optimized models")
    print(f"✓ Expected CV accuracy: ~{np.mean(list(model_scores.values())):.4f}")
    print(f"✓ Robust preprocessing with outlier handling")
    print(f"✓ 21-class equipment status classification")

    print(f"\nPrediction distribution:")
    pred_counts = pd.Series(ensemble_predictions).value_counts().sort_index()
    for class_id, count in pred_counts.items():
        print(f"Class {class_id:2d}: {count:4d} samples")

    print(f"\nFile saved: final_submission.csv")
    return np.mean(list(model_scores.values()))

if __name__ == "__main__":
    final_score = create_final_ensemble()