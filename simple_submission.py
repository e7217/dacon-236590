import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def main():
    """간단한 베이스라인 모델 및 제출 파일 생성"""
    print("Loading data...")
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    # 특성 추출
    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X = train[feature_cols]
    y = train['target']
    X_test = test[feature_cols]

    print(f"Train: {X.shape}, Test: {X_test.shape}")

    # 간단한 전처리
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest 모델
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 교차검증으로 성능 확인
    cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 전체 데이터로 훈련
    print("\nTraining final model...")
    rf_model.fit(X_scaled, y)

    # 테스트 예측
    predictions = rf_model.predict(X_test_scaled)

    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': predictions
    })

    submission.to_csv('submission.csv', index=False)

    print(f"\n✅ Submission created!")
    print(f"CV Accuracy: {cv_scores.mean():.4f}")
    print("\nPrediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index())

    return cv_scores.mean()

if __name__ == "__main__":
    main()