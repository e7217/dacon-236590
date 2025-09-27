import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    """빠른 베이스라인 모델 평가"""
    print("Loading processed data...")

    # 기본 전처리 데이터 로드
    data = joblib.load('processed_data_basic.pkl')
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # 빠른 모델들
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
    }

    results = []
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        # 훈련
        model.fit(X_train, y_train)

        # 예측 및 평가
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Overfitting: {train_acc - val_acc:.4f}")

        results.append({
            'Model': name,
            'Train_Acc': train_acc,
            'Val_Acc': val_acc,
            'Overfitting': train_acc - val_acc
        })

        if val_acc > best_score:
            best_score = val_acc
            best_model = model
            best_name = name

    # 결과 요약
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Val_Acc', ascending=False)

    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(results_df.to_string(index=False, float_format='%.4f'))

    # 제출 파일 생성
    print(f"\nCreating submission with best model: {best_name}")

    # 전체 데이터로 재훈련
    train_full = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    feature_cols = [col for col in train_full.columns if col.startswith('X_')]
    X_full = train_full[feature_cols]
    y_full = train_full['target']
    X_test = test[feature_cols]

    # 전처리기 적용
    preprocessor = joblib.load('preprocessor_basic.pkl')
    X_full_processed = preprocessor.fit_transform(X_full, y_full)
    X_test_processed = preprocessor.transform(X_test)

    # 최고 모델로 재훈련
    best_model.fit(X_full_processed, y_full)

    # 예측
    predictions = best_model.predict(X_test_processed)

    # 제출 파일
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': predictions
    })

    submission.to_csv('submission.csv', index=False)

    print(f"\n✅ Complete!")
    print(f"Best model: {best_name} (Val Acc: {best_score:.4f})")
    print("Files created: submission.csv")

    # 예측 분포 확인
    print(f"\nPrediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index())

if __name__ == "__main__":
    main()