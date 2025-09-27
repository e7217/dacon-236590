import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(preprocessor_name='basic'):
    """전처리된 데이터 로드"""
    data = joblib.load(f'processed_data_{preprocessor_name}.pkl')
    return data

def create_baseline_models():
    """베이스라인 모델들 생성"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        ),
        'SVM': SVC(
            random_state=42,
            probability=True
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=500
        )
    }
    return models

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """모델 평가"""
    print(f"\n=== Evaluating {model_name} ===")

    # 모델 훈련
    model.fit(X_train, y_train)

    # 예측
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 정확도 계산
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Overfitting: {train_acc - val_acc:.4f}")

    return {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'y_val_pred': y_val_pred
    }

def compare_preprocessors():
    """전처리 방법별 성능 비교"""
    preprocessors = ['basic', 'aggressive', 'feature_selected']
    results = {}

    for prep_name in preprocessors:
        print(f"\n{'='*50}")
        print(f"Testing with {prep_name} preprocessor")
        print(f"{'='*50}")

        # 데이터 로드
        data = load_processed_data(prep_name)
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']

        # 빠른 모델들로만 테스트
        quick_models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
        }

        prep_results = {}
        for model_name, model in quick_models.items():
            result = evaluate_model(model, X_train, X_val, y_train, y_val, model_name)
            prep_results[model_name] = result

        results[prep_name] = prep_results

    return results

def full_model_evaluation(preprocessor_name='basic'):
    """전체 모델 평가"""
    print(f"\n{'='*60}")
    print(f"Full Model Evaluation with {preprocessor_name} preprocessor")
    print(f"{'='*60}")

    # 데이터 로드
    data = load_processed_data(preprocessor_name)
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    # 모델 생성
    models = create_baseline_models()

    # 결과 저장
    results = {}
    performance_summary = []

    for model_name, model in models.items():
        try:
            result = evaluate_model(model, X_train, X_val, y_train, y_val, model_name)
            results[model_name] = result

            performance_summary.append({
                'Model': model_name,
                'Train Acc': result['train_acc'],
                'Val Acc': result['val_acc'],
                'Overfitting': result['train_acc'] - result['val_acc']
            })

        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue

    # 성능 요약 테이블
    summary_df = pd.DataFrame(performance_summary)
    summary_df = summary_df.sort_values('Val Acc', ascending=False)

    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    return results, summary_df

def create_submission(best_model, preprocessor_name='basic'):
    """제출 파일 생성"""
    print(f"\nCreating submission file...")

    # 전체 훈련 데이터로 재훈련
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X_full = train[feature_cols]
    y_full = train['target']
    X_test = test[feature_cols]

    # 전처리기 로드 및 적용
    preprocessor = joblib.load(f'preprocessor_{preprocessor_name}.pkl')
    X_full_processed = preprocessor.fit_transform(X_full, y_full)
    X_test_processed = preprocessor.transform(X_test)

    # 모델 재훈련
    best_model.fit(X_full_processed, y_full)

    # 예측
    test_predictions = best_model.predict(X_test_processed)

    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': test_predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission file saved: submission.csv")

    return submission

def main():
    """메인 실행 함수"""
    print("Starting baseline model evaluation...")

    # 1. 전처리 방법 비교 (빠른 테스트)
    print("Step 1: Comparing preprocessing methods...")
    prep_results = compare_preprocessors()

    # 최고 성능 전처리 방법 찾기
    best_prep = None
    best_val_acc = 0

    for prep_name, results in prep_results.items():
        avg_val_acc = np.mean([result['val_acc'] for result in results.values()])
        print(f"{prep_name} average validation accuracy: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_prep = prep_name

    print(f"\nBest preprocessing method: {best_prep} ({best_val_acc:.4f})")

    # 2. 최고 전처리 방법으로 전체 모델 평가
    print(f"\nStep 2: Full model evaluation with {best_prep} preprocessing...")
    results, summary_df = full_model_evaluation(best_prep)

    # 최고 모델 선택
    best_model_name = summary_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name}")
    print(f"Validation accuracy: {summary_df.iloc[0]['Val Acc']:.4f}")

    # 3. 제출 파일 생성
    submission = create_submission(best_model, best_prep)

    # 결과 저장
    joblib.dump(best_model, 'best_model.pkl')
    summary_df.to_csv('model_performance_summary.csv', index=False)

    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION COMPLETE")
    print("="*60)
    print(f"✓ Best preprocessing: {best_prep}")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Best validation accuracy: {summary_df.iloc[0]['Val Acc']:.4f}")
    print("✓ Files saved: best_model.pkl, submission.csv, model_performance_summary.csv")

if __name__ == "__main__":
    main()