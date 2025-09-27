import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class VotingEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1] * len(models)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])

        # 가중 다수결 투표
        weighted_preds = []
        for i in range(X.shape[0]):
            votes = {}
            for j, pred in enumerate(predictions[:, i]):
                votes[pred] = votes.get(pred, 0) + self.weights[j]
            weighted_preds.append(max(votes, key=votes.get))

        return np.array(weighted_preds)

def create_optimized_models():
    """최적화된 모델들 생성"""
    models = {
        'RF_100': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'RF_200': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=43,
            n_jobs=-1
        ),
        'ET_100': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'ET_200': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=43,
            n_jobs=-1
        ),
        'GB_100': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
    }
    return models

def evaluate_ensemble_combinations():
    """다양한 앙상블 조합 평가"""
    print("Loading and preprocessing data...")
    train = pd.read_csv('data/open/train.csv')

    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X = train[feature_cols]
    y = train['target']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 개별 모델 생성
    models = create_optimized_models()

    print("Evaluating individual models...")
    individual_scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
        individual_scores[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # 상위 모델들로 앙상블 구성
    sorted_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
    top_3_names = [name for name, _ in sorted_models[:3]]
    top_5_names = [name for name, _ in sorted_models[:5]]

    print(f"\nTop 3 models: {top_3_names}")
    print(f"Top 5 models: {top_5_names}")

    ensemble_configs = [
        {'name': 'Top3_Equal', 'models': top_3_names, 'weights': [1, 1, 1]},
        {'name': 'Top3_Weighted', 'models': top_3_names,
         'weights': [sorted_models[i][1] for i in range(3)]},
        {'name': 'Top5_Equal', 'models': top_5_names, 'weights': [1, 1, 1, 1, 1]},
        {'name': 'All_Equal', 'models': list(models.keys()), 'weights': None}
    ]

    ensemble_results = {}

    print("\nEvaluating ensemble combinations...")
    for config in ensemble_configs:
        selected_models = [models[name] for name in config['models']]
        ensemble = VotingEnsemble(selected_models, config['weights'])

        scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=1)
        ensemble_results[config['name']] = scores.mean()
        print(f"{config['name']}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return individual_scores, ensemble_results, models, scaler

def create_final_submission():
    """최종 제출 파일 생성"""
    print("\nCreating final submission...")

    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]

    # 전처리
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 최고 성능 개별 모델
    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    # 다양성을 위한 추가 모델들
    models = [
        best_rf,
        ExtraTreesClassifier(n_estimators=150, max_depth=18, random_state=43, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=44)
    ]

    # 앙상블 생성
    ensemble = VotingEnsemble(models, weights=[0.4, 0.35, 0.25])

    # 훈련 및 예측
    ensemble.fit(X_train_scaled, y_train)
    predictions = ensemble.predict(X_test_scaled)

    # 제출 파일
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': predictions
    })

    submission.to_csv('ensemble_submission.csv', index=False)

    print("Final ensemble submission created: ensemble_submission.csv")
    print("\nPrediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index())

    return ensemble

def main():
    """메인 실행"""
    print("Advanced Ensemble Model Development")
    print("="*50)

    # 앙상블 조합 평가
    individual_scores, ensemble_results, models, scaler = evaluate_ensemble_combinations()

    # 결과 요약
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")

    print("\nIndividual Model Performance:")
    for name, score in sorted(individual_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")

    print("\nEnsemble Performance:")
    for name, score in sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {score:.4f}")

    # 최고 성능 확인
    best_individual = max(individual_scores.values())
    best_ensemble = max(ensemble_results.values())

    print(f"\nBest Individual: {best_individual:.4f}")
    print(f"Best Ensemble: {best_ensemble:.4f}")
    print(f"Improvement: {best_ensemble - best_individual:.4f}")

    # 최종 제출 파일 생성
    final_ensemble = create_final_submission()

    print(f"\n🎯 Complete! Expected CV score: ~{best_ensemble:.4f}")

if __name__ == "__main__":
    main()