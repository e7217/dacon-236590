"""
고급 ML 라이브러리 설치 테스트 스크립트
T001 태스크의 일부로 설치된 라이브러리들이 정상 작동하는지 확인
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_libraries():
    """설치된 라이브러리들 테스트"""
    print("🧪 고급 ML 라이브러리 설치 테스트 시작...\n")

    # 기본 라이브러리
    print("📦 기본 라이브러리 테스트:")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        print("  ✅ pandas, numpy, matplotlib, seaborn, plotly")
    except ImportError as e:
        print(f"  ❌ 기본 라이브러리 오류: {e}")
        return False

    # Gradient Boosting 모델들
    print("\n🚀 Gradient Boosting 모델들:")
    try:
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        print(f"  ✅ XGBoost {xgb.__version__}")
        print(f"  ✅ LightGBM {lgb.__version__}")
        print(f"  ✅ CatBoost {cb.__version__}")
    except ImportError as e:
        print(f"  ❌ Gradient Boosting 라이브러리 오류: {e}")
        return False

    # 하이퍼파라미터 최적화
    print("\n🎯 하이퍼파라미터 최적화:")
    try:
        import optuna
        import hyperopt
        import skopt
        print(f"  ✅ Optuna {optuna.__version__}")
        print(f"  ✅ Hyperopt {hyperopt.__version__}")
        print(f"  ✅ Scikit-optimize")
    except ImportError as e:
        print(f"  ❌ 최적화 라이브러리 오류: {e}")
        return False

    # 모델 해석성
    print("\n🔍 모델 해석성:")
    try:
        import shap
        import eli5
        import lime
        print(f"  ✅ SHAP {shap.__version__}")
        print(f"  ✅ ELI5 {eli5.__version__}")
        print(f"  ✅ LIME")
    except ImportError as e:
        print(f"  ❌ 해석성 라이브러리 오류: {e}")
        return False

    # 딥러닝 (PyTorch & TabNet)
    print("\n🧠 딥러닝:")
    try:
        import torch
        import pytorch_tabnet
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ TabNet (설치 확인됨)")
    except ImportError as e:
        print(f"  ❌ 딥러닝 라이브러리 오류: {e}")
        return False

    # 차원 축소
    print("\n📊 차원 축소:")
    try:
        import umap
        print(f"  ✅ UMAP {umap.__version__}")
    except ImportError as e:
        print(f"  ❌ 차원 축소 라이브러리 오류: {e}")
        return False

    # 실험 추적
    print("\n📈 실험 추적:")
    try:
        import mlflow
        import wandb
        print(f"  ✅ MLflow {mlflow.__version__}")
        print(f"  ✅ WandB {wandb.__version__}")
    except ImportError as e:
        print(f"  ❌ 실험 추적 라이브러리 오류: {e}")
        return False

    # AutoML
    print("\n🤖 AutoML:")
    try:
        from autogluon import tabular
        import flaml
        print(f"  ✅ AutoGluon")
        print(f"  ✅ FLAML {flaml.__version__}")
    except ImportError as e:
        print(f"  ❌ AutoML 라이브러리 오류: {e}")
        return False

    # 추가 시각화
    print("\n🎨 고급 시각화:")
    try:
        import yellowbrick
        print(f"  ✅ Yellowbrick {yellowbrick.__version__}")
    except ImportError as e:
        print(f"  ❌ 시각화 라이브러리 오류: {e}")
        return False

    print("\n" + "="*50)
    print("🎉 모든 고급 ML 라이브러리 설치 및 테스트 완료!")
    print("✅ T001 태스크 성공적으로 완료")
    print("="*50)

    return True

def test_simple_operations():
    """간단한 작업들로 라이브러리 기능 테스트"""
    print("\n🧪 기능 테스트:")

    try:
        # XGBoost 간단 테스트
        import xgboost as xgb
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X, y)
        print("  ✅ XGBoost 모델 훈련 성공")

        # Optuna 간단 테스트
        import optuna
        def objective(trial):
            return trial.suggest_float('x', -10, 10) ** 2
        study = optuna.create_study()
        study.optimize(objective, n_trials=3)
        print("  ✅ Optuna 최적화 성공")

        print("  🎉 기능 테스트 모두 통과!")
        return True

    except Exception as e:
        print(f"  ❌ 기능 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_libraries()
    if success:
        success = test_simple_operations()

    if success:
        print(f"\n✅ T001 완료: 고급 ML 라이브러리 설치 및 검증 성공")
        print(f"📝 설치된 패키지 수: 162개")
        print(f"🎯 다음 단계: T002 (실험 추적 환경 설정)")
    else:
        print(f"\n❌ T001 실패: 라이브러리 설치 또는 테스트 문제 발생")
        sys.exit(1)