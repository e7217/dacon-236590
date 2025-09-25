def main():
    """T010: Confusion Matrix 심화 분석을 통한 클래스별 성능 평가 테스트"""
    print("🔍 T010: Confusion Matrix 심화 분석을 통한 클래스별 성능 평가를 시작합니다...")

    from src.analysis.confusion_matrix_analyzer import ConfusionMatrixAnalyzer
    from src.utils.config import Config
    from src.tracking.experiment_tracker import ExperimentTracker
    import pandas as pd
    import numpy as np
    from pathlib import Path

    try:
        # 설정 로드
        config = Config()

        # 실험 추적기 초기화
        tracker = ExperimentTracker(
            project_name="dacon-smartmh-02",
            experiment_name="T010_confusion_matrix_analysis"
        )

        # 데이터 로드 또는 생성
        data_path = Path("data")

        if not (data_path / "train.csv").exists():
            print("📝 SHAP 분석용 샘플 데이터를 생성합니다...")
            np.random.seed(42)
            n_samples = 2000  # SHAP 계산 속도를 위해 작게 설정
            n_features = 52

            # 클래스 불균형 데이터 생성
            class_probs = np.array([0.1, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04,
                                   0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
            class_probs = class_probs / class_probs.sum()

            # 특성 데이터 생성 (센서 데이터 시뮬레이션)
            X = np.random.normal(0, 1, (n_samples, n_features))

            # 타겟 생성 (클래스별로 다른 패턴)
            y = np.random.choice(21, size=n_samples, p=class_probs)

            # 클래스별로 특성에 패턴 추가 (해석가능한 패턴 생성)
            for class_id in range(21):
                mask = y == class_id
                if mask.sum() > 0:
                    # 각 클래스마다 특정 특성들에 고유한 패턴 부여
                    pattern_features = np.random.choice(n_features, size=7, replace=False)
                    for feat in pattern_features:
                        X[mask, feat] += np.random.normal(class_id * 0.4, 0.4, mask.sum())

            # DataFrame 생성
            feature_names = [f'feature_{i:02d}' for i in range(n_features)]
            train_data = pd.DataFrame(X, columns=feature_names)
            train_data['target'] = y

            print(f"✅ SHAP 분석용 데이터 생성 완료: {len(train_data)}행, {train_data['target'].nunique()}개 클래스")

        else:
            print("📂 기존 데이터를 로드합니다...")
            train_data = pd.read_csv(data_path / "train.csv")

        # 특성과 레이블 분리
        X_train = train_data.drop(['target'], axis=1, errors='ignore')
        y_train = train_data['target']

        print(f"📊 데이터 정보: {len(X_train)}행 × {len(X_train.columns)}열")
        print(f"🎯 타겟 클래스: {y_train.nunique()}개 클래스")

        # Confusion Matrix 분석기 초기화
        analyzer = ConfusionMatrixAnalyzer(config=config, experiment_tracker=tracker)

        # Confusion Matrix 분석 실행
        print("\n🔍 Confusion Matrix 심화 분석을 실행합니다...")
        results = analyzer.analyze_confusion_matrix_comprehensive(
            X_train=X_train,
            y_train=y_train
        )

        # 결과 요약 출력
        analyzer.print_summary(results)

        print("\n🎉 T010: Confusion Matrix 분석이 성공적으로 완료되었습니다!")
        return results

    except Exception as e:
        print(f"❌ T010 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
