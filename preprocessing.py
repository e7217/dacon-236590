import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self, remove_corr_threshold=0.95, variance_threshold=0.01, n_features=None):
        self.remove_corr_threshold = remove_corr_threshold
        self.variance_threshold = variance_threshold
        self.n_features = n_features

        self.scaler = RobustScaler()
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        self.corr_features_to_remove = []
        self.feature_selector = None
        self.pca = None

    def remove_high_correlation(self, X, threshold=0.95):
        """높은 상관관계 특성 제거"""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 상관계수가 threshold 이상인 특성 쌍 찾기
        high_corr_pairs = [column for column in upper_tri.columns
                          if any(upper_tri[column] > threshold)]

        return high_corr_pairs

    def fit(self, X, y=None):
        """전처리기 학습"""
        print("=== Fitting Preprocessor ===")

        # 1. 스케일링
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        print(f"1. Scaling: {X_scaled.shape}")

        # 2. 낮은 분산 특성 제거
        X_variance = pd.DataFrame(
            self.variance_selector.fit_transform(X_scaled),
            columns=X_scaled.columns[self.variance_selector.get_support()]
        )
        print(f"2. Variance filtering: {X_variance.shape}")

        # 3. 높은 상관관계 특성 제거
        self.corr_features_to_remove = self.remove_high_correlation(
            X_variance, self.remove_corr_threshold
        )
        X_corr = X_variance.drop(columns=self.corr_features_to_remove)
        print(f"3. Correlation filtering: {X_corr.shape} (removed {len(self.corr_features_to_remove)} features)")

        # 4. 특성 선택 (옵션)
        if self.n_features and y is not None:
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            X_selected = self.feature_selector.fit_transform(X_corr, y)
            print(f"4. Feature selection: {X_selected.shape}")

        print("Preprocessing fit complete!")
        return self

    def transform(self, X):
        """데이터 변환"""
        # 1. 스케일링
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )

        # 2. 낮은 분산 특성 제거
        X_variance = pd.DataFrame(
            self.variance_selector.transform(X_scaled),
            columns=X_scaled.columns[self.variance_selector.get_support()]
        )

        # 3. 높은 상관관계 특성 제거
        X_corr = X_variance.drop(columns=self.corr_features_to_remove)

        # 4. 특성 선택 (옵션)
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_corr)
            return X_selected

        return X_corr.values

    def fit_transform(self, X, y=None):
        """학습 및 변환"""
        return self.fit(X, y).transform(X)

def create_preprocessors():
    """다양한 전처리 파이프라인 생성"""
    preprocessors = {
        'basic': DataPreprocessor(
            remove_corr_threshold=0.95,
            variance_threshold=0.01
        ),
        'aggressive': DataPreprocessor(
            remove_corr_threshold=0.90,
            variance_threshold=0.02
        ),
        'feature_selected': DataPreprocessor(
            remove_corr_threshold=0.95,
            variance_threshold=0.01,
            n_features=30
        )
    }
    return preprocessors

def main():
    """메인 실행 함수"""
    print("Loading data...")
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')

    feature_cols = [col for col in train.columns if col.startswith('X_')]
    X_train_full = train[feature_cols]
    y_train_full = train['target']
    X_test = test[feature_cols]

    # 검증 세트 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # 전처리기 생성
    preprocessors = create_preprocessors()

    # 각 전처리기 적용 및 저장
    for name, preprocessor in preprocessors.items():
        print(f"\n=== Processing with {name} preprocessor ===")

        # 전처리 학습 및 적용
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # 결과 저장
        processed_data = {
            'X_train': X_train_processed,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_val': y_val
        }

        joblib.dump(processed_data, f'processed_data_{name}.pkl')
        joblib.dump(preprocessor, f'preprocessor_{name}.pkl')

        print(f"Saved: processed_data_{name}.pkl, preprocessor_{name}.pkl")
        print(f"Processed shape: {X_train_processed.shape}")

if __name__ == "__main__":
    main()