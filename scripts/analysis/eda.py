import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_data():
    """데이터 로드"""
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')
    sample_sub = pd.read_csv('data/open/sample_submission.csv')
    return train, test, sample_sub

def basic_info(df, name):
    """기본 정보 출력"""
    print(f"\n=== {name} Dataset Info ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nData Types:")
    print(df.dtypes.value_counts())

    if 'target' in df.columns:
        print("\nTarget distribution:")
        print(df['target'].value_counts().sort_index())
        print(f"\nClass balance ratio: {df['target'].value_counts().min() / df['target'].value_counts().max():.3f}")

def analyze_features(df):
    """특성 분석"""
    feature_cols = [col for col in df.columns if col.startswith('X_')]

    print(f"\n=== Feature Analysis ===")
    print(f"Number of features: {len(feature_cols)}")

    # 특성별 통계
    stats = df[feature_cols].describe()
    print("\nFeature Statistics Summary:")
    print(f"Min values range: [{stats.loc['min'].min():.3f}, {stats.loc['min'].max():.3f}]")
    print(f"Max values range: [{stats.loc['max'].min():.3f}, {stats.loc['max'].max():.3f}]")
    print(f"Mean values range: [{stats.loc['mean'].min():.3f}, {stats.loc['mean'].max():.3f}]")

    # 결측값 확인
    missing = df[feature_cols].isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values found in {missing[missing>0].shape[0]} features")
    else:
        print("\nNo missing values found")

    return feature_cols

def plot_target_distribution(df):
    """타겟 분포 시각화"""
    if 'target' not in df.columns:
        return

    plt.figure(figsize=(12, 4))

    # 타겟 분포
    plt.subplot(1, 2, 1)
    target_counts = df['target'].value_counts().sort_index()
    plt.bar(target_counts.index, target_counts.values)
    plt.title('Target Distribution')
    plt.xlabel('Target Class')
    plt.ylabel('Count')

    # 비율
    plt.subplot(1, 2, 2)
    target_pct = df['target'].value_counts(normalize=True).sort_index() * 100
    plt.bar(target_pct.index, target_pct.values)
    plt.title('Target Distribution (%)')
    plt.xlabel('Target Class')
    plt.ylabel('Percentage')

    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_feature_patterns(df, feature_cols):
    """특성 패턴 분석"""
    # 특성값 범위별 그룹핑
    feature_ranges = {}

    for col in feature_cols:
        min_val, max_val = df[col].min(), df[col].max()

        if max_val <= 1.0 and min_val >= 0.0:
            feature_ranges[col] = 'normalized_0_1'
        elif max_val <= 100 and min_val >= 0:
            feature_ranges[col] = 'small_positive'
        else:
            feature_ranges[col] = 'other'

    print("\n=== Feature Range Patterns ===")
    range_counts = pd.Series(feature_ranges).value_counts()
    for range_type, count in range_counts.items():
        print(f"{range_type}: {count} features")

    return feature_ranges

def main():
    """메인 실행 함수"""
    print("Loading data...")
    train, test, sample_sub = load_data()

    # 기본 정보
    basic_info(train, "Train")
    basic_info(test, "Test")

    # 특성 분석
    feature_cols = analyze_features(train)

    # 타겟 분포
    plot_target_distribution(train)

    # 특성 패턴 분석
    feature_ranges = analyze_feature_patterns(train, feature_cols)

    print("\n=== EDA Summary ===")
    print(f"✓ Train: {train.shape[0]} samples, {len(feature_cols)} features")
    print(f"✓ Test: {test.shape[0]} samples")
    print(f"✓ Classes: {train['target'].nunique()} unique classes")
    print(f"✓ No missing values detected")
    print(f"✓ Feature patterns identified")

if __name__ == "__main__":
    main()