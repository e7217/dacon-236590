import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv('data/open/train.csv')
    return train

def analyze_feature_importance(df):
    """특성 중요도 분석"""
    feature_cols = [col for col in df.columns if col.startswith('X_')]

    # Random Forest로 특성 중요도 계산
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_cols], df['target'])

    # 특성 중요도 데이터프레임 생성
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("=== Top 10 Most Important Features ===")
    print(importance_df.head(10))

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    top_features = importance_df.head(15)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.xlabel('Importance')

    plt.subplot(2, 1, 2)
    plt.hist(importance_df['importance'], bins=20, alpha=0.7)
    plt.title('Feature Importance Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return importance_df

def correlation_analysis(df):
    """상관관계 분석"""
    feature_cols = [col for col in df.columns if col.startswith('X_')]

    # 특성 간 상관계수 계산
    corr_matrix = df[feature_cols].corr()

    # 높은 상관관계 쌍 찾기
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:  # 0.8 이상의 상관관계
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    print(f"\n=== High Correlation Pairs (>0.8) ===")
    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print("No high correlation pairs found")

    # 상관관계 히트맵 (상위 20개 특성만)
    top_features = analyze_feature_importance(df).head(20)['feature'].values
    corr_subset = df[top_features].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap (Top 20 Features)')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

def class_distribution_analysis(df):
    """클래스별 특성 분포 분석"""
    feature_cols = [col for col in df.columns if col.startswith('X_')]

    # 주요 특성들의 클래스별 분포 시각화
    importance_df = analyze_feature_importance(df)
    top_features = importance_df.head(6)['feature'].values

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)

        # 각 클래스별로 특성값 분포 플롯
        for class_id in range(0, 21, 5):  # 0, 5, 10, 15, 20 클래스만 표시
            class_data = df[df['target'] == class_id][feature]
            plt.hist(class_data, alpha=0.6, label=f'Class {class_id}', bins=30)

        plt.title(f'{feature} Distribution by Class')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        if i == 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def dimensionality_analysis(df):
    """차원 축소 분석"""
    feature_cols = [col for col in df.columns if col.startswith('X_')]

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # PCA 분석
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)

    # 누적 분산 비율 계산
    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)

    # 95% 분산을 설명하는 주성분 개수
    n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
    print(f"\n=== PCA Analysis ===")
    print(f"Components needed for 95% variance: {n_components_95}")
    print(f"Components needed for 90% variance: {np.argmax(cumsum_ratio >= 0.90) + 1}")

    # 시각화
    plt.figure(figsize=(15, 5))

    # PCA 누적 분산 비율
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=n_components_95, color='r', linestyle='--', alpha=0.7)
    plt.title('PCA Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.grid(True, alpha=0.3)

    # 첫 두 주성분으로 데이터 시각화
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['target'],
                         cmap='tab20', alpha=0.6, s=1)
    plt.title('PCA: First 2 Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter)

    # t-SNE (샘플링해서 시각화)
    if len(df) > 5000:
        sample_idx = np.random.choice(len(df), 5000, replace=False)
        X_sample = X_scaled[sample_idx]
        y_sample = df['target'].iloc[sample_idx]
    else:
        X_sample = X_scaled
        y_sample = df['target']

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_sample)

    plt.subplot(1, 3, 3)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_sample,
                         cmap='tab20', alpha=0.6, s=1)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.savefig('dimensionality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("Loading data...")
    train = load_data()

    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(train)

    print("Analyzing correlations...")
    correlation_analysis(train)

    print("Analyzing class distributions...")
    class_distribution_analysis(train)

    print("Analyzing dimensionality...")
    dimensionality_analysis(train)

    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("- feature_importance.png")
    print("- correlation_heatmap.png")
    print("- class_distribution.png")
    print("- dimensionality_analysis.png")

if __name__ == "__main__":
    main()