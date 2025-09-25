"""
Adversarial Validation 구현
T004 태스크: Train/Test 데이터 분포 차이 분석

목적:
- CV 점수와 실제 제출 점수 격차(9.4%)의 원인 규명
- Train과 Test 데이터 분포 차이 정량화
- 데이터 도메인 변화(Domain Shift) 탐지
- 신뢰할 수 있는 검증 전략 제시

방법론:
1. Train + Test 데이터를 합쳐서 이진 분류 문제로 변환
2. Train=0, Test=1로 라벨링하여 구별 가능성 측정
3. AUC가 0.5에 가까우면 분포 유사, 1.0에 가까우면 분포 상이
4. 중요한 특성들을 분석하여 분포 차이의 원인 파악
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.config import get_config, get_paths


class AdversarialValidator:
    """
    Adversarial Validation 분석기
    Train/Test 분포 차이를 체계적으로 분석하여 CV-실제 점수 격차 원인 규명
    """

    def __init__(self, use_tracking: bool = True):
        """
        Adversarial Validator 초기화

        Args:
            use_tracking: 실험 추적 사용 여부
        """
        self.use_tracking = use_tracking
        self.results = {}
        self.figures = {}

        # 경로 설정
        self.paths = get_paths()
        self.plots_dir = self.paths['experiments_dir'] / 'plots' / 'adversarial'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 실험 추적 설정
        if self.use_tracking:
            self.tracker = ExperimentTracker(
                project_name=get_config('tracking.project_name'),
                experiment_name="adversarial_validation"
            )

        print("🎭 Adversarial Validation 분석기 초기화 완료")

    def validate_data_distribution(self, train_data, test_data, target_col=None):
        """
        Train/Test 데이터 분포 차이 종합 분석

        Args:
            train_data: 훈련 데이터 (DataFrame)
            test_data: 테스트 데이터 (DataFrame)
            target_col: 타겟 컬럼명 (train_data에서 제외할 컬럼)

        Returns:
            Dict: 분석 결과 딕셔너리
        """
        print("\n🎭 Adversarial Validation 종합 분석 시작...")
        print("=" * 60)

        # 실험 추적 시작
        if self.use_tracking:
            self.tracker.start_run(
                run_name="adversarial_validation_analysis",
                description="Train/Test 데이터 분포 차이 분석 - CV 점수 격차 원인 규명",
                tags={
                    "analysis_type": "adversarial_validation",
                    "task": "T004",
                    "purpose": "domain_shift_detection"
                }
            )

        # 데이터 전처리 및 준비
        X_train, X_test = self._prepare_data(train_data, test_data, target_col)

        # 1. 기본 분포 차이 분석
        basic_stats = self._analyze_basic_statistics(X_train, X_test)

        # 2. Adversarial Validation 수행
        adversarial_results = self._perform_adversarial_validation(X_train, X_test)

        # 3. 특성별 중요도 분석
        feature_importance = self._analyze_feature_importance(
            X_train, X_test, adversarial_results['model']
        )

        # 4. 분포 시각화 생성
        visualization_results = self._create_distribution_visualizations(
            X_train, X_test, feature_importance
        )

        # 5. 검증 전략 제안
        validation_strategy = self._propose_validation_strategy(adversarial_results)

        # 결과 종합
        comprehensive_results = {
            'basic_statistics': basic_stats,
            'adversarial_results': adversarial_results,
            'feature_importance': feature_importance,
            'visualizations': visualization_results,
            'validation_strategy': validation_strategy,
            'data_info': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'feature_names': X_train.columns.tolist()
            }
        }

        self.results = comprehensive_results

        # 실험 추적 로깅
        if self.use_tracking:
            self._log_adversarial_results(comprehensive_results)

        print("\n✅ Adversarial Validation 분석 완료!")
        self._print_analysis_summary(comprehensive_results)

        return comprehensive_results

    def _prepare_data(self, train_data, test_data, target_col):
        """데이터 전처리 및 준비"""
        print("\n📋 1단계: 데이터 전처리")

        # 타겟 컬럼 제거 (있는 경우)
        if target_col and target_col in train_data.columns:
            X_train = train_data.drop(columns=[target_col])
        else:
            X_train = train_data.copy()

        X_test = test_data.copy()

        # 공통 컬럼만 사용
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        # 결측값 처리 (간단한 중앙값 대체)
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)

        print(f"  • Train 데이터: {X_train.shape}")
        print(f"  • Test 데이터: {X_test.shape}")
        print(f"  • 공통 특성: {len(common_cols)}개")

        return X_train, X_test

    def _analyze_basic_statistics(self, X_train, X_test):
        """기본 통계량 비교 분석"""
        print("\n📊 2단계: 기본 통계량 분석")

        stats_comparison = {}

        for col in X_train.columns:
            train_stats = X_train[col].describe()
            test_stats = X_test[col].describe()

            # 통계적 차이 계산
            mean_diff = abs(train_stats['mean'] - test_stats['mean'])
            std_diff = abs(train_stats['std'] - test_stats['std'])

            # 정규화된 차이 (평균 대비)
            normalized_mean_diff = mean_diff / (abs(train_stats['mean']) + 1e-8)
            normalized_std_diff = std_diff / (train_stats['std'] + 1e-8)

            stats_comparison[col] = {
                'train_mean': train_stats['mean'],
                'test_mean': test_stats['mean'],
                'train_std': train_stats['std'],
                'test_std': test_stats['std'],
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'normalized_mean_diff': normalized_mean_diff,
                'normalized_std_diff': normalized_std_diff
            }

        # 가장 차이가 큰 특성들 식별
        mean_diffs = {k: v['normalized_mean_diff'] for k, v in stats_comparison.items()}
        top_different_features = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)[:10]

        results = {
            'stats_comparison': stats_comparison,
            'top_different_features': top_different_features,
            'overall_mean_diff': np.mean(list(mean_diffs.values())),
            'overall_std_diff': np.std(list(mean_diffs.values()))
        }

        print(f"  • 전체 평균 차이: {results['overall_mean_diff']:.4f}")
        print(f"  • 차이 표준편차: {results['overall_std_diff']:.4f}")
        print(f"  • 가장 다른 특성: {top_different_features[0][0]} ({top_different_features[0][1]:.4f})")

        return results

    def _perform_adversarial_validation(self, X_train, X_test):
        """Adversarial Validation 수행"""
        print("\n🎭 3단계: Adversarial Validation 수행")

        # 데이터 결합 및 라벨 생성
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        y_combined = np.concatenate([
            np.zeros(len(X_train)),  # Train: 0
            np.ones(len(X_test))     # Test: 1
        ])

        # 여러 모델로 Adversarial Validation 수행
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=get_config('models.random_seed'),
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=get_config('models.random_seed'),
                max_iter=1000
            )
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n  🤖 {model_name} 분석:")

            # 특성 스케일링 (로지스틱 회귀용)
            if model_name == 'LogisticRegression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_combined)
                X_model = pd.DataFrame(X_scaled, columns=X_combined.columns)
            else:
                X_model = X_combined

            # 교차검증으로 AUC 계산
            cv_scores = cross_val_score(
                model, X_model, y_combined,
                cv=5, scoring='roc_auc', n_jobs=-1
            )

            # 전체 데이터로 모델 학습 (특성 중요도용)
            model.fit(X_model, y_combined)

            # ROC Curve 계산
            y_pred_proba = model.predict_proba(X_model)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_combined, y_pred_proba)
            auc_score = roc_auc_score(y_combined, y_pred_proba)

            results[model_name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'full_auc': auc_score,
                'fpr': fpr,
                'tpr': tpr,
                'model': model,
                'predictions': y_pred_proba
            }

            print(f"    • CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    • Full AUC: {auc_score:.4f}")

        # 최고 성능 모델 선택
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_auc_mean'])
        best_result = results[best_model_name]

        # ROC Curve 시각화
        self._plot_roc_curve(results)

        # Domain shift 정도 해석
        domain_shift_severity = self._interpret_domain_shift(best_result['cv_auc_mean'])

        final_results = {
            'all_models': results,
            'best_model_name': best_model_name,
            'best_auc': best_result['cv_auc_mean'],
            'domain_shift_severity': domain_shift_severity,
            'model': best_result['model']
        }

        print(f"\n  🏆 최고 성능: {best_model_name} (AUC: {best_result['cv_auc_mean']:.4f})")
        print(f"  📊 Domain Shift 정도: {domain_shift_severity}")

        return final_results

    def _plot_roc_curve(self, results):
        """ROC Curve 시각화"""
        plt.figure(figsize=(10, 8))

        colors = ['blue', 'red', 'green', 'orange']

        for i, (model_name, result) in enumerate(results.items()):
            plt.plot(
                result['fpr'], result['tpr'],
                color=colors[i % len(colors)],
                linewidth=2,
                label=f'{model_name} (AUC = {result["full_auc"]:.3f})'
            )

        # 대각선 (랜덤 분류기)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Adversarial Validation\n(Train vs Test 분포 구별 성능)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 그래프 저장
        roc_path = self.plots_dir / 'adversarial_roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        self.figures['roc_curve'] = plt.gcf()
        plt.close()

    def _interpret_domain_shift(self, auc_score):
        """Domain Shift 정도 해석"""
        if auc_score < 0.53:
            return "매우 낮음 (분포 거의 동일)"
        elif auc_score < 0.57:
            return "낮음 (작은 분포 차이)"
        elif auc_score < 0.65:
            return "보통 (주의 필요)"
        elif auc_score < 0.75:
            return "높음 (심각한 분포 차이)"
        else:
            return "매우 높음 (심각한 도메인 변화)"

    def _analyze_feature_importance(self, X_train, X_test, model):
        """특성 중요도 분석"""
        print("\n🔍 4단계: 특성 중요도 분석")

        # 특성 중요도 추출
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("  ⚠️ 해당 모델에서 특성 중요도를 추출할 수 없습니다.")
            return {}

        # 특성 중요도 정렬
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # 상위 20개 특성 시각화
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Features Distinguishing Train vs Test\n(High importance = Large distribution difference)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # 그래프 저장
        importance_path = self.plots_dir / 'feature_importance.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        self.figures['feature_importance'] = plt.gcf()
        plt.close()

        # 가장 중요한 특성들의 분포 시각화
        self._plot_top_feature_distributions(X_train, X_test, top_features['feature'].head(6))

        results = {
            'importance_df': importance_df,
            'top_features': top_features['feature'].head(10).tolist(),
            'top_importances': top_features['importance'].head(10).tolist(),
            'plot_path': str(importance_path)
        }

        print(f"  • 가장 중요한 특성: {results['top_features'][0]} ({results['top_importances'][0]:.4f})")
        print(f"  • 상위 3개 특성: {', '.join(results['top_features'][:3])}")

        return results

    def _plot_top_feature_distributions(self, X_train, X_test, top_features):
        """상위 특성들의 분포 시각화"""
        n_features = len(top_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, feature in enumerate(top_features):
            if i >= 6:
                break

            ax = axes[i]

            # 히스토그램
            ax.hist(X_train[feature], bins=30, alpha=0.7, label='Train', density=True, color='blue')
            ax.hist(X_test[feature], bins=30, alpha=0.7, label='Test', density=True, color='red')

            ax.set_title(f'{feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 그래프 저장
        distributions_path = self.plots_dir / 'top_features_distributions.png'
        plt.savefig(distributions_path, dpi=300, bbox_inches='tight')
        self.figures['feature_distributions'] = plt.gcf()
        plt.close()

    def _create_distribution_visualizations(self, X_train, X_test, feature_importance):
        """분포 시각화 생성"""
        print("\n📈 5단계: 분포 시각화 생성")

        # 전체 데이터의 차원 축소 시각화 (PCA)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 데이터 결합
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        labels = ['Train'] * len(X_train) + ['Test'] * len(X_test)

        # 스케일링 및 PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        pca = PCA(n_components=2, random_state=get_config('models.random_seed'))
        X_pca = pca.fit_transform(X_scaled)

        # PCA 시각화
        plt.figure(figsize=(12, 8))

        train_mask = np.array(labels) == 'Train'
        test_mask = np.array(labels) == 'Test'

        plt.scatter(X_pca[train_mask, 0], X_pca[train_mask, 1],
                   alpha=0.6, label='Train', color='blue', s=20)
        plt.scatter(X_pca[test_mask, 0], X_pca[test_mask, 1],
                   alpha=0.6, label='Test', color='red', s=20)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization: Train vs Test Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 그래프 저장
        pca_path = self.plots_dir / 'pca_train_test_distribution.png'
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        self.figures['pca_distribution'] = plt.gcf()
        plt.close()

        results = {
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'pca_plot_path': str(pca_path)
        }

        print(f"  • PCA 설명 분산: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
        print(f"  • 총 설명 분산: {sum(pca.explained_variance_ratio_):.2%}")

        return results

    def _propose_validation_strategy(self, adversarial_results):
        """검증 전략 제안"""
        print("\n💡 6단계: 검증 전략 제안")

        auc_score = adversarial_results['best_auc']
        strategies = []

        # AUC 점수에 따른 전략 제안
        if auc_score >= 0.75:
            strategies.extend([
                "🚨 심각한 Domain Shift 감지",
                "• TimeSeriesSplit 또는 GroupKFold 사용 고려",
                "• Test 데이터와 유사한 분포의 validation set 구성",
                "• Domain Adaptation 기법 적용",
                "• 특성 분포를 test와 유사하게 조정"
            ])
        elif auc_score >= 0.65:
            strategies.extend([
                "⚠️ 주의 필요한 분포 차이 감지",
                "• Stratified sampling 강화",
                "• Cross-validation 폴드 수 증가",
                "• 주요 차별 특성에 대한 추가 분석 필요"
            ])
        elif auc_score >= 0.57:
            strategies.extend([
                "📊 작은 분포 차이 감지",
                "• 현재 CV 전략 유지 가능",
                "• 정기적인 adversarial validation 모니터링",
                "• 앙상블 방법으로 안정성 향상"
            ])
        else:
            strategies.extend([
                "✅ 분포 차이 거의 없음",
                "• 현재 CV 전략이 신뢰할 만함",
                "• CV 점수가 실제 성능을 잘 반영할 것으로 예상"
            ])

        # 추가 일반적 권장사항
        strategies.extend([
            "\n🔧 추가 권장사항:",
            "• 중요한 차별 특성들에 대한 도메인 전문가 검토",
            "• 데이터 수집 과정에서의 systematic bias 조사",
            "• Pseudo-labeling 기법 고려 (신중하게 적용)"
        ])

        results = {
            'auc_score': auc_score,
            'severity_level': adversarial_results['domain_shift_severity'],
            'strategies': strategies,
            'recommended_cv': self._recommend_cv_strategy(auc_score)
        }

        print(f"  • 심각도 수준: {results['severity_level']}")
        print(f"  • 권장 CV 전략: {results['recommended_cv']}")

        return results

    def _recommend_cv_strategy(self, auc_score):
        """AUC 점수 기반 CV 전략 추천"""
        if auc_score >= 0.75:
            return "TimeSeriesSplit or Custom Domain-Aware Split"
        elif auc_score >= 0.65:
            return "StratifiedKFold with increased folds (10+)"
        elif auc_score >= 0.57:
            return "StratifiedKFold (current strategy acceptable)"
        else:
            return "Standard StratifiedKFold (highly reliable)"

    def _log_adversarial_results(self, results):
        """실험 추적에 결과 로깅"""
        # 주요 메트릭 로깅
        metrics = {
            'adversarial_auc': results['adversarial_results']['best_auc'],
            'domain_shift_severity_score': self._get_severity_score(
                results['adversarial_results']['domain_shift_severity']
            ),
            'top_feature_importance': results['feature_importance']['top_importances'][0] if results['feature_importance']['top_importances'] else 0,
            'overall_distribution_difference': results['basic_statistics']['overall_mean_diff'],
            'pca_explained_variance_total': sum(results['visualizations']['pca_explained_variance'])
        }

        # 파라미터 로깅
        params = {
            'best_model': results['adversarial_results']['best_model_name'],
            'train_samples': results['data_info']['train_shape'][0],
            'test_samples': results['data_info']['test_shape'][0],
            'n_features': results['data_info']['train_shape'][1],
            'recommended_cv': results['validation_strategy']['recommended_cv']
        }

        self.tracker.log_metrics(metrics)
        self.tracker.log_params(params)

        # 시각화 로깅
        for fig_name, fig in self.figures.items():
            self.tracker.log_figure(fig, fig_name)

        self.tracker.end_run()

    def _get_severity_score(self, severity_text):
        """심각도 텍스트를 숫자로 변환"""
        severity_map = {
            "매우 낮음 (분포 거의 동일)": 1,
            "낮음 (작은 분포 차이)": 2,
            "보통 (주의 필요)": 3,
            "높음 (심각한 분포 차이)": 4,
            "매우 높음 (심각한 도메인 변화)": 5
        }
        return severity_map.get(severity_text, 3)

    def _print_analysis_summary(self, results):
        """분석 결과 요약 출력"""
        print("\n" + "="*70)
        print("🎭 Adversarial Validation 분석 결과 요약")
        print("="*70)

        adv_results = results['adversarial_results']
        print(f"\n🎯 핵심 결과:")
        print(f"  • Adversarial AUC: {adv_results['best_auc']:.4f}")
        print(f"  • Domain Shift 정도: {adv_results['domain_shift_severity']}")
        print(f"  • 최고 성능 모델: {adv_results['best_model_name']}")

        feat_imp = results['feature_importance']
        if feat_imp['top_features']:
            print(f"\n🔍 주요 차별 특성:")
            for i, (feature, importance) in enumerate(zip(feat_imp['top_features'][:3], feat_imp['top_importances'][:3])):
                print(f"  {i+1}. {feature}: {importance:.4f}")

        val_strategy = results['validation_strategy']
        print(f"\n💡 권장 검증 전략:")
        print(f"  • {val_strategy['recommended_cv']}")

        print(f"\n📊 데이터 정보:")
        data_info = results['data_info']
        print(f"  • Train: {data_info['train_shape']}")
        print(f"  • Test: {data_info['test_shape']}")

        print("\n🎨 생성된 시각화:")
        print(f"  • ROC Curve: {self.plots_dir}/adversarial_roc_curve.png")
        print(f"  • 특성 중요도: {self.plots_dir}/feature_importance.png")
        print(f"  • 특성 분포 비교: {self.plots_dir}/top_features_distributions.png")
        print(f"  • PCA 시각화: {self.plots_dir}/pca_train_test_distribution.png")

        print("="*70)

    def get_recommendations(self):
        """분석 결과를 바탕으로 한 구체적 권장사항"""
        if not self.results:
            print("⚠️ 먼저 분석을 실행해주세요.")
            return []

        recommendations = []
        auc_score = self.results['adversarial_results']['best_auc']

        # AUC 점수 기반 권장사항
        if auc_score >= 0.75:
            recommendations.extend([
                {
                    'priority': 'CRITICAL',
                    'category': 'validation_strategy',
                    'issue': f'심각한 Domain Shift 감지 (AUC: {auc_score:.3f})',
                    'recommendation': 'TimeSeriesSplit 또는 domain-aware validation 전략 적용',
                    'expected_impact': 'CV 신뢰도 대폭 향상'
                },
                {
                    'priority': 'HIGH',
                    'category': 'feature_engineering',
                    'issue': 'Train/Test 분포 차이가 매우 큼',
                    'recommendation': 'Domain adaptation 기법 또는 분포 정렬 기법 적용',
                    'expected_impact': '일반화 성능 향상'
                }
            ])
        elif auc_score >= 0.65:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'validation_strategy',
                'issue': f'주의 필요한 분포 차이 (AUC: {auc_score:.3f})',
                'recommendation': 'StratifiedKFold 폴드 수 증가, 더 신중한 검증',
                'expected_impact': 'CV 안정성 향상'
            })

        # 중요 특성 기반 권장사항
        if self.results['feature_importance']['top_features']:
            top_feature = self.results['feature_importance']['top_features'][0]
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'data_analysis',
                'issue': f'주요 차별 특성 발견: {top_feature}',
                'recommendation': f'{top_feature} 특성의 분포 차이 원인 조사',
                'expected_impact': '데이터 품질 향상'
            })

        return recommendations


if __name__ == "__main__":
    """Adversarial Validation 테스트"""
    print("🧪 Adversarial Validation 테스트 시작...")

    # 더미 데이터로 테스트 (실제와 유사하게 분포 차이 생성)
    np.random.seed(42)

    # Train 데이터 생성
    n_train, n_test = 800, 200
    n_features = 20

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i:02d}' for i in range(n_features)]
    )

    # Test 데이터 생성 (일부 특성에 분포 변화 추가)
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i:02d}' for i in range(n_features)]
    )

    # 인위적인 분포 차이 생성 (처음 3개 특성)
    X_test.iloc[:, :3] += 0.5  # 평균 이동
    X_test.iloc[:, :3] *= 1.2  # 분산 증가

    # 분석 실행
    validator = AdversarialValidator()
    results = validator.validate_data_distribution(X_train, X_test)

    # 권장사항 출력
    recommendations = validator.get_recommendations()
    if recommendations:
        print("\n🎯 개선 권장사항:")
        for rec in recommendations:
            print(f"  [{rec['priority']}] {rec['issue']}")
            print(f"      → {rec['recommendation']}")

    print("\n✅ T004 완료: Adversarial Validation 구현 성공!")
    print("🎯 다음 단계: T005 (교차검증 전략 비교 분석)")