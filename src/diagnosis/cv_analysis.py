"""
교차검증 전략 비교 분석
T005 태스크: 다양한 CV 전략의 성능과 신뢰도 비교

목적:
- 현재 CV 전략의 문제점 진단
- 다양한 CV 방법론의 성능 비교
- Adversarial Validation 결과를 반영한 최적 전략 도출
- CV 점수와 실제 성능 간 격차 최소화

비교할 CV 전략:
1. StratifiedKFold (현재 사용)
2. TimeSeriesSplit (시간 순서 고려)
3. GroupKFold (그룹 기반)
4. RepeatedStratifiedKFold (반복 검증)
5. Custom Domain-Aware Split
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold,
    RepeatedStratifiedKFold, cross_val_score, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.config import get_config, get_paths


class CrossValidationAnalyzer:
    """
    교차검증 전략 비교 분석기
    다양한 CV 방법론을 체계적으로 비교하고 최적 전략 도출
    """

    def __init__(self, use_tracking: bool = True):
        """
        CV 분석기 초기화

        Args:
            use_tracking: 실험 추적 사용 여부
        """
        self.use_tracking = use_tracking
        self.results = {}
        self.figures = {}

        # 경로 설정
        self.paths = get_paths()
        self.plots_dir = self.paths['experiments_dir'] / 'plots' / 'cv_analysis'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 실험 추적 설정
        if self.use_tracking:
            self.tracker = ExperimentTracker(
                project_name=get_config('tracking.project_name'),
                experiment_name="cv_strategy_analysis"
            )

        print("📊 교차검증 전략 분석기 초기화 완료")

    def compare_cv_strategies(self, X, y, models=None, adversarial_auc=None):
        """
        다양한 교차검증 전략 종합 비교

        Args:
            X: 특성 데이터
            y: 타겟 데이터
            models: 비교할 모델들 (기본값: RandomForest, LogisticRegression)
            adversarial_auc: Adversarial Validation AUC 점수

        Returns:
            Dict: 분석 결과 딕셔너리
        """
        print("\n📊 교차검증 전략 종합 비교 시작...")
        print("=" * 60)

        # 실험 추적 시작
        if self.use_tracking:
            self.tracker.start_run(
                run_name="cv_strategy_comparison",
                description="다양한 CV 전략 성능 및 신뢰도 비교 분석",
                tags={
                    "analysis_type": "cv_comparison",
                    "task": "T005",
                    "purpose": "optimal_cv_selection"
                }
            )

        # 기본 모델 설정
        if models is None:
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

        # 1. CV 전략 정의
        cv_strategies = self._define_cv_strategies(X, y, adversarial_auc)

        # 2. 각 전략별 성능 평가
        strategy_results = self._evaluate_cv_strategies(X, y, cv_strategies, models)

        # 3. 안정성 분석
        stability_analysis = self._analyze_cv_stability(X, y, cv_strategies, models)

        # 4. 통계적 유의성 검증
        statistical_analysis = self._perform_statistical_tests(strategy_results, models)

        # 5. 시각화 생성
        visualization_results = self._create_cv_visualizations(
            strategy_results, stability_analysis
        )

        # 6. 최적 전략 추천
        recommendations = self._recommend_optimal_strategy(
            strategy_results, stability_analysis, statistical_analysis, adversarial_auc
        )

        # 결과 종합
        comprehensive_results = {
            'cv_strategies': cv_strategies,
            'strategy_results': strategy_results,
            'stability_analysis': stability_analysis,
            'statistical_analysis': statistical_analysis,
            'visualizations': visualization_results,
            'recommendations': recommendations,
            'data_info': {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'n_classes': len(np.unique(y)),
                'class_distribution': np.bincount(y).tolist()
            }
        }

        self.results = comprehensive_results

        # 실험 추적 로깅
        if self.use_tracking:
            self._log_cv_results(comprehensive_results)

        print("\n✅ 교차검증 전략 비교 분석 완료!")
        self._print_analysis_summary(comprehensive_results)

        return comprehensive_results

    def _define_cv_strategies(self, X, y, adversarial_auc):
        """CV 전략 정의"""
        print("\n🎯 1단계: CV 전략 정의")

        random_state = get_config('models.random_seed')
        strategies = {}

        # 1. StratifiedKFold (기본)
        strategies['StratifiedKFold_5'] = {
            'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            'description': '기본 5-fold 계층화 교차검증',
            'type': 'stratified'
        }

        # 2. StratifiedKFold (더 많은 폴드)
        strategies['StratifiedKFold_10'] = {
            'cv': StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state),
            'description': '10-fold 계층화 교차검증 (더 안정적)',
            'type': 'stratified'
        }

        # 3. RepeatedStratifiedKFold
        strategies['RepeatedStratifiedKFold'] = {
            'cv': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state),
            'description': '반복 계층화 교차검증 (5-fold x 3회)',
            'type': 'repeated'
        }

        # 4. TimeSeriesSplit (시간 순서 고려)
        if len(X) >= 50:  # 최소 샘플 수 확인
            n_splits = min(5, len(X) // 10)  # 적절한 split 수 계산
            strategies['TimeSeriesSplit'] = {
                'cv': TimeSeriesSplit(n_splits=n_splits),
                'description': f'시간 순서 기반 {n_splits}-fold 교차검증',
                'type': 'temporal'
            }

        # 5. Custom Domain-Aware Split (Adversarial 결과 반영)
        if adversarial_auc and adversarial_auc > 0.6:
            strategies['DomainAwareSplit'] = {
                'cv': self._create_domain_aware_split(X, y, adversarial_auc),
                'description': 'Domain Shift를 고려한 맞춤형 분할',
                'type': 'domain_aware'
            }

        # 6. GroupKFold (가능한 경우)
        if len(X) >= 100:  # 그룹을 만들기에 충분한 샘플
            groups = self._create_artificial_groups(X, y)
            strategies['GroupKFold'] = {
                'cv': GroupKFold(n_splits=5),
                'description': '그룹 기반 5-fold 교차검증',
                'type': 'group',
                'groups': groups
            }

        print(f"  • 정의된 CV 전략: {len(strategies)}개")
        for name, info in strategies.items():
            print(f"    - {name}: {info['description']}")

        return strategies

    def _create_domain_aware_split(self, X, y, adversarial_auc):
        """Domain-aware 분할 전략 생성"""
        # Adversarial AUC가 높을수록 더 보수적인 분할 사용
        if adversarial_auc > 0.75:
            # 매우 보수적: 시간 순서 기반
            return TimeSeriesSplit(n_splits=3)
        else:
            # 보통 수준: 더 많은 폴드로 안정성 향상
            return StratifiedKFold(n_splits=8, shuffle=True,
                                 random_state=get_config('models.random_seed'))

    def _create_artificial_groups(self, X, y):
        """인위적 그룹 생성 (실제 프로젝트에서는 도메인 지식 활용)"""
        # 간단한 클러스터링 기반 그룹 생성
        from sklearn.cluster import KMeans

        n_groups = min(10, len(X) // 20)  # 적절한 그룹 수
        if n_groups < 3:
            n_groups = 3

        kmeans = KMeans(n_clusters=n_groups, random_state=get_config('models.random_seed'))
        groups = kmeans.fit_predict(X)

        return groups

    def _evaluate_cv_strategies(self, X, y, cv_strategies, models):
        """각 CV 전략별 성능 평가"""
        print("\n⚡ 2단계: CV 전략별 성능 평가")

        results = {}

        for strategy_name, strategy_info in cv_strategies.items():
            print(f"\n  📋 {strategy_name} 평가 중...")

            strategy_results = {'models': {}}
            cv = strategy_info['cv']

            for model_name, model in models.items():
                try:
                    # GroupKFold의 경우 groups 파라미터 필요
                    if strategy_info.get('type') == 'group':
                        groups = strategy_info['groups']
                        cv_results = cross_validate(
                            model, X, y, cv=cv, groups=groups,
                            scoring=['f1_macro', 'accuracy'],
                            return_train_score=True, n_jobs=-1
                        )
                    else:
                        cv_results = cross_validate(
                            model, X, y, cv=cv,
                            scoring=['f1_macro', 'accuracy'],
                            return_train_score=True, n_jobs=-1
                        )

                    # 결과 정리
                    model_results = {
                        'f1_macro_test': cv_results['test_f1_macro'],
                        'f1_macro_train': cv_results['train_f1_macro'],
                        'accuracy_test': cv_results['test_accuracy'],
                        'accuracy_train': cv_results['train_accuracy'],
                        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
                        'f1_macro_std': cv_results['test_f1_macro'].std(),
                        'accuracy_mean': cv_results['test_accuracy'].mean(),
                        'accuracy_std': cv_results['test_accuracy'].std(),
                        'overfitting_gap': (cv_results['train_f1_macro'].mean() -
                                          cv_results['test_f1_macro'].mean())
                    }

                    strategy_results['models'][model_name] = model_results

                    print(f"    • {model_name}: F1={model_results['f1_macro_mean']:.4f}±{model_results['f1_macro_std']:.4f}")

                except Exception as e:
                    print(f"    ❌ {model_name} 오류: {str(e)}")
                    strategy_results['models'][model_name] = None

            strategy_results['description'] = strategy_info['description']
            strategy_results['type'] = strategy_info['type']
            results[strategy_name] = strategy_results

        return results

    def _analyze_cv_stability(self, X, y, cv_strategies, models):
        """CV 안정성 분석 (여러 시드로 반복 실행)"""
        print("\n🔄 3단계: CV 안정성 분석")

        stability_results = {}
        test_seeds = [42, 123, 456, 789, 999]  # 5개 시드로 안정성 테스트

        # 주요 전략들만 안정성 테스트 (시간 절약)
        key_strategies = ['StratifiedKFold_5', 'StratifiedKFold_10', 'RepeatedStratifiedKFold']
        key_strategies = [s for s in key_strategies if s in cv_strategies]

        for strategy_name in key_strategies:
            print(f"  🔄 {strategy_name} 안정성 테스트...")

            strategy_stability = {'models': {}}

            for model_name, base_model in models.items():
                seed_results = []

                for seed in test_seeds:
                    # 모델과 CV 모두 시드 변경
                    model = base_model.__class__(**base_model.get_params())
                    if hasattr(model, 'random_state'):
                        model.set_params(random_state=seed)

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

                    try:
                        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
                        seed_results.append(scores.mean())
                    except:
                        pass

                if seed_results:
                    strategy_stability['models'][model_name] = {
                        'seed_scores': seed_results,
                        'mean_score': np.mean(seed_results),
                        'std_score': np.std(seed_results),
                        'min_score': np.min(seed_results),
                        'max_score': np.max(seed_results),
                        'score_range': np.max(seed_results) - np.min(seed_results),
                        'stability_coefficient': np.std(seed_results) / (np.mean(seed_results) + 1e-8)
                    }

                    print(f"    • {model_name}: 범위={strategy_stability['models'][model_name]['score_range']:.4f}")

            stability_results[strategy_name] = strategy_stability

        return stability_results

    def _perform_statistical_tests(self, strategy_results, models):
        """통계적 유의성 검증"""
        print("\n📈 4단계: 통계적 유의성 검증")

        statistical_results = {}

        # 각 모델별로 전략 간 성능 비교
        for model_name in models.keys():
            print(f"  📊 {model_name} 전략 간 비교:")

            model_comparisons = {}
            strategy_scores = {}

            # 각 전략의 CV 점수 수집
            for strategy_name, strategy_result in strategy_results.items():
                if (strategy_result['models'][model_name] and
                    strategy_result['models'][model_name] is not None):

                    f1_scores = strategy_result['models'][model_name]['f1_macro_test']
                    strategy_scores[strategy_name] = f1_scores

            # 전략 간 t-test 수행
            strategy_names = list(strategy_scores.keys())

            if len(strategy_names) >= 2:
                # 베스트 전략과 다른 전략들 비교
                best_strategy = max(strategy_names,
                                  key=lambda x: strategy_results[x]['models'][model_name]['f1_macro_mean'])

                best_scores = strategy_scores[best_strategy]

                for strategy_name in strategy_names:
                    if strategy_name != best_strategy:
                        other_scores = strategy_scores[strategy_name]

                        # t-test 수행 (길이가 같을 때만)
                        if (len(best_scores) > 1 and len(other_scores) > 1 and
                            len(best_scores) == len(other_scores)):
                            statistic, p_value = stats.ttest_rel(best_scores, other_scores)
                        else:
                            # 길이가 다르면 독립 t-test 사용
                            statistic, p_value = stats.ttest_ind(best_scores, other_scores)

                            model_comparisons[f'{best_strategy}_vs_{strategy_name}'] = {
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'better_strategy': best_strategy if statistic > 0 else strategy_name
                            }

                            significance = "유의함" if p_value < 0.05 else "비유의함"
                            print(f"    • {best_strategy} vs {strategy_name}: p={p_value:.4f} ({significance})")

            statistical_results[model_name] = {
                'best_strategy': best_strategy if 'best_strategy' in locals() else None,
                'comparisons': model_comparisons
            }

        return statistical_results

    def _create_cv_visualizations(self, strategy_results, stability_analysis):
        """CV 비교 시각화 생성"""
        print("\n📊 5단계: CV 비교 시각화 생성")

        # 1. 전략별 성능 비교 박스플롯
        self._plot_strategy_performance_comparison(strategy_results)

        # 2. 안정성 분석 시각화
        if stability_analysis:
            self._plot_stability_analysis(stability_analysis)

        # 3. 과적합 분석 시각화
        self._plot_overfitting_analysis(strategy_results)

        return {
            'performance_comparison': str(self.plots_dir / 'strategy_performance_comparison.png'),
            'stability_analysis': str(self.plots_dir / 'stability_analysis.png'),
            'overfitting_analysis': str(self.plots_dir / 'overfitting_analysis.png')
        }

    def _plot_strategy_performance_comparison(self, strategy_results):
        """전략별 성능 비교 박스플롯"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        models = ['RandomForest', 'LogisticRegression']

        for idx, model_name in enumerate(models):
            ax = axes[idx]

            box_data = []
            box_labels = []

            for strategy_name, strategy_result in strategy_results.items():
                if (strategy_result['models'].get(model_name) and
                    strategy_result['models'][model_name] is not None):

                    f1_scores = strategy_result['models'][model_name]['f1_macro_test']
                    box_data.append(f1_scores)
                    box_labels.append(strategy_name.replace('StratifiedKFold_', 'SK'))

            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

                # 색상 설정
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                ax.set_title(f'{model_name} - CV Strategy Performance')
                ax.set_ylabel('F1-Score (Macro)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 저장
        perf_path = self.plots_dir / 'strategy_performance_comparison.png'
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        self.figures['performance_comparison'] = plt.gcf()
        plt.close()

    def _plot_stability_analysis(self, stability_analysis):
        """안정성 분석 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        models = ['RandomForest', 'LogisticRegression']

        for idx, model_name in enumerate(models):
            ax = axes[idx]

            strategies = []
            stability_scores = []
            score_ranges = []

            for strategy_name, strategy_result in stability_analysis.items():
                if strategy_result['models'].get(model_name):
                    model_stability = strategy_result['models'][model_name]
                    strategies.append(strategy_name.replace('StratifiedKFold_', 'SK'))
                    stability_scores.append(model_stability['stability_coefficient'])
                    score_ranges.append(model_stability['score_range'])

            if strategies:
                x_pos = np.arange(len(strategies))

                # 안정성 계수 (낮을수록 안정적)
                bars1 = ax.bar(x_pos - 0.2, stability_scores, 0.4,
                             label='Stability Coefficient', alpha=0.7, color='blue')

                # 점수 범위 (낮을수록 안정적)
                ax2 = ax.twinx()
                bars2 = ax2.bar(x_pos + 0.2, score_ranges, 0.4,
                              label='Score Range', alpha=0.7, color='red')

                ax.set_xlabel('CV Strategy')
                ax.set_ylabel('Stability Coefficient', color='blue')
                ax2.set_ylabel('Score Range', color='red')
                ax.set_title(f'{model_name} - CV Strategy Stability')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(strategies, rotation=45)

                # 범례
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')

                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 저장
        stability_path = self.plots_dir / 'stability_analysis.png'
        plt.savefig(stability_path, dpi=300, bbox_inches='tight')
        self.figures['stability_analysis'] = plt.gcf()
        plt.close()

    def _plot_overfitting_analysis(self, strategy_results):
        """과적합 분석 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))

        models = ['RandomForest', 'LogisticRegression']
        colors = ['blue', 'red']

        for model_idx, (model_name, color) in enumerate(zip(models, colors)):
            strategies = []
            overfitting_gaps = []

            for strategy_name, strategy_result in strategy_results.items():
                if (strategy_result['models'].get(model_name) and
                    strategy_result['models'][model_name] is not None):

                    gap = strategy_result['models'][model_name]['overfitting_gap']
                    strategies.append(strategy_name.replace('StratifiedKFold_', 'SK'))
                    overfitting_gaps.append(gap)

            if strategies:
                x_pos = np.arange(len(strategies)) + model_idx * 0.4
                ax.bar(x_pos, overfitting_gaps, 0.4,
                      label=model_name, alpha=0.7, color=color)

        ax.set_xlabel('CV Strategy')
        ax.set_ylabel('Overfitting Gap (Train - Test F1)')
        ax.set_title('Overfitting Analysis by CV Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # x축 라벨 설정
        if strategies:  # strategies가 정의되어 있다면
            ax.set_xticks(np.arange(len(strategies)) + 0.2)
            ax.set_xticklabels(strategies, rotation=45)

        plt.tight_layout()

        # 저장
        overfitting_path = self.plots_dir / 'overfitting_analysis.png'
        plt.savefig(overfitting_path, dpi=300, bbox_inches='tight')
        self.figures['overfitting_analysis'] = plt.gcf()
        plt.close()

    def _recommend_optimal_strategy(self, strategy_results, stability_analysis,
                                   statistical_analysis, adversarial_auc):
        """최적 CV 전략 추천"""
        print("\n🎯 6단계: 최적 CV 전략 추천")

        recommendations = {}

        # 각 모델별 최적 전략 분석
        for model_name in ['RandomForest', 'LogisticRegression']:
            print(f"\n  🤖 {model_name} 최적 전략 분석:")

            # 성능 기반 랭킹
            performance_ranking = []
            for strategy_name, strategy_result in strategy_results.items():
                if (strategy_result['models'].get(model_name) and
                    strategy_result['models'][model_name] is not None):

                    model_result = strategy_result['models'][model_name]
                    performance_ranking.append({
                        'strategy': strategy_name,
                        'f1_mean': model_result['f1_macro_mean'],
                        'f1_std': model_result['f1_macro_std'],
                        'overfitting_gap': model_result['overfitting_gap']
                    })

            # 성능순 정렬
            performance_ranking.sort(key=lambda x: x['f1_mean'], reverse=True)

            # 안정성 고려
            stability_bonus = {}
            if model_name in stability_analysis.get('StratifiedKFold_5', {}).get('models', {}):
                for strategy_name, strategy_stability in stability_analysis.items():
                    if strategy_stability['models'].get(model_name):
                        stability_coef = strategy_stability['models'][model_name]['stability_coefficient']
                        # 낮은 안정성 계수(더 안정)에 보너스
                        stability_bonus[strategy_name] = max(0, 0.02 - stability_coef)

            # 종합 점수 계산
            final_ranking = []
            for perf in performance_ranking:
                strategy = perf['strategy']
                base_score = perf['f1_mean']
                stability_boost = stability_bonus.get(strategy, 0)
                overfitting_penalty = max(0, perf['overfitting_gap'] - 0.1) * 0.1

                final_score = base_score + stability_boost - overfitting_penalty

                final_ranking.append({
                    'strategy': strategy,
                    'final_score': final_score,
                    'base_performance': base_score,
                    'stability_boost': stability_boost,
                    'overfitting_penalty': overfitting_penalty
                })

            final_ranking.sort(key=lambda x: x['final_score'], reverse=True)

            # 최적 전략 선정
            if final_ranking:
                best_strategy = final_ranking[0]

                recommendations[model_name] = {
                    'recommended_strategy': best_strategy['strategy'],
                    'final_score': best_strategy['final_score'],
                    'ranking': final_ranking[:3],  # 상위 3개
                    'reason': self._generate_recommendation_reason(
                        best_strategy, adversarial_auc
                    )
                }

                print(f"    🏆 추천: {best_strategy['strategy']} (점수: {best_strategy['final_score']:.4f})")
                print(f"    💭 이유: {recommendations[model_name]['reason']}")

        # 전체 추천사항
        overall_recommendation = self._generate_overall_recommendation(
            recommendations, adversarial_auc
        )

        return {
            'model_specific': recommendations,
            'overall': overall_recommendation
        }

    def _generate_recommendation_reason(self, best_strategy, adversarial_auc):
        """추천 이유 생성"""
        strategy_name = best_strategy['strategy']
        reasons = []

        if 'StratifiedKFold_10' in strategy_name:
            reasons.append("더 많은 폴드로 안정성 확보")
        if 'Repeated' in strategy_name:
            reasons.append("반복 검증으로 신뢰성 향상")
        if 'TimeSeriesSplit' in strategy_name:
            reasons.append("시간 순서 고려로 데이터 누수 방지")
        if 'DomainAware' in strategy_name:
            reasons.append("Domain Shift 고려한 맞춤형 전략")

        if adversarial_auc and adversarial_auc > 0.7:
            reasons.append("높은 Domain Shift 대응")

        if best_strategy['overfitting_penalty'] < 0.01:
            reasons.append("낮은 과적합 위험")

        return "; ".join(reasons) if reasons else "최고 성능"

    def _generate_overall_recommendation(self, model_recommendations, adversarial_auc):
        """전체 추천사항 생성"""
        # 가장 많이 추천된 전략 찾기
        strategy_votes = {}
        for model_rec in model_recommendations.values():
            strategy = model_rec['recommended_strategy']
            strategy_votes[strategy] = strategy_votes.get(strategy, 0) + 1

        if strategy_votes:
            best_overall_strategy = max(strategy_votes, key=strategy_votes.get)
        else:
            best_overall_strategy = "StratifiedKFold_5"

        # 추가 조건부 추천
        additional_recommendations = []

        if adversarial_auc and adversarial_auc > 0.75:
            additional_recommendations.append("심각한 Domain Shift → TimeSeriesSplit 우선 고려")
        elif adversarial_auc and adversarial_auc > 0.65:
            additional_recommendations.append("보통 Domain Shift → StratifiedKFold 폴드 수 증가")

        return {
            'best_strategy': best_overall_strategy,
            'consensus_level': strategy_votes.get(best_overall_strategy, 0),
            'additional_recommendations': additional_recommendations
        }

    def _log_cv_results(self, results):
        """실험 추적에 결과 로깅"""
        # 주요 메트릭 로깅
        strategy_results = results['strategy_results']

        metrics = {}

        # 각 전략별 성능 로깅
        for strategy_name, strategy_result in strategy_results.items():
            for model_name, model_result in strategy_result['models'].items():
                if model_result:
                    prefix = f"{strategy_name}_{model_name}"
                    metrics[f"{prefix}_f1_mean"] = model_result['f1_macro_mean']
                    metrics[f"{prefix}_f1_std"] = model_result['f1_macro_std']
                    metrics[f"{prefix}_overfitting_gap"] = model_result['overfitting_gap']

        # 추천 전략 로깅
        for model_name, recommendation in results['recommendations']['model_specific'].items():
            metrics[f"best_strategy_{model_name}"] = 1  # 카테고리 변수는 숫자로

        self.tracker.log_metrics(metrics)

        # 파라미터 로깅
        params = {
            'n_strategies': len(strategy_results),
            'overall_best_strategy': results['recommendations']['overall']['best_strategy'],
            'n_samples': results['data_info']['n_samples'],
            'n_classes': results['data_info']['n_classes']
        }
        self.tracker.log_params(params)

        # 시각화 로깅
        for fig_name, fig in self.figures.items():
            self.tracker.log_figure(fig, fig_name)

        self.tracker.end_run()

    def _print_analysis_summary(self, results):
        """분석 결과 요약 출력"""
        print("\n" + "="*70)
        print("📊 교차검증 전략 분석 결과 요약")
        print("="*70)

        # 전체 추천
        overall_rec = results['recommendations']['overall']
        print(f"\n🏆 전체 추천 전략: {overall_rec['best_strategy']}")
        print(f"   합의 수준: {overall_rec['consensus_level']}/2 모델")

        # 모델별 추천
        print(f"\n🤖 모델별 추천:")
        for model_name, rec in results['recommendations']['model_specific'].items():
            print(f"  • {model_name}: {rec['recommended_strategy']}")
            print(f"    └─ 이유: {rec['reason']}")

        # 성능 요약
        print(f"\n📊 성능 요약:")
        best_performances = {}
        for strategy_name, strategy_result in results['strategy_results'].items():
            for model_name, model_result in strategy_result['models'].items():
                if model_result:
                    key = f"{model_name}"
                    if key not in best_performances:
                        best_performances[key] = []
                    best_performances[key].append({
                        'strategy': strategy_name,
                        'score': model_result['f1_macro_mean']
                    })

        for model_name, performances in best_performances.items():
            best_perf = max(performances, key=lambda x: x['score'])
            print(f"  • {model_name} 최고: {best_perf['score']:.4f} ({best_perf['strategy']})")

        # 추가 권장사항
        if overall_rec['additional_recommendations']:
            print(f"\n💡 추가 권장사항:")
            for rec in overall_rec['additional_recommendations']:
                print(f"  • {rec}")

        print("="*70)


if __name__ == "__main__":
    """교차검증 전략 분석 테스트"""
    print("🧪 교차검증 전략 분석 테스트 시작...")

    # 더미 데이터 생성
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_informative=15,
        n_clusters_per_class=1,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(20)])
    y_series = pd.Series(y)

    # 분석 실행 (Adversarial AUC 시뮬레이션)
    analyzer = CrossValidationAnalyzer()
    results = analyzer.compare_cv_strategies(
        X_df, y_series,
        adversarial_auc=0.72  # 높은 Domain Shift 시뮬레이션
    )

    print("\n✅ T005 완료: 교차검증 전략 비교 분석 성공!")
    print("🎯 다음 단계: T006 (과적합 진단 시스템 구현)")