"""
성능 격차 분석기
T004-T010 태스크의 핵심 성능 진단을 위한 통합 분석 시스템

현재 상황:
- CV 점수: ~77% (Random Forest 기준)
- 실제 제출 점수: 67.6%
- 성능 격차: 9.4%

분석 목표:
1. CV vs 실제 점수 격차 원인 규명
2. 과적합/과소적합 진단
3. 데이터 리키지 탐지
4. 검증 전략 최적화 방안 제시
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.config import get_config, get_paths


class PerformanceGapAnalyzer:
    """
    성능 격차 종합 분석기
    CV 점수와 실제 성능 간의 격차를 체계적으로 분석
    """

    def __init__(self, use_tracking: bool = True):
        """
        성능 격차 분석기 초기화

        Args:
            use_tracking: 실험 추적 사용 여부
        """
        self.use_tracking = use_tracking
        self.results = {}
        self.figures = {}

        # 경로 설정
        self.paths = get_paths()
        self.plots_dir = self.paths['experiments_dir'] / 'plots' / 'diagnosis'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 실험 추적 설정
        if self.use_tracking:
            self.tracker = ExperimentTracker(
                project_name=get_config('tracking.project_name'),
                experiment_name="performance_gap_analysis"
            )

        print("🔍 성능 격차 분석기 초기화 완료")

    def analyze_performance_gap(self, X_train, y_train, X_val=None, y_val=None,
                              model=None, cv_folds=5):
        """
        종합적인 성능 격차 분석 수행

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            X_val: 검증 데이터 특성 (선택사항)
            y_val: 검증 데이터 레이블 (선택사항)
            model: 분석할 모델 (기본값: RandomForest)
            cv_folds: 교차검증 폴드 수

        Returns:
            Dict: 분석 결과 딕셔너리
        """
        print("\n🚀 성능 격차 종합 분석 시작...")
        print("=" * 50)

        # 기본 모델 설정
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=get_config('models.random_seed'),
                n_jobs=-1
            )

        # 실험 추적 시작
        if self.use_tracking:
            self.tracker.start_run(
                run_name="performance_gap_comprehensive",
                description="CV vs 실제 성능 격차 종합 분석",
                tags={"analysis_type": "performance_gap", "task": "T004-T010"}
            )

        # 1. 기본 CV 성능 분석
        cv_results = self._analyze_cv_performance(X_train, y_train, model, cv_folds)

        # 2. 홀드아웃 검증 분석 (검증 데이터가 있는 경우)
        holdout_results = None
        if X_val is not None and y_val is not None:
            holdout_results = self._analyze_holdout_performance(
                X_train, y_train, X_val, y_val, model
            )

        # 3. 학습 곡선 분석
        learning_results = self._analyze_learning_curves(X_train, y_train, model)

        # 4. 검증 곡선 분석
        validation_results = self._analyze_validation_curves(X_train, y_train, model)

        # 5. 데이터 누수 검사
        leakage_results = self._check_data_leakage(X_train, y_train, model)

        # 6. 특성 안정성 분석
        stability_results = self._analyze_feature_stability(X_train, y_train)

        # 결과 종합
        comprehensive_results = {
            'cv_analysis': cv_results,
            'holdout_analysis': holdout_results,
            'learning_curves': learning_results,
            'validation_curves': validation_results,
            'leakage_check': leakage_results,
            'stability_analysis': stability_results,
            'summary': self._generate_summary(cv_results, holdout_results)
        }

        self.results = comprehensive_results

        # 실험 추적 로깅
        if self.use_tracking:
            self._log_analysis_results(comprehensive_results)

        print("\n✅ 성능 격차 분석 완료!")
        self._print_analysis_summary(comprehensive_results)

        return comprehensive_results

    def _analyze_cv_performance(self, X, y, model, cv_folds):
        """교차검증 성능 상세 분석"""
        print("\n📊 1단계: 교차검증 성능 분석")

        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=get_config('models.random_seed')
        )

        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)

        # 폴드별 세부 분석
        fold_details = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_fold_train, y_fold_train)
            y_pred = model_fold.predict(X_fold_val)

            fold_score = f1_score(y_fold_val, y_pred, average='macro')
            fold_details.append({
                'fold': fold_idx + 1,
                'score': fold_score,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })

        results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max(),
            'scores': cv_scores.tolist(),
            'fold_details': fold_details,
            'cv_range': cv_scores.max() - cv_scores.min()
        }

        print(f"  • 평균 CV 점수: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        print(f"  • 점수 범위: {results['min_score']:.4f} - {results['max_score']:.4f}")
        print(f"  • CV 변동성: {results['cv_range']:.4f}")

        return results

    def _analyze_holdout_performance(self, X_train, y_train, X_val, y_val, model):
        """홀드아웃 검증 성능 분석"""
        print("\n📈 2단계: 홀드아웃 검증 분석")

        model_holdout = model.__class__(**model.get_params())
        model_holdout.fit(X_train, y_train)
        y_pred = model_holdout.predict(X_val)

        holdout_score = f1_score(y_val, y_pred, average='macro')

        results = {
            'holdout_score': holdout_score,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }

        print(f"  • 홀드아웃 점수: {results['holdout_score']:.4f}")

        return results

    def _analyze_learning_curves(self, X, y, model):
        """학습 곡선 분석으로 과적합/과소적합 진단"""
        print("\n📚 3단계: 학습 곡선 분석")

        train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=get_config('models.random_seed')
        )

        # 학습 곡선 시각화
        plt.figure(figsize=(10, 6))

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Macro F1 Score')
        plt.title('Learning Curves - Overfitting/Underfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 그래프 저장
        learning_curve_path = self.plots_dir / 'learning_curves.png'
        plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
        self.figures['learning_curves'] = plt.gcf()
        plt.close()

        # 과적합/과소적합 진단
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        gap = final_train_score - final_val_score

        if gap > 0.05:
            diagnosis = "과적합 (Overfitting)"
        elif gap < 0.01:
            diagnosis = "과소적합 (Underfitting)"
        else:
            diagnosis = "적절한 적합 (Good Fit)"

        results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'final_gap': gap,
            'diagnosis': diagnosis,
            'plot_path': str(learning_curve_path)
        }

        print(f"  • 최종 훈련 점수: {final_train_score:.4f}")
        print(f"  • 최종 검증 점수: {final_val_score:.4f}")
        print(f"  • 성능 격차: {gap:.4f}")
        print(f"  • 진단 결과: {diagnosis}")

        return results

    def _analyze_validation_curves(self, X, y, model):
        """검증 곡선으로 하이퍼파라미터 민감도 분석"""
        print("\n🎛️ 4단계: 검증 곡선 분석")

        # RandomForest의 주요 파라미터 분석
        if hasattr(model, 'n_estimators'):
            param_name = 'n_estimators'
            param_range = [10, 50, 100, 200, 500]
        else:
            param_name = 'C'
            param_range = [0.001, 0.01, 0.1, 1, 10, 100]

        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )

        # 검증 곡선 시각화
        plt.figure(figsize=(10, 6))

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')

        plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel(param_name)
        plt.ylabel('Macro F1 Score')
        plt.title(f'Validation Curve - {param_name} Sensitivity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 그래프 저장
        validation_curve_path = self.plots_dir / f'validation_curve_{param_name}.png'
        plt.savefig(validation_curve_path, dpi=300, bbox_inches='tight')
        self.figures[f'validation_curve_{param_name}'] = plt.gcf()
        plt.close()

        # 최적 파라미터 찾기
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]

        results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': train_mean.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'best_param': best_param,
            'best_score': best_score,
            'param_sensitivity': val_mean.std(),
            'plot_path': str(validation_curve_path)
        }

        print(f"  • 분석 파라미터: {param_name}")
        print(f"  • 최적값: {best_param}")
        print(f"  • 최적 점수: {best_score:.4f}")
        print(f"  • 파라미터 민감도: {results['param_sensitivity']:.4f}")

        return results

    def _check_data_leakage(self, X, y, model):
        """데이터 누수 검사"""
        print("\n🚰 5단계: 데이터 누수 검사")

        # 랜덤 라벨로 학습해서 성능이 높으면 누수 의심
        y_random = np.random.permutation(y.values)

        cv_scores_original = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        cv_scores_random = cross_val_score(model, X, y_random, cv=5, scoring='f1_macro')

        original_mean = cv_scores_original.mean()
        random_mean = cv_scores_random.mean()

        # 랜덤 성능이 너무 높으면 데이터 누수 의심
        leakage_suspected = random_mean > 0.3  # 21클래스 분류에서 랜덤은 ~0.048

        results = {
            'original_score': original_mean,
            'random_label_score': random_mean,
            'leakage_suspected': leakage_suspected,
            'baseline_expected': 1.0 / len(np.unique(y)),  # 클래스 수의 역수
            'leakage_ratio': random_mean / (1.0 / len(np.unique(y)))
        }

        print(f"  • 원본 라벨 점수: {original_mean:.4f}")
        print(f"  • 랜덤 라벨 점수: {random_mean:.4f}")
        print(f"  • 예상 베이스라인: {results['baseline_expected']:.4f}")
        print(f"  • 누수 의심 여부: {'의심됨' if leakage_suspected else '정상'}")

        return results

    def _analyze_feature_stability(self, X, y):
        """특성 안정성 분석"""
        print("\n📊 6단계: 특성 안정성 분석")

        # 특성별 통계 안정성 확인
        feature_stats = {}

        for column in X.columns:
            feature_data = X[column]
            stats = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'skew': feature_data.skew(),
                'kurtosis': feature_data.kurtosis(),
                'missing_ratio': feature_data.isnull().sum() / len(feature_data)
            }
            feature_stats[column] = stats

        # 불안정한 특성 식별 (높은 skew, kurtosis)
        unstable_features = []
        for feature, stats in feature_stats.items():
            if abs(stats['skew']) > 3 or abs(stats['kurtosis']) > 10:
                unstable_features.append(feature)

        results = {
            'feature_stats': feature_stats,
            'unstable_features': unstable_features,
            'stability_score': 1 - (len(unstable_features) / len(X.columns))
        }

        print(f"  • 총 특성 수: {len(X.columns)}")
        print(f"  • 불안정한 특성 수: {len(unstable_features)}")
        print(f"  • 안정성 점수: {results['stability_score']:.4f}")

        return results

    def _generate_summary(self, cv_results, holdout_results):
        """분석 결과 종합 요약"""
        summary = {
            'cv_mean': cv_results['mean_score'],
            'cv_std': cv_results['std_score'],
            'cv_stability': cv_results['cv_range']
        }

        if holdout_results:
            summary['holdout_score'] = holdout_results['holdout_score']
            summary['cv_holdout_gap'] = abs(cv_results['mean_score'] - holdout_results['holdout_score'])

        # 현재 알려진 실제 제출 점수와 비교
        current_submission_score = get_config('targets.current_score', 0.67596)
        summary['submission_score'] = current_submission_score
        summary['cv_submission_gap'] = cv_results['mean_score'] - current_submission_score

        return summary

    def _log_analysis_results(self, results):
        """실험 추적 시스템에 결과 로깅"""
        # 주요 메트릭 로깅
        metrics = {
            'cv_mean_score': results['cv_analysis']['mean_score'],
            'cv_std_score': results['cv_analysis']['std_score'],
            'cv_range': results['cv_analysis']['cv_range'],
            'learning_gap': results['learning_curves']['final_gap'],
            'leakage_ratio': results['leakage_check']['leakage_ratio'],
            'stability_score': results['stability_analysis']['stability_score']
        }

        if results['holdout_analysis']:
            metrics['holdout_score'] = results['holdout_analysis']['holdout_score']
            metrics['cv_holdout_gap'] = results['summary']['cv_holdout_gap']

        metrics['cv_submission_gap'] = results['summary']['cv_submission_gap']

        self.tracker.log_metrics(metrics)

        # 그래프들 로깅
        for fig_name, fig in self.figures.items():
            self.tracker.log_figure(fig, fig_name)

        # 분석 완료
        self.tracker.end_run()

    def _print_analysis_summary(self, results):
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("📋 성능 격차 분석 결과 요약")
        print("="*60)

        print(f"\n🎯 교차검증 성능:")
        print(f"  • 평균 점수: {results['cv_analysis']['mean_score']:.4f} ± {results['cv_analysis']['std_score']:.4f}")
        print(f"  • 점수 범위: {results['cv_analysis']['cv_range']:.4f}")

        if results['holdout_analysis']:
            print(f"\n📊 홀드아웃 검증:")
            print(f"  • 홀드아웃 점수: {results['holdout_analysis']['holdout_score']:.4f}")
            print(f"  • CV-홀드아웃 격차: {results['summary']['cv_holdout_gap']:.4f}")

        print(f"\n🚨 핵심 문제점:")
        print(f"  • CV-실제제출 격차: {results['summary']['cv_submission_gap']:.4f}")
        print(f"  • 학습곡선 진단: {results['learning_curves']['diagnosis']}")
        print(f"  • 데이터 누수 의심: {'예' if results['leakage_check']['leakage_suspected'] else '아니오'}")

        print(f"\n💡 권장사항:")
        if results['summary']['cv_submission_gap'] > 0.05:
            print("  • CV 전략 재검토 필요 (시간 기반 분할 고려)")
        if results['learning_curves']['diagnosis'] == "과적합 (Overfitting)":
            print("  • 정규화 강화 또는 모델 복잡도 감소")
        if results['leakage_check']['leakage_suspected']:
            print("  • 데이터 누수 조사 필요")

        print("="*60)

    def get_recommendations(self):
        """분석 결과를 바탕으로 한 개선 권장사항 반환"""
        if not self.results:
            print("⚠️ 먼저 분석을 실행해주세요.")
            return []

        recommendations = []
        results = self.results

        # CV-제출 점수 격차가 큰 경우
        if results['summary']['cv_submission_gap'] > 0.05:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'validation_strategy',
                'issue': 'CV와 실제 점수 격차가 큼',
                'recommendation': 'TimeSeriesSplit 또는 GroupKFold 사용 검토',
                'expected_impact': 'CV 신뢰도 향상'
            })

        # 과적합 진단인 경우
        if results['learning_curves']['diagnosis'] == "과적합 (Overfitting)":
            recommendations.append({
                'priority': 'HIGH',
                'category': 'model_complexity',
                'issue': '모델이 과적합됨',
                'recommendation': '정규화 강화, early stopping, 데이터 증강',
                'expected_impact': '일반화 성능 향상'
            })

        # 데이터 누수 의심인 경우
        if results['leakage_check']['leakage_suspected']:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'data_quality',
                'issue': '데이터 누수 의심',
                'recommendation': '특성 생성 과정 재검토 및 시간 순서 확인',
                'expected_impact': '올바른 성능 평가'
            })

        # CV 변동성이 큰 경우
        if results['cv_analysis']['cv_range'] > 0.1:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'model_stability',
                'issue': 'CV 점수 변동성이 큼',
                'recommendation': '앙상블 방법 사용 또는 더 많은 폴드 사용',
                'expected_impact': '모델 안정성 향상'
            })

        return recommendations


if __name__ == "__main__":
    """성능 격차 분석기 테스트"""
    print("🧪 성능 격차 분석기 테스트 시작...")

    # 더미 데이터로 테스트
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=52,
        n_classes=21,
        n_informative=30,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(52)])
    y_df = pd.Series(y)

    # 분석기 생성 및 실행
    analyzer = PerformanceGapAnalyzer()
    results = analyzer.analyze_performance_gap(X_df, y_df)

    # 권장사항 출력
    recommendations = analyzer.get_recommendations()

    if recommendations:
        print("\n🎯 개선 권장사항:")
        for rec in recommendations:
            print(f"  [{rec['priority']}] {rec['issue']}")
            print(f"      → {rec['recommendation']}")

    print("\n✅ T003 완료: 성능 진단 디렉토리 구조 생성 성공!")
    print("🎯 다음 단계: T004 (Adversarial Validation 구현)")