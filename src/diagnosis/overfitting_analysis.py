"""
과적합 진단 시스템 구현
T006 태스크: 과적합/과소적합 정밀 진단 및 해결방안 제시

목적:
- 현재 77% CV vs 67.6% 실제 점수 격차의 과적합 원인 정밀 분석
- 다양한 각도에서 과적합 정도 정량화
- 모델별, 특성별, 데이터 크기별 과적합 패턴 분석
- 구체적이고 실행 가능한 과적합 해결 방안 제시

분석 방법론:
1. Learning Curves (데이터 크기별 성능)
2. Validation Curves (하이퍼파라미터별 성능)
3. Feature Learning Curves (특성 수별 성능)
4. Regularization Analysis (정규화 효과)
5. Early Stopping Analysis (최적 중단점)
6. Ensemble Diversity Analysis (앙상블 다양성)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.config import get_config, get_paths


class OverfittingAnalyzer:
    """
    과적합 진단 및 해결 시스템
    다각도 과적합 분석으로 CV-실제 점수 격차 원인 규명 및 해결방안 도출
    """

    def __init__(self, use_tracking: bool = True):
        """
        과적합 분석기 초기화

        Args:
            use_tracking: 실험 추적 사용 여부
        """
        self.use_tracking = use_tracking
        self.results = {}
        self.figures = {}

        # 경로 설정
        self.paths = get_paths()
        self.plots_dir = self.paths['experiments_dir'] / 'plots' / 'overfitting'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 실험 추적 설정
        if self.use_tracking:
            self.tracker = ExperimentTracker(
                project_name=get_config('tracking.project_name'),
                experiment_name="overfitting_analysis"
            )

        print("🔍 과적합 진단 시스템 초기화 완료")

    def comprehensive_overfitting_analysis(self, X, y, models=None, target_score_gap=0.096):
        """
        종합적인 과적합 분석 수행

        Args:
            X: 특성 데이터
            y: 타겟 데이터
            models: 분석할 모델들
            target_score_gap: 목표로 하는 CV-실제 점수 격차 (9.6%)

        Returns:
            Dict: 분석 결과 딕셔너리
        """
        print("\n🔍 종합적인 과적합 분석 시작...")
        print("=" * 60)

        # 실험 추적 시작
        if self.use_tracking:
            self.tracker.start_run(
                run_name="comprehensive_overfitting_analysis",
                description="과적합/과소적합 정밀 진단 및 해결방안 도출",
                tags={
                    "analysis_type": "overfitting_diagnosis",
                    "task": "T006",
                    "purpose": "cv_score_gap_resolution"
                }
            )

        # 기본 모델 설정
        if models is None:
            models = self._define_analysis_models()

        # 1. 학습 곡선 분석 (데이터 크기별)
        learning_analysis = self._analyze_learning_curves(X, y, models)

        # 2. 검증 곡선 분석 (하이퍼파라미터별)
        validation_analysis = self._analyze_validation_curves(X, y, models)

        # 3. 특성 학습 곡선 분석 (특성 수별)
        feature_analysis = self._analyze_feature_learning_curves(X, y, models)

        # 4. 정규화 효과 분석
        regularization_analysis = self._analyze_regularization_effects(X, y)

        # 5. Early Stopping 분석
        early_stopping_analysis = self._analyze_early_stopping(X, y)

        # 6. 앙상블 다양성 분석
        ensemble_analysis = self._analyze_ensemble_diversity(X, y)

        # 7. 과적합 심각도 평가
        severity_assessment = self._assess_overfitting_severity(
            learning_analysis, validation_analysis, target_score_gap
        )

        # 8. 해결방안 생성
        solutions = self._generate_overfitting_solutions(
            learning_analysis, validation_analysis, feature_analysis,
            regularization_analysis, severity_assessment
        )

        # 결과 종합
        comprehensive_results = {
            'learning_curves': learning_analysis,
            'validation_curves': validation_analysis,
            'feature_curves': feature_analysis,
            'regularization': regularization_analysis,
            'early_stopping': early_stopping_analysis,
            'ensemble_diversity': ensemble_analysis,
            'severity_assessment': severity_assessment,
            'solutions': solutions,
            'data_info': {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'n_classes': len(np.unique(y)),
                'target_score_gap': target_score_gap
            }
        }

        self.results = comprehensive_results

        # 실험 추적 로깅
        if self.use_tracking:
            self._log_overfitting_results(comprehensive_results)

        print("\n✅ 종합적인 과적합 분석 완료!")
        self._print_analysis_summary(comprehensive_results)

        return comprehensive_results

    def _define_analysis_models(self):
        """분석용 모델들 정의"""
        random_state = get_config('models.random_seed')

        models = {
            'RandomForest_Default': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            'RandomForest_Regularized': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1
            ),
            'LogisticRegression_L2': LogisticRegression(
                C=1.0,
                random_state=random_state,
                max_iter=1000
            ),
            'LogisticRegression_L1': LogisticRegression(
                penalty='l1',
                C=1.0,
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            )
        }

        return models

    def _analyze_learning_curves(self, X, y, models):
        """학습 곡선 분석"""
        print("\n📚 1단계: 학습 곡선 분석")

        results = {}
        train_sizes = np.linspace(0.1, 1.0, 10)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (model_name, model) in enumerate(models.items()):
            if idx >= 4:
                break

            print(f"  📊 {model_name} 학습 곡선 분석...")

            try:
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model, X, y,
                    train_sizes=train_sizes,
                    cv=5,
                    scoring='f1_macro',
                    n_jobs=-1,
                    random_state=get_config('models.random_seed')
                )

                # 통계 계산
                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                val_mean = val_scores.mean(axis=1)
                val_std = val_scores.std(axis=1)

                # 과적합 지표 계산
                final_gap = train_mean[-1] - val_mean[-1]
                convergence_point = self._find_convergence_point(train_mean, val_mean)
                optimal_size = self._find_optimal_training_size(train_sizes_abs, val_mean)

                results[model_name] = {
                    'train_sizes': train_sizes_abs.tolist(),
                    'train_scores_mean': train_mean.tolist(),
                    'train_scores_std': train_std.tolist(),
                    'val_scores_mean': val_mean.tolist(),
                    'val_scores_std': val_std.tolist(),
                    'final_gap': final_gap,
                    'convergence_point': convergence_point,
                    'optimal_training_size': optimal_size,
                    'overfitting_severity': self._classify_overfitting_severity(final_gap)
                }

                # 시각화
                ax = axes[idx]
                ax.plot(train_sizes_abs, train_mean, 'o-', color='blue',
                       label=f'Training Score')
                ax.fill_between(train_sizes_abs, train_mean - train_std,
                              train_mean + train_std, alpha=0.1, color='blue')

                ax.plot(train_sizes_abs, val_mean, 'o-', color='red',
                       label=f'Validation Score')
                ax.fill_between(train_sizes_abs, val_mean - val_std,
                              val_mean + val_std, alpha=0.1, color='red')

                ax.set_title(f'{model_name}\nGap: {final_gap:.3f} ({results[model_name]["overfitting_severity"]})')
                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('F1-Score (Macro)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                print(f"    • 최종 격차: {final_gap:.4f} ({results[model_name]['overfitting_severity']})")
                print(f"    • 최적 데이터 크기: {optimal_size}")

            except Exception as e:
                print(f"    ❌ {model_name} 오류: {str(e)}")
                results[model_name] = None

        plt.tight_layout()

        # 그래프 저장
        learning_path = self.plots_dir / 'learning_curves_analysis.png'
        plt.savefig(learning_path, dpi=300, bbox_inches='tight')
        self.figures['learning_curves'] = plt.gcf()
        plt.close()

        return results

    def _analyze_validation_curves(self, X, y, models):
        """검증 곡선 분석 (하이퍼파라미터별)"""
        print("\n🎛️ 2단계: 검증 곡선 분석")

        results = {}

        # RandomForest n_estimators 분석
        if 'RandomForest_Default' in models:
            rf_model = models['RandomForest_Default']
            param_range = [10, 50, 100, 200, 500, 1000]

            print("  🌲 RandomForest n_estimators 분석...")

            train_scores, val_scores = validation_curve(
                rf_model, X, y,
                param_name='n_estimators',
                param_range=param_range,
                cv=5, scoring='f1_macro', n_jobs=-1
            )

            results['RandomForest_n_estimators'] = self._process_validation_curve(
                param_range, train_scores, val_scores, 'n_estimators'
            )

        # LogisticRegression C 분석
        if 'LogisticRegression_L2' in models:
            lr_model = models['LogisticRegression_L2']
            param_range = [0.001, 0.01, 0.1, 1, 10, 100]

            print("  📊 LogisticRegression C 분석...")

            train_scores, val_scores = validation_curve(
                lr_model, X, y,
                param_name='C',
                param_range=param_range,
                cv=5, scoring='f1_macro', n_jobs=-1
            )

            results['LogisticRegression_C'] = self._process_validation_curve(
                param_range, train_scores, val_scores, 'C'
            )

        # 검증 곡선 시각화
        self._plot_validation_curves(results)

        return results

    def _process_validation_curve(self, param_range, train_scores, val_scores, param_name):
        """검증 곡선 결과 처리"""
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # 최적 파라미터 찾기
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]

        # 과적합 구간 찾기
        gaps = train_mean - val_mean
        overfitting_threshold = 0.05
        overfitting_params = [param_range[i] for i, gap in enumerate(gaps) if gap > overfitting_threshold]

        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'best_param': best_param,
            'best_score': best_score,
            'overfitting_params': overfitting_params,
            'gaps': gaps.tolist()
        }

    def _plot_validation_curves(self, results):
        """검증 곡선 시각화"""
        n_plots = len(results)
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        for idx, (curve_name, curve_data) in enumerate(results.items()):
            ax = axes[idx]

            param_range = curve_data['param_range']
            train_mean = np.array(curve_data['train_scores_mean'])
            train_std = np.array(curve_data['train_scores_std'])
            val_mean = np.array(curve_data['val_scores_mean'])
            val_std = np.array(curve_data['val_scores_std'])

            ax.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(param_range, train_mean - train_std,
                          train_mean + train_std, alpha=0.1, color='blue')

            ax.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(param_range, val_mean - val_std,
                          val_mean + val_std, alpha=0.1, color='red')

            # 최적 파라미터 표시
            best_param = curve_data['best_param']
            ax.axvline(x=best_param, color='green', linestyle='--', alpha=0.7,
                      label=f'Best: {best_param}')

            ax.set_title(f'Validation Curve: {curve_name}')
            ax.set_xlabel(curve_data['param_name'])
            ax.set_ylabel('F1-Score (Macro)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 저장
        validation_path = self.plots_dir / 'validation_curves_analysis.png'
        plt.savefig(validation_path, dpi=300, bbox_inches='tight')
        self.figures['validation_curves'] = plt.gcf()
        plt.close()

    def _analyze_feature_learning_curves(self, X, y, models):
        """특성 학습 곡선 분석"""
        print("\n🔍 3단계: 특성 학습 곡선 분석")

        results = {}

        # 특성 수별 성능 분석
        feature_counts = [5, 10, 15, 20, min(30, len(X.columns)), len(X.columns)]
        feature_counts = [f for f in feature_counts if f <= len(X.columns)]

        # 주요 모델 하나만 사용 (시간 절약)
        main_model = RandomForestClassifier(
            n_estimators=100,
            random_state=get_config('models.random_seed'),
            n_jobs=-1
        )

        print("  📊 특성 수별 성능 분석...")

        feature_scores = []
        feature_train_scores = []

        for n_features in feature_counts:
            # 특성 선택
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)

            # CV 점수 계산
            cv_scores = cross_val_score(main_model, X_selected, y, cv=5, scoring='f1_macro')

            # 훈련 점수 계산 (과적합 정도 확인용)
            main_model.fit(X_selected, y)
            train_pred = main_model.predict(X_selected)
            train_score = f1_score(y, train_pred, average='macro')

            feature_scores.append(cv_scores.mean())
            feature_train_scores.append(train_score)

            print(f"    • {n_features}개 특성: CV={cv_scores.mean():.4f}, Train={train_score:.4f}")

        # 결과 저장
        results = {
            'feature_counts': feature_counts,
            'val_scores': feature_scores,
            'train_scores': feature_train_scores,
            'gaps': [train - val for train, val in zip(feature_train_scores, feature_scores)],
            'optimal_features': feature_counts[np.argmax(feature_scores)]
        }

        # 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(feature_counts, feature_train_scores, 'o-', color='blue', label='Training Score')
        plt.plot(feature_counts, feature_scores, 'o-', color='red', label='Validation Score')
        plt.axvline(x=results['optimal_features'], color='green', linestyle='--',
                   label=f'Optimal: {results["optimal_features"]} features')

        plt.xlabel('Number of Features')
        plt.ylabel('F1-Score (Macro)')
        plt.title('Feature Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 저장
        feature_path = self.plots_dir / 'feature_learning_curve.png'
        plt.savefig(feature_path, dpi=300, bbox_inches='tight')
        self.figures['feature_learning'] = plt.gcf()
        plt.close()

        print(f"  🎯 최적 특성 수: {results['optimal_features']}개")

        return results

    def _analyze_regularization_effects(self, X, y):
        """정규화 효과 분석"""
        print("\n🛡️ 4단계: 정규화 효과 분석")

        results = {}

        # L1, L2 정규화 강도별 분석
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]

        for penalty in ['l1', 'l2']:
            print(f"  📊 {penalty.upper()} 정규화 분석...")

            penalty_results = {
                'C_values': C_values,
                'val_scores': [],
                'train_scores': [],
                'gaps': []
            }

            for C in C_values:
                try:
                    if penalty == 'l1':
                        model = LogisticRegression(
                            penalty='l1', C=C, solver='liblinear',
                            random_state=get_config('models.random_seed'),
                            max_iter=1000
                        )
                    else:
                        model = LogisticRegression(
                            penalty='l2', C=C,
                            random_state=get_config('models.random_seed'),
                            max_iter=1000
                        )

                    # 교차검증 점수
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
                    val_score = cv_scores.mean()

                    # 훈련 점수
                    model.fit(X, y)
                    train_pred = model.predict(X)
                    train_score = f1_score(y, train_pred, average='macro')

                    penalty_results['val_scores'].append(val_score)
                    penalty_results['train_scores'].append(train_score)
                    penalty_results['gaps'].append(train_score - val_score)

                except:
                    penalty_results['val_scores'].append(0)
                    penalty_results['train_scores'].append(0)
                    penalty_results['gaps'].append(0)

            # 최적 C 찾기
            best_idx = np.argmax(penalty_results['val_scores'])
            penalty_results['best_C'] = C_values[best_idx]
            penalty_results['best_score'] = penalty_results['val_scores'][best_idx]
            penalty_results['best_gap'] = penalty_results['gaps'][best_idx]

            results[penalty] = penalty_results

            print(f"    • 최적 C: {penalty_results['best_C']}")
            print(f"    • 최고 점수: {penalty_results['best_score']:.4f}")
            print(f"    • 격차: {penalty_results['best_gap']:.4f}")

        # 시각화
        self._plot_regularization_effects(results)

        return results

    def _plot_regularization_effects(self, results):
        """정규화 효과 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (penalty, penalty_data) in enumerate(results.items()):
            ax = axes[idx]

            C_values = penalty_data['C_values']
            train_scores = penalty_data['train_scores']
            val_scores = penalty_data['val_scores']

            ax.semilogx(C_values, train_scores, 'o-', color='blue', label='Training Score')
            ax.semilogx(C_values, val_scores, 'o-', color='red', label='Validation Score')

            # 최적 C 표시
            best_C = penalty_data['best_C']
            ax.axvline(x=best_C, color='green', linestyle='--',
                      label=f'Best C: {best_C}')

            ax.set_title(f'{penalty.upper()} Regularization Effect')
            ax.set_xlabel('C (Inverse Regularization Strength)')
            ax.set_ylabel('F1-Score (Macro)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 저장
        reg_path = self.plots_dir / 'regularization_effects.png'
        plt.savefig(reg_path, dpi=300, bbox_inches='tight')
        self.figures['regularization'] = plt.gcf()
        plt.close()

    def _analyze_early_stopping(self, X, y):
        """Early Stopping 분석"""
        print("\n⏰ 5단계: Early Stopping 분석")

        # RandomForest n_estimators별 성능 분석 (조기 종료 시뮬레이션)
        n_estimators_range = range(10, 201, 10)
        val_scores = []
        train_scores = []

        print("  📊 RandomForest 트리 수별 성능 분석...")

        for n_est in n_estimators_range:
            model = RandomForestClassifier(
                n_estimators=n_est,
                random_state=get_config('models.random_seed'),
                n_jobs=-1
            )

            # 교차검증 점수
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
            val_score = cv_scores.mean()

            # 훈련 점수
            model.fit(X, y)
            train_pred = model.predict(X)
            train_score = f1_score(y, train_pred, average='macro')

            val_scores.append(val_score)
            train_scores.append(train_score)

        # 최적 중단점 찾기
        gaps = np.array(train_scores) - np.array(val_scores)
        val_scores_array = np.array(val_scores)

        # 검증 점수가 개선되지 않는 지점 찾기
        patience = 3
        best_score = 0
        best_iter = 0
        no_improve_count = 0
        early_stop_point = len(n_estimators_range)

        for i, score in enumerate(val_scores):
            if score > best_score:
                best_score = score
                best_iter = i
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience and early_stop_point == len(n_estimators_range):
                    early_stop_point = i

        optimal_n_estimators = list(n_estimators_range)[best_iter]
        early_stop_n_estimators = list(n_estimators_range)[min(early_stop_point, len(n_estimators_range)-1)]

        results = {
            'n_estimators_range': list(n_estimators_range),
            'val_scores': val_scores,
            'train_scores': train_scores,
            'gaps': gaps.tolist(),
            'optimal_n_estimators': optimal_n_estimators,
            'optimal_score': best_score,
            'early_stop_point': early_stop_n_estimators,
            'early_stop_score': val_scores[early_stop_point] if early_stop_point < len(val_scores) else val_scores[-1]
        }

        # 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(n_estimators_range, train_scores, 'o-', color='blue', label='Training Score')
        plt.plot(n_estimators_range, val_scores, 'o-', color='red', label='Validation Score')
        plt.axvline(x=optimal_n_estimators, color='green', linestyle='--',
                   label=f'Optimal: {optimal_n_estimators}')
        plt.axvline(x=early_stop_n_estimators, color='orange', linestyle='--',
                   label=f'Early Stop: {early_stop_n_estimators}')

        plt.xlabel('Number of Estimators')
        plt.ylabel('F1-Score (Macro)')
        plt.title('Early Stopping Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 저장
        early_stop_path = self.plots_dir / 'early_stopping_analysis.png'
        plt.savefig(early_stop_path, dpi=300, bbox_inches='tight')
        self.figures['early_stopping'] = plt.gcf()
        plt.close()

        print(f"  🎯 최적 n_estimators: {optimal_n_estimators}")
        print(f"  ⏰ Early Stop 지점: {early_stop_n_estimators}")

        return results

    def _analyze_ensemble_diversity(self, X, y):
        """앙상블 다양성 분석"""
        print("\n🎭 6단계: 앙상블 다양성 분석")

        # 다양한 랜덤 시드로 모델 학습
        n_models = 10
        models = []
        predictions = []

        print("  🎲 다양한 시드로 모델 학습...")

        for seed in range(n_models):
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X, y)
            pred = model.predict(X)
            models.append(model)
            predictions.append(pred)

        # 다양성 측정
        diversity_metrics = self._calculate_ensemble_diversity(predictions, y)

        # 앙상블 성능 (다수결 투표)
        ensemble_pred = np.array(predictions).T
        final_pred = []

        for pred_row in ensemble_pred:
            # 다수결 투표
            unique, counts = np.unique(pred_row, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            final_pred.append(majority_vote)

        ensemble_score = f1_score(y, final_pred, average='macro')

        # 개별 모델 성능
        individual_scores = [f1_score(y, pred, average='macro') for pred in predictions]

        results = {
            'n_models': n_models,
            'individual_scores': individual_scores,
            'individual_mean': np.mean(individual_scores),
            'individual_std': np.std(individual_scores),
            'ensemble_score': ensemble_score,
            'ensemble_improvement': ensemble_score - np.mean(individual_scores),
            'diversity_metrics': diversity_metrics
        }

        print(f"  📊 개별 모델 평균: {results['individual_mean']:.4f} ± {results['individual_std']:.4f}")
        print(f"  🎭 앙상블 성능: {results['ensemble_score']:.4f}")
        print(f"  📈 개선 효과: +{results['ensemble_improvement']:.4f}")

        return results

    def _calculate_ensemble_diversity(self, predictions, y_true):
        """앙상블 다양성 메트릭 계산"""
        predictions = np.array(predictions)
        n_models = len(predictions)

        # Disagreement measure (모델 간 불일치 정도)
        disagreements = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)

        avg_disagreement = np.mean(disagreements)

        # Q-statistic (두 모델 간 상관관계)
        q_statistics = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                # 일치/불일치 표 생성
                both_correct = np.sum((predictions[i] == y_true) & (predictions[j] == y_true))
                both_wrong = np.sum((predictions[i] != y_true) & (predictions[j] != y_true))
                i_correct_j_wrong = np.sum((predictions[i] == y_true) & (predictions[j] != y_true))
                i_wrong_j_correct = np.sum((predictions[i] != y_true) & (predictions[j] == y_true))

                # Q-statistic 계산
                if (both_correct * both_wrong + i_correct_j_wrong * i_wrong_j_correct) != 0:
                    q_stat = (both_correct * both_wrong - i_correct_j_wrong * i_wrong_j_correct) / \
                            (both_correct * both_wrong + i_correct_j_wrong * i_wrong_j_correct)
                    q_statistics.append(q_stat)

        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0

        return {
            'average_disagreement': avg_disagreement,
            'average_q_statistic': avg_q_statistic,
            'diversity_score': avg_disagreement  # 높을수록 다양성 높음
        }

    def _find_convergence_point(self, train_scores, val_scores):
        """학습/검증 점수 수렴점 찾기"""
        gaps = train_scores - val_scores
        # 격차가 안정화되는 지점 찾기
        convergence_threshold = 0.01

        for i in range(1, len(gaps)):
            if abs(gaps[i] - gaps[i-1]) < convergence_threshold:
                return i

        return len(gaps) - 1

    def _find_optimal_training_size(self, train_sizes, val_scores):
        """최적 훈련 데이터 크기 찾기"""
        best_idx = np.argmax(val_scores)
        return train_sizes[best_idx]

    def _classify_overfitting_severity(self, gap):
        """과적합 심각도 분류"""
        if gap < 0.02:
            return "매우 낮음"
        elif gap < 0.05:
            return "낮음"
        elif gap < 0.1:
            return "보통"
        elif gap < 0.2:
            return "높음"
        else:
            return "매우 높음"

    def _assess_overfitting_severity(self, learning_analysis, validation_analysis, target_gap):
        """과적합 심각도 종합 평가"""
        print("\n📊 7단계: 과적합 심각도 종합 평가")

        severity_scores = []
        model_assessments = {}

        for model_name, model_result in learning_analysis.items():
            if model_result is None:
                continue

            gap = model_result['final_gap']
            severity = model_result['overfitting_severity']

            # 점수화 (0-5 스케일)
            severity_map = {
                "매우 낮음": 1,
                "낮음": 2,
                "보통": 3,
                "높음": 4,
                "매우 높음": 5
            }

            score = severity_map.get(severity, 3)
            severity_scores.append(score)

            model_assessments[model_name] = {
                'gap': gap,
                'severity': severity,
                'severity_score': score,
                'exceeds_target': gap > target_gap
            }

        # 전체 평가
        overall_severity_score = np.mean(severity_scores) if severity_scores else 3
        overall_severity = self._score_to_severity(overall_severity_score)

        # 목표 격차와 비교
        models_exceeding_target = sum(1 for assessment in model_assessments.values()
                                    if assessment['exceeds_target'])

        results = {
            'overall_severity_score': overall_severity_score,
            'overall_severity': overall_severity,
            'target_gap': target_gap,
            'models_exceeding_target': models_exceeding_target,
            'model_assessments': model_assessments,
            'requires_action': overall_severity_score > 3 or models_exceeding_target > 0
        }

        print(f"  📊 전체 심각도: {overall_severity} (점수: {overall_severity_score:.2f})")
        print(f"  🎯 목표 격차 초과 모델: {models_exceeding_target}개")
        print(f"  ⚠️ 조치 필요: {'예' if results['requires_action'] else '아니오'}")

        return results

    def _score_to_severity(self, score):
        """점수를 심각도 텍스트로 변환"""
        if score <= 1.5:
            return "매우 낮음"
        elif score <= 2.5:
            return "낮음"
        elif score <= 3.5:
            return "보통"
        elif score <= 4.5:
            return "높음"
        else:
            return "매우 높음"

    def _generate_overfitting_solutions(self, learning_analysis, validation_analysis,
                                       feature_analysis, regularization_analysis,
                                       severity_assessment):
        """과적합 해결방안 생성"""
        print("\n💡 8단계: 과적합 해결방안 생성")

        solutions = []

        # 심각도에 따른 기본 전략
        severity_score = severity_assessment['overall_severity_score']

        if severity_score > 4:  # 매우 높음/높음
            solutions.extend([
                {
                    'priority': 'HIGH',
                    'category': 'model_complexity',
                    'solution': '모델 복잡도 대폭 감소',
                    'details': [
                        'RandomForest: max_depth=3-5, min_samples_leaf=20-50',
                        'LogisticRegression: 강한 L1 정규화 (C=0.01-0.1)',
                        '특성 수 줄이기 (SelectKBest, RFE)'
                    ],
                    'expected_impact': '과적합 격차 50% 이상 감소'
                },
                {
                    'priority': 'HIGH',
                    'category': 'data_augmentation',
                    'solution': '데이터 증강 및 정규화',
                    'details': [
                        '교차검증 폴드 수 증가 (10-15 folds)',
                        'Dropout, Early Stopping 적용',
                        '앙상블 다양성 증가'
                    ],
                    'expected_impact': '일반화 성능 향상'
                }
            ])
        elif severity_score > 3:  # 보통
            solutions.extend([
                {
                    'priority': 'MEDIUM',
                    'category': 'regularization',
                    'solution': '적절한 정규화 적용',
                    'details': [],
                    'expected_impact': '과적합 격차 20-30% 감소'
                }
            ])

        # 특성 분석 기반 솔루션
        if feature_analysis:
            optimal_features = feature_analysis['optimal_features']
            current_features = len(feature_analysis['feature_counts']) - 1
            if optimal_features < feature_analysis['feature_counts'][-1]:
                solutions.append({
                    'priority': 'MEDIUM',
                    'category': 'feature_selection',
                    'solution': f'특성 수를 {optimal_features}개로 줄이기',
                    'details': [
                        'SelectKBest, RFE, 또는 L1 정규화 사용',
                        '특성 중요도 기반 선택',
                        '상관관계 높은 특성 제거'
                    ],
                    'expected_impact': '차원의 저주 방지, 일반화 성능 향상'
                })

        # 정규화 분석 기반 솔루션
        if regularization_analysis:
            for penalty, penalty_data in regularization_analysis.items():
                if penalty_data['best_gap'] < 0.05:  # 좋은 정규화 효과
                    solutions.append({
                        'priority': 'MEDIUM',
                        'category': 'regularization',
                        'solution': f'{penalty.upper()} 정규화 적용 (C={penalty_data["best_C"]})',
                        'details': [
                            f'검증된 최적 하이퍼파라미터 사용',
                            f'예상 격차: {penalty_data["best_gap"]:.3f}'
                        ],
                        'expected_impact': '과적합 제어 및 성능 안정화'
                    })

        # 앙상블 기반 솔루션
        solutions.append({
            'priority': 'LOW',
            'category': 'ensemble',
            'solution': '앙상블 다양성 증가',
            'details': [
                '다양한 시드로 여러 모델 학습',
                '서로 다른 알고리즘 조합',
                'Bagging, Boosting 기법 활용'
            ],
            'expected_impact': '개별 모델 과적합 상쇄'
        })

        # 우선순위별 정렬
        priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        solutions.sort(key=lambda x: priority_order.get(x['priority'], 3))

        print(f"  💡 생성된 해결방안: {len(solutions)}개")
        for i, solution in enumerate(solutions[:3], 1):
            print(f"    {i}. [{solution['priority']}] {solution['solution']}")

        return solutions

    def _log_overfitting_results(self, results):
        """실험 추적에 결과 로깅"""
        # 주요 메트릭 로깅
        severity = results['severity_assessment']
        metrics = {
            'overall_severity_score': severity['overall_severity_score'],
            'models_exceeding_target': severity['models_exceeding_target'],
            'requires_action': 1 if severity['requires_action'] else 0
        }

        # 학습 곡선 결과
        for model_name, model_result in results['learning_curves'].items():
            if model_result:
                metrics[f'{model_name}_final_gap'] = model_result['final_gap']
                metrics[f'{model_name}_optimal_size'] = model_result['optimal_training_size']

        # 특성 학습 곡선 결과
        if results['feature_curves']:
            metrics['optimal_features'] = results['feature_curves']['optimal_features']

        # 앙상블 다양성 결과
        if results['ensemble_diversity']:
            ensemble = results['ensemble_diversity']
            metrics['ensemble_improvement'] = ensemble['ensemble_improvement']
            metrics['ensemble_diversity'] = ensemble['diversity_metrics']['diversity_score']

        self.tracker.log_metrics(metrics)

        # 파라미터 로깅
        params = {
            'target_score_gap': results['data_info']['target_score_gap'],
            'n_samples': results['data_info']['n_samples'],
            'n_features': results['data_info']['n_features'],
            'n_classes': results['data_info']['n_classes'],
            'n_solutions': len(results['solutions'])
        }
        self.tracker.log_params(params)

        # 시각화 로깅
        for fig_name, fig in self.figures.items():
            self.tracker.log_figure(fig, fig_name)

        self.tracker.end_run()

    def _print_analysis_summary(self, results):
        """분석 결과 요약 출력"""
        print("\n" + "="*70)
        print("🔍 과적합 진단 시스템 분석 결과 요약")
        print("="*70)

        severity = results['severity_assessment']
        print(f"\n🎯 전체 과적합 심각도: {severity['overall_severity']} ({severity['overall_severity_score']:.2f}/5.0)")
        print(f"📊 목표 격차({severity['target_gap']:.3f}) 초과 모델: {severity['models_exceeding_target']}개")
        print(f"⚠️ 조치 필요 여부: {'예' if severity['requires_action'] else '아니오'}")

        # 모델별 격차
        print(f"\n🤖 모델별 과적합 격차:")
        for model_name, assessment in severity['model_assessments'].items():
            status = "⚠️ 초과" if assessment['exceeds_target'] else "✅ 양호"
            print(f"  • {model_name}: {assessment['gap']:.3f} ({assessment['severity']}) {status}")

        # 주요 발견사항
        if results['feature_curves']:
            print(f"\n🔍 특성 분석:")
            print(f"  • 최적 특성 수: {results['feature_curves']['optimal_features']}개")

        if results['ensemble_diversity']:
            ensemble = results['ensemble_diversity']
            print(f"\n🎭 앙상블 분석:")
            print(f"  • 개별 모델: {ensemble['individual_mean']:.3f} ± {ensemble['individual_std']:.3f}")
            print(f"  • 앙상블 성능: {ensemble['ensemble_score']:.3f}")
            print(f"  • 개선 효과: +{ensemble['ensemble_improvement']:.3f}")

        # 핵심 해결방안
        print(f"\n💡 핵심 해결방안:")
        high_priority_solutions = [sol for sol in results['solutions'] if sol['priority'] == 'HIGH']
        if high_priority_solutions:
            for i, solution in enumerate(high_priority_solutions[:3], 1):
                print(f"  {i}. {solution['solution']}")
                print(f"     → {solution['expected_impact']}")
        else:
            medium_solutions = [sol for sol in results['solutions'] if sol['priority'] == 'MEDIUM'][:2]
            for i, solution in enumerate(medium_solutions, 1):
                print(f"  {i}. {solution['solution']}")

        print("="*70)

    def get_actionable_recommendations(self):
        """실행 가능한 권장사항 반환"""
        if not self.results:
            print("⚠️ 먼저 분석을 실행해주세요.")
            return []

        return self.results['solutions']


if __name__ == "__main__":
    """과적합 진단 시스템 테스트"""
    print("🧪 과적합 진단 시스템 테스트 시작...")

    # 더미 데이터 생성 (과적합이 발생하도록)
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,      # 적은 샘플
        n_features=50,      # 많은 특성 (과적합 유발)
        n_classes=5,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(50)])
    y_series = pd.Series(y)

    # 분석 실행
    analyzer = OverfittingAnalyzer()
    results = analyzer.comprehensive_overfitting_analysis(
        X_df, y_series,
        target_score_gap=0.096  # 9.6% 목표 격차
    )

    # 실행 가능한 권장사항 출력
    recommendations = analyzer.get_actionable_recommendations()
    if recommendations:
        print("\n🎯 실행 가능한 권장사항:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. [{rec['priority']}] {rec['solution']}")
            if rec['details']:
                for detail in rec['details'][:2]:
                    print(f"      - {detail}")

    print("\n✅ T006 완료: 과적합 진단 시스템 구현 성공!")
    print("🎯 다음 단계: T007 (클래스별 성능 분석)")