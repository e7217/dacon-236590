"""
T009: Permutation Importance 분석을 통한 특성 중요도 평가

이 모듈은 Permutation Importance를 계산하여 모델의 특성 중요도를 분석합니다.
SHAP 분석과 함께 사용하여 모델 해석성을 종합적으로 평가할 수 있습니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import warnings
from pathlib import Path
import mlflow
import wandb
from typing import Dict, List, Tuple, Optional, Union
import logging

class PermutationAnalyzer:
    """Permutation Importance를 통한 특성 중요도 분석 클래스"""

    def __init__(self, config=None, experiment_tracker=None):
        """
        Permutation Importance 분석기 초기화

        Args:
            config: 설정 객체
            experiment_tracker: 실험 추적기 객체
        """
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.logger = logging.getLogger(__name__)

        # 결과 저장 경로 설정
        if config:
            self.plots_dir = Path(config.get_paths().get('plots', 'plots'))
            self.experiments_dir = Path(config.get_paths().get('experiments', 'experiments'))
        else:
            self.plots_dir = Path('plots')
            self.experiments_dir = Path('experiments')

        self.permutation_dir = self.plots_dir / 'permutation_analysis'
        self.permutation_dir.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        warnings.filterwarnings('ignore', category=UserWarning)

        self.results = {}

    def calculate_permutation_importance(
        self,
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scoring: str = 'f1_macro',
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict:
        """
        Permutation Importance 계산

        Args:
            model: 학습된 모델
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 레이블
            scoring: 평가 지표
            n_repeats: 반복 횟수
            random_state: 랜덤 시드

        Returns:
            Permutation Importance 결과
        """
        try:
            self.logger.info(f"🔄 Permutation Importance 계산 시작...")
            self.logger.info(f"   - 특성 수: {len(X_val.columns)}")
            self.logger.info(f"   - 검증 데이터 크기: {len(X_val)}")
            self.logger.info(f"   - 반복 횟수: {n_repeats}")
            self.logger.info(f"   - 평가 지표: {scoring}")

            # Permutation Importance 계산
            perm_importance = permutation_importance(
                model, X_val, y_val,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )

            # 결과 정리
            importance_mean = perm_importance.importances_mean
            importance_std = perm_importance.importances_std

            # DataFrame으로 변환
            perm_df = pd.DataFrame({
                'feature': X_val.columns,
                'importance_mean': importance_mean,
                'importance_std': importance_std,
                'importance_abs': np.abs(importance_mean)  # 절댓값
            })

            # 중요도 순으로 정렬
            perm_df = perm_df.sort_values('importance_abs', ascending=False).reset_index(drop=True)

            results = {
                'permutation_df': perm_df,
                'raw_importances': perm_importance.importances,
                'baseline_score': self._get_baseline_score(model, X_val, y_val, scoring),
                'scoring_metric': scoring,
                'n_repeats': n_repeats
            }

            self.logger.info(f"✅ Permutation Importance 계산 완료")
            self.logger.info(f"   - 가장 중요한 특성: {perm_df.iloc[0]['feature']} (중요도: {perm_df.iloc[0]['importance_mean']:.4f})")
            self.logger.info(f"   - 베이스라인 점수: {results['baseline_score']:.4f}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Permutation Importance 계산 오류: {str(e)}")
            raise

    def _get_baseline_score(self, model, X_val, y_val, scoring):
        """베이스라인 점수 계산"""
        try:
            if scoring == 'f1_macro':
                y_pred = model.predict(X_val)
                return f1_score(y_val, y_pred, average='macro')
            else:
                return model.score(X_val, y_val)
        except Exception as e:
            self.logger.warning(f"베이스라인 점수 계산 실패: {str(e)}")
            return 0.0

    def create_permutation_plots(
        self,
        perm_results: Dict,
        model_name: str = "RandomForest",
        top_k: int = 20
    ) -> Dict[str, str]:
        """
        Permutation Importance 시각화

        Args:
            perm_results: Permutation Importance 결과
            model_name: 모델 이름
            top_k: 상위 K개 특성

        Returns:
            생성된 플롯 파일 경로들
        """
        plot_paths = {}
        perm_df = perm_results['permutation_df']
        model_dir = self.permutation_dir / f"{model_name.lower()}-permutation"
        model_dir.mkdir(exist_ok=True)

        try:
            # 1. 상위 특성 중요도 바 플롯
            plt.figure(figsize=(12, 8))
            top_features = perm_df.head(top_k)

            bars = plt.barh(range(len(top_features)), top_features['importance_mean'],
                           xerr=top_features['importance_std'], alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Permutation Importance')
            plt.title(f'{model_name}\nTop {top_k} Feature Permutation Importance')
            plt.gca().invert_yaxis()

            # 색상 구분 (양수/음수)
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance_mean'])):
                if importance >= 0:
                    bar.set_color('skyblue')
                else:
                    bar.set_color('lightcoral')

            plt.tight_layout()
            plot_path = model_dir / 'top_features_importance.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['top_features'] = str(plot_path)

            # 2. 중요도 분포 히스토그램
            plt.figure(figsize=(10, 6))
            plt.hist(perm_df['importance_mean'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Importance')
            plt.xlabel('Permutation Importance')
            plt.ylabel('Number of Features')
            plt.title(f'{model_name}\nDistribution of Permutation Importance')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = model_dir / 'importance_distribution.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['distribution'] = str(plot_path)

            # 3. 중요도 vs 표준편차 산점도
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(perm_df['importance_mean'], perm_df['importance_std'],
                                alpha=0.6, c=perm_df['importance_abs'], cmap='viridis')
            plt.colorbar(scatter, label='Absolute Importance')

            # 상위 특성 라벨링
            for i, row in perm_df.head(10).iterrows():
                plt.annotate(row['feature'],
                           (row['importance_mean'], row['importance_std']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

            plt.xlabel('Mean Permutation Importance')
            plt.ylabel('Standard Deviation')
            plt.title(f'{model_name}\nPermutation Importance: Mean vs Std')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = model_dir / 'importance_scatter.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['scatter'] = str(plot_path)

            # 4. 박스플롯 (상위 특성들의 중요도 분포)
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(perm_df))
            box_data = []
            box_labels = []

            for idx in range(top_n):
                feature_idx = list(perm_df['feature']).index(perm_df.iloc[idx]['feature'])
                feature_importances = perm_results['raw_importances'][feature_idx]
                box_data.append(feature_importances)
                box_labels.append(perm_df.iloc[idx]['feature'])

            plt.boxplot(box_data, labels=box_labels)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Permutation Importance')
            plt.title(f'{model_name}\nTop {top_n} Features Importance Distribution')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = model_dir / 'importance_boxplot.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['boxplot'] = str(plot_path)

            self.logger.info(f"✅ {len(plot_paths)}개 Permutation 시각화 완료: {model_dir}")

        except Exception as e:
            self.logger.error(f"❌ Permutation 시각화 오류: {str(e)}")

        return plot_paths

    def compare_with_shap(
        self,
        perm_df: pd.DataFrame,
        shap_df: pd.DataFrame,
        model_name: str = "RandomForest"
    ) -> Dict:
        """
        Permutation Importance와 SHAP 중요도 비교

        Args:
            perm_df: Permutation Importance DataFrame
            shap_df: SHAP 중요도 DataFrame
            model_name: 모델 이름

        Returns:
            비교 결과 및 시각화 경로
        """
        try:
            self.logger.info("🔄 Permutation vs SHAP 비교 분석 시작...")

            # 공통 특성만 선택
            common_features = set(perm_df['feature']).intersection(set(shap_df['feature']))

            if not common_features:
                self.logger.warning("공통 특성이 없습니다.")
                return {}

            # 데이터 정리
            perm_common = perm_df[perm_df['feature'].isin(common_features)].set_index('feature')
            shap_common = shap_df[shap_df['feature'].isin(common_features)].set_index('feature')

            # 상관관계 계산
            correlation = np.corrcoef(
                perm_common.loc[common_features, 'importance_abs'].values,
                shap_common.loc[common_features, 'importance'].values
            )[0, 1]

            # 비교 DataFrame 생성
            comparison_df = pd.DataFrame({
                'feature': list(common_features),
                'permutation_importance': [perm_common.loc[f, 'importance_abs'] for f in common_features],
                'shap_importance': [shap_common.loc[f, 'importance'] for f in common_features]
            })

            # 순위 계산
            comparison_df['perm_rank'] = comparison_df['permutation_importance'].rank(ascending=False)
            comparison_df['shap_rank'] = comparison_df['shap_importance'].rank(ascending=False)
            comparison_df['rank_diff'] = abs(comparison_df['perm_rank'] - comparison_df['shap_rank'])

            # 시각화
            model_dir = self.permutation_dir / f"{model_name.lower()}-comparison"
            model_dir.mkdir(exist_ok=True)

            # 1. 산점도
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(comparison_df['permutation_importance'],
                                comparison_df['shap_importance'],
                                alpha=0.6, s=60)

            # 상위 특성 라벨링
            top_features = comparison_df.nlargest(10, 'permutation_importance')
            for _, row in top_features.iterrows():
                plt.annotate(row['feature'],
                           (row['permutation_importance'], row['shap_importance']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

            plt.xlabel('Permutation Importance')
            plt.ylabel('SHAP Importance')
            plt.title(f'{model_name}\nPermutation vs SHAP Importance\n(Correlation: {correlation:.3f})')

            # 대각선 추가
            max_val = max(comparison_df['permutation_importance'].max(),
                         comparison_df['shap_importance'].max())
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            scatter_path = model_dir / 'importance_comparison.png'
            plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 2. 순위 비교
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(comparison_df))
            top_comparison = comparison_df.nsmallest(top_n, 'rank_diff')

            x = range(len(top_comparison))
            width = 0.35

            plt.bar([i - width/2 for i in x], top_comparison['perm_rank'],
                   width, label='Permutation Rank', alpha=0.7)
            plt.bar([i + width/2 for i in x], top_comparison['shap_rank'],
                   width, label='SHAP Rank', alpha=0.7)

            plt.xlabel('Features')
            plt.ylabel('Importance Rank')
            plt.title(f'{model_name}\nRank Comparison (Top {top_n} Most Consistent)')
            plt.xticks(x, top_comparison['feature'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            rank_path = model_dir / 'rank_comparison.png'
            plt.savefig(rank_path, dpi=150, bbox_inches='tight')
            plt.close()

            results = {
                'comparison_df': comparison_df,
                'correlation': correlation,
                'common_features_count': len(common_features),
                'plot_paths': {
                    'scatter': str(scatter_path),
                    'rank_comparison': str(rank_path)
                }
            }

            self.logger.info(f"✅ Permutation vs SHAP 비교 완료")
            self.logger.info(f"   - 상관관계: {correlation:.3f}")
            self.logger.info(f"   - 공통 특성 수: {len(common_features)}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Permutation vs SHAP 비교 오류: {str(e)}")
            return {}

    def analyze_feature_stability(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_params: Dict = None,
        n_splits: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        교차검증을 통한 특성 중요도 안정성 분석

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            model_params: 모델 파라미터
            n_splits: 교차검증 분할 수
            random_state: 랜덤 시드

        Returns:
            안정성 분석 결과
        """
        try:
            self.logger.info("🔄 특성 중요도 안정성 분석 시작...")

            from sklearn.model_selection import KFold

            if model_params is None:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': random_state,
                    'class_weight': 'balanced'
                }

            # KFold 설정
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            importance_across_folds = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                self.logger.info(f"   - Fold {fold + 1}/{n_splits} 처리 중...")

                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]

                # 모델 훈련
                model = RandomForestClassifier(**model_params)
                model.fit(X_fold_train, y_fold_train)

                # Permutation Importance 계산
                perm_result = self.calculate_permutation_importance(
                    model, X_fold_val, y_fold_val,
                    scoring='f1_macro', n_repeats=5
                )

                fold_importance = perm_result['permutation_df'][['feature', 'importance_mean']].copy()
                fold_importance['fold'] = fold
                importance_across_folds.append(fold_importance)

            # 결과 통합
            all_importance = pd.concat(importance_across_folds, ignore_index=True)

            # 안정성 통계 계산
            stability_stats = all_importance.groupby('feature')['importance_mean'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()

            stability_stats['cv'] = stability_stats['std'] / (stability_stats['mean'] + 1e-8)  # 변동계수
            stability_stats = stability_stats.sort_values('mean', ascending=False)

            # 시각화
            model_dir = self.permutation_dir / "stability_analysis"
            model_dir.mkdir(exist_ok=True)

            # 1. 안정성 히트맵
            pivot_data = all_importance.pivot(index='feature', columns='fold', values='importance_mean')

            plt.figure(figsize=(12, 16))
            sns.heatmap(pivot_data.iloc[:30], annot=False, cmap='viridis',
                       cbar_kws={'label': 'Permutation Importance'})
            plt.title('Feature Importance Across CV Folds\n(Top 30 Features)')
            plt.xlabel('Fold')
            plt.ylabel('Feature')

            plt.tight_layout()
            heatmap_path = model_dir / 'stability_heatmap.png'
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 2. 변동계수 vs 평균 중요도
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(stability_stats['mean'], stability_stats['cv'],
                                alpha=0.6, s=60)

            # 상위 특성 라벨링
            top_stable = stability_stats.head(10)
            for _, row in top_stable.iterrows():
                plt.annotate(row['feature'],
                           (row['mean'], row['cv']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

            plt.xlabel('Mean Importance')
            plt.ylabel('Coefficient of Variation')
            plt.title('Feature Stability Analysis\nMean Importance vs Variability')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            stability_path = model_dir / 'stability_scatter.png'
            plt.savefig(stability_path, dpi=150, bbox_inches='tight')
            plt.close()

            results = {
                'stability_stats': stability_stats,
                'all_importance': all_importance,
                'n_splits': n_splits,
                'plot_paths': {
                    'heatmap': str(heatmap_path),
                    'stability': str(stability_path)
                }
            }

            self.logger.info("✅ 특성 중요도 안정성 분석 완료")
            self.logger.info(f"   - 가장 안정한 특성: {stability_stats.iloc[0]['feature']}")
            self.logger.info(f"   - 평균 변동계수: {stability_stats['cv'].mean():.3f}")

            return results

        except Exception as e:
            self.logger.error(f"❌ 안정성 분석 오류: {str(e)}")
            return {}

    def analyze_permutation_interpretability(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        model_params: Dict = None,
        shap_results: Dict = None
    ) -> Dict:
        """
        종합적인 Permutation Importance 해석성 분석

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 레이블
            model_params: 모델 파라미터
            shap_results: SHAP 분석 결과 (있는 경우)

        Returns:
            종합 분석 결과
        """
        try:
            self.logger.info("🚀 T009: Permutation Importance 종합 분석 시작")

            # 기본 모델 파라미터
            if model_params is None:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'class_weight': 'balanced'
                }

            # 테스트 데이터가 없으면 훈련 데이터에서 분할
            if X_test is None or y_test is None:
                from sklearn.model_selection import train_test_split
                X_train_split, X_test, y_train_split, y_test = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
                )
            else:
                X_train_split = X_train
                y_train_split = y_train

            # 모델 훈련
            self.logger.info("🔄 RandomForest 모델 훈련...")
            model = RandomForestClassifier(**model_params)
            model.fit(X_train_split, y_train_split)

            # 1. Permutation Importance 계산
            perm_results = self.calculate_permutation_importance(
                model, X_test, y_test,
                scoring='f1_macro', n_repeats=10
            )

            # 2. 시각화 생성
            plot_paths = self.create_permutation_plots(
                perm_results, model_name="RandomForest-Balanced"
            )

            # 3. SHAP과 비교 (결과가 있는 경우)
            comparison_results = {}
            if shap_results:
                self.logger.info("🔄 SHAP vs Permutation 비교...")
                comparison_results = self.compare_with_shap(
                    perm_results['permutation_df'],
                    shap_results.get('global_importance', pd.DataFrame()),
                    model_name="RandomForest-Balanced"
                )

            # 4. 안정성 분석
            stability_results = self.analyze_feature_stability(
                X_train_split, y_train_split, model_params
            )

            # 5. 결과 통합
            results = {
                'permutation_importance': perm_results,
                'plot_paths': plot_paths,
                'comparison_with_shap': comparison_results,
                'stability_analysis': stability_results,
                'model_performance': {
                    'baseline_f1_macro': perm_results['baseline_score'],
                    'model_params': model_params
                }
            }

            # 실험 추적
            if self.experiment_tracker:
                self._log_to_experiment_tracker(results)

            self.results = results

            self.logger.info("🎉 T009: Permutation Importance 분석 완료!")
            return results

        except Exception as e:
            self.logger.error(f"❌ T009 분석 오류: {str(e)}")
            raise

    def _log_to_experiment_tracker(self, results: Dict):
        """실험 추적 시스템에 결과 로그"""
        try:
            if not self.experiment_tracker:
                return

            # MLflow 로깅
            if hasattr(self.experiment_tracker, 'mlflow_client'):
                with mlflow.start_run(run_name="T009_Permutation_Analysis"):
                    # 메트릭 로깅
                    mlflow.log_metric("baseline_f1_macro", results['model_performance']['baseline_f1_macro'])
                    mlflow.log_metric("n_features", len(results['permutation_importance']['permutation_df']))

                    # 상위 특성 로깅
                    top_features = results['permutation_importance']['permutation_df'].head(5)
                    for idx, row in top_features.iterrows():
                        mlflow.log_metric(f"top_{idx+1}_importance", row['importance_mean'])
                        mlflow.log_param(f"top_{idx+1}_feature", row['feature'])

                    # 시각화 로깅
                    for plot_name, plot_path in results['plot_paths'].items():
                        mlflow.log_artifact(plot_path, f"permutation_plots/{plot_name}")

            # WandB 로깅
            if hasattr(self.experiment_tracker, 'wandb_run'):
                wandb.log({
                    "permutation/baseline_f1_macro": results['model_performance']['baseline_f1_macro'],
                    "permutation/n_features": len(results['permutation_importance']['permutation_df']),
                })

                # 시각화 로깅
                for plot_name, plot_path in results['plot_paths'].items():
                    wandb.log({f"permutation/{plot_name}": wandb.Image(plot_path)})

        except Exception as e:
            self.logger.warning(f"실험 추적 로깅 실패: {str(e)}")

    def print_summary(self, results: Dict):
        """분석 결과 요약 출력"""
        try:
            print("\n" + "="*80)
            print("🎯 T009: PERMUTATION IMPORTANCE 분석 결과 요약")
            print("="*80)

            perm_df = results['permutation_importance']['permutation_df']
            baseline_score = results['model_performance']['baseline_f1_macro']

            print(f"📊 기본 정보:")
            print(f"   • 베이스라인 F1 Macro: {baseline_score:.4f}")
            print(f"   • 분석된 특성 수: {len(perm_df)}")
            print(f"   • 양의 중요도 특성: {(perm_df['importance_mean'] > 0).sum()}개")
            print(f"   • 음의 중요도 특성: {(perm_df['importance_mean'] < 0).sum()}개")

            print(f"\n🏆 상위 10개 중요한 특성:")
            for idx, row in perm_df.head(10).iterrows():
                print(f"   {idx+1:2d}. {row['feature']:15s} : {row['importance_mean']:8.4f} (±{row['importance_std']:.4f})")

            print(f"\n📉 하위 5개 특성 (성능 저하 요인):")
            bottom_features = perm_df[perm_df['importance_mean'] < 0].tail(5)
            for idx, row in bottom_features.iterrows():
                print(f"   • {row['feature']:15s} : {row['importance_mean']:8.4f} (±{row['importance_std']:.4f})")

            # SHAP 비교 결과
            if results.get('comparison_with_shap'):
                correlation = results['comparison_with_shap']['correlation']
                print(f"\n🔗 SHAP vs Permutation 상관관계: {correlation:.3f}")
                if correlation > 0.7:
                    print("   ✅ 두 방법이 높은 일치성을 보입니다.")
                elif correlation > 0.5:
                    print("   ⚠️  두 방법이 중간 정도 일치성을 보입니다.")
                else:
                    print("   ❌ 두 방법 간 차이가 큽니다. 추가 분석 필요.")

            # 안정성 분석 결과
            if results.get('stability_analysis'):
                stability_stats = results['stability_analysis']['stability_stats']
                avg_cv = stability_stats['cv'].mean()
                print(f"\n📊 특성 중요도 안정성:")
                print(f"   • 평균 변동계수: {avg_cv:.3f}")
                if avg_cv < 0.3:
                    print("   ✅ 특성 중요도가 안정적입니다.")
                elif avg_cv < 0.5:
                    print("   ⚠️  특성 중요도가 보통 수준으로 안정적입니다.")
                else:
                    print("   ❌ 특성 중요도가 불안정합니다.")

            print(f"\n📈 주요 인사이트:")
            # 성능에 가장 도움되는 특성
            best_feature = perm_df.iloc[0]
            print(f"   • 가장 중요한 특성: {best_feature['feature']} (성능 기여도: {best_feature['importance_mean']:.4f})")

            # 노이즈 특성
            noise_features = perm_df[perm_df['importance_mean'] < 0]
            if len(noise_features) > 0:
                print(f"   • 노이즈 특성 수: {len(noise_features)}개 (제거 고려 대상)")

            # 중요 특성 집중도
            top_10_importance = perm_df.head(10)['importance_mean'].sum()
            total_positive_importance = perm_df[perm_df['importance_mean'] > 0]['importance_mean'].sum()
            if total_positive_importance > 0:
                concentration = top_10_importance / total_positive_importance
                print(f"   • 상위 10개 특성 집중도: {concentration:.1%}")

            print(f"\n📁 생성된 시각화:")
            for plot_name, plot_path in results['plot_paths'].items():
                print(f"   • {plot_name}: {plot_path}")

            print("\n" + "="*80)

        except Exception as e:
            print(f"❌ 요약 출력 오류: {str(e)}")