"""
T010: Confusion Matrix 심화 분석을 통한 클래스별 성능 평가

이 모듈은 Confusion Matrix를 활용하여 다중 클래스 분류 모델의 성능을
클래스별로 상세히 분석합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings
from pathlib import Path
import mlflow
import wandb
from typing import Dict, List, Tuple, Optional, Union
import logging
from itertools import combinations
from scipy import stats

class ConfusionMatrixAnalyzer:
    """Confusion Matrix를 통한 클래스별 성능 심화 분석 클래스"""

    def __init__(self, config=None, experiment_tracker=None):
        """
        Confusion Matrix 분석기 초기화

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

        self.confusion_dir = self.plots_dir / 'confusion_analysis'
        self.confusion_dir.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        warnings.filterwarnings('ignore', category=UserWarning)

        self.results = {}

    def calculate_confusion_matrix(
        self,
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        class_names: List[str] = None
    ) -> Dict:
        """
        Confusion Matrix 계산 및 기본 메트릭 추출

        Args:
            model: 학습된 모델
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 레이블
            class_names: 클래스 이름 리스트

        Returns:
            Confusion Matrix 분석 결과
        """
        try:
            self.logger.info("🔄 Confusion Matrix 계산 시작...")

            # 예측 수행
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            # Confusion Matrix 계산
            cm = confusion_matrix(y_val, y_pred)

            # 클래스 이름 설정
            if class_names is None:
                unique_classes = sorted(set(list(y_val.unique()) + list(y_pred)))
                class_names = [f"Class_{i}" for i in unique_classes]

            # 정규화된 Confusion Matrix (행 단위)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

            # 기본 메트릭 계산
            precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average=None)

            # 전체 메트릭
            overall_metrics = {
                'accuracy': (y_pred == y_val).mean(),
                'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
                'macro_f1': f1.mean(),
                'weighted_f1': np.average(f1, weights=support),
                'cohen_kappa': cohen_kappa_score(y_val, y_pred)
            }

            # 클래스별 메트릭
            class_metrics = pd.DataFrame({
                'class': class_names[:len(precision)],
                'class_id': sorted(set(list(y_val.unique()) + list(y_pred)))[:len(precision)],
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            })

            results = {
                'confusion_matrix': cm,
                'confusion_matrix_normalized': cm_normalized,
                'class_names': class_names,
                'class_metrics': class_metrics,
                'overall_metrics': overall_metrics,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'true_labels': y_val
            }

            self.logger.info(f"✅ Confusion Matrix 계산 완료")
            self.logger.info(f"   - 전체 정확도: {overall_metrics['accuracy']:.4f}")
            self.logger.info(f"   - 균형 정확도: {overall_metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"   - Macro F1: {overall_metrics['macro_f1']:.4f}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Confusion Matrix 계산 오류: {str(e)}")
            raise

    def analyze_misclassification_patterns(self, cm_results: Dict) -> Dict:
        """
        오분류 패턴 분석

        Args:
            cm_results: Confusion Matrix 분석 결과

        Returns:
            오분류 패턴 분석 결과
        """
        try:
            self.logger.info("🔄 오분류 패턴 분석 시작...")

            cm = cm_results['confusion_matrix']
            cm_norm = cm_results['confusion_matrix_normalized']
            class_names = cm_results['class_names']

            n_classes = len(class_names)

            # 1. 가장 빈번한 오분류 찾기
            misclassifications = []
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j and cm[i, j] > 0:
                        misclassifications.append({
                            'true_class': class_names[i],
                            'predicted_class': class_names[j],
                            'count': cm[i, j],
                            'rate': cm_norm[i, j],
                            'true_class_id': i,
                            'predicted_class_id': j
                        })

            # 빈도순으로 정렬
            misclassifications.sort(key=lambda x: x['count'], reverse=True)

            # 2. 클래스별 오분류 통계
            class_error_stats = []
            for i, class_name in enumerate(class_names):
                if i < cm.shape[0]:  # 인덱스 범위 확인
                    correct = cm[i, i] if i < cm.shape[1] else 0
                    total = cm[i, :].sum() if cm[i, :].sum() > 0 else 1

                    # 가장 흔한 오분류 클래스
                    misclass_counts = [(j, cm[i, j]) for j in range(n_classes) if j != i and j < cm.shape[1]]
                    most_confused_with = max(misclass_counts, key=lambda x: x[1]) if misclass_counts else (None, 0)

                    class_error_stats.append({
                        'class': class_name,
                        'class_id': i,
                        'correct_predictions': correct,
                        'total_instances': total,
                        'error_rate': 1 - (correct / total),
                        'most_confused_with': class_names[most_confused_with[0]] if most_confused_with[0] is not None else 'None',
                        'most_confused_count': most_confused_with[1]
                    })

            # 3. 대칭적 오분류 찾기 (A→B와 B→A가 모두 높은 경우)
            symmetric_errors = []
            for i in range(n_classes):
                for j in range(i+1, n_classes):
                    if i < cm.shape[0] and j < cm.shape[1] and i < cm.shape[1] and j < cm.shape[0]:
                        error_ij = cm[i, j]
                        error_ji = cm[j, i]
                        if error_ij > 0 and error_ji > 0:
                            symmetric_errors.append({
                                'class_pair': (class_names[i], class_names[j]),
                                'mutual_confusion_score': min(error_ij, error_ji),
                                'total_confusion': error_ij + error_ji,
                                'asymmetry': abs(error_ij - error_ji) / max(error_ij + error_ji, 1)
                            })

            symmetric_errors.sort(key=lambda x: x['mutual_confusion_score'], reverse=True)

            results = {
                'top_misclassifications': misclassifications[:20],  # 상위 20개
                'class_error_statistics': class_error_stats,
                'symmetric_errors': symmetric_errors[:10],  # 상위 10개
                'total_misclassifications': len([x for x in misclassifications if x['count'] > 0])
            }

            self.logger.info("✅ 오분류 패턴 분석 완료")
            self.logger.info(f"   - 총 오분류 패턴 수: {results['total_misclassifications']}")
            if misclassifications:
                top_error = misclassifications[0]
                self.logger.info(f"   - 가장 빈번한 오분류: {top_error['true_class']} → {top_error['predicted_class']} ({top_error['count']}회)")

            return results

        except Exception as e:
            self.logger.error(f"❌ 오분류 패턴 분석 오류: {str(e)}")
            return {}

    def create_confusion_matrix_plots(
        self,
        cm_results: Dict,
        model_name: str = "RandomForest"
    ) -> Dict[str, str]:
        """
        Confusion Matrix 시각화

        Args:
            cm_results: Confusion Matrix 분석 결과
            model_name: 모델 이름

        Returns:
            생성된 플롯 파일 경로들
        """
        plot_paths = {}
        model_dir = self.confusion_dir / f"{model_name.lower()}-confusion"
        model_dir.mkdir(exist_ok=True)

        try:
            cm = cm_results['confusion_matrix']
            cm_norm = cm_results['confusion_matrix_normalized']
            class_names = cm_results['class_names']

            # 클래스 수가 많은 경우 라벨 단순화
            display_labels = class_names
            if len(class_names) > 15:
                display_labels = [f"C{i}" for i in range(len(class_names))]

            # 1. 원본 Confusion Matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=display_labels, yticklabels=display_labels)
            plt.title(f'{model_name}\nConfusion Matrix (Counts)')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.tight_layout()

            plot_path = model_dir / 'confusion_matrix_counts.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['counts'] = str(plot_path)

            # 2. 정규화된 Confusion Matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=display_labels, yticklabels=display_labels,
                       vmin=0, vmax=1)
            plt.title(f'{model_name}\nNormalized Confusion Matrix (Proportions)')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.tight_layout()

            plot_path = model_dir / 'confusion_matrix_normalized.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['normalized'] = str(plot_path)

            # 3. 클래스별 성능 메트릭
            class_metrics = cm_results['class_metrics']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Precision
            axes[0,0].bar(range(len(class_metrics)), class_metrics['precision'], alpha=0.7, color='skyblue')
            axes[0,0].set_title('Precision by Class')
            axes[0,0].set_xlabel('Class')
            axes[0,0].set_ylabel('Precision')
            axes[0,0].set_xticks(range(len(class_metrics)))
            axes[0,0].set_xticklabels(display_labels[:len(class_metrics)], rotation=45)
            axes[0,0].grid(True, alpha=0.3)

            # Recall
            axes[0,1].bar(range(len(class_metrics)), class_metrics['recall'], alpha=0.7, color='lightcoral')
            axes[0,1].set_title('Recall by Class')
            axes[0,1].set_xlabel('Class')
            axes[0,1].set_ylabel('Recall')
            axes[0,1].set_xticks(range(len(class_metrics)))
            axes[0,1].set_xticklabels(display_labels[:len(class_metrics)], rotation=45)
            axes[0,1].grid(True, alpha=0.3)

            # F1-Score
            axes[1,0].bar(range(len(class_metrics)), class_metrics['f1_score'], alpha=0.7, color='lightgreen')
            axes[1,0].set_title('F1-Score by Class')
            axes[1,0].set_xlabel('Class')
            axes[1,0].set_ylabel('F1-Score')
            axes[1,0].set_xticks(range(len(class_metrics)))
            axes[1,0].set_xticklabels(display_labels[:len(class_metrics)], rotation=45)
            axes[1,0].grid(True, alpha=0.3)

            # Support
            axes[1,1].bar(range(len(class_metrics)), class_metrics['support'], alpha=0.7, color='gold')
            axes[1,1].set_title('Support (Sample Count) by Class')
            axes[1,1].set_xlabel('Class')
            axes[1,1].set_ylabel('Sample Count')
            axes[1,1].set_xticks(range(len(class_metrics)))
            axes[1,1].set_xticklabels(display_labels[:len(class_metrics)], rotation=45)
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = model_dir / 'class_performance_metrics.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['metrics'] = str(plot_path)

            # 4. 오분류 히트맵 (대각선 제외)
            cm_errors = cm.copy()
            np.fill_diagonal(cm_errors, 0)  # 정확한 예측 제거

            if cm_errors.sum() > 0:  # 오분류가 있는 경우만
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm_errors, annot=True, fmt='d', cmap='Reds',
                           xticklabels=display_labels, yticklabels=display_labels)
                plt.title(f'{model_name}\nMisclassification Patterns')
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.tight_layout()

                plot_path = model_dir / 'misclassification_heatmap.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['misclassification'] = str(plot_path)

            self.logger.info(f"✅ {len(plot_paths)}개 Confusion Matrix 시각화 완료: {model_dir}")

        except Exception as e:
            self.logger.error(f"❌ Confusion Matrix 시각화 오류: {str(e)}")

        return plot_paths

    def analyze_class_difficulty(self, cm_results: Dict, pattern_results: Dict) -> Dict:
        """
        클래스별 분류 난이도 분석

        Args:
            cm_results: Confusion Matrix 분석 결과
            pattern_results: 오분류 패턴 분석 결과

        Returns:
            클래스 난이도 분석 결과
        """
        try:
            self.logger.info("🔄 클래스 난이도 분석 시작...")

            class_metrics = cm_results['class_metrics']
            class_errors = pattern_results['class_error_statistics']

            # 난이도 점수 계산 (여러 지표의 조합)
            difficulty_analysis = []

            for i, row in class_metrics.iterrows():
                error_info = next((x for x in class_errors if x['class_id'] == row['class_id']), {})

                # 난이도 점수 계산 (0-1, 높을수록 어려움)
                precision_penalty = 1 - row['precision']
                recall_penalty = 1 - row['recall']
                f1_penalty = 1 - row['f1_score']
                error_rate = error_info.get('error_rate', 0)

                # 가중 평균으로 종합 난이도 계산
                difficulty_score = (
                    0.3 * precision_penalty +
                    0.3 * recall_penalty +
                    0.2 * f1_penalty +
                    0.2 * error_rate
                )

                # 샘플 수 대비 성능 (불균형 영향)
                sample_ratio = row['support'] / class_metrics['support'].sum()
                imbalance_penalty = 1 / (sample_ratio + 0.01)  # 작은 클래스일수록 높은 penalty

                difficulty_analysis.append({
                    'class': row['class'],
                    'class_id': row['class_id'],
                    'difficulty_score': difficulty_score,
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score'],
                    'support': row['support'],
                    'sample_ratio': sample_ratio,
                    'imbalance_penalty': imbalance_penalty,
                    'error_rate': error_rate,
                    'most_confused_with': error_info.get('most_confused_with', 'None'),
                    'difficulty_level': self._categorize_difficulty(difficulty_score)
                })

            # 난이도순 정렬
            difficulty_analysis.sort(key=lambda x: x['difficulty_score'], reverse=True)

            # 통계 계산
            difficulty_stats = {
                'most_difficult_classes': difficulty_analysis[:5],
                'easiest_classes': difficulty_analysis[-5:],
                'avg_difficulty': np.mean([x['difficulty_score'] for x in difficulty_analysis]),
                'difficulty_distribution': {
                    'very_hard': len([x for x in difficulty_analysis if x['difficulty_level'] == 'Very Hard']),
                    'hard': len([x for x in difficulty_analysis if x['difficulty_level'] == 'Hard']),
                    'medium': len([x for x in difficulty_analysis if x['difficulty_level'] == 'Medium']),
                    'easy': len([x for x in difficulty_analysis if x['difficulty_level'] == 'Easy']),
                    'very_easy': len([x for x in difficulty_analysis if x['difficulty_level'] == 'Very Easy'])
                }
            }

            results = {
                'class_difficulty_analysis': difficulty_analysis,
                'difficulty_statistics': difficulty_stats
            }

            self.logger.info("✅ 클래스 난이도 분석 완료")
            if difficulty_analysis:
                hardest = difficulty_analysis[0]
                easiest = difficulty_analysis[-1]
                self.logger.info(f"   - 가장 어려운 클래스: {hardest['class']} (난이도: {hardest['difficulty_score']:.3f})")
                self.logger.info(f"   - 가장 쉬운 클래스: {easiest['class']} (난이도: {easiest['difficulty_score']:.3f})")

            return results

        except Exception as e:
            self.logger.error(f"❌ 클래스 난이도 분석 오류: {str(e)}")
            return {}

    def _categorize_difficulty(self, score: float) -> str:
        """난이도 점수를 카테고리로 분류"""
        if score >= 0.8:
            return 'Very Hard'
        elif score >= 0.6:
            return 'Hard'
        elif score >= 0.4:
            return 'Medium'
        elif score >= 0.2:
            return 'Easy'
        else:
            return 'Very Easy'

    def analyze_confusion_matrix_comprehensive(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        model_params: Dict = None,
        class_names: List[str] = None
    ) -> Dict:
        """
        종합적인 Confusion Matrix 분석

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 레이블
            model_params: 모델 파라미터
            class_names: 클래스 이름 리스트

        Returns:
            종합 분석 결과
        """
        try:
            self.logger.info("🚀 T010: Confusion Matrix 종합 분석 시작")

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

            # 클래스 이름 설정
            if class_names is None:
                unique_classes = sorted(y_train.unique())
                class_names = [f"Class_{i}" for i in unique_classes]

            # 모델 훈련
            self.logger.info("🔄 RandomForest 모델 훈련...")
            model = RandomForestClassifier(**model_params)
            model.fit(X_train_split, y_train_split)

            # 1. Confusion Matrix 계산
            cm_results = self.calculate_confusion_matrix(
                model, X_test, y_test, class_names
            )

            # 2. 오분류 패턴 분석
            pattern_results = self.analyze_misclassification_patterns(cm_results)

            # 3. 클래스 난이도 분석
            difficulty_results = self.analyze_class_difficulty(cm_results, pattern_results)

            # 4. 시각화 생성
            plot_paths = self.create_confusion_matrix_plots(
                cm_results, model_name="RandomForest-Balanced"
            )

            # 5. 결과 통합
            results = {
                'confusion_matrix_results': cm_results,
                'misclassification_patterns': pattern_results,
                'class_difficulty_analysis': difficulty_results,
                'plot_paths': plot_paths,
                'model_performance': {
                    'overall_metrics': cm_results['overall_metrics'],
                    'model_params': model_params
                }
            }

            # 실험 추적
            if self.experiment_tracker:
                self._log_to_experiment_tracker(results)

            self.results = results

            self.logger.info("🎉 T010: Confusion Matrix 분석 완료!")
            return results

        except Exception as e:
            self.logger.error(f"❌ T010 분석 오류: {str(e)}")
            raise

    def _log_to_experiment_tracker(self, results: Dict):
        """실험 추적 시스템에 결과 로그"""
        try:
            if not self.experiment_tracker:
                return

            overall_metrics = results['model_performance']['overall_metrics']

            # MLflow 로깅
            if hasattr(self.experiment_tracker, 'mlflow_client'):
                with mlflow.start_run(run_name="T010_Confusion_Matrix_Analysis"):
                    # 전체 메트릭 로깅
                    for metric_name, metric_value in overall_metrics.items():
                        mlflow.log_metric(f"overall_{metric_name}", metric_value)

                    # 클래스별 메트릭 로깅
                    class_metrics = results['confusion_matrix_results']['class_metrics']
                    for idx, row in class_metrics.head(10).iterrows():  # 상위 10개 클래스만
                        mlflow.log_metric(f"class_{row['class_id']}_f1", row['f1_score'])
                        mlflow.log_metric(f"class_{row['class_id']}_precision", row['precision'])
                        mlflow.log_metric(f"class_{row['class_id']}_recall", row['recall'])

                    # 시각화 로깅
                    for plot_name, plot_path in results['plot_paths'].items():
                        mlflow.log_artifact(plot_path, f"confusion_plots/{plot_name}")

            # WandB 로깅
            if hasattr(self.experiment_tracker, 'wandb_run'):
                # 전체 메트릭
                wandb_metrics = {f"confusion/{k}": v for k, v in overall_metrics.items()}
                wandb.log(wandb_metrics)

                # 시각화 로깅
                for plot_name, plot_path in results['plot_paths'].items():
                    wandb.log({f"confusion/{plot_name}": wandb.Image(plot_path)})

        except Exception as e:
            self.logger.warning(f"실험 추적 로깅 실패: {str(e)}")

    def print_summary(self, results: Dict):
        """분석 결과 요약 출력"""
        try:
            print("\n" + "="*80)
            print("🎯 T010: CONFUSION MATRIX 분석 결과 요약")
            print("="*80)

            overall_metrics = results['model_performance']['overall_metrics']
            class_metrics = results['confusion_matrix_results']['class_metrics']
            pattern_results = results['misclassification_patterns']
            difficulty_results = results['class_difficulty_analysis']

            print(f"📊 전체 성능:")
            print(f"   • 정확도 (Accuracy): {overall_metrics['accuracy']:.4f}")
            print(f"   • 균형 정확도 (Balanced Accuracy): {overall_metrics['balanced_accuracy']:.4f}")
            print(f"   • Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
            print(f"   • Weighted F1-Score: {overall_metrics['weighted_f1']:.4f}")
            print(f"   • Cohen's Kappa: {overall_metrics['cohen_kappa']:.4f}")

            print(f"\n🏆 클래스별 성능 (상위 5개):")
            top_classes = class_metrics.nlargest(5, 'f1_score')
            for idx, row in top_classes.iterrows():
                print(f"   {idx+1}. {row['class']:12s}: F1={row['f1_score']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f} (n={int(row['support'])})")

            print(f"\n📉 성능이 낮은 클래스 (하위 5개):")
            bottom_classes = class_metrics.nsmallest(5, 'f1_score')
            for idx, row in bottom_classes.iterrows():
                print(f"   • {row['class']:12s}: F1={row['f1_score']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f} (n={int(row['support'])})")

            print(f"\n🔄 주요 오분류 패턴:")
            top_misclass = pattern_results['top_misclassifications'][:5]
            for i, error in enumerate(top_misclass):
                print(f"   {i+1}. {error['true_class']} → {error['predicted_class']}: {error['count']}회 ({error['rate']:.1%})")

            print(f"\n😰 가장 어려운 클래스 (상위 5개):")
            difficult_stats = difficulty_results['difficulty_statistics']
            for i, cls in enumerate(difficult_stats['most_difficult_classes']):
                print(f"   {i+1}. {cls['class']:12s}: 난이도 {cls['difficulty_score']:.3f} ({cls['difficulty_level']})")
                print(f"      └─ 주요 혼동 대상: {cls['most_confused_with']}")

            print(f"\n😊 가장 쉬운 클래스 (상위 5개):")
            for i, cls in enumerate(difficult_stats['easiest_classes']):
                print(f"   {i+1}. {cls['class']:12s}: 난이도 {cls['difficulty_score']:.3f} ({cls['difficulty_level']})")

            print(f"\n📊 난이도 분포:")
            diff_dist = difficult_stats['difficulty_distribution']
            total_classes = sum(diff_dist.values())
            for level, count in diff_dist.items():
                percentage = (count / total_classes) * 100 if total_classes > 0 else 0
                print(f"   • {level}: {count}개 ({percentage:.1f}%)")

            print(f"\n🔍 추가 인사이트:")

            # 클래스 불균형 분석
            support_std = class_metrics['support'].std()
            support_mean = class_metrics['support'].mean()
            imbalance_ratio = support_std / support_mean if support_mean > 0 else 0

            if imbalance_ratio > 0.5:
                print(f"   • 심각한 클래스 불균형 감지 (변동계수: {imbalance_ratio:.2f})")
                min_class = class_metrics.loc[class_metrics['support'].idxmin()]
                max_class = class_metrics.loc[class_metrics['support'].idxmax()]
                ratio = max_class['support'] / max(min_class['support'], 1)
                print(f"     └─ 최대/최소 비율: {ratio:.1f}:1 ({max_class['class']} vs {min_class['class']})")

            # 대칭 오분류 패턴
            symmetric_errors = pattern_results['symmetric_errors']
            if symmetric_errors:
                print(f"   • 상호 혼동되는 클래스 쌍: {len(symmetric_errors)}쌍")
                top_symmetric = symmetric_errors[0]
                print(f"     └─ 가장 문제되는 쌍: {top_symmetric['class_pair'][0]} ↔ {top_symmetric['class_pair'][1]}")

            print(f"\n📁 생성된 시각화:")
            for plot_name, plot_path in results['plot_paths'].items():
                print(f"   • {plot_name}: {plot_path}")

            print("\n" + "="*80)

        except Exception as e:
            print(f"❌ 요약 출력 오류: {str(e)}")