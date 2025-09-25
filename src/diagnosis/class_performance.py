"""
클래스별 성능 분석 모듈 (Class Performance Analysis)

이 모듈은 21개 클래스의 개별 성능을 분석하고 시각화합니다.
- 클래스별 F1-score, Precision, Recall 분석
- 혼동행렬(Confusion Matrix) 시각화
- 클래스 불균형 영향 분석
- 성능 개선 권장사항 제공

작성자: Claude
날짜: 2024-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from ..tracking.experiment_tracker import ExperimentTracker
from ..utils.config import Config

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

class ClassPerformanceAnalyzer:
    """
    21개 클래스의 개별 성능을 분석하는 클래스

    주요 기능:
    1. 클래스별 상세 성능 메트릭 계산
    2. 클래스 불균형 영향 분석
    3. 혼동행렬 시각화
    4. 성능 개선 권장사항 생성
    """

    def __init__(self, config: Config = None, experiment_tracker: ExperimentTracker = None):
        self.config = config or Config()
        self.tracker = experiment_tracker
        self.class_performance = {}
        self.confusion_matrices = {}
        self.recommendations = []

        # 21개 클래스 정보
        self.class_names = [f'Class_{i}' for i in range(21)]

    def analyze_class_performance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        models: Dict[str, Any] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        클래스별 성능을 종합 분석합니다.

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 레이블
            models: 분석할 모델 딕셔너리
            cv_folds: 교차검증 폴드 수

        Returns:
            Dict: 클래스별 성능 분석 결과
        """
        print("🎯 클래스별 성능 분석을 시작합니다...")

        if self.tracker:
            self.tracker.start_run(
                run_name="class_performance_analysis",
                description="21개 클래스의 개별 성능 분석",
                tags={"analysis_type": "class_performance", "task": "T007"}
            )

        # 기본 모델 설정
        if models is None:
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'RandomForest_Balanced': RandomForestClassifier(
                    n_estimators=100, class_weight='balanced', random_state=42
                )
            }

        # 클래스 분포 분석
        class_distribution = self._analyze_class_distribution(y_train)

        # 각 모델별 클래스 성능 분석
        model_results = {}
        for model_name, model in models.items():
            print(f"\n📊 {model_name} 모델 클래스별 성능 분석 중...")

            # 교차검증으로 예측 수행
            y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_folds, method='predict')

            # 클래스별 성능 계산
            class_metrics = self._calculate_class_metrics(y_train, y_pred_cv)

            # 혼동행렬 생성
            cm = confusion_matrix(y_train, y_pred_cv)

            model_results[model_name] = {
                'class_metrics': class_metrics,
                'confusion_matrix': cm,
                'overall_f1': f1_score(y_train, y_pred_cv, average='macro')
            }

        # 클래스별 성능 비교 분석
        comparison_analysis = self._compare_class_performance(model_results)

        # 시각화 생성
        self._create_visualizations(class_distribution, model_results, comparison_analysis)

        # 권장사항 생성
        self._generate_recommendations(class_distribution, comparison_analysis)

        # 결과 통합
        analysis_result = {
            'class_distribution': class_distribution,
            'model_results': model_results,
            'comparison_analysis': comparison_analysis,
            'recommendations': self.recommendations,
            'summary': self._generate_summary(model_results, comparison_analysis)
        }

        # 실험 추적에 결과 기록
        if self.tracker:
            self._log_results_to_tracker(analysis_result)

        print("✅ 클래스별 성능 분석이 완료되었습니다!")
        return analysis_result

    def _analyze_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """클래스 분포를 분석합니다."""
        print("📈 클래스 분포 분석 중...")

        class_counts = y.value_counts().sort_index()
        total_samples = len(y)

        distribution_stats = {
            'counts': class_counts.to_dict(),
            'percentages': (class_counts / total_samples * 100).to_dict(),
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'min_samples': class_counts.min(),
            'max_samples': class_counts.max(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'std_deviation': class_counts.std()
        }

        # 클래스 불균형 심각도 평가
        imbalance_severity = self._assess_imbalance_severity(distribution_stats)
        distribution_stats['imbalance_severity'] = imbalance_severity

        return distribution_stats

    def _assess_imbalance_severity(self, stats: Dict) -> str:
        """클래스 불균형 심각도를 평가합니다."""
        ratio = stats['imbalance_ratio']

        if ratio <= 2:
            return "낮음 (Balanced)"
        elif ratio <= 5:
            return "보통 (Moderate)"
        elif ratio <= 10:
            return "높음 (High)"
        else:
            return "매우 높음 (Severe)"

    def _calculate_class_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """클래스별 상세 메트릭을 계산합니다."""

        # 전체 분류 보고서
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        class_metrics = {}
        for class_idx in range(21):
            class_key = str(class_idx)
            if class_key in report:
                class_metrics[class_idx] = {
                    'f1_score': report[class_key]['f1-score'],
                    'precision': report[class_key]['precision'],
                    'recall': report[class_key]['recall'],
                    'support': int(report[class_key]['support'])
                }
            else:
                # 해당 클래스가 예측되지 않은 경우
                class_metrics[class_idx] = {
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'support': 0
                }

        return class_metrics

    def _compare_class_performance(self, model_results: Dict) -> Dict[str, Any]:
        """모델 간 클래스별 성능을 비교 분석합니다."""
        print("⚖️ 모델 간 클래스별 성능 비교 중...")

        comparison = {
            'best_model_per_class': {},
            'worst_performing_classes': [],
            'best_performing_classes': [],
            'performance_variance': {},
            'class_difficulty_ranking': {}
        }

        # 각 클래스별로 최고 성능 모델 찾기
        for class_idx in range(21):
            best_f1 = 0
            best_model = None
            f1_scores = []

            for model_name, results in model_results.items():
                class_f1 = results['class_metrics'].get(class_idx, {}).get('f1_score', 0)
                f1_scores.append(class_f1)

                if class_f1 > best_f1:
                    best_f1 = class_f1
                    best_model = model_name

            comparison['best_model_per_class'][class_idx] = {
                'model': best_model,
                'f1_score': best_f1
            }

            # 성능 분산 계산
            comparison['performance_variance'][class_idx] = np.std(f1_scores)

        # 최고/최저 성능 클래스 식별
        class_best_f1s = [(idx, data['f1_score']) for idx, data in comparison['best_model_per_class'].items()]
        class_best_f1s.sort(key=lambda x: x[1])

        comparison['worst_performing_classes'] = [
            {'class': idx, 'f1_score': f1} for idx, f1 in class_best_f1s[:5]
        ]
        comparison['best_performing_classes'] = [
            {'class': idx, 'f1_score': f1} for idx, f1 in class_best_f1s[-5:]
        ]

        # 클래스 난이도 순위
        comparison['class_difficulty_ranking'] = {
            idx: {'rank': rank + 1, 'f1_score': f1}
            for rank, (idx, f1) in enumerate(class_best_f1s)
        }

        return comparison

    def _create_visualizations(
        self,
        class_distribution: Dict,
        model_results: Dict,
        comparison_analysis: Dict
    ):
        """클래스 성능 관련 시각화를 생성합니다."""
        print("📊 클래스 성능 시각화 생성 중...")

        # 시각화 저장 경로
        plots_dir = self.config.get_paths()['plots'] / 'class_performance'
        plots_dir.mkdir(exist_ok=True)

        # 1. 클래스 분포 시각화
        self._plot_class_distribution(class_distribution, plots_dir)

        # 2. 클래스별 F1-score 비교
        self._plot_class_f1_comparison(model_results, plots_dir)

        # 3. 혼동행렬 히트맵
        self._plot_confusion_matrices(model_results, plots_dir)

        # 4. 클래스 난이도 분석
        self._plot_class_difficulty_analysis(comparison_analysis, class_distribution, plots_dir)

        # 5. 성능-샘플수 상관관계
        self._plot_performance_vs_samples(comparison_analysis, class_distribution, plots_dir)

    def _plot_class_distribution(self, class_distribution: Dict, plots_dir):
        """클래스 분포를 시각화합니다."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 클래스별 샘플 수
        classes = list(range(21))
        counts = [class_distribution['counts'].get(i, 0) for i in classes]

        ax1.bar(classes, counts, color='skyblue', alpha=0.7)
        ax1.set_title('클래스별 샘플 수 분포', fontsize=14, fontweight='bold')
        ax1.set_xlabel('클래스 번호')
        ax1.set_ylabel('샘플 수')
        ax1.grid(True, alpha=0.3)

        # 불균형 정도 시각화 (비율)
        percentages = [class_distribution['percentages'].get(i, 0) for i in classes]
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(classes)))

        ax2.bar(classes, percentages, color=colors)
        ax2.set_title('클래스별 비율 분포', fontsize=14, fontweight='bold')
        ax2.set_xlabel('클래스 번호')
        ax2.set_ylabel('비율 (%)')
        ax2.grid(True, alpha=0.3)

        # 통계 정보 추가
        stats_text = f"""불균형 심각도: {class_distribution['imbalance_severity']}
        불균형 비율: {class_distribution['imbalance_ratio']:.2f}
        최소 샘플: {class_distribution['min_samples']}
        최대 샘플: {class_distribution['max_samples']}"""

        fig.suptitle('클래스 분포 분석', fontsize=16, fontweight='bold')
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        plt.tight_layout()
        plt.savefig(plots_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(plots_dir / 'class_distribution.png'), "class_distribution")

    def _plot_class_f1_comparison(self, model_results: Dict, plots_dir):
        """클래스별 F1-score를 비교 시각화합니다."""

        fig, ax = plt.subplots(figsize=(16, 8))

        classes = list(range(21))
        x = np.arange(len(classes))
        width = 0.35

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (model_name, results) in enumerate(model_results.items()):
            f1_scores = [results['class_metrics'].get(cls, {}).get('f1_score', 0) for cls in classes]
            offset = (i - len(model_results)/2 + 0.5) * width / len(model_results)

            bars = ax.bar(x + offset, f1_scores, width/len(model_results),
                         label=model_name, color=colors[i % len(colors)], alpha=0.8)

            # 값 표시 (0.5 이하만)
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height < 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_title('클래스별 F1-Score 비교', fontsize=16, fontweight='bold')
        ax.set_xlabel('클래스 번호', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # 목표 성능 라인 추가
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='목표 성능 (0.9)')
        ax.axhline(y=0.67596, color='orange', linestyle='--', alpha=0.7, label='현재 전체 성능')

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / 'class_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(plots_dir / 'class_f1_comparison.png'), "class_f1_comparison")

    def _plot_confusion_matrices(self, model_results: Dict, plots_dir):
        """혼동행렬들을 시각화합니다."""

        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]

        for i, (model_name, results) in enumerate(model_results.items()):
            cm = results['confusion_matrix']

            # 정규화된 혼동행렬
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')

            # 컬러바 추가
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # 축 레이블
            axes[i].set_xlabel('예측 클래스')
            axes[i].set_ylabel('실제 클래스')

            # 틱 설정
            tick_marks = np.arange(21)
            axes[i].set_xticks(tick_marks[::2])  # 2개씩 건너뛰기
            axes[i].set_yticks(tick_marks[::2])
            axes[i].set_xticklabels([str(i) for i in tick_marks[::2]])
            axes[i].set_yticklabels([str(i) for i in tick_marks[::2]])

        plt.suptitle('모델별 혼동행렬 비교', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(plots_dir / 'confusion_matrices.png'), "confusion_matrices")

    def _plot_class_difficulty_analysis(self, comparison_analysis: Dict, class_distribution: Dict, plots_dir):
        """클래스 난이도 분석을 시각화합니다."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 클래스 난이도 순위
        difficulty_data = comparison_analysis['class_difficulty_ranking']
        classes = list(difficulty_data.keys())
        f1_scores = [difficulty_data[cls]['f1_score'] for cls in classes]
        ranks = [difficulty_data[cls]['rank'] for cls in classes]

        # 난이도별 색상 (낮은 F1 = 어려운 클래스 = 빨간색)
        colors = plt.cm.RdYlGn([score for score in f1_scores])

        bars1 = ax1.bar(classes, f1_scores, color=colors, alpha=0.8)
        ax1.set_title('클래스별 난이도 분석\n(낮은 F1-Score = 어려운 클래스)', fontweight='bold')
        ax1.set_xlabel('클래스 번호')
        ax1.set_ylabel('최고 F1-Score')
        ax1.grid(True, alpha=0.3)

        # 어려운 클래스 강조
        worst_classes = [item['class'] for item in comparison_analysis['worst_performing_classes']]
        for i, (bar, cls) in enumerate(zip(bars1, classes)):
            if cls in worst_classes:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'어려움\n{bar.get_height():.2f}',
                        ha='center', va='bottom', fontweight='bold', color='red')

        # 2. 성능 vs 샘플 수 상관관계
        sample_counts = [class_distribution['counts'].get(cls, 0) for cls in classes]

        scatter = ax2.scatter(sample_counts, f1_scores, c=f1_scores, cmap='RdYlGn',
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        # 상관계수 계산
        correlation = np.corrcoef(sample_counts, f1_scores)[0, 1]

        ax2.set_title(f'성능 vs 샘플 수 상관관계\n상관계수: {correlation:.3f}', fontweight='bold')
        ax2.set_xlabel('샘플 수')
        ax2.set_ylabel('최고 F1-Score')
        ax2.grid(True, alpha=0.3)

        # 컬러바
        plt.colorbar(scatter, ax=ax2, label='F1-Score')

        # 문제 클래스 레이블링
        for cls in worst_classes[:3]:  # 상위 3개만
            idx = classes.index(cls)
            ax2.annotate(f'Class {cls}',
                        (sample_counts[idx], f1_scores[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        plt.savefig(plots_dir / 'class_difficulty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(plots_dir / 'class_difficulty_analysis.png'), "class_difficulty_analysis")

    def _plot_performance_vs_samples(self, comparison_analysis: Dict, class_distribution: Dict, plots_dir):
        """성능과 샘플 수의 관계를 상세 분석합니다."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        classes = list(range(21))
        f1_scores = [comparison_analysis['best_model_per_class'][cls]['f1_score'] for cls in classes]
        sample_counts = [class_distribution['counts'].get(cls, 0) for cls in classes]

        # 1. 산점도 with 회귀선
        ax1.scatter(sample_counts, f1_scores, c=f1_scores, cmap='RdYlGn', s=100, alpha=0.7)

        # 회귀선 추가
        z = np.polyfit(sample_counts, f1_scores, 1)
        p = np.poly1d(z)
        ax1.plot(sample_counts, p(sample_counts), "r--", alpha=0.8, linewidth=2)

        correlation = np.corrcoef(sample_counts, f1_scores)[0, 1]
        ax1.set_title(f'성능 vs 샘플 수 (상관계수: {correlation:.3f})', fontweight='bold')
        ax1.set_xlabel('샘플 수')
        ax1.set_ylabel('최고 F1-Score')
        ax1.grid(True, alpha=0.3)

        # 2. 샘플 수 구간별 성능 분포
        # 샘플 수를 4분위로 나누기
        sample_quartiles = np.percentile(sample_counts, [25, 50, 75])

        q1_classes = [cls for cls in classes if sample_counts[cls] <= sample_quartiles[0]]
        q2_classes = [cls for cls in classes if sample_quartiles[0] < sample_counts[cls] <= sample_quartiles[1]]
        q3_classes = [cls for cls in classes if sample_quartiles[1] < sample_counts[cls] <= sample_quartiles[2]]
        q4_classes = [cls for cls in classes if sample_counts[cls] > sample_quartiles[2]]

        quartile_f1s = [
            [f1_scores[cls] for cls in q1_classes],
            [f1_scores[cls] for cls in q2_classes],
            [f1_scores[cls] for cls in q3_classes],
            [f1_scores[cls] for cls in q4_classes]
        ]

        box_plot = ax2.boxplot(quartile_f1s, labels=['Q1 (적음)', 'Q2', 'Q3', 'Q4 (많음)'], patch_artist=True)
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax2.set_title('샘플 수 분위별 성능 분포', fontweight='bold')
        ax2.set_xlabel('샘플 수 분위')
        ax2.set_ylabel('F1-Score')
        ax2.grid(True, alpha=0.3)

        # 3. 클래스별 성능 순위 vs 샘플 순위
        sample_ranks = np.argsort(np.argsort(sample_counts)) + 1  # 1부터 시작
        f1_ranks = np.argsort(np.argsort(f1_scores)) + 1

        ax3.scatter(sample_ranks, f1_ranks, c=f1_scores, cmap='RdYlGn', s=100, alpha=0.7)
        ax3.plot([1, 21], [1, 21], 'k--', alpha=0.5, label='완벽한 상관관계')

        rank_correlation = np.corrcoef(sample_ranks, f1_ranks)[0, 1]
        ax3.set_title(f'샘플 순위 vs 성능 순위\n순위 상관계수: {rank_correlation:.3f}', fontweight='bold')
        ax3.set_xlabel('샘플 수 순위 (1=가장 적음)')
        ax3.set_ylabel('성능 순위 (1=가장 낮음)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 문제 클래스 식별 (낮은 성능 + 적은 샘플)
        problem_threshold_f1 = np.percentile(f1_scores, 25)  # 하위 25%
        problem_threshold_samples = np.percentile(sample_counts, 25)  # 하위 25%

        problem_classes = []
        normal_classes = []

        for cls in classes:
            if (f1_scores[cls] <= problem_threshold_f1 and
                sample_counts[cls] <= problem_threshold_samples):
                problem_classes.append(cls)
            else:
                normal_classes.append(cls)

        # 문제 클래스와 일반 클래스 분리해서 표시
        problem_samples = [sample_counts[cls] for cls in problem_classes]
        problem_f1s = [f1_scores[cls] for cls in problem_classes]
        normal_samples = [sample_counts[cls] for cls in normal_classes]
        normal_f1s = [f1_scores[cls] for cls in normal_classes]

        ax4.scatter(normal_samples, normal_f1s, c='blue', label='일반 클래스', s=100, alpha=0.7)
        ax4.scatter(problem_samples, problem_f1s, c='red', label='문제 클래스', s=100, alpha=0.7)

        # 문제 영역 표시
        ax4.axvline(x=problem_threshold_samples, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=problem_threshold_f1, color='red', linestyle='--', alpha=0.5)
        ax4.fill_betweenx([0, problem_threshold_f1], 0, problem_threshold_samples,
                         alpha=0.2, color='red', label='문제 영역')

        ax4.set_title(f'문제 클래스 식별\n문제 클래스: {len(problem_classes)}개', fontweight='bold')
        ax4.set_xlabel('샘플 수')
        ax4.set_ylabel('F1-Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 문제 클래스 레이블링
        for cls in problem_classes:
            ax4.annotate(f'{cls}',
                        (sample_counts[cls], f1_scores[cls]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_vs_samples_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(plots_dir / 'performance_vs_samples_detailed.png'), "performance_vs_samples")

        return problem_classes

    def _generate_recommendations(self, class_distribution: Dict, comparison_analysis: Dict):
        """클래스별 성능 개선 권장사항을 생성합니다."""
        print("💡 클래스별 성능 개선 권장사항 생성 중...")

        recommendations = []

        # 1. 클래스 불균형 관련 권장사항
        imbalance_severity = class_distribution['imbalance_severity']
        if "높음" in imbalance_severity or "매우" in imbalance_severity:
            recommendations.extend([
                {
                    "category": "클래스 불균형 해결",
                    "priority": "높음",
                    "issue": f"클래스 불균형 심각도: {imbalance_severity}",
                    "solution": "SMOTE, ADASYN 등 오버샘플링 기법 적용",
                    "expected_impact": "소수 클래스 재현율 20-30% 향상",
                    "implementation": "imblearn.over_sampling 모듈 활용"
                },
                {
                    "category": "모델 가중치 조정",
                    "priority": "높음",
                    "issue": "불균형 데이터로 인한 편향된 학습",
                    "solution": "class_weight='balanced' 또는 focal loss 적용",
                    "expected_impact": "Macro F1-score 5-10% 향상",
                    "implementation": "sklearn.utils.class_weight.compute_class_weight 활용"
                }
            ])

        # 2. 최저 성능 클래스 관련 권장사항
        worst_classes = comparison_analysis['worst_performing_classes']
        worst_class_ids = [item['class'] for item in worst_classes[:3]]
        worst_avg_f1 = np.mean([item['f1_score'] for item in worst_classes[:3]])

        if worst_avg_f1 < 0.3:
            recommendations.append({
                "category": "최저 성능 클래스 집중 개선",
                "priority": "매우 높음",
                "issue": f"클래스 {worst_class_ids} 평균 F1-score: {worst_avg_f1:.3f}",
                "solution": "1) 특성 엔지니어링 집중, 2) 클래스별 전문 모델 개발, 3) 앙상블 가중치 조정",
                "expected_impact": "해당 클래스들 F1-score 50% 이상 향상",
                "implementation": "클래스별 특화된 특성 생성 및 모델 튜닝"
            })

        # 3. 성능 분산 관련 권장사항
        high_variance_classes = []
        for class_idx, variance in comparison_analysis['performance_variance'].items():
            if variance > 0.1:  # 높은 분산 기준
                high_variance_classes.append((class_idx, variance))

        if len(high_variance_classes) > 5:
            recommendations.append({
                "category": "모델 안정성 개선",
                "priority": "보통",
                "issue": f"{len(high_variance_classes)}개 클래스에서 모델 간 성능 차이 큼",
                "solution": "앙상블 방법 개선 및 교차검증 전략 재검토",
                "expected_impact": "전체 모델 안정성 15-20% 향상",
                "implementation": "Stacking, Voting 앙상블 및 Stratified CV 적용"
            })

        # 4. 특성 엔지니어링 권장사항
        recommendations.append({
            "category": "클래스별 특성 엔지니어링",
            "priority": "보통",
            "issue": "범용 특성으로 인한 클래스별 차별화 부족",
            "solution": "1) 클래스별 중요 특성 식별, 2) 특성 상호작용 생성, 3) 도메인 지식 활용",
            "expected_impact": "전체 Macro F1-score 10-15% 향상",
            "implementation": "SHAP, Permutation Importance를 활용한 클래스별 특성 분석"
        })

        # 5. 평가 메트릭 다양화
        recommendations.append({
            "category": "평가 메트릭 보완",
            "priority": "낮음",
            "issue": "Macro F1-score만으로는 클래스별 문제 파악 한계",
            "solution": "클래스별 Precision, Recall, F1-score 개별 모니터링",
            "expected_impact": "모델 개선 방향성 명확화",
            "implementation": "classification_report 및 클래스별 메트릭 추적"
        })

        self.recommendations = recommendations

    def _generate_summary(self, model_results: Dict, comparison_analysis: Dict) -> Dict[str, Any]:
        """분석 결과 요약을 생성합니다."""

        # 전체 성능 통계
        overall_f1s = [results['overall_f1'] for results in model_results.values()]
        best_overall_f1 = max(overall_f1s)
        best_model = max(model_results.items(), key=lambda x: x[1]['overall_f1'])[0]

        # 클래스별 성능 통계
        all_class_f1s = []
        for results in model_results.values():
            for class_metrics in results['class_metrics'].values():
                all_class_f1s.append(class_metrics['f1_score'])

        # 최고/최저 성능 클래스
        worst_classes = comparison_analysis['worst_performing_classes'][:3]
        best_classes = comparison_analysis['best_performing_classes'][-3:]

        summary = {
            "전체_성능": {
                "최고_Macro_F1": round(best_overall_f1, 4),
                "최고_성능_모델": best_model,
                "목표_달성률": round((best_overall_f1 / 0.9) * 100, 1),
                "현재_대비_개선률": round(((best_overall_f1 / 0.67596) - 1) * 100, 1)
            },
            "클래스별_성능_분포": {
                "평균_F1": round(np.mean(all_class_f1s), 4),
                "중앙값_F1": round(np.median(all_class_f1s), 4),
                "표준편차": round(np.std(all_class_f1s), 4),
                "최고_클래스_F1": round(max(all_class_f1s), 4),
                "최저_클래스_F1": round(min(all_class_f1s), 4)
            },
            "문제_클래스": {
                "최저_성능_3개": [
                    f"Class {item['class']} (F1: {item['f1_score']:.3f})"
                    for item in worst_classes
                ],
                "zero_f1_클래스_수": sum(1 for f1 in all_class_f1s if f1 == 0),
                "목표_미달_클래스_수": sum(1 for f1 in all_class_f1s if f1 < 0.9)
            },
            "우수_클래스": {
                "최고_성능_3개": [
                    f"Class {item['class']} (F1: {item['f1_score']:.3f})"
                    for item in best_classes
                ],
                "목표_달성_클래스_수": sum(1 for f1 in all_class_f1s if f1 >= 0.9)
            },
            "개선_권장사항_수": len(self.recommendations),
            "즉시_조치_필요": len([r for r in self.recommendations if r['priority'] in ['매우 높음', '높음']])
        }

        return summary

    def _log_results_to_tracker(self, analysis_result: Dict[str, Any]):
        """분석 결과를 실험 추적 시스템에 기록합니다."""
        if not self.tracker:
            return

        # 메트릭 기록
        summary = analysis_result['summary']

        metrics = {
            "class_performance/best_macro_f1": summary['전체_성능']['최고_Macro_F1'],
            "class_performance/target_achievement_rate": summary['전체_성능']['목표_달성률'],
            "class_performance/improvement_rate": summary['전체_성능']['현재_대비_개선률'],
            "class_performance/avg_class_f1": summary['클래스별_성능_분포']['평균_F1'],
            "class_performance/class_f1_std": summary['클래스별_성능_분포']['표준편차'],
            "class_performance/zero_f1_classes": summary['문제_클래스']['zero_f1_클래스_수'],
            "class_performance/below_target_classes": summary['문제_클래스']['목표_미달_클래스_수'],
            "class_performance/target_achieved_classes": summary['우수_클래스']['목표_달성_클래스_수'],
            "class_performance/high_priority_recommendations": summary['즉시_조치_필요']
        }

        self.tracker.log_metrics(metrics)

        # 파라미터 기록
        params = {
            "analysis_type": "class_performance",
            "num_classes": 21,
            "best_model": summary['전체_성능']['최고_성능_모델'],
            "imbalance_severity": analysis_result['class_distribution']['imbalance_severity']
        }

        self.tracker.log_params(params)

        print(f"📈 실험 결과가 추적 시스템에 기록되었습니다.")

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """생성된 권장사항을 반환합니다."""
        return self.recommendations

    def print_summary(self, analysis_result: Dict[str, Any]):
        """분석 결과 요약을 출력합니다."""

        summary = analysis_result['summary']

        print("\n" + "="*60)
        print("🎯 클래스별 성능 분석 결과 요약")
        print("="*60)

        # 전체 성능
        print(f"\n📊 전체 성능:")
        print(f"  • 최고 Macro F1-score: {summary['전체_성능']['최고_Macro_F1']:.4f}")
        print(f"  • 최고 성능 모델: {summary['전체_성능']['최고_성능_모델']}")
        print(f"  • 목표 달성률: {summary['전체_성능']['목표_달성률']:.1f}%")
        print(f"  • 현재 대비 개선률: {summary['전체_성능']['현재_대비_개선률']:.1f}%")

        # 클래스별 성능 분포
        print(f"\n📈 클래스별 성능 분포:")
        print(f"  • 평균 F1-score: {summary['클래스별_성능_분포']['평균_F1']:.4f}")
        print(f"  • 성능 편차 (표준편차): {summary['클래스별_성능_분포']['표준편차']:.4f}")
        print(f"  • 최고 클래스 F1: {summary['클래스별_성능_분포']['최고_클래스_F1']:.4f}")
        print(f"  • 최저 클래스 F1: {summary['클래스별_성능_분포']['최저_클래스_F1']:.4f}")

        # 문제 클래스
        print(f"\n🚨 문제 클래스:")
        print(f"  • Zero F1 클래스: {summary['문제_클래스']['zero_f1_클래스_수']}개")
        print(f"  • 목표 미달 클래스: {summary['문제_클래스']['목표_미달_클래스_수']}/21개")
        print(f"  • 최저 성능 클래스:")
        for class_info in summary['문제_클래스']['최저_성능_3개']:
            print(f"    - {class_info}")

        # 우수 클래스
        print(f"\n🏆 우수 클래스:")
        print(f"  • 목표 달성 클래스: {summary['우수_클래스']['목표_달성_클래스_수']}/21개")
        print(f"  • 최고 성능 클래스:")
        for class_info in summary['우수_클래스']['최고_성능_3개']:
            print(f"    - {class_info}")

        # 개선 권장사항
        print(f"\n💡 개선 권장사항:")
        print(f"  • 총 권장사항: {summary['개선_권장사항_수']}개")
        print(f"  • 즉시 조치 필요: {summary['즉시_조치_필요']}개")

        high_priority_recs = [r for r in self.recommendations if r['priority'] in ['매우 높음', '높음']]
        for i, rec in enumerate(high_priority_recs[:3], 1):
            print(f"    {i}. [{rec['priority']}] {rec['category']}: {rec['solution']}")

        print("\n" + "="*60)
        print("✅ 클래스별 성능 분석이 완료되었습니다!")
        print("📁 상세 시각화는 experiments/plots/class_performance/ 에서 확인하세요.")
        print("="*60)


def main():
    """메인 실행 함수"""
    from ..utils.config import Config
    from ..tracking.experiment_tracker import ExperimentTracker
    import pandas as pd

    # 설정 로드
    config = Config()

    # 실험 추적기 초기화
    tracker = ExperimentTracker(
        project_name="dacon-smartmh-02",
        experiment_name="class_performance_analysis"
    )

    # 클래스 성능 분석기 초기화
    analyzer = ClassPerformanceAnalyzer(config=config, experiment_tracker=tracker)

    try:
        # 데이터 로드 (실제 경로에 맞게 수정 필요)
        data_path = config.get_paths()['data']
        train_data = pd.read_csv(data_path / 'train.csv')

        # 특성과 레이블 분리
        X_train = train_data.drop(['target'], axis=1, errors='ignore')
        y_train = train_data['target']

        # 클래스별 성능 분석 실행
        results = analyzer.analyze_class_performance(X_train, y_train)

        # 결과 요약 출력
        analyzer.print_summary(results)

        return results

    except Exception as e:
        print(f"❌ 클래스별 성능 분석 중 오류 발생: {str(e)}")
        return None


if __name__ == "__main__":
    main()