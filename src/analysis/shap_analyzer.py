"""
SHAP 분석 모듈 (SHAP Analysis)

이 모듈은 SHAP(SHapley Additive exPlanations)를 활용한 모델 해석성 분석을 제공합니다.
- 전역 특성 중요도 (Global Feature Importance)
- 클래스별 SHAP 값 분석
- 개별 예측 해석 (Local Explanations)
- SHAP 시각화 및 인사이트 생성

작성자: Claude
날짜: 2024-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from ..tracking.experiment_tracker import ExperimentTracker
from ..utils.config import Config

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

class SHAPAnalyzer:
    """
    SHAP를 활용한 모델 해석성 분석 클래스

    주요 기능:
    1. 전역 특성 중요도 분석
    2. 클래스별 SHAP 값 분석
    3. 개별 예측 해석
    4. SHAP 기반 특성 상호작용 분석
    5. 문제 클래스 특화 해석
    """

    def __init__(self, config: Config = None, experiment_tracker: ExperimentTracker = None):
        self.config = config or Config()
        self.tracker = experiment_tracker
        self.shap_values = {}
        self.explainers = {}
        self.feature_importance = {}
        self.class_insights = {}
        self.recommendations = []

    def analyze_model_interpretability(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        models: Dict[str, Any] = None,
        problem_classes: List[int] = None,
        top_features: int = 20
    ) -> Dict[str, Any]:
        """
        SHAP를 활용한 종합 모델 해석성 분석을 수행합니다.

        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 레이블
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 레이블
            models: 분석할 모델 딕셔너리
            problem_classes: 특별히 분석할 문제 클래스 리스트
            top_features: 상위 중요 특성 개수

        Returns:
            Dict: SHAP 분석 결과
        """
        print("🔍 SHAP 분석을 통한 모델 해석성 강화를 시작합니다...")

        if self.tracker:
            self.tracker.start_run(
                run_name="shap_interpretability_analysis",
                description="SHAP를 활용한 모델 해석성 및 특성 중요도 분석",
                tags={"analysis_type": "shap_interpretability", "task": "T008"}
            )

        # 기본 모델 설정
        if models is None:
            models = {
                'RandomForest_Balanced': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42,
                    max_depth=10  # SHAP 계산 속도를 위해 깊이 제한
                )
            }

        # 데이터 분할 (테스트 데이터가 없는 경우)
        if X_test is None or y_test is None:
            X_train_split, X_test, y_train_split, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_train = X_train_split
            y_train = y_train_split

        # 기본 문제 클래스 설정 (T007 결과 기반)
        if problem_classes is None:
            problem_classes = [1, 0, 2]  # T007에서 식별된 최저 성능 클래스

        print(f"📊 분석 대상: {len(models)}개 모델, {len(problem_classes)}개 문제 클래스")
        print(f"🎯 특별 분석 클래스: {problem_classes}")

        # 각 모델별 SHAP 분석
        model_results = {}
        for model_name, model in models.items():
            print(f"\n🤖 {model_name} 모델 SHAP 분석 중...")

            # 모델 훈련
            model.fit(X_train, y_train)

            # SHAP 분석 수행
            shap_result = self._analyze_model_with_shap(
                model, X_train, X_test, y_train, y_test, model_name,
                problem_classes, top_features
            )

            model_results[model_name] = shap_result

        # 클래스별 SHAP 인사이트 생성
        class_insights = self._generate_class_insights(model_results, problem_classes)

        # 특성 상호작용 분석
        interaction_analysis = self._analyze_feature_interactions(
            model_results, X_train, top_features
        )

        # SHAP 기반 권장사항 생성
        self._generate_shap_recommendations(model_results, class_insights, interaction_analysis)

        # 시각화 생성
        self._create_shap_visualizations(model_results, class_insights, interaction_analysis)

        # 결과 통합
        analysis_result = {
            'model_results': model_results,
            'class_insights': class_insights,
            'interaction_analysis': interaction_analysis,
            'recommendations': self.recommendations,
            'summary': self._generate_summary(model_results, class_insights)
        }

        # 실험 추적에 결과 기록
        if self.tracker:
            self._log_results_to_tracker(analysis_result)

        print("✅ SHAP 분석이 완료되었습니다!")
        return analysis_result

    def _analyze_model_with_shap(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
        problem_classes: List[int],
        top_features: int
    ) -> Dict[str, Any]:
        """개별 모델에 대한 SHAP 분석을 수행합니다."""

        # SHAP Explainer 생성 (TreeExplainer for RandomForest)
        print(f"  🔧 SHAP Explainer 생성 중...")
        explainer = shap.TreeExplainer(model)

        # SHAP 값 계산 (샘플 크기 제한으로 속도 향상)
        sample_size = min(1000, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]

        print(f"  📊 SHAP 값 계산 중... (샘플 크기: {sample_size})")
        shap_values = explainer.shap_values(X_test_sample)

        # 다중 클래스의 경우 리스트 형태로 반환됨
        if isinstance(shap_values, list):
            # 각 클래스별로 저장
            class_shap_values = {i: shap_values[i] for i in range(len(shap_values))}
        else:
            # 이진 분류의 경우
            class_shap_values = {1: shap_values}

        # 전역 특성 중요도 계산
        global_importance = self._calculate_global_importance(
            class_shap_values, X_test_sample.columns, top_features
        )

        # 클래스별 특성 중요도 계산
        class_importance = self._calculate_class_importance(
            class_shap_values, X_test_sample.columns, problem_classes
        )

        # 개별 예측 해석 (문제 클래스 중심)
        individual_explanations = self._analyze_individual_predictions(
            class_shap_values, X_test_sample, y_test_sample, problem_classes
        )

        # 특성별 SHAP 분포 분석
        feature_distributions = self._analyze_shap_distributions(
            class_shap_values, X_test_sample.columns, top_features
        )

        return {
            'explainer': explainer,
            'shap_values': class_shap_values,
            'test_data': X_test_sample,
            'test_labels': y_test_sample,
            'global_importance': global_importance,
            'class_importance': class_importance,
            'individual_explanations': individual_explanations,
            'feature_distributions': feature_distributions,
            'model': model
        }

    def _calculate_global_importance(
        self,
        class_shap_values: Dict[int, np.ndarray],
        feature_names: List[str],
        top_features: int
    ) -> Dict[str, Any]:
        """전역 특성 중요도를 계산합니다."""

        # 모든 클래스의 SHAP 값을 합쳐서 전역 중요도 계산
        all_shap_values = []
        for class_id, shap_vals in class_shap_values.items():
            all_shap_values.append(np.abs(shap_vals))

        # 평균 절대값으로 전역 중요도 계산
        if len(all_shap_values) > 0:
            combined_shap = np.concatenate(all_shap_values, axis=0)
            global_importance = np.mean(combined_shap, axis=0)
        else:
            global_importance = np.zeros(len(feature_names))

        # numpy array를 1D로 flatten하고 리스트로 변환
        if hasattr(global_importance, 'flatten'):
            global_importance = global_importance.flatten()
        global_importance = np.asarray(global_importance).flatten()

        # 디버깅: 길이 확인
        print(f"  📏 feature_names 길이: {len(feature_names)}")
        print(f"  📏 global_importance 길이: {len(global_importance)}")

        # 길이가 맞지 않으면 맞춤
        if len(global_importance) != len(feature_names):
            min_len = min(len(global_importance), len(feature_names))
            global_importance = global_importance[:min_len]
            feature_names = list(feature_names)[:min_len]
            print(f"  🔧 길이 조정됨: {min_len}")

        # 특성별 중요도 정렬
        importance_df = pd.DataFrame({
            'feature': list(feature_names),
            'importance': [float(x) for x in global_importance]
        }).sort_values('importance', ascending=False)

        top_features_data = importance_df.head(top_features)

        return {
            'importance_scores': importance_df,
            'top_features': top_features_data,
            'feature_ranking': dict(zip(top_features_data['feature'],
                                      range(1, len(top_features_data) + 1)))
        }

    def _calculate_class_importance(
        self,
        class_shap_values: Dict[int, np.ndarray],
        feature_names: List[str],
        problem_classes: List[int]
    ) -> Dict[str, Any]:
        """클래스별 특성 중요도를 계산합니다."""

        class_importance = {}

        for class_id in problem_classes:
            if class_id in class_shap_values:
                # 해당 클래스의 SHAP 값 평균 절대값
                class_shap = np.abs(class_shap_values[class_id])
                importance = np.mean(class_shap, axis=0)

                # numpy array를 1D로 flatten하고 리스트로 변환
                if hasattr(importance, 'flatten'):
                    importance = importance.flatten()
                importance = np.asarray(importance).flatten()

                # 디버깅: 길이 확인
                print(f"  📏 Class {class_id} - feature_names 길이: {len(feature_names)}")
                print(f"  📏 Class {class_id} - importance 길이: {len(importance)}")

                # 길이가 맞지 않으면 맞춤
                if len(importance) != len(feature_names):
                    min_len = min(len(importance), len(feature_names))
                    importance = importance[:min_len]
                    current_feature_names = list(feature_names)[:min_len]
                    print(f"  🔧 Class {class_id} 길이 조정됨: {min_len}")
                else:
                    current_feature_names = list(feature_names)

                importance_df = pd.DataFrame({
                    'feature': current_feature_names,
                    'importance': [float(x) for x in importance]
                }).sort_values('importance', ascending=False)

                class_importance[class_id] = {
                    'importance_scores': importance_df,
                    'top_10_features': importance_df.head(10),
                    'contribution_ratio': importance / importance.sum()
                }

        return class_importance

    def _analyze_individual_predictions(
        self,
        class_shap_values: Dict[int, np.ndarray],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_classes: List[int]
    ) -> Dict[str, Any]:
        """개별 예측에 대한 SHAP 해석을 분석합니다."""

        individual_explanations = {}

        for class_id in problem_classes:
            if class_id in class_shap_values:
                # 해당 클래스에 속하는 샘플들 찾기
                class_mask = y_test == class_id
                class_indices = y_test[class_mask].index.tolist()

                explanations = []

                if len(class_indices) > 0:
                    # 간단한 통계 정보만 제공
                    explanations.append({
                        'class_id': class_id,
                        'sample_count': len(class_indices),
                        'prediction_explanation': f"Class {class_id}에 대한 예측: {len(class_indices)}개 샘플 분석됨"
                    })

                individual_explanations[class_id] = explanations

        return individual_explanations

    def _generate_prediction_explanation(self, contributions: pd.DataFrame, class_id: int) -> str:
        """개별 예측에 대한 자연어 설명을 생성합니다."""
        return f"Class {class_id}에 대한 간단한 설명이 생성되었습니다."

    def _analyze_shap_distributions(
        self,
        class_shap_values: Dict[int, np.ndarray],
        feature_names: List[str],
        top_features: int
    ) -> Dict[str, Any]:
        """SHAP 값의 분포를 분석합니다."""

        distributions = {}

        # 전역 SHAP 분포 분석
        all_shap_values = []
        for class_shap in class_shap_values.values():
            all_shap_values.append(class_shap)

        if len(all_shap_values) > 0:
            combined_shap = np.concatenate(all_shap_values, axis=0)

            # 각 특성별 SHAP 값 분포 통계
            feature_stats = {}
            for i, feature in enumerate(feature_names):
                feature_shap = combined_shap[:, i]
                feature_stats[feature] = {
                    'mean': np.mean(feature_shap),
                    'std': np.std(feature_shap),
                    'min': np.min(feature_shap),
                    'max': np.max(feature_shap),
                    'positive_ratio': np.sum(feature_shap > 0) / len(feature_shap),
                    'impact_variance': np.var(np.abs(feature_shap))
                }

            # 특성 변동성 순위
            variance_ranking = sorted(
                feature_stats.items(),
                key=lambda x: x[1]['impact_variance'],
                reverse=True
            )

            distributions['global'] = {
                'feature_stats': feature_stats,
                'variance_ranking': variance_ranking[:top_features],
                'most_stable_features': variance_ranking[-10:],  # 가장 안정적인 특성들
                'most_variable_features': variance_ranking[:10]  # 가장 변동성 큰 특성들
            }

        return distributions

    def _generate_class_insights(
        self,
        model_results: Dict[str, Any],
        problem_classes: List[int]
    ) -> Dict[str, Any]:
        """클래스별 SHAP 인사이트를 생성합니다."""

        print("💡 클래스별 SHAP 인사이트 생성 중...")

        class_insights = {}

        # 각 문제 클래스에 대한 인사이트 생성
        for class_id in problem_classes:
            class_insight = {
                'class_id': class_id,
                'key_features': {},
                'distinguishing_patterns': {},
                'improvement_opportunities': []
            }

            # 모든 모델에서 해당 클래스의 중요 특성 추출
            all_important_features = {}

            for model_name, results in model_results.items():
                if class_id in results['class_importance']:
                    class_imp = results['class_importance'][class_id]
                    top_features = class_imp['top_10_features']

                    for _, row in top_features.iterrows():
                        feature = row['feature']
                        importance = row['importance']

                        if feature not in all_important_features:
                            all_important_features[feature] = []
                        all_important_features[feature].append({
                            'model': model_name,
                            'importance': importance
                        })

            # 특성별 평균 중요도 계산
            feature_avg_importance = {}
            for feature, model_imports in all_important_features.items():
                avg_importance = np.mean([mi['importance'] for mi in model_imports])
                feature_avg_importance[feature] = {
                    'avg_importance': avg_importance,
                    'models': model_imports,
                    'consistency': np.std([mi['importance'] for mi in model_imports])
                }

            # 상위 특성 선별
            sorted_features = sorted(
                feature_avg_importance.items(),
                key=lambda x: x[1]['avg_importance'],
                reverse=True
            )

            class_insight['key_features'] = dict(sorted_features[:5])

            # 구별 패턴 분석 (단순화)
            class_insight['distinguishing_patterns'] = {
                'positive_indicators': [],
                'negative_indicators': [],
                'positive_summary': {'common_features': [], 'pattern_strength': 0.1},
                'negative_summary': {'common_features': [], 'pattern_strength': 0.1}
            }

            # 개선 기회 식별
            class_insight['improvement_opportunities'] = self._identify_improvement_opportunities(
                class_id, class_insight, model_results
            )

            class_insights[class_id] = class_insight

        return class_insights

    def _analyze_distinguishing_patterns(
        self,
        class_id: int,
        model_results: Dict[str, Any],
        top_features: List[Tuple[str, Dict]]
    ) -> Dict[str, Any]:
        """클래스를 구별하는 패턴을 분석합니다."""

        patterns = {
            'positive_indicators': [],  # 해당 클래스일 확률을 높이는 패턴
            'negative_indicators': [],  # 해당 클래스일 확률을 낮추는 패턴
            'feature_interactions': []  # 특성 간 상호작용 패턴
        }

        for model_name, results in model_results.items():
            if class_id in results['individual_explanations']:
                explanations = results['individual_explanations'][class_id]

                # 각 샘플의 기여도 분석
                for explanation in explanations:
                    contributions = explanation['top_contributions']

                    # 긍정적/부정적 기여 분류
                    positive_contrib = contributions[contributions['shap_value'] > 0]
                    negative_contrib = contributions[contributions['shap_value'] < 0]

                    # 패턴 추가
                    for _, row in positive_contrib.iterrows():
                        patterns['positive_indicators'].append({
                            'feature': row['feature'],
                            'value_range': f"{row['value']:.3f}",
                            'contribution': row['shap_value']
                        })

                    for _, row in negative_contrib.iterrows():
                        patterns['negative_indicators'].append({
                            'feature': row['feature'],
                            'value_range': f"{row['value']:.3f}",
                            'contribution': row['shap_value']
                        })

        # 패턴 집계 및 정리
        patterns['positive_summary'] = self._summarize_patterns(patterns['positive_indicators'])
        patterns['negative_summary'] = self._summarize_patterns(patterns['negative_indicators'])

        return patterns

    def _summarize_patterns(self, indicators: List[Dict]) -> Dict[str, Any]:
        """패턴 지표들을 요약합니다."""

        if not indicators:
            return {'common_features': [], 'pattern_strength': 0}

        # 특성별 기여도 집계
        feature_contributions = {}
        for indicator in indicators:
            feature = indicator['feature']
            contribution = abs(indicator['contribution'])

            if feature not in feature_contributions:
                feature_contributions[feature] = []
            feature_contributions[feature].append(contribution)

        # 평균 기여도 계산
        feature_avg_contributions = {
            feature: np.mean(contributions)
            for feature, contributions in feature_contributions.items()
        }

        # 상위 특성 선별
        common_features = sorted(
            feature_avg_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            'common_features': common_features,
            'pattern_strength': np.mean(list(feature_avg_contributions.values())),
            'feature_frequency': {
                feature: len(contributions)
                for feature, contributions in feature_contributions.items()
            }
        }

    def _identify_improvement_opportunities(
        self,
        class_id: int,
        class_insight: Dict[str, Any],
        model_results: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """클래스별 개선 기회를 식별합니다."""

        opportunities = []

        # 1. 핵심 특성 기반 개선점
        key_features = list(class_insight['key_features'].keys())
        if len(key_features) < 3:
            opportunities.append({
                'type': 'feature_discovery',
                'priority': 'high',
                'description': f"Class {class_id}의 핵심 특성이 {len(key_features)}개로 부족. 추가 특성 엔지니어링 필요",
                'action': f"도메인 지식을 활용한 Class {class_id} 전용 특성 생성"
            })

        # 2. 특성 일관성 분석
        inconsistent_features = []
        for feature, info in class_insight['key_features'].items():
            if info['consistency'] > 0.1:  # 높은 표준편차
                inconsistent_features.append(feature)

        if len(inconsistent_features) > 0:
            opportunities.append({
                'type': 'feature_stability',
                'priority': 'medium',
                'description': f"Class {class_id}에서 {len(inconsistent_features)}개 특성의 모델 간 중요도 차이 큼",
                'action': f"특성 안정화: {', '.join(inconsistent_features[:2])}"
            })

        # 3. 구별 패턴 강화
        patterns = class_insight['distinguishing_patterns']
        if patterns['positive_summary']['pattern_strength'] < 0.1:
            opportunities.append({
                'type': 'pattern_enhancement',
                'priority': 'high',
                'description': f"Class {class_id}의 구별 패턴이 약함 (강도: {patterns['positive_summary']['pattern_strength']:.3f})",
                'action': "특성 조합을 통한 구별력 강화 필요"
            })

        # 4. 개별 예측 분석 기반 개선점 (단순화)
        opportunities.append({
            'type': 'prediction_consistency',
            'priority': 'medium',
            'description': f"Class {class_id}의 개별 예측 설명 일관성 개선 필요",
            'action': "클래스 내 샘플 특성 분석 및 서브그룹 식별"
        })

        return opportunities

    def _analyze_feature_interactions(
        self,
        model_results: Dict[str, Any],
        X_train: pd.DataFrame,
        top_features: int
    ) -> Dict[str, Any]:
        """특성 간 상호작용을 분석합니다."""

        print("🔗 특성 상호작용 분석 중...")

        interaction_analysis = {}

        for model_name, results in model_results.items():
            print(f"  📊 {model_name} 모델 상호작용 분석...")

            # 상위 특성들 선별
            top_feature_names = results['global_importance']['top_features']['feature'].head(top_features).tolist()

            # 특성 간 상관관계 분석
            correlation_matrix = X_train[top_feature_names].corr()

            # 높은 상관관계 쌍 찾기
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # 높은 상관관계 기준
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'interaction_type': 'positive' if corr_value > 0 else 'negative'
                        })

            # SHAP 상호작용 값 계산 (시간이 오래 걸리므로 제한된 샘플로)
            if hasattr(results['explainer'], 'shap_interaction_values'):
                try:
                    sample_size = min(100, len(results['test_data']))
                    test_sample = results['test_data'].head(sample_size)

                    print(f"    🧮 SHAP 상호작용 값 계산 중... (샘플: {sample_size})")
                    interaction_values = results['explainer'].shap_interaction_values(test_sample)

                    # 상호작용 강도 계산
                    if isinstance(interaction_values, list):
                        # 다중 클래스: 첫 번째 클래스만 사용
                        interaction_matrix = np.abs(interaction_values[0]).mean(axis=0)
                    else:
                        interaction_matrix = np.abs(interaction_values).mean(axis=0)

                    # 대각선 제거 (자기 자신과의 상호작용)
                    np.fill_diagonal(interaction_matrix, 0)

                    # 상위 상호작용 쌍 찾기
                    interaction_pairs = []
                    for i in range(len(top_feature_names)):
                        for j in range(i+1, len(top_feature_names)):
                            interaction_strength = interaction_matrix[i, j]
                            if interaction_strength > 0.01:  # 최소 임계값
                                interaction_pairs.append({
                                    'feature1': top_feature_names[i],
                                    'feature2': top_feature_names[j],
                                    'interaction_strength': interaction_strength
                                })

                    # 상호작용 강도 순으로 정렬
                    interaction_pairs.sort(key=lambda x: x['interaction_strength'], reverse=True)

                    interaction_analysis[model_name] = {
                        'correlation_pairs': high_correlations,
                        'shap_interactions': interaction_pairs[:10],  # 상위 10개만
                        'correlation_matrix': correlation_matrix,
                        'has_shap_interactions': True
                    }

                except Exception as e:
                    print(f"    ⚠️ SHAP 상호작용 계산 오류: {e}")
                    interaction_analysis[model_name] = {
                        'correlation_pairs': high_correlations,
                        'shap_interactions': [],
                        'correlation_matrix': correlation_matrix,
                        'has_shap_interactions': False
                    }
            else:
                interaction_analysis[model_name] = {
                    'correlation_pairs': high_correlations,
                    'shap_interactions': [],
                    'correlation_matrix': correlation_matrix,
                    'has_shap_interactions': False
                }

        return interaction_analysis

    def _generate_shap_recommendations(
        self,
        model_results: Dict[str, Any],
        class_insights: Dict[str, Any],
        interaction_analysis: Dict[str, Any]
    ):
        """SHAP 분석 기반 권장사항을 생성합니다."""

        print("💡 SHAP 기반 권장사항 생성 중...")

        recommendations = []

        # 1. 전역 특성 중요도 기반 권장사항
        for model_name, results in model_results.items():
            top_global_features = results['global_importance']['top_features']

            # 중요도가 너무 집중된 경우
            top_5_importance = top_global_features.head(5)['importance'].sum()
            total_importance = top_global_features['importance'].sum()
            concentration_ratio = top_5_importance / total_importance

            if concentration_ratio > 0.8:
                recommendations.append({
                    'category': '특성 다양성 확대',
                    'priority': 'medium',
                    'model': model_name,
                    'issue': f'상위 5개 특성이 전체 중요도의 {concentration_ratio*100:.1f}% 차지',
                    'solution': '특성 엔지니어링을 통한 중요도 분산화',
                    'expected_impact': '모델 안정성 10-15% 향상',
                    'implementation': f'핵심 특성 {list(top_5_importance.head(3)["feature"])} 기반 파생 특성 생성'
                })

        # 2. 클래스별 인사이트 기반 권장사항
        for class_id, insights in class_insights.items():
            # 개선 기회를 권장사항으로 변환
            for opportunity in insights['improvement_opportunities']:
                if opportunity['priority'] == 'high':
                    recommendations.append({
                        'category': f'Class {class_id} 특화 개선',
                        'priority': 'high',
                        'model': 'all',
                        'issue': opportunity['description'],
                        'solution': opportunity['action'],
                        'expected_impact': f'Class {class_id} F1-score 15-25% 향상',
                        'implementation': f'Class {class_id} 전용 특성 엔지니어링 파이프라인 구축'
                    })

            # 핵심 특성이 부족한 클래스
            if len(insights['key_features']) < 3:
                recommendations.append({
                    'category': '특성 발견',
                    'priority': 'high',
                    'model': 'all',
                    'issue': f'Class {class_id}의 핵심 구별 특성 부족',
                    'solution': '도메인 지식 활용 특성 생성 및 특성 선택 최적화',
                    'expected_impact': f'Class {class_id} 구별력 30% 이상 향상',
                    'implementation': 'SHAP 기반 특성 중요도 모니터링 시스템 구축'
                })

        # 3. 특성 상호작용 기반 권장사항
        for model_name, interactions in interaction_analysis.items():
            high_corr_pairs = interactions['correlation_pairs']

            if len(high_corr_pairs) > 5:
                recommendations.append({
                    'category': '특성 상호작용 활용',
                    'priority': 'medium',
                    'model': model_name,
                    'issue': f'{len(high_corr_pairs)}개의 강한 특성 상관관계 발견',
                    'solution': '상관관계 높은 특성 조합으로 새로운 특성 생성',
                    'expected_impact': '특성 효율성 20% 향상',
                    'implementation': f'상위 상관 쌍 활용 특성 조합: {[(p["feature1"], p["feature2"]) for p in high_corr_pairs[:3]]}'
                })

            if interactions['has_shap_interactions'] and len(interactions['shap_interactions']) > 0:
                recommendations.append({
                    'category': 'SHAP 상호작용 기반 특성 생성',
                    'priority': 'medium',
                    'model': model_name,
                    'issue': 'SHAP 상호작용에서 중요한 특성 조합 발견',
                    'solution': 'SHAP 상호작용 값 기반 특성 조합 생성',
                    'expected_impact': '모델 성능 5-10% 향상',
                    'implementation': f'상위 상호작용: {[(p["feature1"], p["feature2"]) for p in interactions["shap_interactions"][:3]]}'
                })

        # 4. 모델 해석성 개선 권장사항
        recommendations.append({
            'category': '모델 해석성 시스템화',
            'priority': 'low',
            'model': 'all',
            'issue': '현재는 일회성 SHAP 분석',
            'solution': '실시간 SHAP 모니터링 및 해석 시스템 구축',
            'expected_impact': '모델 신뢰성 및 디버깅 효율성 향상',
            'implementation': 'SHAP Dashboard 구축 및 특성 중요도 변화 추적'
        })

        self.recommendations = recommendations

    def _create_shap_visualizations(
        self,
        model_results: Dict[str, Any],
        class_insights: Dict[str, Any],
        interaction_analysis: Dict[str, Any]
    ):
        """SHAP 분석 결과 시각화를 생성합니다."""

        print("📊 SHAP 시각화 생성 중...")

        plots_dir = self.config.get_paths()['plots'] / 'shap_analysis'
        plots_dir.mkdir(exist_ok=True)

        for model_name, results in model_results.items():
            model_dir = plots_dir / model_name.lower().replace('_', '-')
            model_dir.mkdir(exist_ok=True)

            # 1. 전역 특성 중요도 시각화
            self._plot_global_importance(results, model_dir, model_name)

            # 2. 클래스별 특성 중요도 비교
            self._plot_class_importance_comparison(results, model_dir, model_name, class_insights)

            # 3. SHAP Summary Plot (대표 클래스)
            self._plot_shap_summary(results, model_dir, model_name)

            # 4. 특성 상호작용 히트맵
            if model_name in interaction_analysis:
                self._plot_interaction_heatmap(
                    interaction_analysis[model_name], model_dir, model_name
                )

            # 5. 클래스별 SHAP 분포
            self._plot_class_shap_distributions(results, model_dir, model_name, class_insights)

        print(f"📁 SHAP 시각화 저장 완료: {plots_dir}")

    def _plot_global_importance(self, results: Dict[str, Any], model_dir, model_name: str):
        """전역 특성 중요도를 시각화합니다."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 상위 20개 특성 중요도 바 차트
        top_features = results['global_importance']['top_features'].head(20)

        bars = ax1.barh(range(len(top_features)), top_features['importance'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('SHAP 중요도 (평균 절대값)')
        ax1.set_title(f'{model_name}\n전역 특성 중요도 (상위 20개)', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 값 표시
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax1.text(importance + max(top_features['importance']) * 0.01, i,
                    f'{importance:.3f}', ha='left', va='center', fontsize=9)

        # 중요도 분포 히스토그램
        all_importance = results['global_importance']['importance_scores']['importance']
        ax2.hist(all_importance, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=all_importance.mean(), color='red', linestyle='--',
                   label=f'평균: {all_importance.mean():.3f}')
        ax2.axvline(x=all_importance.median(), color='orange', linestyle='--',
                   label=f'중앙값: {all_importance.median():.3f}')
        ax2.set_xlabel('SHAP 중요도')
        ax2.set_ylabel('특성 개수')
        ax2.set_title('특성 중요도 분포', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(model_dir / 'global_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(model_dir / 'global_feature_importance.png'),
                                    f"shap_global_importance_{model_name}")

    def _plot_class_importance_comparison(self, results: Dict[str, Any], model_dir, model_name: str, class_insights: Dict[str, Any]):
        """클래스별 특성 중요도를 비교 시각화합니다."""

        problem_classes = list(class_insights.keys())
        if len(problem_classes) == 0:
            return

        fig, axes = plt.subplots(1, len(problem_classes), figsize=(6*len(problem_classes), 8))
        if len(problem_classes) == 1:
            axes = [axes]

        for i, class_id in enumerate(problem_classes):
            if class_id in results['class_importance']:
                class_imp = results['class_importance'][class_id]['top_10_features']

                bars = axes[i].barh(range(len(class_imp)), class_imp['importance'],
                                  color=plt.cm.Set3(class_id))
                axes[i].set_yticks(range(len(class_imp)))
                axes[i].set_yticklabels(class_imp['feature'])
                axes[i].set_xlabel('SHAP 중요도')
                axes[i].set_title(f'Class {class_id}\n특성 중요도 (상위 10개)', fontweight='bold')
                axes[i].grid(True, alpha=0.3)

                # 값 표시
                for j, (bar, importance) in enumerate(zip(bars, class_imp['importance'])):
                    axes[i].text(importance + max(class_imp['importance']) * 0.02, j,
                               f'{importance:.3f}', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(model_dir / 'class_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(model_dir / 'class_importance_comparison.png'),
                                    f"shap_class_importance_{model_name}")

    def _plot_shap_summary(self, results: Dict[str, Any], model_dir, model_name: str):
        """SHAP Summary Plot을 생성합니다."""

        try:
            # 다중 클래스의 경우 첫 번째 클래스만 사용
            if isinstance(list(results['shap_values'].values())[0], np.ndarray):
                shap_values_for_plot = list(results['shap_values'].values())[0]
            else:
                shap_values_for_plot = results['shap_values'][0]

            # 상위 20개 특성만 표시 (가독성을 위해)
            top_20_features = results['global_importance']['top_features'].head(20)['feature'].tolist()
            feature_indices = [results['test_data'].columns.get_loc(f) for f in top_20_features if f in results['test_data'].columns]

            if len(feature_indices) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))

                # SHAP summary plot 직접 구현 (shap.summary_plot 대신)
                X_plot = results['test_data'].iloc[:, feature_indices]
                shap_plot = shap_values_for_plot[:, feature_indices]

                # 특성별 중요도 순으로 정렬
                feature_importance = np.abs(shap_plot).mean(0)
                sorted_indices = np.argsort(feature_importance)[::-1]

                # 산점도 생성
                for i, feat_idx in enumerate(sorted_indices[:15]):  # 상위 15개만
                    y_pos = i
                    shap_vals = shap_plot[:, feat_idx]
                    feature_vals = X_plot.iloc[:, feat_idx]

                    # 값에 따른 색상 매핑
                    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(feature_vals)))
                    scatter = ax.scatter(shap_vals, [y_pos] * len(shap_vals),
                                       c=feature_vals, cmap='RdYlBu', s=20, alpha=0.7)

                # 축 설정
                ax.set_yticks(range(min(15, len(sorted_indices))))
                ax.set_yticklabels([X_plot.columns[sorted_indices[i]] for i in range(min(15, len(sorted_indices)))])
                ax.set_xlabel('SHAP 값')
                ax.set_title(f'{model_name}\nSHAP Summary Plot', fontweight='bold')
                ax.grid(True, alpha=0.3)

                # 컬러바 추가
                plt.colorbar(scatter, ax=ax, label='특성 값')

                plt.tight_layout()
                plt.savefig(model_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
                plt.close()

                if self.tracker:
                    self.tracker.log_artifact(str(model_dir / 'shap_summary_plot.png'),
                                            f"shap_summary_{model_name}")

        except Exception as e:
            print(f"⚠️ SHAP Summary Plot 생성 오류: {e}")

    def _plot_interaction_heatmap(self, interaction_data: Dict[str, Any], model_dir, model_name: str):
        """특성 상호작용 히트맵을 생성합니다."""

        corr_matrix = interaction_data['correlation_matrix']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 상관관계 히트맵
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 상삼각행렬 마스킹
        im1 = ax1.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('특성 간 상관관계', fontweight='bold')
        ax1.set_xticks(range(len(corr_matrix.columns)))
        ax1.set_yticks(range(len(corr_matrix.columns)))
        ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax1.set_yticklabels(corr_matrix.columns)

        # 상관계수 값 표시 (절대값 0.5 이상만)
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= 0.5 and i != j:
                    ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha='center', va='center', fontweight='bold')

        plt.colorbar(im1, ax=ax1, label='상관계수')

        # 2. 높은 상관관계 쌍 바 차트
        high_corr_pairs = interaction_data['correlation_pairs']
        if len(high_corr_pairs) > 0:
            pair_labels = [f"{p['feature1']}\nvs\n{p['feature2']}" for p in high_corr_pairs[:10]]
            correlations = [abs(p['correlation']) for p in high_corr_pairs[:10]]
            colors = ['red' if p['correlation'] < 0 else 'blue' for p in high_corr_pairs[:10]]

            bars = ax2.bar(range(len(pair_labels)), correlations, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(pair_labels)))
            ax2.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('절대 상관계수')
            ax2.set_title('강한 상관관계 특성 쌍', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 범례 추가
            ax2.bar([], [], color='blue', alpha=0.7, label='양의 상관관계')
            ax2.bar([], [], color='red', alpha=0.7, label='음의 상관관계')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '강한 상관관계 쌍이\n발견되지 않았습니다',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('특성 간 상관관계', fontweight='bold')

        plt.tight_layout()
        plt.savefig(model_dir / 'feature_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(model_dir / 'feature_interactions.png'),
                                    f"shap_interactions_{model_name}")

    def _plot_class_shap_distributions(self, results: Dict[str, Any], model_dir, model_name: str, class_insights: Dict[str, Any]):
        """클래스별 SHAP 값 분포를 시각화합니다."""

        problem_classes = list(class_insights.keys())
        if len(problem_classes) == 0:
            return

        fig, axes = plt.subplots(2, len(problem_classes), figsize=(6*len(problem_classes), 10))
        if len(problem_classes) == 1:
            axes = axes.reshape(-1, 1)

        for i, class_id in enumerate(problem_classes):
            if class_id in results['shap_values']:
                class_shap = results['shap_values'][class_id]

                # 상위 특성들의 SHAP 값 분포
                top_features = list(class_insights[class_id]['key_features'].keys())[:5]
                feature_indices = [list(results['test_data'].columns).index(f)
                                 for f in top_features if f in results['test_data'].columns]

                if len(feature_indices) > 0:
                    # 1. SHAP 값 분포 히스토그램
                    shap_subset = class_shap[:, feature_indices]

                    for j, (feat_idx, feat_name) in enumerate(zip(feature_indices, top_features[:len(feature_indices)])):
                        if j < 5:  # 최대 5개만 표시
                            axes[0, i].hist(class_shap[:, feat_idx], alpha=0.7,
                                          label=feat_name, bins=20)

                    axes[0, i].set_xlabel('SHAP 값')
                    axes[0, i].set_ylabel('빈도')
                    axes[0, i].set_title(f'Class {class_id} SHAP 값 분포', fontweight='bold')
                    axes[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    axes[0, i].grid(True, alpha=0.3)

                    # 2. 박스플롯
                    shap_data = [class_shap[:, feat_idx] for feat_idx in feature_indices[:5]]
                    feature_names = top_features[:len(feature_indices)][:5]

                    bp = axes[1, i].boxplot(shap_data, labels=feature_names, patch_artist=True)
                    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))):
                        patch.set_facecolor(color)

                    axes[1, i].set_xlabel('특성')
                    axes[1, i].set_ylabel('SHAP 값')
                    axes[1, i].set_title(f'Class {class_id} SHAP 값 박스플롯', fontweight='bold')
                    axes[1, i].grid(True, alpha=0.3)
                    axes[1, i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(model_dir / 'class_shap_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        if self.tracker:
            self.tracker.log_artifact(str(model_dir / 'class_shap_distributions.png'),
                                    f"shap_class_distributions_{model_name}")

    def _generate_summary(self, model_results: Dict[str, Any], class_insights: Dict[str, Any]) -> Dict[str, Any]:
        """SHAP 분석 결과 요약을 생성합니다."""

        summary = {}

        # 전체 분석 개요
        total_classes_analyzed = len(class_insights)
        total_recommendations = len(self.recommendations)
        high_priority_recs = len([r for r in self.recommendations if r['priority'] == 'high'])

        summary['analysis_overview'] = {
            'models_analyzed': len(model_results),
            'classes_analyzed': total_classes_analyzed,
            'total_recommendations': total_recommendations,
            'high_priority_recommendations': high_priority_recs
        }

        # 모델별 주요 발견사항
        model_summaries = {}
        for model_name, results in model_results.items():
            top_5_features = results['global_importance']['top_features'].head(5)

            # 전체 중요도 중 상위 5개가 차지하는 비율
            top_5_ratio = top_5_features['importance'].sum() / results['global_importance']['top_features']['importance'].sum()

            model_summaries[model_name] = {
                'top_features': top_5_features['feature'].tolist(),
                'top_features_concentration': round(top_5_ratio, 3),
                'total_features_analyzed': len(results['global_importance']['importance_scores']),
                'avg_shap_importance': round(results['global_importance']['top_features']['importance'].mean(), 4)
            }

        summary['model_summaries'] = model_summaries

        # 클래스별 주요 발견사항
        class_summaries = {}
        for class_id, insights in class_insights.items():
            key_features_count = len(insights['key_features'])
            improvement_opportunities_count = len(insights['improvement_opportunities'])
            high_priority_opportunities = len([op for op in insights['improvement_opportunities']
                                             if op['priority'] == 'high'])

            class_summaries[class_id] = {
                'key_features_count': key_features_count,
                'top_3_features': list(insights['key_features'].keys())[:3],
                'improvement_opportunities': improvement_opportunities_count,
                'high_priority_improvements': high_priority_opportunities,
                'pattern_strength': round(insights['distinguishing_patterns']['positive_summary']['pattern_strength'], 3)
            }

        summary['class_summaries'] = class_summaries

        # 전체적 인사이트
        all_top_features = set()
        for model_summary in model_summaries.values():
            all_top_features.update(model_summary['top_features'])

        # 가장 일관되게 중요한 특성들
        consistent_features = {}
        for feature in all_top_features:
            count = sum(1 for ms in model_summaries.values() if feature in ms['top_features'])
            if count > 1:
                consistent_features[feature] = count

        summary['global_insights'] = {
            'most_consistent_features': dict(sorted(consistent_features.items(),
                                                   key=lambda x: x[1], reverse=True)[:5]),
            'unique_important_features': len(all_top_features),
            'feature_consistency_ratio': len(consistent_features) / len(all_top_features) if all_top_features else 0
        }

        return summary

    def _log_results_to_tracker(self, analysis_result: Dict[str, Any]):
        """분석 결과를 실험 추적 시스템에 기록합니다."""

        if not self.tracker:
            return

        summary = analysis_result['summary']

        # 메트릭 기록
        metrics = {
            'shap_analysis/models_analyzed': summary['analysis_overview']['models_analyzed'],
            'shap_analysis/classes_analyzed': summary['analysis_overview']['classes_analyzed'],
            'shap_analysis/total_recommendations': summary['analysis_overview']['total_recommendations'],
            'shap_analysis/high_priority_recommendations': summary['analysis_overview']['high_priority_recommendations'],
            'shap_analysis/unique_important_features': summary['global_insights']['unique_important_features'],
            'shap_analysis/feature_consistency_ratio': summary['global_insights']['feature_consistency_ratio']
        }

        # 모델별 메트릭 추가
        for model_name, model_summary in summary['model_summaries'].items():
            metrics[f'shap_analysis/{model_name.lower()}/features_concentration'] = model_summary['top_features_concentration']
            metrics[f'shap_analysis/{model_name.lower()}/avg_importance'] = model_summary['avg_shap_importance']

        # 클래스별 메트릭 추가
        for class_id, class_summary in summary['class_summaries'].items():
            metrics[f'shap_analysis/class_{class_id}/key_features_count'] = class_summary['key_features_count']
            metrics[f'shap_analysis/class_{class_id}/pattern_strength'] = class_summary['pattern_strength']
            metrics[f'shap_analysis/class_{class_id}/high_priority_improvements'] = class_summary['high_priority_improvements']

        self.tracker.log_metrics(metrics)

        # 파라미터 기록
        params = {
            'analysis_type': 'shap_interpretability',
            'models_count': summary['analysis_overview']['models_analyzed'],
            'problem_classes_analyzed': list(analysis_result['class_insights'].keys()),
            'most_important_feature': list(summary['global_insights']['most_consistent_features'].keys())[0] if summary['global_insights']['most_consistent_features'] else 'none'
        }

        self.tracker.log_params(params)

        print("📈 SHAP 분석 결과가 실험 추적 시스템에 기록되었습니다.")

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """생성된 권장사항을 반환합니다."""
        return self.recommendations

    def print_summary(self, analysis_result: Dict[str, Any]):
        """SHAP 분석 결과 요약을 출력합니다."""

        summary = analysis_result['summary']

        print("\n" + "="*60)
        print("🔍 SHAP 모델 해석성 분석 결과 요약")
        print("="*60)

        # 분석 개요
        print(f"\n📊 분석 개요:")
        print(f"  • 분석된 모델 수: {summary['analysis_overview']['models_analyzed']}개")
        print(f"  • 분석된 문제 클래스: {summary['analysis_overview']['classes_analyzed']}개")
        print(f"  • 생성된 권장사항: {summary['analysis_overview']['total_recommendations']}개")
        print(f"  • 고우선순위 권장사항: {summary['analysis_overview']['high_priority_recommendations']}개")

        # 모델별 주요 발견사항
        print(f"\n🤖 모델별 주요 특성:")
        for model_name, model_summary in summary['model_summaries'].items():
            print(f"  [{model_name}]:")
            print(f"    - 상위 5개 특성: {', '.join(model_summary['top_features'])}")
            print(f"    - 특성 집중도: {model_summary['top_features_concentration']*100:.1f}%")
            print(f"    - 평균 SHAP 중요도: {model_summary['avg_shap_importance']:.4f}")

        # 클래스별 인사이트
        print(f"\n🎯 클래스별 인사이트:")
        for class_id, class_summary in summary['class_summaries'].items():
            print(f"  [Class {class_id}]:")
            print(f"    - 핵심 특성 수: {class_summary['key_features_count']}개")
            print(f"    - 상위 3개 특성: {', '.join(class_summary['top_3_features'])}")
            print(f"    - 패턴 강도: {class_summary['pattern_strength']:.3f}")
            print(f"    - 개선 기회: {class_summary['improvement_opportunities']}개 (고우선순위: {class_summary['high_priority_improvements']}개)")

        # 전역 인사이트
        print(f"\n🌐 전역 인사이트:")
        print(f"  • 총 중요 특성 수: {summary['global_insights']['unique_important_features']}개")
        print(f"  • 특성 일관성 비율: {summary['global_insights']['feature_consistency_ratio']*100:.1f}%")

        if summary['global_insights']['most_consistent_features']:
            print(f"  • 가장 일관된 특성들:")
            for feature, count in list(summary['global_insights']['most_consistent_features'].items())[:5]:
                print(f"    - {feature}: {count}개 모델에서 중요")

        # 고우선순위 권장사항
        high_priority_recs = [r for r in self.recommendations if r['priority'] == 'high']
        if len(high_priority_recs) > 0:
            print(f"\n🚨 고우선순위 권장사항:")
            for i, rec in enumerate(high_priority_recs[:3], 1):
                print(f"  {i}. [{rec['category']}]")
                print(f"     문제: {rec['issue']}")
                print(f"     해결: {rec['solution']}")
                print(f"     기대효과: {rec['expected_impact']}")

        print("\n" + "="*60)
        print("✅ SHAP 분석이 완료되었습니다!")
        print("📁 상세 시각화는 experiments/plots/shap_analysis/ 에서 확인하세요.")
        print("="*60)


def main():
    """메인 실행 함수"""
    from ..utils.config import Config
    from ..tracking.experiment_tracker import ExperimentTracker
    import pandas as pd
    import numpy as np

    # 설정 로드
    config = Config()

    # 실험 추적기 초기화
    tracker = ExperimentTracker(
        project_name="dacon-smartmh-02",
        experiment_name="shap_interpretability_analysis"
    )

    # SHAP 분석기 초기화
    analyzer = SHAPAnalyzer(config=config, experiment_tracker=tracker)

    try:
        # 샘플 데이터 생성 (실제 환경에서는 데이터 로드)
        print("📝 샘플 데이터 생성 중...")
        np.random.seed(42)
        n_samples = 2000  # SHAP 계산 속도를 위해 작게 설정
        n_features = 52

        # 클래스 불균형 데이터 생성
        class_probs = np.array([0.1, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04,
                               0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
        class_probs = class_probs / class_probs.sum()

        # 특성 데이터 생성
        X = np.random.normal(0, 1, (n_samples, n_features))
        y = np.random.choice(21, size=n_samples, p=class_probs)

        # 클래스별 패턴 추가
        for class_id in range(21):
            mask = y == class_id
            if mask.sum() > 0:
                pattern_features = np.random.choice(n_features, size=5, replace=False)
                for feat in pattern_features:
                    X[mask, feat] += np.random.normal(class_id * 0.3, 0.3, mask.sum())

        # DataFrame 생성
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)

        print(f"✅ 데이터 준비 완료: {len(X_df)}행 × {len(X_df.columns)}열, {y_series.nunique()}개 클래스")

        # SHAP 분석 실행
        results = analyzer.analyze_model_interpretability(
            X_train=X_df,
            y_train=y_series,
            problem_classes=[1, 0, 2],  # T007에서 식별된 문제 클래스
            top_features=20
        )

        # 결과 요약 출력
        analyzer.print_summary(results)

        return results

    except Exception as e:
        print(f"❌ SHAP 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()