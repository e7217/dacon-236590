"""
성능 격차 분석기 테스트
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.diagnosis.performance_gap_analyzer import PerformanceGapAnalyzer


class TestPerformanceGapAnalyzer(unittest.TestCase):
    """성능 격차 분석기 단위 테스트"""

    def setUp(self):
        """테스트 설정"""
        # 더미 데이터 생성
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=5,
            n_informative=15,
            random_state=42
        )

        self.X_df = pd.DataFrame(X, columns=[f'feature_{i:02d}' for i in range(20)])
        self.y_df = pd.Series(y)

        # 분석기 생성 (추적 비활성화)
        self.analyzer = PerformanceGapAnalyzer(use_tracking=False)

    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        self.assertIsInstance(self.analyzer, PerformanceGapAnalyzer)
        self.assertFalse(self.analyzer.use_tracking)

    def test_performance_gap_analysis(self):
        """성능 격차 분석 테스트"""
        results = self.analyzer.analyze_performance_gap(
            self.X_df, self.y_df, cv_folds=3
        )

        # 결과 구조 확인
        expected_keys = [
            'cv_analysis', 'learning_curves', 'validation_curves',
            'leakage_check', 'stability_analysis', 'summary'
        ]
        for key in expected_keys:
            self.assertIn(key, results)

        # CV 분석 결과 확인
        cv_results = results['cv_analysis']
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        self.assertIsInstance(cv_results['mean_score'], float)
        self.assertTrue(0 <= cv_results['mean_score'] <= 1)

    def test_recommendations_generation(self):
        """권장사항 생성 테스트"""
        # 먼저 분석 실행
        self.analyzer.analyze_performance_gap(self.X_df, self.y_df, cv_folds=3)

        # 권장사항 생성
        recommendations = self.analyzer.get_recommendations()
        self.assertIsInstance(recommendations, list)

        # 권장사항이 있다면 구조 확인
        if recommendations:
            for rec in recommendations:
                self.assertIn('priority', rec)
                self.assertIn('category', rec)
                self.assertIn('recommendation', rec)

    def tearDown(self):
        """테스트 정리"""
        pass


if __name__ == '__main__':
    unittest.main()