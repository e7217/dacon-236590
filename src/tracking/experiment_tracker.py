"""
실험 추적 시스템
T002 태스크의 일부로 MLflow와 WandB를 통합한 실험 추적 환경 구성
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import wandb
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


class ExperimentTracker:
    """
    통합 실험 추적 클래스
    MLflow와 WandB를 동시에 사용하여 모든 실험을 추적
    """

    def __init__(self, project_name: str = "dacon-smartmh-02",
                 experiment_name: str = None,
                 use_wandb: bool = True,
                 use_mlflow: bool = True):
        """
        실험 추적기 초기화

        Args:
            project_name: 프로젝트 이름
            experiment_name: 실험 이름 (자동 생성 시 None)
            use_wandb: WandB 사용 여부
            use_mlflow: MLflow 사용 여부
        """
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 추적 디렉토리 생성
        self.tracking_dir = Path("experiments")
        self.tracking_dir.mkdir(exist_ok=True)

        # MLflow 설정
        if self.use_mlflow:
            self._setup_mlflow()

        # WandB 설정
        if self.use_wandb:
            self._setup_wandb()

    def _setup_mlflow(self):
        """MLflow 환경 설정"""
        # MLflow 추적 디렉토리 설정
        mlflow_dir = self.tracking_dir / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)

        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")

        # 실험 설정 또는 생성
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)
        print(f"✅ MLflow 설정 완료: {self.experiment_name}")

    def _setup_wandb(self):
        """WandB 환경 설정"""
        try:
            # WandB 초기화 (오프라인 모드로 시작)
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                mode="offline",  # 인터넷 연결 없이도 작동
                dir=str(self.tracking_dir)
            )
            print(f"✅ WandB 설정 완료: {self.project_name}/{self.experiment_name}")
        except Exception as e:
            print(f"⚠️ WandB 설정 중 오류 (오프라인 모드 사용): {e}")
            self.use_wandb = False

    def start_run(self, run_name: str, description: str = "", tags: Dict[str, str] = None):
        """
        새로운 실험 실행 시작

        Args:
            run_name: 실행 이름
            description: 실행 설명
            tags: 태그 딕셔너리
        """
        self.current_run_name = run_name

        # MLflow 실행 시작
        if self.use_mlflow:
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
            if description:
                mlflow.set_tag("description", description)

        # WandB 실행 정보 업데이트
        if self.use_wandb and wandb.run:
            wandb.run.name = run_name
            wandb.run.notes = description
            if tags:
                wandb.config.update(tags)

        print(f"🚀 실험 실행 시작: {run_name}")

    def log_params(self, params: Dict[str, Any]):
        """하이퍼파라미터 로그"""
        if self.use_mlflow:
            mlflow.log_params(params)

        if self.use_wandb and wandb.run:
            wandb.config.update(params)

        print(f"📝 파라미터 로그: {len(params)}개 항목")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int = None):
        """메트릭 로그"""
        if self.use_mlflow:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

        if self.use_wandb and wandb.run:
            wandb.log(metrics, step=step)

        print(f"📊 메트릭 로그: {metrics}")

    def log_model(self, model, model_name: str, signature=None, input_example=None):
        """모델 로그"""
        if self.use_mlflow:
            # 모델 타입에 따른 적절한 MLflow 로깅
            if hasattr(model, 'booster'):  # XGBoost
                mlflow.xgboost.log_model(model, model_name, signature=signature, input_example=input_example)
            elif hasattr(model, '_Booster'):  # LightGBM
                mlflow.lightgbm.log_model(model, model_name, signature=signature, input_example=input_example)
            elif hasattr(model, '_get_tags'):  # CatBoost
                mlflow.catboost.log_model(model, model_name, signature=signature, input_example=input_example)
            else:  # Scikit-learn
                mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)

        print(f"🤖 모델 로그 완료: {model_name}")

    def log_artifact(self, filepath: str, artifact_path: str = None):
        """아티팩트 (파일) 로그"""
        if self.use_mlflow:
            mlflow.log_artifact(filepath, artifact_path)

        if self.use_wandb and wandb.run:
            wandb.save(filepath)

        print(f"📁 아티팩트 로그: {filepath}")

    def log_figure(self, figure, name: str):
        """matplotlib/seaborn 그림 로그"""
        # 임시 파일로 저장 후 로그
        temp_path = self.tracking_dir / f"temp_{name}.png"
        figure.savefig(temp_path, dpi=300, bbox_inches='tight')

        if self.use_mlflow:
            mlflow.log_artifact(str(temp_path))

        if self.use_wandb and wandb.run:
            wandb.log({name: wandb.Image(str(temp_path))})

        # 임시 파일 삭제
        temp_path.unlink()
        print(f"📈 그림 로그: {name}")

    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """혼동행렬 로그"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        self.log_figure(plt.gcf(), 'confusion_matrix')
        plt.close()

    def log_feature_importance(self, model, feature_names):
        """특성 중요도 로그"""
        import matplotlib.pyplot as plt

        # 모델 타입에 따른 특성 중요도 추출
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            print("⚠️ 특성 중요도를 추출할 수 없는 모델입니다.")
            return

        # 상위 20개 특성만 시각화
        indices = np.argsort(importance)[::-1][:20]

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Top 20)')
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()

        self.log_figure(plt.gcf(), 'feature_importance')
        plt.close()

        # 특성 중요도 데이터도 로그
        importance_dict = {f"importance_{feature_names[i]}": importance[i] for i in indices}
        self.log_metrics(importance_dict)

    def end_run(self):
        """실험 실행 종료"""
        if self.use_mlflow and hasattr(self, 'mlflow_run'):
            mlflow.end_run()

        print(f"🏁 실험 실행 종료: {self.current_run_name}")

    def finish(self):
        """전체 실험 세션 종료"""
        if self.use_wandb and wandb.run:
            wandb.finish()

        print(f"✅ 실험 추적 세션 종료: {self.experiment_name}")


def get_experiment_summary(tracking_dir: str = "experiments") -> pd.DataFrame:
    """
    실험 결과 요약 조회

    Args:
        tracking_dir: 추적 디렉토리 경로

    Returns:
        실험 결과 요약 DataFrame
    """
    tracking_path = Path(tracking_dir)

    if not tracking_path.exists():
        print("⚠️ 실험 추적 디렉토리가 존재하지 않습니다.")
        return pd.DataFrame()

    # MLflow에서 실험 결과 조회
    mlflow.set_tracking_uri(f"file://{(tracking_path / 'mlruns').absolute()}")

    try:
        experiments = mlflow.search_experiments()
        all_runs = []

        for experiment in experiments:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                all_runs.append(runs)

        if all_runs:
            summary_df = pd.concat(all_runs, ignore_index=True)
            print(f"📋 총 {len(summary_df)}개의 실험 실행 결과를 조회했습니다.")
            return summary_df
        else:
            print("📋 저장된 실험 실행 결과가 없습니다.")
            return pd.DataFrame()

    except Exception as e:
        print(f"⚠️ 실험 결과 조회 중 오류: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    """실험 추적기 테스트"""
    print("🧪 실험 추적기 테스트 시작...")

    # 테스트 추적기 생성
    tracker = ExperimentTracker(
        project_name="dacon-smartmh-02",
        experiment_name="test_tracking"
    )

    # 테스트 실행
    tracker.start_run(
        run_name="test_run",
        description="실험 추적기 기능 테스트",
        tags={"model_type": "test", "version": "1.0"}
    )

    # 테스트 파라미터 및 메트릭 로그
    tracker.log_params({
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100
    })

    tracker.log_metrics({
        "accuracy": 0.85,
        "f1_score": 0.82,
        "precision": 0.87,
        "recall": 0.78
    })

    # 실행 종료
    tracker.end_run()
    tracker.finish()

    print("✅ 실험 추적기 테스트 완료!")
    print("🎯 다음 단계: T003 (성능 진단 디렉토리 구조 생성)")