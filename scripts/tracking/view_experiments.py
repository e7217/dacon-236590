"""
실험 결과 조회 스크립트
MLflow와 WandB 실험 결과를 조회하는 유틸리티
"""

import mlflow
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config import get_config


def view_mlflow_experiments():
    """MLflow 실험 결과 조회"""
    print("🔍 MLflow 실험 결과 조회")
    print("=" * 50)

    # MLflow 추적 URI 설정
    tracking_dir = project_root / "experiments" / "mlruns"
    mlflow.set_tracking_uri(f"file://{tracking_dir.absolute()}")

    try:
        # 모든 실험 조회
        experiments = mlflow.search_experiments()

        if not experiments:
            print("📝 저장된 실험이 없습니다.")
            return

        print(f"📊 총 {len(experiments)}개의 실험 발견")

        for exp in experiments:
            print(f"\n🧪 실험: {exp.name}")
            print(f"   ID: {exp.experiment_id}")

            # 해당 실험의 런들 조회
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

            if runs.empty:
                print("   📝 실행 기록이 없습니다.")
                continue

            print(f"   🚀 총 {len(runs)}개의 실행")

            # 주요 메트릭 출력
            for idx, run in runs.iterrows():
                print(f"\n   🏃 Run: {run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else 'Unnamed'}")
                print(f"      Status: {run['status']}")
                print(f"      Start Time: {run['start_time']}")

                # 메트릭 출력
                metric_cols = [col for col in run.index if col.startswith('metrics.')]
                if metric_cols:
                    print("      📊 Metrics:")
                    for metric_col in metric_cols:
                        metric_name = metric_col.replace('metrics.', '')
                        value = run[metric_col]
                        if pd.notna(value):
                            print(f"         {metric_name}: {value:.4f}")

                # 파라미터 출력
                param_cols = [col for col in run.index if col.startswith('params.')]
                if param_cols:
                    print("      🎛️ Parameters:")
                    for param_col in param_cols[:5]:  # 처음 5개만 표시
                        param_name = param_col.replace('params.', '')
                        value = run[param_col]
                        if pd.notna(value):
                            print(f"         {param_name}: {value}")

    except Exception as e:
        print(f"❌ MLflow 조회 오류: {e}")


def view_wandb_experiments():
    """WandB 실험 결과 조회"""
    print("\n🎨 WandB 실험 결과 조회")
    print("=" * 50)

    wandb_dir = project_root / "experiments" / "wandb"

    if not wandb_dir.exists():
        print("📝 WandB 실험 디렉토리가 없습니다.")
        return

    # 오프라인 런 디렉토리들 조회
    offline_runs = list(wandb_dir.glob("offline-run-*"))

    if not offline_runs:
        print("📝 저장된 WandB 실행이 없습니다.")
        return

    print(f"📊 총 {len(offline_runs)}개의 오프라인 실행 발견")

    for run_dir in offline_runs:
        print(f"\n🚀 Run: {run_dir.name}")

        # 설정 파일 조회
        config_file = run_dir / "files" / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print("   🎛️ Configuration:")
                for key, value in list(config.items())[:5]:  # 처음 5개만 표시
                    print(f"      {key}: {value}")
            except:
                pass

        # 요약 파일 조회
        summary_file = run_dir / "files" / "wandb-summary.json"
        if summary_file.exists():
            try:
                import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print("   📊 Metrics:")
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")
            except:
                pass


def show_sync_commands():
    """WandB 동기화 명령어 안내"""
    print("\n🔄 WandB 온라인 동기화 방법")
    print("=" * 50)

    wandb_dir = project_root / "experiments" / "wandb"
    offline_runs = list(wandb_dir.glob("offline-run-*"))

    if offline_runs:
        print("다음 명령어로 WandB 클라우드에 동기화할 수 있습니다:")
        print("\n# 모든 오프라인 실행을 동기화")
        for run_dir in offline_runs:
            print(f"wandb sync {run_dir}")

        print(f"\n# 또는 WandB 온라인 모드로 전환")
        print("wandb online")
    else:
        print("동기화할 오프라인 실행이 없습니다.")


if __name__ == "__main__":
    print("🔍 실험 결과 조회 도구")
    print("=" * 60)

    # MLflow 실험 조회
    view_mlflow_experiments()

    # WandB 실험 조회
    view_wandb_experiments()

    # 동기화 방법 안내
    show_sync_commands()

    print("\n" + "=" * 60)
    print("💡 추가 정보:")
    print("• MLflow UI: `uv run mlflow ui --backend-store-uri file://experiments/mlruns`")
    print("• 브라우저에서 http://localhost:5000 접속")
    print("• WandB 온라인 동기화: 위의 wandb sync 명령어 사용")
    print("=" * 60)