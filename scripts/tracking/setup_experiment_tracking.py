"""
실험 추적 환경 통합 설정 스크립트
T002 태스크 완료를 위한 원스톱 설정
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.tracking.experiment_tracker import ExperimentTracker
from src.utils.config import Config


def setup_experiment_environment():
    """실험 추적 환경 전체 설정"""
    print("🚀 실험 추적 환경 설정 시작...")
    print("=" * 60)

    # 1. 설정 관리 시스템 초기화
    print("\n📋 1단계: 설정 관리 시스템 초기화")
    config = Config()
    config.print_summary()

    # 경로 유효성 검사
    if not config.validate_paths():
        print("❌ 필수 파일들이 누락되어 있습니다. 데이터 파일을 확인해주세요.")
        return False

    # 2. 실험 추적 디렉토리 구조 생성
    print("\n📁 2단계: 실험 추적 디렉토리 구조 생성")
    paths = config.get_paths()

    directories_to_create = [
        paths['experiments_dir'],
        paths['experiments_dir'] / 'models',
        paths['experiments_dir'] / 'plots',
        paths['experiments_dir'] / 'logs',
        paths['data_dir'],
        paths['data_dir'] / 'features',
        paths['data_dir'] / 'processed'
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")

    # 3. 실험 추적기 테스트
    print("\n🧪 3단계: 실험 추적기 기능 테스트")
    tracker = ExperimentTracker(
        project_name=config.get('tracking.project_name'),
        experiment_name="setup_validation_test",
        use_wandb=config.get('tracking.use_wandb'),
        use_mlflow=config.get('tracking.use_mlflow')
    )

    # 테스트 실행
    tracker.start_run(
        run_name="validation_test",
        description="T002 태스크 - 실험 추적 환경 설정 검증",
        tags={"task": "T002", "type": "setup_validation"}
    )

    # 테스트 데이터 로깅
    test_params = {
        "setup_date": "2025-09-25",
        "task_id": "T002",
        "random_seed": config.get('models.random_seed'),
        "cv_folds": config.get('models.cv_folds'),
        "target_score": config.get('targets.target_score')
    }
    tracker.log_params(test_params)

    test_metrics = {
        "current_score": config.get('targets.current_score'),
        "target_score": config.get('targets.target_score'),
        "improvement_needed": config.get('targets.improvement_needed'),
        "setup_success": 1.0
    }
    tracker.log_metrics(test_metrics)

    tracker.end_run()
    tracker.finish()

    # 4. 최종 확인
    print("\n✅ 4단계: 설정 완료 확인")

    success_checks = {
        "MLflow 디렉토리": (paths['experiments_dir'] / 'mlruns').exists(),
        "WandB 디렉토리": (paths['experiments_dir'] / 'wandb').exists(),
        "설정 파일": (paths['config_dir'] / 'experiment_config.yaml').exists(),
        "실험 추적기": True,  # 위에서 성공적으로 실행됨
        "모델 저장 디렉토리": (paths['experiments_dir'] / 'models').exists(),
        "플롯 저장 디렉토리": (paths['experiments_dir'] / 'plots').exists()
    }

    all_success = True
    for check_name, status in success_checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {check_name}")
        if not status:
            all_success = False

    print("\n" + "=" * 60)
    if all_success:
        print("🎉 T002 태스크 완료: 실험 추적 환경 설정 성공!")
        print("\n📋 설정된 기능들:")
        print("  • MLflow: 모델 및 메트릭 추적")
        print("  • WandB: 실험 시각화 및 비교 (오프라인 모드)")
        print("  • 통합 설정 관리: YAML 기반 중앙집중식 설정")
        print("  • 자동 실험 이름 생성")
        print("  • 모델 자동 저장 및 로드")
        print("  • 그래프 및 메트릭 시각화")
        print("\n🎯 다음 단계: T003 (성능 진단 디렉토리 구조 생성)")
        return True
    else:
        print("❌ T002 태스크 실패: 일부 설정이 완료되지 않았습니다.")
        return False


if __name__ == "__main__":
    success = setup_experiment_environment()
    sys.exit(0 if success else 1)