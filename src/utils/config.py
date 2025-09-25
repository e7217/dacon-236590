"""
설정 관리 유틸리티
T002 태스크의 일부로 실험 설정을 중앙집중식으로 관리
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class Config:
    """
    프로젝트 설정 관리 클래스
    """

    def __init__(self, config_path: str = None):
        """
        설정 초기화

        Args:
            config_path: 설정 파일 경로 (기본값: config/experiment_config.yaml)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = config_path or self.project_root / "config" / "experiment_config.yaml"

        # 기본 설정
        self.default_config = {
            # 실험 추적 설정
            "tracking": {
                "use_mlflow": True,
                "use_wandb": True,
                "project_name": "dacon-smartmh-02",
                "experiment_dir": "experiments",
                "save_models": True,
                "save_plots": True
            },

            # 데이터 경로 설정
            "data": {
                "train_path": "data/open/train.csv",
                "test_path": "data/open/test.csv",
                "sample_submission_path": "data/open/sample_submission.csv",
                "processed_dir": "data/processed",
                "features_dir": "data/features"
            },

            # 모델 설정
            "models": {
                "random_seed": 42,
                "cv_folds": 5,
                "test_size": 0.2,
                "stratify": True,
                "shuffle": True
            },

            # 성능 목표
            "targets": {
                "current_score": 0.67596,
                "target_score": 0.90,
                "improvement_needed": 0.22404,
                "metric": "macro_f1"
            },

            # 하이퍼파라미터 최적화 설정
            "optimization": {
                "n_trials": 100,
                "timeout": 3600,  # 1시간
                "n_jobs": -1,
                "random_state": 42
            },

            # 출력 및 로깅 설정
            "output": {
                "verbose": True,
                "save_intermediate": True,
                "plot_results": True,
                "log_level": "INFO"
            }
        }

        # 설정 로드
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """설정 파일 로드 또는 기본 설정 생성"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ 설정 파일 로드됨: {self.config_path}")
                return self._merge_config(self.default_config, config)
            except Exception as e:
                print(f"⚠️ 설정 파일 로드 오류: {e}")
                print("기본 설정을 사용합니다.")
        else:
            print("설정 파일이 없어 기본 설정을 사용합니다.")

        # 기본 설정 파일 생성
        self.save_config(self.default_config)
        return self.default_config.copy()

    def save_config(self, config: Dict[str, Any] = None):
        """설정을 파일로 저장"""
        config = config or self.config

        # 디렉토리 생성
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"✅ 설정 파일 저장됨: {self.config_path}")
        except Exception as e:
            print(f"❌ 설정 파일 저장 오류: {e}")

    def _merge_config(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """기본 설정과 사용자 설정 병합"""
        result = default.copy()
        for key, value in custom.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점 표기법으로 설정 값 조회

        Args:
            key_path: 'section.key' 형식의 경로
            default: 기본값

        Returns:
            설정 값
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        점 표기법으로 설정 값 변경

        Args:
            key_path: 'section.key' 형식의 경로
            value: 설정할 값
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        print(f"🔧 설정 변경: {key_path} = {value}")

    def get_paths(self) -> Dict[str, Path]:
        """프로젝트 주요 경로들 반환"""
        return {
            'project_root': self.project_root,
            'data': self.project_root / self.get('data.processed_dir', 'data'),
            'train_path': self.project_root / self.get('data.train_path'),
            'test_path': self.project_root / self.get('data.test_path'),
            'experiments_dir': self.project_root / self.get('tracking.experiment_dir', 'experiments'),
            'config_dir': self.config_path.parent,
            'answers_dir': self.project_root / 'answers',
            'src_dir': self.project_root / 'src',
            'plots': self.project_root / self.get('tracking.experiment_dir', 'experiments') / 'plots'
        }

    def create_experiment_name(self, prefix: str = "exp") -> str:
        """실험 이름 자동 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def validate_paths(self) -> bool:
        """중요 경로들 존재 여부 확인"""
        paths = self.get_paths()
        missing_paths = []

        # 필수 파일 확인
        required_files = ['train_path', 'test_path']
        for file_key in required_files:
            if not paths[file_key].exists():
                missing_paths.append(f"{file_key}: {paths[file_key]}")

        if missing_paths:
            print("❌ 누락된 필수 파일들:")
            for path in missing_paths:
                print(f"  - {path}")
            return False
        else:
            print("✅ 모든 필수 파일이 존재합니다.")
            return True

    def print_summary(self):
        """현재 설정 요약 출력"""
        print("\n" + "="*50)
        print("📋 현재 실험 설정 요약")
        print("="*50)

        paths = self.get_paths()
        print(f"📁 프로젝트 루트: {paths['project_root']}")
        print(f"📊 학습 데이터: {paths['train_path']}")
        print(f"🧪 실험 디렉토리: {paths['experiments_dir']}")

        print(f"\n🎯 성능 목표:")
        print(f"  - 현재 점수: {self.get('targets.current_score')}")
        print(f"  - 목표 점수: {self.get('targets.target_score')}")
        print(f"  - 개선 필요: +{self.get('targets.improvement_needed'):.3f}")

        print(f"\n🔧 모델 설정:")
        print(f"  - 랜덤 시드: {self.get('models.random_seed')}")
        print(f"  - CV 폴드: {self.get('models.cv_folds')}")
        print(f"  - 테스트 비율: {self.get('models.test_size')}")

        print(f"\n⚡ 최적화 설정:")
        print(f"  - 시행 횟수: {self.get('optimization.n_trials')}")
        print(f"  - 제한 시간: {self.get('optimization.timeout')}초")

        print("="*50)


# 전역 설정 인스턴스
config = Config()

# 편의 함수들
def get_config(key_path: str, default: Any = None) -> Any:
    """설정 값 조회 편의 함수"""
    return config.get(key_path, default)

def set_config(key_path: str, value: Any):
    """설정 값 변경 편의 함수"""
    config.set(key_path, value)

def get_paths() -> Dict[str, Path]:
    """경로 정보 조회 편의 함수"""
    return config.get_paths()


if __name__ == "__main__":
    """설정 관리 테스트"""
    print("🧪 설정 관리 테스트 시작...")

    # 설정 요약 출력
    config.print_summary()

    # 경로 검증
    config.validate_paths()

    # 실험 이름 생성 테스트
    exp_name = config.create_experiment_name("test")
    print(f"\n🏷️ 생성된 실험 이름: {exp_name}")

    print("\n✅ 설정 관리 테스트 완료!")