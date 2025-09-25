# 🎯 마스터 플랜: Macro F1-Score 0.90+ 달성 전략

**작성일**: 2025-09-25
**현재 성능**: 0.67596 (67.6%)
**목표 성능**: 0.90+ (90%+)
**성능 격차**: +22.4% 개선 필요

## 📊 상황 분석

### 현재 달성 상황
- ✅ **EDA 완료**: 52개 특성, 21개 클래스, 완벽한 클래스 균형
- ✅ **전처리 파이프라인**: 3가지 전략 (Basic, Aggressive, Feature Selected)
- ✅ **베이스라인 모델**: Random Forest ~77% CV 정확도
- ✅ **기본 앙상블**: 다수결 투표 방식
- ❌ **성능 병목**: 실제 제출 점수가 CV 점수보다 현저히 낮음 (과적합 의심)

### 핵심 문제점 진단
1. **과적합 문제**: CV 77% vs 실제 제출 67.6%
2. **모델 복잡도 부족**: 단순한 tree 기반 모델만 사용
3. **특성 엔지니어링 한계**: 기본적인 상관관계 제거만 수행
4. **앙상블 전략 미흡**: 단순 다수결 투표
5. **검증 전략 부재**: 적절한 모델 선택 기준 부재

## 🚀 5단계 성능 향상 전략

### Phase 1: 📊 데이터 기반 진단 및 분석 (1-2일)
**헌법 원칙**: 데이터 기반 분석 + 문서화 우선

**목표 성능**: 0.70+ (현재 모델 최적화)

#### 1.1 성능 격차 근본 원인 분석
- [ ] 교차검증 vs 실제 제출 점수 차이 원인 분석
- [ ] 학습/검증 데이터 분포 비교 (Adversarial Validation)
- [ ] 과적합/과소적합 진단 (learning curve 분석)
- [ ] 클래스별 개별 F1-score 분석 (21개 클래스)

#### 1.2 모델 해석성 강화
- [ ] SHAP 분석으로 특성 중요도 재평가
- [ ] Permutation Importance 계산
- [ ] 특성 상호작용 분석
- [ ] 오분류 패턴 심화 분석 (confusion matrix 분석)

#### 1.3 검증 전략 개선
- [ ] 시간 기반 분할 교차검증 적용
- [ ] Stratified K-Fold 재검토
- [ ] Multiple random seed 실험으로 안정성 확인

**결과물**: `answers/chapter_01_diagnosis.md`

---

### Phase 2: 🔧 고급 특성 엔지니어링 (2-3일)
**헌법 원칙**: 경쟁 우수성 + 재현 가능한 연구

**목표 성능**: 0.75+ (특성 개선으로 +5% 향상)

#### 2.1 도메인 기반 특성 생성
- [ ] **센서 그룹별 통계 특성**
  - 온도 관련 센서 그룹 (X_01, X_05, X_12 추정)
  - 압력 관련 센서 그룹 (X_11, X_19, X_37 추정)
  - 진동 관련 센서 그룹 (X_40, X_46 추정)
  - 각 그룹별 mean, std, skew, kurtosis 계산

- [ ] **센서 간 관계 특성**
  - 주요 센서 간 비율 특성 (X_40/X_11, X_46/X_40)
  - 센서 간 차이 특성 (X_40-X_11, X_46-X_40)
  - 센서 조합 특성 (X_40*X_11, X_46+X_40)

- [ ] **통계적 특성**
  - 센서 값들의 중앙값, 최대값, 최소값
  - Quantile 기반 특성 (25%, 75% percentile)
  - Z-score 정규화 후 극값 특성

#### 2.2 고급 차원 축소 및 특성 선택
- [ ] **차원 축소 특성**
  - PCA 주성분 (상위 10개) + 원본 특성 조합
  - UMAP 2D/3D 임베딩 특성
  - t-SNE 특성 (계산 비용 고려)

- [ ] **특성 선택 최적화**
  - Recursive Feature Elimination with Cross-Validation
  - SelectKBest with mutual_info_classif
  - Lasso 정규화 기반 특성 선택
  - Tree 기반 특성 중요도 앙상블

**결과물**: `answers/chapter_02_features.md`

---

### Phase 3: 🤖 최신 모델 적용 (3-4일)
**헌법 원칙**: 경쟁 우수성 + 지속적 학습

**목표 성능**: 0.85+ (고급 모델로 +10% 향상)

#### 3.1 Gradient Boosting 계열 모델
- [ ] **XGBoost**
  - Optuna 기반 하이퍼파라미터 최적화
  - early_stopping_rounds 적용
  - class_weight 조정으로 클래스 균형 최적화

- [ ] **LightGBM**
  - dart boosting type 실험
  - categorical_feature 활용
  - feature_fraction, bagging_fraction 튜닝

- [ ] **CatBoost**
  - 범주형 특성 자동 처리
  - robust 손실 함수 실험
  - bootstrap_type 최적화

#### 3.2 딥러닝 모델
- [ ] **TabNet**
  - attention mechanism 활용
  - feature selection mask 분석
  - 다단계 decision step 최적화

- [ ] **Neural Network**
  - Batch Normalization + Dropout
  - ResNet 구조 적용 (skip connection)
  - Label Smoothing 적용

#### 3.3 기타 고급 모델
- [ ] **Wide & Deep Learning**
  - Wide: 선형 특성 조합
  - Deep: 비선형 특성 학습

- [ ] **AutoML 접근법**
  - AutoGluon Tabular 실험
  - FLAML 활용한 자동 모델 선택

**결과물**: `answers/chapter_03_models.md`

---

### Phase 4: 🎯 앙상블 및 최적화 (2-3일)
**헌법 원칙**: 데이터 기반 분석 + 경쟁 우수성

**목표 성능**: 0.90+ (앙상블로 +5% 최종 향상)

#### 4.1 고급 앙상블 전략
- [ ] **Stacking Ensemble**
  - Level 1: XGBoost, LightGBM, CatBoost, TabNet, RF
  - Level 2: 로지스틱 회귀, Linear SVM, Ridge 메타 모델
  - Out-of-fold 예측으로 과적합 방지

- [ ] **Blending**
  - 최적 가중치 탐색 (scipy.optimize)
  - 베이지안 최적화 기반 가중치 조합
  - Dynamic weighted 평균

- [ ] **Voting Ensemble**
  - Soft voting with probability averaging
  - Hard voting 비교 실험
  - 모델별 신뢰도 기반 가중치

#### 4.2 하이퍼파라미터 대규모 최적화
- [ ] **개별 모델 최적화**
  - Optuna/Hyperopt 베이지안 최적화
  - TPE (Tree Parzen Estimator) 활용
  - Multi-objective 최적화 (성능 + 속도)

- [ ] **앙상블 최적화**
  - 모델 조합 최적화
  - 앙상블 가중치 최적화
  - Stacking meta-model 하이퍼파라미터 튜닝

#### 4.3 앙상블 다양성 증대
- [ ] **모델 다양성 분석**
  - 모델 간 상관관계 분석
  - Diversity 측정 지표 계산
  - 낮은 상관관계 모델 조합 우선선택

- [ ] **Bootstrap Aggregation**
  - 서로 다른 시드로 여러 모델 학습
  - Feature subsampling 다양화
  - Sample subsampling 전략

**결과물**: `answers/chapter_04_ensemble.md`

---

### Phase 5: ✅ 검증 및 최종 제출 (1-2일)
**헌법 원칙**: 재현 가능한 연구 + 문서화 우선

**목표 성능**: 0.90+ (안정성 확보 및 검증)

#### 5.1 강건성 검증
- [ ] **Multiple Random Seed 실험**
  - 10개 다른 시드로 모델 학습
  - 성능 분산 분석 및 안정성 평가
  - 평균 성능 vs 최고 성능 분석

- [ ] **교차검증 다양화**
  - 시간 기반 분할 (TimeSeriesSplit)
  - Group K-Fold (센서 그룹 기반)
  - Repeated Stratified K-Fold

- [ ] **Adversarial Validation 심화**
  - Train vs Test 분포 차이 재확인
  - Domain adaptation 필요성 검토
  - Pseudo-labeling 활용 가능성

#### 5.2 최종 모델 선택 및 검증
- [ ] **모델 선택 기준 수립**
  - CV 점수 vs 리더보드 점수 상관관계 분석
  - 안정성 지표 (std, min, max) 고려
  - 복잡도 vs 성능 trade-off 분석

- [ ] **리더보드 과적합 방지**
  - Hold-out validation set 별도 구성
  - Early stopping 기준 재조정
  - Ensemble pruning (성능 기여도 낮은 모델 제거)

#### 5.3 최종 제출 전략
- [ ] **다중 제출 전략**
  - 여러 앙상블 조합으로 다양한 제출파일 생성
  - Conservative vs Aggressive 앙상블 비교
  - 안전한 제출 vs 도전적 제출

- [ ] **코드 최종 정리**
  - 재현성 확보 (random seed, 환경 설정)
  - 코드 주석 및 문서화 완료
  - 실험 로그 정리 및 백업

**결과물**: `answers/chapter_05_results.md`

---

## 📈 단계별 성능 목표 및 일정

| Phase | 기간 | 목표 성능 | 핵심 전략 | 예상 향상폭 | 위험 요소 |
|-------|------|-----------|----------|------------|-----------|
| Phase 1 | 1-2일 | 0.70+ | 현재 모델 진단 & 최적화 | +2.4% | 과적합 원인 미해결 |
| Phase 2 | 2-3일 | 0.75+ | 특성 엔지니어링 혁신 | +5% | 특성 폭발, 차원의 저주 |
| Phase 3 | 3-4일 | 0.85+ | 최신 모델 도입 | +10% | 모델 복잡도, 과적합 |
| Phase 4 | 2-3일 | 0.90+ | 고급 앙상블 최적화 | +5% | 계산 비용, 앙상블 과적합 |
| Phase 5 | 1-2일 | 0.90+ | 검증 및 안정화 | 안정성 확보 | 시간 부족, 리더보드 변동 |

**총 소요 기간**: 9-14일
**최종 목표**: Macro F1-Score 0.90+ 달성

---

## 🔧 기술 스택 및 환경 설정

### 필수 라이브러리 업그레이드
```bash
# 고급 ML 라이브러리
pip install xgboost lightgbm catboost
pip install optuna hyperopt scikit-optimize
pip install shap eli5 lime
pip install pytorch-tabnet
pip install umap-learn

# 실험 관리
pip install mlflow wandb
pip install joblib pickle

# 시각화
pip install plotly seaborn
pip install yellowbrick

# AutoML
pip install autogluon flaml
```

### 하드웨어 요구사항
- **CPU**: 멀티코어 (병렬 학습용)
- **GPU**: CUDA 지원 (TabNet, Neural Network용)
- **메모리**: 16GB+ (대용량 앙상블용)
- **저장공간**: 10GB+ (모델 저장, 실험 로그용)

---

## 📚 학습 및 문서화 계획

### 챕터별 문서 구조
각 Phase 완료 후 해당 챕터를 `/answers` 폴더에 생성:

1. **Chapter 1**: `chapter_01_diagnosis.md` - 성능 진단 및 분석
2. **Chapter 2**: `chapter_02_features.md` - 특성 엔지니어링
3. **Chapter 3**: `chapter_03_models.md` - 고급 모델링
4. **Chapter 4**: `chapter_04_ensemble.md` - 앙상블 전략
5. **Chapter 5**: `chapter_05_results.md` - 최종 검증 및 결과

### 문서화 원칙 (헌법 준수)
- **실증 기반**: 모든 결정을 실험 결과로 뒷받침
- **과정 기록**: 실패한 실험도 포함하여 학습 과정 기록
- **재현성**: 코드와 파라미터를 정확히 기록
- **해석성**: 왜 그런 결정을 했는지 명확한 근거 제시

---

## ⚠️ 위험 요소 및 대응 방안

### 주요 위험 요소
1. **과적합 심화**: 복잡한 모델 도입 시 과적합 위험 증가
2. **계산 비용**: 대규모 하이퍼파라미터 최적화의 시간 비용
3. **특성 폭발**: 특성 엔지니어링으로 인한 차원 증가
4. **앙상블 복잡도**: 너무 복잡한 앙상블로 인한 불안정성
5. **리더보드 과적합**: Public LB 점수에 과도한 최적화

### 대응 전략
1. **점진적 접근**: 각 Phase별로 검증하며 단계적 개선
2. **조기 종료**: 성능 향상이 없으면 빠른 전략 전환
3. **다양한 검증**: 여러 CV 전략으로 robust한 평가
4. **백업 계획**: 각 단계별 최고 성능 모델 저장
5. **시간 관리**: 각 Phase별 데드라인 엄격 준수

---

## 🎯 성공 지표 및 중간 점검

### 정량적 지표
- **주요 지표**: Macro F1-Score 0.90+ 달성
- **중간 목표**: 각 Phase별 목표 성능 달성
- **안정성**: 10회 시드 실험에서 표준편차 < 0.01
- **일관성**: CV 점수와 리더보드 점수 차이 < 0.05

### 정성적 지표
- **학습 효과**: 각 기법에 대한 깊은 이해 습득
- **재현성**: 타인이 동일한 결과를 재현할 수 있는 수준
- **문서화 품질**: 전 과정이 명확히 기록되고 설명됨
- **코드 품질**: 읽기 쉽고 모듈화된 코드 구조

---

**작성자**: Claude Code Assistant
**검토자**: 사용자
**승인일**: 2025-09-25
**다음 리뷰**: Phase 1 완료 후

---

*"성능 향상은 마라톤이지 단거리가 아니다. 체계적이고 점진적인 접근으로 목표를 달성하자!"* 🚀