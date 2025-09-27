# 📊 Chapter 0: 현재 코드베이스 분석 및 이해

**작성일**: 2025-09-25
**목적**: 기존 완료된 작업들에 대한 체계적 분석 및 이해를 통한 학습
**범위**: EDA, 전처리, 베이스라인 모델링 단계

---

## 🎯 프로젝트 개요 재확인

### 문제 정의
- **과제명**: Dacon 스마트 제조 장비 이상 감지 AI 경진대회
- **문제 유형**: 21개 클래스 다중 분류 (Multi-class Classification)
- **평가 지표**: Accuracy (하지만 목표는 Macro F1-score 0.90+)
- **데이터**: 52개 센서 특성 (X_01 ~ X_52), 블랙박스 환경

### 데이터셋 특성
- **Training Set**: 21,693개 샘플 (각 클래스 1,033개씩 완벽한 균형)
- **Test Set**: 15,004개 샘플
- **특징**: 결측값 없음, 메모리 효율적 (~17MB)

---

## 📈 1단계: 탐색적 데이터 분석 (EDA) 분석

### 1.1 기본 EDA (`eda.py`) 분석

**구현된 기능들**:

#### `load_data()` 함수
```python
def load_data():
    train = pd.read_csv('data/open/train.csv')
    test = pd.read_csv('data/open/test.csv')
    sample_sub = pd.read_csv('data/open/sample_submission.csv')
    return train, test, sample_sub
```
- **목적**: 표준화된 데이터 로딩
- **장점**: 일관된 데이터 접근 방식
- **학습**: 함수형 접근으로 재사용성 확보

#### `basic_info()` 함수
```python
def basic_info(df, name):
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("Data Types:")
    print(df.dtypes.value_counts())

    if 'target' in df.columns:
        print("Target distribution:")
        print(df['target'].value_counts().sort_index())
        print(f"Class balance ratio: {df['target'].value_counts().min() / df['target'].value_counts().max():.3f}")
```

**발견된 핵심 통찰**:
- ✅ **완벽한 클래스 균형**: 21개 클래스 각각 정확히 1,033개 샘플
- ✅ **메모리 효율성**: 전체 데이터셋 약 17MB
- ✅ **깨끗한 데이터**: 결측값 없음

#### `analyze_features()` 함수
특성 통계 요약을 통해 발견한 패턴:
```python
Min values range: [-0.235, 0.000]
Max values range: [0.037, 100.241]
Mean values range: [0.018, 50.956]
```

**학습 포인트**:
- 특성들의 스케일이 크게 다름 → 스케일링 필수
- 일부 특성은 0-1 범위, 일부는 더 큰 범위 → 도메인 특성 차이 시사

#### `analyze_feature_patterns()` 함수
특성을 범위별로 분류:
- **정규화된 특성** (47개): 0-1 사이 값들 → 대부분의 센서 데이터
- **다른 범위 특성** (5개): X_11, X_19, X_37, X_40 등 → 특별한 의미 가능성

**중요한 인사이트**:
- X_40, X_11 등은 다른 스케일 → 중요 센서일 가능성
- 범위 패턴으로 센서 그룹화 가능

### 1.2 고급 분석 (`advanced_analysis.py`) 분석

#### 특성 중요도 분석
```python
def analyze_feature_importance(df):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_cols], df['target'])

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
```

**발견된 중요 특성들**:
1. **X_40**: 8.6% - 가장 중요한 특성
2. **X_11**: 5.6%
3. **X_46**: 5.5%
4. **X_36**: 5.3%
5. **X_34**: 3.9%

**학습 포인트**:
- Random Forest의 feature importance는 Gini 불순도 기반
- 상위 5개 특성이 전체 중요도의 약 31% 차지
- 특성 간 중요도 차이가 명확 → 특성 선택 효과 기대

#### 상관관계 분석
```python
def correlation_analysis(df):
    corr_matrix = df[feature_cols].corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:  # 높은 상관관계 감지
                high_corr_pairs.append(...)
```

**발견된 다중공선성 문제**:
- **47개 고상관 쌍** 발견 (상관계수 > 0.8)
- 주요 예시:
  - X_06 ↔ X_45: 1.000 (완전 상관)
  - X_04 ↔ X_39: 0.991
  - X_05 ↔ X_25: 0.998

**중요한 인사이트**:
- 다중공선성이 심각 → 차원 축소 필요
- 완전 상관(1.000) 특성들은 중복 제거 가능
- 정보 손실 없이 특성 수를 크게 줄일 수 있음

#### 차원 축소 분석
```python
def dimensionality_analysis(df):
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)

    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
```

**PCA 분석 결과**:
- **95% 분산을 18개 주성분으로 설명** 가능
- 52개 → 18개로 65% 차원 축소 가능
- t-SNE로 클래스 간 분리도 양호 확인

**학습 포인트**:
- PCA는 선형 변환으로 해석성 일부 손실
- 하지만 차원 축소 효과는 매우 큼
- 원본 특성 + PCA 특성 조합 전략 고려 가능

---

## 🔧 2단계: 데이터 전처리 파이프라인 분석

### 2.1 `DataPreprocessor` 클래스 구조 분석

#### 클래스 설계 철학
```python
class DataPreprocessor:
    def __init__(self, remove_corr_threshold=0.95, variance_threshold=0.01, n_features=None):
        self.scaler = RobustScaler()  # 아웃라이어 대응
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        self.corr_features_to_remove = []
        self.feature_selector = None
```

**설계 우수성**:
- ✅ **파라미터화**: threshold 값들을 조정 가능
- ✅ **RobustScaler 선택**: 아웃라이어에 강건한 스케일링
- ✅ **모듈화**: 각 전처리 단계가 분리되어 있음
- ✅ **상태 저장**: fit 후 transform 가능한 scikit-learn 스타일

#### 전처리 파이프라인 순서
```python
def fit(self, X, y=None):
    # 1. 스케일링 (RobustScaler)
    X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

    # 2. 낮은 분산 특성 제거 (VarianceThreshold)
    X_variance = pd.DataFrame(
        self.variance_selector.fit_transform(X_scaled),
        columns=X_scaled.columns[self.variance_selector.get_support()]
    )

    # 3. 높은 상관관계 특성 제거 (Custom Implementation)
    self.corr_features_to_remove = self.remove_high_correlation(X_variance, self.remove_corr_threshold)
    X_corr = X_variance.drop(columns=self.corr_features_to_remove)

    # 4. 특성 선택 (SelectKBest with f_classif)
    if self.n_features and y is not None:
        self.feature_selector = SelectKBest(f_classif, k=self.n_features)
        X_selected = self.feature_selector.fit_transform(X_corr, y)
```

**처리 순서의 합리성 분석**:
1. **스케일링 우선**: 상관계수 계산 전에 스케일 정규화
2. **분산 필터링**: 정보가 없는 특성 조기 제거
3. **상관관계 제거**: 중복 정보 제거로 차원 축소
4. **특성 선택**: 목표 변수와의 관련성 기반 최종 선택

**학습된 베스트 프랙티스**:
- 전처리 순서가 결과에 큰 영향을 미침
- DataFrame 형태 유지로 특성 이름 추적
- fit/transform 패턴으로 data leakage 방지

### 2.2 3가지 전처리 전략 분석

#### 전략별 특징
```python
def create_preprocessors():
    preprocessors = {
        'basic': DataPreprocessor(
            remove_corr_threshold=0.95,    # 보수적 상관관계 제거
            variance_threshold=0.01
        ),
        'aggressive': DataPreprocessor(
            remove_corr_threshold=0.90,    # 적극적 상관관계 제거
            variance_threshold=0.02
        ),
        'feature_selected': DataPreprocessor(
            remove_corr_threshold=0.95,
            variance_threshold=0.01,
            n_features=30                   # 최종 30개 특성으로 제한
        )
    }
```

**전략별 기대 효과**:

| 전략 | 특성 수 변화 | 장점 | 단점 |
|------|-------------|------|------|
| **Basic** | 52→40 | 정보 보존, 안정성 | 다중공선성 잔존 |
| **Aggressive** | 52→38 | 다중공선성 적극 제거 | 정보 손실 위험 |
| **Feature Selected** | 52→30 | 최적 특성만 선택 | 과적합 위험 |

**학습 포인트**:
- 다양한 전처리 전략을 실험적으로 비교
- 하이퍼파라미터처럼 전처리도 튜닝 대상
- 도메인 특성에 따라 최적 전략이 달라짐

---

## 🤖 3단계: 베이스라인 모델링 분석

### 3.1 모델 선택 전략 분석

#### 포함된 모델들과 목적
```python
def create_baseline_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
```

**모델 선택 철학 분석**:

1. **Tree-based 모델들** (RF, ET, GB):
   - 특성 선택 자동화
   - 비선형 관계 포착
   - 다중 클래스 분류에 강함

2. **Linear 모델들** (LR, SVM):
   - 해석 가능성 높음
   - 고차원에서도 안정적
   - 베이스라인 성능 확인용

3. **Instance-based** (KNN):
   - 지역적 패턴 포착
   - 단순하지만 효과적

4. **Probabilistic** (Naive Bayes):
   - 빠른 학습/예측
   - 특성 독립성 가정

5. **Neural Network**:
   - 복잡한 패턴 학습
   - 딥러닝 전 단계

### 3.2 평가 시스템 분석

#### 평가 메트릭과 과적합 감지
```python
def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    # 예측
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 정확도 계산
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Overfitting: {train_acc - val_acc:.4f}")
```

**평가 시스템의 우수성**:
- ✅ **과적합 지표**: train_acc - val_acc 차이로 정량화
- ✅ **일관된 평가**: 모든 모델을 동일한 방식으로 평가
- ✅ **결과 저장**: 재현성을 위한 예측값 보관

**개선 여지**:
- ❌ **단일 지표**: Accuracy만 사용 (Macro F1-score 필요)
- ❌ **단일 분할**: 하나의 train/val split만 사용
- ❌ **통계적 검정 부재**: 성능 차이의 유의성 검정 없음

### 3.3 성능 결과 분석

#### PROJECT_SUMMARY.md에서 확인된 성능
- **현재 최고 성능**: 76.96% (Random Forest 5-Fold CV)
- **실제 제출 점수**: 67.596% (약 9.4% 성능 격차)

**성능 격차 원인 추정**:
1. **Cross-Validation vs Single Split**: CV가 더 낙관적
2. **과적합**: 모델이 훈련 데이터에 과도하게 적응
3. **리더보드 과적합**: Public test set에 특화
4. **전처리 불일치**: Train/Test 분포 차이

### 3.4 제출 시스템 분석

#### `create_submission()` 함수
```python
def create_submission(best_model, preprocessor_name='basic'):
    # 전체 훈련 데이터로 재훈련
    X_full_processed = preprocessor.fit_transform(X_full, y_full)
    X_test_processed = preprocessor.transform(X_test)

    # 모델 재훈련
    best_model.fit(X_full_processed, y_full)

    # 예측 및 저장
    test_predictions = best_model.predict(X_test_processed)
    submission = pd.DataFrame({
        'ID': test['ID'],
        'target': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
```

**구현의 합리성**:
- ✅ **전체 데이터 활용**: 최종 모델은 모든 훈련 데이터 사용
- ✅ **일관된 전처리**: 동일한 preprocessor 객체 사용
- ✅ **표준 형식**: 대회 요구사항에 맞는 제출 파일

---

## 📊 4단계: 현재 성능 분석 및 문제 진단

### 4.1 성능 격차 분석

#### 확인된 성능 지표들
- **Cross-Validation**: 76.96% (Random Forest)
- **실제 리더보드**: 67.596%
- **성능 격차**: -9.36% (상당한 차이)

**가능한 원인들**:

1. **과적합 (Overfitting)**:
   ```python
   print(f"Overfitting: {train_acc - val_acc:.4f}")
   ```
   - 훈련 정확도 1.0000 vs 검증 정확도 0.7624
   - 과적합도: 0.2376 (23.76%)

2. **교차검증 편향**:
   - Single hold-out validation vs 5-fold CV
   - CV가 더 낙관적인 추정을 제공

3. **데이터 분포 차이**:
   - Train vs Test set의 분포가 다를 가능성
   - Domain shift 또는 temporal shift

4. **리더보드 특성**:
   - Public LB의 일부분만 공개
   - Private LB와 다른 분포일 가능성

### 4.2 모델 복잡도 분석

#### 현재 사용 중인 모델들의 복잡도
- **Random Forest**: n_estimators=100 (중간 복잡도)
- **Extra Trees**: n_estimators=100 (중간 복잡도)
- **Gradient Boosting**: n_estimators=100 (높은 복잡도)

**복잡도 vs 성능 trade-off**:
- 높은 복잡도 → 훈련 성능 상승, 일반화 성능 하락
- 적절한 regularization 부재
- Early stopping, cross-validation 기반 tuning 필요

### 4.3 특성 엔지니어링의 한계

#### 현재 특성 처리 수준
1. **기본적 전처리**만 수행:
   - 스케일링 (RobustScaler)
   - 상관관계 제거
   - 분산 필터링

2. **부재한 고급 기법들**:
   - 도메인 기반 특성 생성
   - 센서 간 상호작용 특성
   - 시간 윈도우 기반 특성 (if applicable)
   - 비선형 변환 특성

#### 개선 기회
- **센서 그룹화**: X_40, X_11, X_46 등 중요 센서들의 조합
- **통계적 특성**: 센서 그룹별 mean, std, skew, kurtosis
- **비율 특성**: 중요 센서들 간의 비율 관계

---

## 🎓 학습된 핵심 통찰들

### 5.1 데이터 사이언스 워크플로우 이해

#### 체계적 접근법의 중요성
1. **EDA 단계**: 데이터 이해가 모든 후속 작업의 기반
2. **전처리 전략**: 단순한 정리가 아닌 전략적 선택
3. **모델 비교**: 다양한 접근법의 체계적 실험
4. **성능 검증**: 단순한 정확도 이상의 종합적 평가

#### 재현가능한 연구의 실천
- ✅ **random_state 일관성**: 모든 모델에서 42 사용
- ✅ **함수 모듈화**: 재사용 가능한 코드 구조
- ✅ **결과 저장**: 모델과 전처리기 pickle 저장
- ✅ **파라미터 기록**: 실험 설정의 명시적 기록

### 5.2 머신러닝 모델링 통찰

#### Tree-based 모델의 효과성
- **Random Forest**: 가장 안정적인 성능
- **Extra Trees**: 더 많은 randomness로 일반화 향상
- **Gradient Boosting**: 높은 성능이지만 과적합 위험

#### 앙상블의 필요성
- 단일 모델로는 0.90+ 목표 달성 어려움
- 서로 다른 특성의 모델 조합 필요
- 다양성(diversity)과 성능의 균형

### 5.3 경진대회 특화 인사이트

#### CV vs LB 점수 격차
- **일반적 현상**: 대부분의 경진대회에서 관찰
- **관리 전략**: 보수적인 모델 선택, robust validation
- **리더보드 과적합 방지**: 제출 횟수 제한, 안정성 우선

#### 클래스 균형의 이점
- **완벽한 균형**: 각 클래스 1,033개씩
- **단순한 평가**: 복잡한 클래스 가중치 불필요
- **Macro F1-score 유리**: 모든 클래스 동등한 중요도

---

## 🚀 다음 단계를 위한 개선 방향

### 6.1 즉시 실행 가능한 개선사항

1. **평가 지표 개선**:
   ```python
   from sklearn.metrics import f1_score
   macro_f1 = f1_score(y_val, y_val_pred, average='macro')
   ```

2. **교차검증 강화**:
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```

3. **과적합 방지**:
   ```python
   # Random Forest 정규화
   rf = RandomForestClassifier(
       max_depth=20,           # 깊이 제한
       min_samples_split=10,   # 분할 최소 샘플
       min_samples_leaf=5      # 잎 노드 최소 샘플
   )
   ```

### 6.2 중기 개선 전략

1. **고급 특성 엔지니어링**:
   - 센서 그룹별 통계적 특성
   - 주요 센서 간 상호작용 특성
   - PCA + 원본 특성 조합

2. **최신 모델 도입**:
   - XGBoost/LightGBM with hyperparameter tuning
   - TabNet for tabular data
   - Stacking ensemble

3. **검증 전략 고도화**:
   - Time-based split (if temporal order exists)
   - Multiple random splits with different seeds
   - Adversarial validation for train/test distribution comparison

---

## ✅ 현재 코드베이스 강점과 약점

### 강점 (Strengths)
- ✅ **체계적 구조**: EDA → 전처리 → 모델링 순서
- ✅ **모듈화**: 재사용 가능한 함수들
- ✅ **재현성**: random_state와 저장/로딩 시스템
- ✅ **다양성**: 8개 다른 알고리즘 비교
- ✅ **실용성**: 실제 제출까지 완료된 파이프라인

### 약점 (Weaknesses)
- ❌ **단순한 특성 엔지니어링**: 기본적인 전처리만
- ❌ **하이퍼파라미터 최적화 부재**: 기본값만 사용
- ❌ **앙상블 전략 미흡**: 단순 투표 방식
- ❌ **평가 지표 단일화**: Accuracy만 사용
- ❌ **교차검증 부족**: 단일 분할만 사용

---

**결론**: 현재 코드베이스는 견고한 기반을 제공하지만, 0.90+ 목표 달성을 위해서는 특성 엔지니어링, 모델 최적화, 앙상블 전략의 대폭 개선이 필요하다.

**다음 장에서**: Phase 1 성능 진단을 통해 구체적인 개선점들을 실험적으로 검증하고, 단계별 성능 향상 전략을 실행할 예정이다.