# 노트북 개선 계획

**분석 대상**: `notebooks/01_main_analysis.ipynb`
**현재 수준**: 중급 (6.5/10)
**목표 수준**: 고급 (8.5+/10)

---

## 🔴 긴급 개선 항목 (High Priority)

### 1. 하이퍼파라미터 튜닝 부재 ⚠️
**문제**: 모든 모델이 기본 파라미터만 사용, 성능 최적화 미실시
**영향**: 2-5% F1 score 개선 기회 손실

**필요 작업**:
- [ ] RandomizedSearchCV 또는 GridSearchCV 구현
- [ ] LightGBM 파라미터 탐색 (n_estimators, max_depth, learning_rate, num_leaves)
- [ ] XGBoost 파라미터 탐색 (n_estimators, max_depth, learning_rate, subsample)
- [ ] 최적 파라미터로 모델 재학습
- [ ] Cross-validation으로 튜닝 효과 검증

**예상 코드**:
```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 50, 100, 150],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    lgbm, param_distributions,
    n_iter=50, cv=5, scoring='f1_macro',
    random_state=42, n_jobs=-1
)
```

---

### 2. 피처 엔지니어링 부족 ⚠️
**문제**: 원본 52개 피처만 사용, 파생 변수 생성 없음
**영향**: 1-3% F1 score 개선 기회 손실

**필요 작업**:
- [ ] 피처 간 상호작용 항 생성 (multiplication, division)
- [ ] 다항식 피처 생성 (PolynomialFeatures)
- [ ] 통계적 피처 생성 (mean, std, min, max of feature groups)
- [ ] 도메인 지식 기반 피처 생성
- [ ] 피처 중요도 기반 선택 (SelectFromModel)

**예상 코드**:
```python
from sklearn.preprocessing import PolynomialFeatures

# 상위 중요도 피처들 간 상호작용
top_features = ['X_01', 'X_06', 'X_45', ...]
poly = PolynomialFeatures(degree=2, interaction_only=True)
interaction_features = poly.fit_transform(X_scaled[top_features])

# 통계적 피처
X_scaled['feature_mean'] = X_scaled[feature_cols].mean(axis=1)
X_scaled['feature_std'] = X_scaled[feature_cols].std(axis=1)
X_scaled['feature_max'] = X_scaled[feature_cols].max(axis=1)
X_scaled['feature_min'] = X_scaled[feature_cols].min(axis=1)
```

---

### 3. sklearn Pipeline 미사용 ⚠️
**문제**: 수동 전처리로 인한 데이터 유출 위험, 재현성 저하
**영향**: 프로덕션 배포 불가, 검증 신뢰도 문제

**필요 작업**:
- [ ] 전처리 Pipeline 구축 (Scaler → Feature Engineering)
- [ ] 모델 Pipeline 통합
- [ ] Pipeline으로 Cross-validation 재실행
- [ ] Pipeline 저장 (joblib/pickle)

**예상 코드**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('feature_engineer', CustomFeatureEngineering()),
    ('model', lgb.LGBMClassifier())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 🟡 중요 개선 항목 (Medium Priority)

### 4. 앙상블 기법 부재
**문제**: 단일 모델만 사용, 앙상블 성능 향상 기회 미활용
**영향**: 1-2% F1 score 개선 기회 손실

**필요 작업**:
- [ ] VotingClassifier (Soft voting)
- [ ] StackingClassifier (Meta-learner 추가)
- [ ] Blending (holdout 기반 앙상블)
- [ ] 앙상블 성능 비교 및 검증

**예상 코드**:
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Voting
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_tuned),
        ('xgb', xgb_tuned),
        ('rf', RandomForestClassifier())
    ],
    voting='soft'
)

# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('lgbm', lgbm_tuned),
        ('xgb', xgb_tuned)
    ],
    final_estimator=LogisticRegression(multi_class='multinomial'),
    cv=5
)
```

---

### 5. 모델 해석 및 설명가능성 부족
**문제**: 모델 예측 근거 불명확, 디버깅 어려움
**영향**: 모델 신뢰도 저하, 개선 방향 불명확

**필요 작업**:
- [ ] SHAP values 계산 및 시각화
- [ ] Permutation importance
- [ ] Confusion matrix 상세 분석
- [ ] 클래스별 오분류 패턴 분석
- [ ] Feature importance 심층 분석

**예상 코드**:
```python
import shap

# SHAP values
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))
```

---

### 6. 피처 선택 미실시
**문제**: 모든 52개 피처 사용, 노이즈 피처 포함 가능성
**영향**: 과적합 위험, 학습 속도 저하

**필요 작업**:
- [ ] Recursive Feature Elimination (RFE)
- [ ] SelectFromModel (feature importance 기반)
- [ ] Correlation-based feature selection
- [ ] 피처 수에 따른 성능 비교

**예상 코드**:
```python
from sklearn.feature_selection import RFE, SelectFromModel

# RFE
rfe = RFE(estimator=lgbm, n_features_to_select=30, step=1)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]

# Feature importance 기반
selector = SelectFromModel(lgbm, threshold='median')
selector.fit(X_train, y_train)
```

---

## 🟢 권장 개선 항목 (Low Priority)

### 7. 코드 조직화 및 모듈화 부족
**문제**: 반복 코드, 긴 셀, 재사용성 저하
**영향**: 유지보수 어려움, 가독성 저하

**필요 작업**:
- [ ] 반복 코드 함수화 (plotting, evaluation)
- [ ] 유틸리티 함수 분리 (utils.py)
- [ ] 설정값 상수화 (config.py)
- [ ] Docstring 추가

**예상 구조**:
```python
# utils.py
def plot_confusion_matrix(y_true, y_pred, labels):
    """Confusion matrix 시각화"""
    pass

def evaluate_model(model, X_test, y_test):
    """모델 성능 평가 및 출력"""
    pass

def cross_validate_model(model, X, y, cv=5):
    """Cross-validation 수행"""
    pass
```

---

### 8. 딥러닝 모델 미탐색
**문제**: 전통적 ML만 사용, 신경망 성능 비교 없음
**영향**: 최적 알고리즘 선택 불확실

**필요 작업**:
- [ ] Simple MLP (Multi-Layer Perceptron) 구현
- [ ] TabNet 또는 FT-Transformer 탐색
- [ ] 딥러닝 vs 전통 ML 성능 비교
- [ ] 계산 비용 대비 성능 분석

**예상 코드**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

mlp = Sequential([
    Dense(128, activation='relu', input_shape=(52,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])

mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
```

---

### 9. 고급 검증 전략 부재
**문제**: 단일 CV 전략만 사용
**영향**: 검증 신뢰도 제한적

**필요 작업**:
- [ ] Repeated K-Fold CV
- [ ] Nested Cross-Validation (튜닝 편향 제거)
- [ ] Leave-One-Out CV (데이터 적을 경우)
- [ ] 시계열 데이터면 TimeSeriesSplit

**예상 코드**:
```python
from sklearn.model_selection import RepeatedStratifiedKFold

repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(lgbm, X_scaled, y, cv=repeated_cv, scoring='f1_macro')
```

---

### 10. 문서화 및 주석 개선
**문제**: 한영 혼용, 설명 부족
**영향**: 협업 어려움, 재현성 저하

**필요 작업**:
- [ ] 마크다운 섹션 구조화
- [ ] 코드 주석 표준화 (영어 또는 한글 통일)
- [ ] 분석 배경 및 목적 명시
- [ ] 결과 해석 및 비즈니스 의미 추가

---

## 📊 개선 우선순위 요약

| 순위 | 항목 | 예상 효과 | 난이도 | 소요 시간 |
|-----|------|----------|--------|----------|
| 1 | 하이퍼파라미터 튜닝 | 🔥🔥🔥🔥🔥 | 중 | 2-3시간 |
| 2 | 피처 엔지니어링 | 🔥🔥🔥🔥 | 중상 | 3-4시간 |
| 3 | sklearn Pipeline | 🔥🔥🔥 | 하 | 1-2시간 |
| 4 | 앙상블 기법 | 🔥🔥🔥 | 중 | 2-3시간 |
| 5 | 모델 해석 (SHAP) | 🔥🔥 | 중 | 1-2시간 |
| 6 | 피처 선택 | 🔥🔥 | 하 | 1시간 |
| 7 | 코드 리팩토링 | 🔥 | 중 | 2-3시간 |
| 8 | 딥러닝 모델 | 🔥 | 상 | 4-6시간 |
| 9 | 고급 검증 전략 | 🔥 | 하 | 1시간 |
| 10 | 문서화 개선 | 🔥 | 하 | 1-2시간 |

---

## 🎯 단계별 실행 계획

### Phase 1: Quick Wins (1-2일)
1. 하이퍼파라미터 튜닝
2. sklearn Pipeline 구축
3. 피처 선택

**예상 성능 향상**: +3-6% F1 score

### Phase 2: Advanced Techniques (2-3일)
4. 피처 엔지니어링
5. 앙상블 기법
6. 모델 해석 (SHAP)

**예상 성능 향상**: +2-4% F1 score

### Phase 3: Polish & Production (1-2일)
7. 코드 리팩토링
8. 문서화 개선
9. 고급 검증 전략

**예상 성능 향상**: +0-1% F1 score (안정성 향상)

### Phase 4: Experimental (선택)
10. 딥러닝 모델 탐색

---

## 📈 예상 최종 성능

**현재 성능**:
- LightGBM CV: ~0.798 F1-macro
- XGBoost CV: ~0.800 F1-macro

**개선 후 예상 성능**:
- 하이퍼파라미터 튜닝: 0.820-0.840
- 피처 엔지니어링: 0.830-0.850
- 앙상블: 0.840-0.860

**목표**: **0.85+ F1-macro score**