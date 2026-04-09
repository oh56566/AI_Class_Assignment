# 딥러닝 실습 보고서
**Breast Cancer · Diabetes · Wine 데이터셋 — 분류(Classification) 및 회귀(Regression) 실습**

---

## 1. 개요

본 보고서는 세 가지 데이터셋(Breast Cancer, Diabetes, Wine)을 활용하여 딥러닝 기반 분류 및 회귀 모델을 구축하고 성능을 비교·분석한 실습 결과를 정리한 것이다. 각 데이터셋에 대해 분류(Classification)와 회귀(Regression) 실험을 수행하였으며, 상관관계 행렬을 활용한 피처 선택, 결측값 처리, 스케일링 등 데이터 전처리 과정을 체계적으로 적용하였다.

| 실습 파일 | 데이터셋 | 문제 유형 | 타겟 변수 |
|---|---|---|---|
| DL_BC_class | Breast Cancer | 이진 분류 | label (0: 악성, 1: 양성) |
| DL_BC_reg_corr | Breast Cancer | 회귀 | mean area (종양 면적) |
| DL_diabetes_class | Diabetes | 이진 분류 | Outcome (0: 정상, 1: 당뇨) |
| DL_diabetes_reg_corr | Diabetes | 회귀 | BMI (체질량지수) |
| DL_Wine_class | Wine | 다중 분류 | Wine (품종 1·2·3) |
| DL_Wine_reg_corr | Wine | 회귀 | Alcohol (알코올 함량) |

---

## 2. 공통 전처리 파이프라인

모든 실습에 공통으로 적용된 전처리 절차는 다음과 같다.

| 단계 | 내용 | 적용 대상 |
|---|---|---|
| 결측값 처리 | 0값을 의학적 결측으로 간주하여 중앙값으로 대체 | Diabetes (Glucose, BloodPressure, SkinThickness, Insulin, BMI) |
| 피처 선택 | 상관관계 행렬로 타겟과 상관 높은 피처 선택, 다중공선성 제거 | BC_reg, Diabetes_reg, Wine_reg |
| 데이터 분할 | `train_test_split(test_size=0.2, random_state=42)` | 전체 |
| 스케일링 (X) | StandardScaler — train fit, test transform only | 전체 (Data Leakage 방지) |
| 스케일링 (y) | StandardScaler — 회귀 타겟에만 적용, 역변환으로 실제 단위 확인 | BC_reg, Diabetes_reg, Wine_reg |

---

## 3. 실습별 차이점

| 구분 | BC_class | BC_reg | Diabetes_class | Diabetes_reg | Wine_class | Wine_reg |
|---|---|---|---|---|---|---|
| 샘플 수 | 569 | 569 | 768 | 768 | 178 | 178 |
| 입력 피처 수 | 30개 | 6개 | 8개 | 9개 | 13개 | 7개 |
| 피처 선택 | 전체 사용 | 상관관계 기반 | 전체 사용 | 상관관계+파생 | 전체 사용 | 상관관계 기반 |
| 출력층 | Dense(1) sigmoid | Dense(1) 선형 | Dense(1) sigmoid | Dense(1) 선형 | Dense(3) softmax | Dense(1) 선형 |
| 손실함수 | binary_crossentropy | mse | binary_crossentropy | mse | categorical_crossentropy | mse |
| Epochs | 50 | 30 | 100 | 100 | 50 | 75 |
| 특이사항 | Dropout(0.3) | X·y 모두 스케일링 | class_weight 불균형 보정 | 파생 피처 2개 추가 | get_dummies 원핫 | X·y 모두 스케일링, 역변환 |

---

## 4. 모델 성능 비교

### 4.1 분류 모델

| 데이터셋 | Test Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
|---|---|---|---|---|
| Breast Cancer (BC_class) | **97.4%** | 0.97 | 0.97 | 0.97 |
| Diabetes (Diabetes_class) | 70.8% | 0.71 | 0.71 | 0.71 |
| Wine (Wine_class) | **100.0%** | 1.00 | 1.00 | 1.00 |

#### Breast Cancer 분류 (DL_BC_class)

피처 30개 전체를 사용하고 Dropout(0.3)을 적용하였다. 악성(0)을 양성(1)으로 잘못 예측한 False Negative가 3건 발생하였으나, 전반적으로 97.4%의 높은 정확도를 기록하였다. 의료 데이터에서 악성을 놓치는 False Negative는 더 위험하므로, Recall(악성 기준 0.93)에 주목할 필요가 있다.

**혼동 행렬**

| | 예측: 양성(0) | 예측: 악성(1) |
|---|---|---|
| 실제: 양성(0) | 40 | 3 |
| 실제: 악성(1) | 0 | 71 |

#### Diabetes 분류 (DL_diabetes_class)

클래스 불균형(비당뇨 500명 vs 당뇨 268명, 약 1.9:1)을 보정하기 위해 `class_weight='balanced'`를 적용하고 Epochs를 100으로 늘렸다. 그럼에도 정확도는 70.8%에 그쳤는데, 이는 알고리즘 한계가 아닌 데이터 자체의 한계(샘플 768개, 결측률 높음)에 기인한다.

**혼동 행렬**

| | 예측: 정상(0) | 예측: 당뇨(1) |
|---|---|---|
| 실제: 정상(0) | 75 | 24 |
| 실제: 당뇨(1) | 21 | 34 |

#### Wine 분류 (DL_Wine_class)

3클래스 분류 문제로, `get_dummies()`로 원핫인코딩 후 출력층에 softmax를 적용하였다. StandardScaler 전처리 후 완벽한 100% 분류 성능을 달성하였다. Wine 데이터셋은 13개 피처가 품종 간 선형 분리가 명확하여 딥러닝 모델에 매우 적합한 구조임을 확인하였다.

---

### 4.2 회귀 모델

| 데이터셋 | 타겟 변수 | RMSE (실제 단위) | 오차율 (RMSE/평균) | 비고 |
|---|---|---|---|---|
| BC_reg | mean area | ~103 픽셀² | ~15.8% | R² ≈ 0.91, 우수 |
| Diabetes_reg | BMI | ~6.0 kg/m² | ~18.7% | R² ≈ 0.18, 낮음 |
| Wine_reg | Alcohol (%) | 0.55 % | 4.2% | R² ≈ 0.50, X·y 모두 스케일링 적용 |

#### Breast Cancer 회귀 (DL_BC_reg_corr)

label과의 상관관계를 기반으로 30개 피처에서 6개를 선택하였다. 다중공선성(|r| ≥ 0.9) 피처는 제거하였다.

| 선택 피처 | 상관계수 | 선택 이유 |
|---|---|---|
| mean concave points | \|r\|=0.823 | 타겟 상관 1위 |
| area error | \|r\|=0.800 | 크기 변동성 대표 |
| mean concavity | \|r\|=0.686 | 오목도 mean |
| mean compactness | \|r\|=0.499 | 밀도 |
| mean texture | \|r\|=0.321 | 질감 (독립적) |
| mean fractal dimension | \|r\|=0.283 | 경계 복잡도 |

X, y 모두 StandardScaler 적용 후 역변환하여 실제 단위 RMSE를 확인하였다. R² ≈ 0.91로 우수한 성능을 기록하였다.

#### Diabetes 회귀 (DL_diabetes_reg_corr)

BMI와의 상관관계 분석 결과 SkinThickness(0.543)만 중간 이상 상관관계를 보였으며, 나머지 피처들은 0.28 이하의 낮은 값을 나타냈다. 파생 피처(Glucose×Insulin, SkinThickness×BloodPressure)를 추가하여 9개 피처로 학습하였다.

| 선택 피처 | BMI 상관계수 | 비고 |
|---|---|---|
| SkinThickness | \|r\|=0.543 | BMI 상관 1위 |
| BloodPressure | \|r\|=0.281 | 비만-혈압 관련 |
| Glucose | \|r\|=0.231 | 인슐린 저항성 |
| Insulin | \|r\|=0.180 | Glucose와 함께 사용 |
| DiabetesPedigreeFunction | \|r\|=0.153 | 유전적 요인 |
| Age, Pregnancies | \|r\|<0.03 | 참고용 포함 |
| Glucose_Insulin (파생) | — | 인슐린 저항성 근사 |
| SkinThickness_BP (파생) | — | 비만+혈압 복합 |

데이터셋 자체의 BMI 설명력 한계(최고 상관 피처 0.543)로 인해 R² ≈ 0.18의 낮은 성능을 기록하였다. 이는 모델 구조의 문제가 아닌 피처-타겟 간 관계의 복잡성에 기인한다.

#### Wine 회귀 (DL_Wine_reg_corr)

알코올 함량(Alcohol)과의 상관관계를 기반으로 7개 피처를 선택하였다(Proline, Color.int, Acl, Phenols, Mg, Flavanoids, Ash). X와 y 모두 StandardScaler를 적용하고, 역변환을 통해 실제 단위의 RMSE를 확인하였다. RMSE 0.55%로 알코올 평균(13.09%) 대비 오차율 4.2%의 양호한 성능을 기록하였다. R² ≈ 0.50으로 BC_reg(0.91)보다는 낮으나 Diabetes_reg(0.18)보다 높은 설명력을 나타냈다.

---

## 5. 종합평가

### 개선 사항

- **Diabetes_reg 성능 한계**: BMI를 설명하는 피처의 상관관계가 전반적으로 낮아 R² ≈ 0.18에 그쳤다. 체중, 신장 등 직접적인 피처가 데이터셋에 없는 구조적 한계이다.

### 결론

이번 실습을 통해 딥러닝 모델의 성능은 알고리즘 구조보다 데이터 품질과 전처리의 영향이 크다는 것을 확인하였다. Breast Cancer와 Wine처럼 피처-타겟 간 상관관계가 명확한 데이터셋에서는 높은 성능(97~100%)을 달성하였으나, Diabetes처럼 피처 설명력이 낮고 결측률이 높은 데이터셋에서는 전처리를 충실히 수행하더라도 성능에 한계가 존재함을 이해하였다.
