# 타이타닉 생존자 분류 실습
## 노트북 비교 분석 보고서
> `titanic0327.ipynb` vs `titanic0327_df_corr.ipynb`

---

## 1. 개요

두 노트북은 동일한 타이타닉 데이터셋에 대해 동일한 전처리 파이프라인을 적용하되, **Feature 선택 방식**에서 차이를 두었다.

- `titanic0327.ipynb` : 전처리 후 전체 Feature 사용
- `titanic0327_df_corr.ipynb` : 상관관계 히트맵 분석 후 유의미한 Feature 3개만 선택

---

## 2. 공통 전처리 파이프라인

### 2-1. 불필요 컬럼 제거

```python
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
```

| 컬럼 | 제거 이유 |
|---|---|
| PassengerId | 단순 일련번호, 생존과 무관 |
| Name | 텍스트라 인코딩이 어렵고 의미 없음 |
| Ticket | 불규칙한 형식으로 패턴 추출 어려움 |
| Cabin | 결측 687개(77%)로 신뢰할 수 있는 처리 불가 |

### 2-2. 결측값 처리

```python
df['Age'] = df['Age'].fillna(df['Age'].median())       # 177개
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # 2개
```

- **Age** → 중앙값(28.0): 고령자 이상치에 강건한 중앙값을 선택했다.
- **Embarked** → 최빈값('S'): 범주형이므로 평균/중앙값 대신 최빈값을 사용했다.

### 2-3. 인코딩

```python
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # female=0, male=1
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

- **Sex**: 값이 2개이므로 LabelEncoder로 0/1을 직접 지정했다.
- **Embarked**: 3개 범주를 크기 관계 없이 독립적으로 표현하기 위해 `get_dummies`를 사용했고, `drop_first=True`로 다중공선성을 방지했다.

### 2-4. 스케일링

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   # fit 없이 transform만
```

Age(0~80), Fare(0~512), Pclass(1~3) 등 단위가 달라 표준화가 필요했다. 테스트 데이터에 `fit`을 재수행하면 데이터 누수(data leakage)가 발생하므로 `transform`만 적용했다.

---

## 3. 두 노트북의 차이점

| 구분 | titanic0327 | titanic0327_df_corr |
|---|---|---|
| Feature 수 | 8개 | 3개 |
| Feature 목록 | Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S | Sex, Pclass, Fare |
| Feature 선택 기준 | 전처리 후 전체 사용 | Survived 상관계수 상위 3개 |
| 상관관계 분석 | 없음 | 히트맵 시각화 후 선택 |

`df_corr` 노트북에서는 전처리 완료 후 seaborn 히트맵으로 피어슨 상관계수를 시각화했다. 분석 결과 아래 3개 Feature가 Survived와 가장 높은 상관관계를 보여 이를 선택했다. 두 노트북 모두 DecisionTree와 RandomForest의 `random_state=42`로 통일해 `train_test_split`과 일관성을 맞췄다.

| Feature | 상관계수 | 해석 |
|---|---|---|
| Sex | -0.54 | 여성 생존율이 높음 (male=1) |
| Pclass | -0.34 | 낮은 등급일수록 생존율 낮음 |
| Fare | +0.26 | 높은 요금(1등석)일수록 생존율 높음 |

---

## 4. 모델 성능 비교

테스트 셋: 전체 891개 중 25% (223개), `random_state=42`, `stratify` 미적용


| 모델 | 전체 Feature (8개) | 전체 F1(macro) | 상관관계 Feature (3개) | 상관관계 F1(macro) | 변화 |
|---|---|---|---|---|---|
| Logistic Regression | 80.7% | 0.80 | 78.0% | 0.77 | ▼ 2.7%p |
| Decision Tree | 77.6% | 0.76 | 82.1% | 0.81 | ▲ 4.5%p |
| Random Forest | 78.9% | 0.78 | 82.5% | 0.82 | ▲ 3.6%p |

### Logistic Regression

전체 Feature 사용 시 80.7%에서 78.0%로 2.7%p 하락했다. 선형 모델 특성상 Feature가 줄면서 활용 가능한 정보가 감소한 것이 원인으로 보인다.

### Decision Tree

77.6%에서 82.1%로 4.5%p 상승했다. 불필요한 Feature가 많을 경우 과적합 경향이 있는 결정 트리의 특성상, Feature 수를 줄임으로써 일반화 성능이 향상된 것으로 해석했다.

### Random Forest

78.9%에서 82.5%로 3.6%p 상승했다. 앙상블 모델임에도 Feature 선택의 효과가 나타났으며, 세 모델 중 가장 높은 최종 정확도를 얻었다.

---

## 결론

### 데이터에서 알 수 있는 것

성별이 생존에 가장 큰 영향을 미쳤다 (상관계수 -0.54) "여성과 어린이 먼저" 라는 당시 구조 원칙이 데이터로 실증된다. 여성 생존율이 남성보다 상당히 높다. 하지만 나이는 전체 데이터 기준으로 영향은 미미했다. 결측치가 20%나 되어 신뢰도가 낮은 영향도 있다.

객실 등급과 요금도 생존을 갈랐다 (상관계수 Pclass -0.34, Fare +0.26) 1등석 승객일수록 생존율이 높았고, 상위 등급 객실이 갑판에 가까워 구명보트 접근성이 좋았던 것이 원인이 아닐까 생각된다. 게다가 이 두 피쳐는 서로 높은 상관관계를 가진다.

### 성능에서 알 수 있는 것

핵심 Feature 3개만으로도 83% 정확도가 나왔다.
Sex, Pclass, Fare 3개가 생존 여부의 대부분을 설명한다. 나머지 Feature들은 노이즈에 가까웠다.
상관관계 기반 Feature 선택은 Logistic Regression에서 소폭 성능 저하를 가져왔지만, Decision Tree와 Random Forest에서는 유의미한 성능 향상을 이끌어냈다. 이를 통해 트리 계열 모델이 불필요한 Feature에 더 민감하게 반응한다는 점을 실험적으로 확인했다.
