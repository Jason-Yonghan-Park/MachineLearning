############### Estimator API
########## 지도학습 - 선형 회귀
import matplotlib.pyplot as plt
import numpy as np

# 메르센느 트위스터 알고리즘 -> 난수 발생
rng = np.random.RandomState(42)

# 균등분포 -> 0~1 사이의 실수 50개 샘플링
x = 10 * rng.rand(50)

# 표준정규분포 -> 50개 샘플링
y = 2 * x - 1 + rng.randn(50)

# 선형회귀 지도학습모델 순서
# 1. Estimator클래스 선택
from sklearn.linear_model import LinearRegression

# 2. 모델의 하이퍼파라미터 선택 (기본값 설정)
model = LinearRegression(fit_intercept=True)
print(model)

# 3. 데이터 -> data(2차원 배열) & label(1차원 배열)로 구성해 배치
X = x[:, np.newaxis]
print(X)

# 4. 모델에 data를 적합
model.fit(X, y)

# 5. 모델 출력
# scikit learn 사용 시 관례상 학습된 모델의 모수는 뒤에 언더바 붙음
print(model.coef_, "-", model.intercept_)

# 6. 새로운 데이터 모델에 적용 후 검증
# 새로운 데이터
xfit = np.linspace(-1, 11)
print(xfit)

Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# 산점도 및 회귀선 출력
plt.scatter(x, y)
plt.plot(Xfit, yfit)
plt.show()

