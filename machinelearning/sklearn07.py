####################### 비지도학습 - 붓꽃 군집화
################## GMM (가우스 혼합모델)

# 필요한 패키지 임포트
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn iris dataset 읽기
iris = sns.load_dataset("iris")

# 1. 붓꽃 종별 군집화 위한 모델 클래스 선택
from sklearn.mixture import GaussianMixture

# 2. 모델 클래스 인스턴스 생성
model = GaussianMixture(n_components=3)

# 3. 데이터 구조 input(2차원) label(1차원) 구성 후 배치
X_iris = iris.drop("species", axis=1)

# 4. 모델에 데이터 학습
# 비지도 학습 -> train data 에 label 적용 X
model.fit(X_iris)

# 5. 군집 레이블 결정
y_gmm = model.predict(X_iris)
# 군집 정보 -> 기존의 iris에 추가
iris["cluster"] = y_gmm
print(iris.iloc[80:90, :])

# 회귀모델 그래프 출력
sns.lmplot("sepal_width", "petal_width", hue='species', col='cluster', data=iris, fit_reg=False)
plt.show()







