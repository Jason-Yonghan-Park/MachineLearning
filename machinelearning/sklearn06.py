################### 비지도 학습 - 붓꽃 데이터 차원 축소
################## PCA 알고리즘

# 필요한 패키지 임포트
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 패키지가 제공하는 iris dataset 읽어옴
iris = sns.load_dataset("iris")

# 1. 차원 축소
# PCA 모델 클래스 선택
from sklearn.decomposition import PCA

# 2. 모델 클래스 인스턴스 생성 - hyperparameter
model = PCA(n_components=2)

# 3. 데이터 input(2차원 배열), label(1차원 배열) 배치
# train data 에서 필요한 열만 추출
X_iris = iris.drop("species", axis=1)

# 4. 모델 데이터에 학습
# 비지도 학습 -> 학습 데이터만 제공 -> label 제공X
model.fit(X_iris)

# 5. 데이터 2차원으로 축소
X_2D = model.transform(X_iris)
# 기존의 iris DataFrame에 열을 추가 -> 축소 변환된 데이터 저장
iris["PCA1"] = X_2D[:, 0]
iris["PCA2"] = X_2D[:, 1]

sns.lmplot("PCA1", "PCA2", hue="species", data=iris, fit_reg=False)
plt.show()





