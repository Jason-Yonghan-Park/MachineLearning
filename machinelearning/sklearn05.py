########## 지도학습 - 붓꽃분류 / naivebayes model

import pandas as pd

# pandas 이용해 붓꽃 데이터 csv 읽기
csv1 = pd.read_csv('D:/MachineLearning/machinelearning/iris.csv')

# 가우스 분포 이용한 가우스 naivebayes model 임포트
# 1. 가우스 naivebayes model 클래스 선택
from sklearn.naive_bayes import GaussianNB

# 2. 모델 클래스의 인스턴스 생성
model = GaussianNB()

# 3. 데이터 -> input(2차원 배열), label(1차원 배열) 구성 후 배치
X_iris = csv1[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y_iris = csv1["Name"]

# 4. Train Data / Test Data 7:3으로 분할
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

# 4. 모델에 데이터 학습
model.fit(Xtrain, ytrain)

# 5. 새로운 데이터 모델 적용
pre = model.predict(Xtest)

# 6. 예측 정확도 출력
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, pre))








