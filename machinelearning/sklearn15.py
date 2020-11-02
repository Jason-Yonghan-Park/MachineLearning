#################### 모델 검증
################# K-fold cross validation
import pandas as pd

csv1 = pd.read_csv('D:/MachineLearning/machinelearning/iris.csv')
print(csv1)

# 데이터 data, label로 분리
data = csv1.iloc[:, 0:-1]
label = csv1['Name']
print(data);print(label)

# SVM classifier 중 SVC 클래스 선택
from sklearn import svm

# 모델 클래스 생성
model = svm.SVC(gamma='auto')

# cross validation 사용
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data, label, cv=5)
print("5fold 각각의 정답률: {}".format(scores))
print("5fold 평균 정답률: {}".format(scores.mean()))






