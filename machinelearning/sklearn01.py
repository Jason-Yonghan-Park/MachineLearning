############## 파이썬 머신러닝
######### scikit learn
######### XOR 학습
from sklearn import svm, metrics
import pandas as pd

# 학습용 데이터(Training Data) 생성
xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
    ]

# 학습용 데이터를 DataFrame으로 생성 -> 데이터 및 레이블 분리
df = pd.DataFrame(xor_data)
print(df)
data = df.loc[:, 0:1] # 배열의 앞 두자리 -> 문제
label = df.loc[:, 2]  # 배열의 맨 뒷자리 -> 답

# 분류 알고리즘 - SVC인스턴스 생성
classfic = svm.SVC()
classfic.fit(data, label) # 문제 및 답을 학습
pre = classfic.predict(data)
print("예측 결과: {}".format(pre)) # [0, 1, 1, 0]

# 정확도 계산
ac_score = metrics.accuracy_score(label, pre)
print("정답률: {}".format(ac_score))

