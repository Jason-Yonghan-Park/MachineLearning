######################### 손글씨 이미지 랜덤포레스트 분류

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 손글씨 데이터 읽어옴
digits = load_digits()

# 1. 랜덤포레스트 모델 클래스 선택
from sklearn.ensemble import RandomForestClassifier

# 2. 모델 클래스의 인스턴스 생성
model = RandomForestClassifier(n_estimators=1000)

# 3. 데이터 data(2차원), label(1차원) 구성후 배치
Xtrain, Xtest, ytrain, ytest, imgTrain, imgTest = train_test_split(digits.data, digits.target, digits.images, random_state=0)

# 4. 모델 데이터 학습
model.fit(Xtrain, ytrain)

# 5. 새로운 데이터 모델에 적용 후 검증
pre = model.predict(Xtest)

# 6. 예측 결과 출력
# 정확도
from sklearn.metrics import accuracy_score
print("정확도: {}".format(accuracy_score(ytest, pre)))
# 랜덤포레스트 분류결과 보고서 출력
from sklearn.metrics import classification_report
print(classification_report(pre, ytest))
# 오차행렬 생성
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, pre)
# seaborn 히트맵 생성
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('예측데이터')
plt.ylabel('실제데이터')
plt.show()







