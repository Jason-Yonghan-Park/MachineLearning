##################### 손글씨 이미지 숫자 분류
##################### naivebayes 분류기

# 필요한 패키지 임포트
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 손글씨 데이터
digits = load_digits()

# 1. 가우시안 나이브 베이즈 모델 클래스 선택
from sklearn.naive_bayes import GaussianNB

# 2. 모델 클래스 인스턴스 생성
model = GaussianNB()

# 3. 데이터 배치 -> data(2차원) / label(1차원)
# train / test data 7:3 분리 / 이미지 데이터도 분리
Xtrain, Xtest, ytrain, ytest, imgTrain, imgTest = train_test_split(digits.data, digits.target, digits.images, random_state=0)

# 4. 모델 데이터 학습
model.fit(Xtrain, ytrain)

# 5. 모델 새로운 데이터 적용
pre = model.predict(Xtest)

# 6. 예측 결과 정확도 출력
from sklearn.metrics import accuracy_score
print("정확도: {}".format(accuracy_score(ytest, pre)))

# 7. 오차행렬(Confusion Matrix) 생성
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, pre)
# seaborn의 히트맵
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel("real data")
plt.ylabel("predict")
plt.show()

# 잘못 분류된 데이터 시각화
fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgTest[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(pre[i]), transform = ax.transAxes, color = 'green' if (ytest[i] == pre[i]) else 'red')

plt.show()














