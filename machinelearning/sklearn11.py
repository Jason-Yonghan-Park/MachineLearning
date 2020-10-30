################ 유명인사 얼굴 사진 데이터 분류
################ 서포트벡터머신(SVM) 분류

from sklearn.datasets import fetch_lfw_people

# 세계 정치인 중 사진이 60장 이상인 8명 데이터 로드
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces)

print(faces.data.shape)
print(faces.images.shape)

import matplotlib.pyplot as plt

def facesGraph01():
    fig, axes = plt.subplots(3, 5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces.images[i], cmap='bone')
        ax.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    plt.show()

# 1. SVC 모델 클래스 선택
from sklearn.svm import SVC

# 2. 모델 클래스 인스턴스 생성
model = SVC()

# 3. 데이터 분할 data, label
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest, imgTrain, imgTest = train_test_split(faces.data, faces.target, faces.images, random_state=0)

# 4. 모델 데이터 학습
model.fit(Xtrain, ytrain)

# 5. 새로운 데이터 모델 적용
pre = model.predict(Xtest)

# 6. 예측 결과 정확도 측정
from sklearn.metrics import accuracy_score
print("정확도: {}".format(accuracy_score(ytest, pre)))









