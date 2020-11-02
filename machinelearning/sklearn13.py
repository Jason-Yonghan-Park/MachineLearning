#################### 랜덤 포레스트

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# make_blob() -> 가우시안 정규분포 이용 가상데이터 생성
# centers: 클러스터의 수, cluster_std: 클러스터 표준편차 (디폴트 1.0)
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)

# 머신러닝 모델 인스턴스 및 데이터 받아 시각화하는 함수
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    # matplotlib.axes.Axes 객체 인스턴스 구함
    ax = ax or plt.gca()
    # 차트에 산점도 그림
    # 색상의 제한 y값의 최저~최고 / 등고선
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    # 축 및 축라벨 출력 X
    ax.axis('off')
    ax.axis('tight')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 모델에 데이터 학습
    model.fit(X,y)

    # meshgrid() 좌표 구성하는 함수
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))

    # 예측결과 데이터 shape 변환
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # 레이블에서 유니크한 값을 뽑아 개수를 구함(클러스터의 개수)
    n_classes = len(np.unique(y))

    # 등고선(윤관) 그래프 이용 -> 각 클러스터 결정 경계 시각화
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1)-0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# 의사결정 트리의 결정 경계 시각화
tree = DecisionTreeClassifier()
visualize_classifier(tree, X, y)
# 랜덤 포레스트의 결정 경계 시각화 - 의사결정 트리 추정기 100개 지정
model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y)
