####################### 손글씨 이미지 데이터 시각화
###################### isomap 이용 -> 차원 축소 -> 시각화

# 필요한 패키지 임포트
from sklearn.datasets import load_digits

# 손글씨 데이터 읽어오기
digits = load_digits()

# 1. 차원 축소 위한 모델 클래스 선택 - Isomap
from sklearn.manifold import Isomap

# 2. 모델 클래스 인스턴스 생성 -> 2차원 축소
iso = Isomap(n_components=2)

# 3. 데이터 -> data(2차원), label(1차원)
# 비지도 -> label 필요 X, digits.data -> 2차원 구조

# 4. 모델에 데이터 훈련
iso.fit(digits.data)

# 5. 데이터 2차원 축소
data_transf = iso.transform(digits.data)
print(digits.data.shape, "-", data_transf.shape)

# 6. 축소된 데이터 -> 산점도 출력
import matplotlib.pyplot as plt

plt.scatter(data_transf[:, 0], data_transf[:, 1], c=digits.target, cmap=plt.cm.get_cmap('CMRmap', 10),
            alpha=0.7, edgecolor='none')
plt.colorbar(label='digital label', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()





