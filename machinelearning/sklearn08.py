############## 손글씨 숫자 이미지 탐색

# 손글씨 데이터 읽기 위한 모듈 임포트
from sklearn.datasets import load_digits

# 손글씨 숫자 이미지 데이터 읽어오기
digits = load_digits()
print(digits.data.shape)
print(digits.data[0], "-", digits.target[0])
print(digits.images[0], "-", digits.target[0])

# 8*8 픽셀 이미지 가로세로 10*10 한창 출력
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(8,8), subplot_kw={"xticks":[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

# axes.flat -> AxesSubplot 객체 담긴 반복가능한 객체 반환
# 반복문 통해 모든 이미지 출력 가능
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')

plt.show()

print(digits.images.shape)
print(digits.data.shape)





