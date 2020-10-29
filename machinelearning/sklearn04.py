############ 지도학습 예제 - 붓꽃분류

# 필요한 패키지 임포트
from sklearn import svm, metrics
import random, re


# 붓꽃 데이터 csv 파일 읽기
csv1 = []
with open("D:\MachineLearning\machinelearning\iris.csv", "r", encoding="utf-8") as fp:
    # 한줄씩 데이터 접근
    for line in fp:
        line = line.strip() # 줄바꿈 문자 포함 공백 제거
        cols = line.split(",")
        # 숫자 형식의 문자열 데이터 -> 숫자로 변환
        fn = lambda n: float(n) if re.match(r"^[0-9\.]+$", n) else n
        # 다시 리스트화
        cols = list(map(fn, cols))
        csv1.append(cols)
print(csv1)

# 데이터의 헤더 제거
del csv1[0]

# 데이터 섞기
random.shuffle(csv1)

# 섞은 데이터 -> 학습 데이터 / 테스트 데이터 분할 -> 2:1
total_len = len(csv1)
print(total_len)
train_len = int(total_len * 2/3)
print(train_len)

# 지도 학습 -> 모델 학습 or 검증하는 데이터는 항상 input(2차원 배열) / label(1차원 배열) 로 나뉨
train_input = []
train_label = []
test_input = []
test_label = []

for i in range(total_len):
    # 섞은 데이터 지도학습에 적합한 데이터 형태로 분할
    inputData = csv1[i][0:4]
    label = csv1[i][-1]

    if i < train_len:
        train_input.append(inputData)
        train_label.append(label)
    else:
        test_input.append(inputData)
        test_label.append(label)

# 데이터 준비 완료 -> 어떤 estimator 사용할지 결정
# estimator 인스턴스 생성
clf = svm.SVC()

# 데이터 학습
clf.fit(train_input, train_label)

# 새로운 데이터 통해 모델 적용 검증
pre = clf.predict(test_input)

# test 데이터의 모델 적용한 값과 실제 값의 정확도 비교
ac_score = metrics.accuracy_score(test_label, pre)
print("정답률: {}".format(ac_score))







