######## Scikit learn의 데이터 표현
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 패키지가 제공하는 iris dataset
iris = sns.load_dataset("iris")

# scikit-learn 2차원 DataFrame 형식 데이터 많이 사용
# data(2차원 / 특징 배열) , label(1차원 / 대상 배열) 로 분리해서 지정
x_iris = iris.iloc[:, :-1]
# x_iris = iris.drop("species", axis=1)
print(x_iris.head())
y_iris = iris["species"]
print(y_iris.head(), type(y_iris.head()))
print(x_iris.ndim, "-", y_iris.ndim)

sns.pairplot(iris, hue="species")
plt.show()