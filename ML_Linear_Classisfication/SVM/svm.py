from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
X = df
Y = iris.target
x_tr,x_te,y_tr,y_te = train_test_split(X,Y,train_size = 0.8)
model = SVC()
model.fit(x_tr,y_tr)
predict = model.predict(x_te)
print(predict)
print(y_te)