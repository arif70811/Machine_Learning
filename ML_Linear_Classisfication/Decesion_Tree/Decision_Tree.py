from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

path = r'/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Decesion_Tree/Sampl_Data/titanic.csv'

df = pd.read_csv(path)
df = df[['Survived', 'Pclass', 'Sex', 'Age','Fare']]
sex_label = LabelEncoder()
df['sex_l'] = sex_label.fit_transform(df['Sex'])
df = df.drop(['Sex'],axis='columns')
x_train,x_test,y_train,y_test = train_test_split(df[['Pclass', 'sex_l', 'Age','Fare']],df['Survived'],test_size = 0.2)
model = DecisionTreeClassifier()
model = model.fit(x_train,y_train)
# print(model.score(x_test,y_test))
Testing_Prediction = model.predict(x_test)
print(Testing_Prediction)
print(y_test)