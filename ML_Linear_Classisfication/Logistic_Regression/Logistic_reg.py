from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


path_1 = r'/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Logistic_Regression/Sample_Data/HR_comma_sep.csv'
path_2 = r'/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Logistic_Regression/Sample_Data/insurance_data.csv'

df_1 = pd.read_csv(path_1)
df_2 = pd.read_csv(path_2)

x_train,x_test,y_train,y_test = train_test_split(df_2[['age']],df_2[['bought_insurance']],test_size = 0.3)

reg = LogisticRegression()
reg = reg.fit(x_train,y_train)
predicted = reg.predict(x_test)
# print(predicted)
# print(y_test)
print(reg.score(x_train,y_train)*100)

