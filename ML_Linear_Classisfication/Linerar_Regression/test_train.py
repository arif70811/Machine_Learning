from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

path = '/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Linerar_Regression/Sample_Data/carprices.csv'

def test_train():

    df = pd.read_csv(path)
    x = df[['Mileage','Age(yrs)']]
    y = df['Sell Price($)']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    reg = LinearRegression()
    reg = reg.fit(x_train,y_train)
    predicted_value = reg.predict(x_test)
    return predicted_value


if __name__=='__main__':
    predicted_value = test_train()
    print(predicted_value)