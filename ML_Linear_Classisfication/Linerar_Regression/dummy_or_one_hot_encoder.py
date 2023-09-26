from sklearn import linear_model
import pandas as pd
import csv
import os

path = '/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Linerar_Regression/Sample_Data/carprices.csv'

def Lenear_Reg_Dummies(milage,age,BMW_X5,Mercedez_Benz):
    df = pd.read_csv(path)
    x=pd.get_dummies(df, prefix=['Car Model'], drop_first=True)
    x.drop(['Sell Price($)'],axis = 'columns',inplace = True )
    y=df['Sell Price($)']
    reg = linear_model.LinearRegression()
    reg = reg.fit(x,y)
    prediction = reg.predict([[milage,age,BMW_X5,Mercedez_Benz]])
    return prediction

if __name__=='__main__':
    result = Lenear_Reg_Dummies(72000,6,0,0)
    print(result)