from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

Training_data = r'/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Linerar_Regression/Sample_Data/canada_per_capita_income.csv'

def linear_reg(prediction):
    if isinstance(prediction,int)==True:
        df = pd.read_csv(Training_data)
        reg = linear_model.LinearRegression()
        reg = reg.fit(df[['year']],df.price)
        predicted  = reg.predict([[prediction]])
        return  predicted
    else:
        print('Please proviide valid number')
 
if __name__='__main__':
    predicted = linear_reg(2020)
    print(predicted[0])
                    
