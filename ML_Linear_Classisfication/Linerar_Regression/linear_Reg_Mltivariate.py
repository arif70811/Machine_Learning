from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n
import pandas as pd
import numpy as np
import csv

Training_data = r'/media/arif/EE8E4F908E4F4FEF/Projects/Data_Science_Practice/Linerar_Regression/Sample_Data/hiring.csv'

def linear_Reg_Mltivariate(a,b,c):
    df = pd.read_csv(Training_data)
    df.experience = df.experience.fillna("zero")
    df.experience = df.experience.apply(w2n.word_to_num)
    mean = round(df['test_score(out of 10)'].sum()/len(df['test_score(out of 10)']))
    df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(mean)
    print(df.columns)
    reg = linear_model.LinearRegression()
    reg = reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],df['salary($)'])
    predicted  = reg.predict([[a,b,c]])
    return predicted

if __name__='__main__':
    predicted = linear_Reg_Mltivariate(2,9,6)
    print(predicted[0])
