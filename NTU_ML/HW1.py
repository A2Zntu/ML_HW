# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:54:35 2019

@author: Evan
"""

import numpy as np
import math 
import pandas as pd 

df = pd.read_csv(r"C:\Users\Evan\Documents\GitHub\ML_HW\data\train.csv", encoding = "cp950")


length = len(df)

feature_num = 18
months_num = 12
days_num = 20
hours_num = 24
rolling_Days = 9

one_month_hours = days_num*hours_num #480
one_rolling_period_times = one_month_hours - rolling_Days #471
feature_name = df["測項"][0:feature_num].tolist()


dic_df = {}
list_df = []

for i in range(length):
    dic_key = df["測項"][i]
    if dic_key not in dic_df:
        dic_df[dic_key] = len(dic_df)
        list_df.append(list(df.iloc[i, 2:]))  
    else:
        temp = list(df.iloc[i, 3:])
        list_df[dic_df[dic_key]].extend(temp)
        
df1 = pd.DataFrame(list_df)
col_names = list(np.arange(df1.shape[1]))
col_names.insert(0, "feature")
col_names.pop(-1)
df1.columns = col_names
df1 = df1.set_index('feature')
 #rows are feature, columns are days

def flatten_df(df):
    list_flat = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                list_flat.append(float(df.iloc[i, j]))
            except ValueError:
                list_flat.append(np.NAN)
    return list_flat

count = 0    
df2 = pd.DataFrame()
for j in range(months_num): 
    print(count)
    for i in range(one_rolling_period_times): 
        first = j*one_month_hours + i
        last = first + 9
        capture_feat = df1.iloc[:, first:last] #18*9
        cf = flatten_df(capture_feat)
        df2[str(count)] = cf
        count = count + 1
        #the final hours in one month are not recorded because they are for predicted
        
        



    
        
        
        
        
    

    