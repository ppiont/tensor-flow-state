# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:40:28 2020

@author: peterpiontek
"""
import os
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', -1)

df = pd.read_csv("ndw_raw/RWS01_MONIBAS_0021hrl0403ra.csv", header = None)

df.head()

df2 = df.dropna(axis = 1, how = "all")

df2.head(200)

df3 = df2.iloc[:, 2:7]

df3['mean'] = df3.mean(axis=1)

len(df3[(df3['mean'] < -1) & (df3['mean'] > -2)]) / len(df3) * 100

rows_nan = df3.iloc[:, 0:5].min(axis = 1)
rows_nan = rows_nan < -1

df4 = df3[rows_nan]

df4.head(50)

rows_mean = df4.iloc[:, 0:5].mean(axis = 1)
rows_mean = rows_mean != -2.0

df5 = df4[rows_mean]
df5.head()

len(df5) / len(df3) * 100

df5.columns = ["lane 1", "lane 2", "lane 3", "lane 4", "lane 5", "average"]
df5.head(100)
