# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:19:15 2019

@author: peterpiontek
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Set pandas and matplotlib settings
exec(open('modules/Settings.py').read())

# Define directories
datadir = "./data/"
plotdir = './plots/'


waze = pd.read_csv('data/A2_waze.csv')

waze.info()

waze.head()
waze.columns
waze.describe
waze.info()
waze['uuid'].nunique()
waze.city.unique()

points = waze[['lat', 'lon']].drop_duplicates() # remove duplicates\
points = points.dropna()
points = points.loc[(points!=0).any(axis=1)] # remove 0

points.size

sensor = pd.DataFrame({'lat': [52.2435], 'lon': [4.9726]})

plt.scatter(points['lon'], points['lat'])
plt.scatter(sensor['lon'], sensor['lat'])




