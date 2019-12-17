# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:13:45 2019

@author: peterpiontek
"""
import os
import pandas as pd

os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

from modules.pg_connect import pg_connect

connection = pg_connect(host = 'leda.geodan.nl', database = 'ndw')
cursor = connection.cursor()

# Define query
query = "select distinct dgl_loc from telpunten"
        
# Execute query
cursor.execute(query)
df = pd.DataFrame(cursor.fetchall(), columns = ['id'])

print(df.head())

# Set pandas and matplotlib settings
exec(open('modules/Settings.py').read())

mapper = pd.read_csv('data/index_mapping.csv', header = 0)

flow = mapper[mapper.measurement_type == 'trafficFlow']['ndw_index'].to_list()

import re

