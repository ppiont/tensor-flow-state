# Import libraries
import os
import pandas as pd
from datetime import datetime as dt

# Set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# Define directories
datadir = "./data/"
plotdir = './plots/'

# Don't limit, truncate or wrap columns displayed by pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
# Display decimals instead of scientific with pandas
pd.options.display.float_format = '{:.2f}'.format





df = pd.read_pickle(datadir + 'RWS01_MONIBAS_0021hrl0414ra_jun_oct.pkl')

# # Create train and test df
# train_data = df[df['date'] < '2019-11']
# test_data = df[df['date'] > '2019-08']

# print(train_data['date'].unique())


