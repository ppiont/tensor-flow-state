# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:10:16 2019

@author: peterpiontek
"""
## fix this trainwreck

import pandas as pd
import matplotlib.pyplot as plt

def set_all():
    dir_settings()
    pandas_settings()
    mpl_settings()

datadir = ''
plotdir = ''
def dir_settings():
    global datadir
    global plotdir
    # Define directories
    datadir = "./data/"
    plotdir = './plots/'

def pandas_settings():
    # Set pandas options
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 200) # Accepted line width before wrapping
    pd.set_option('display.max_colwidth', -1)
    pd.options.display.float_format = '{:.2f}'.format

def mpl_settings():
    # Set mpl options
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['image.cmap'] = 'viridis'


    