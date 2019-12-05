# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:10:16 2019

@author: peterpiontek
"""
## fix this trainwreck

import pandas as pd
print("pandas imported as pd")
import matplotlib.pyplot as plt
print("matplotlib.pyplot imported as plt")

# Set pandas options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200) # Accepted line width before wrapping
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:.2f}'.format
print("pandas options set:\n\
      display.max_columns = 500\n\
      display.max_rows = 500\n\
      display.width = 200\n\
      display.max_colwidth = -1\n\
      display.float_format = :.2f")

# Set matplotlib options
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['image.cmap'] = 'viridis'
print("matplotlib options set:\n\
      figure.figsize = (16, 9)\n\
      figure.dpi = 300\n\
      savefig.dpi= 600\n\
      image.cmap = 'viridis'")