# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:44:20 2019

@author: peterpiontek
"""

import numpy as np

# Generates sequential 3D batches to feed to the model
def generator(data, lookback, delay, min_index = 0, max_index = None, 
              shuffle = False, batch_size = 128, step = 1, target_col = 0):
    # If max index not given, subtract prediction horizon - 1 (len to index) from last data point
    if max_index is None:
        max_index = len(data) - delay - 1
    # Set i to first idx with valid lookback length behind it
    i = min_index + lookback
    while 1:
        # Use shuffle for non-sequential data
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size = batch_size)
        # Else for sequential (time series)
        else:
            # Check if adding batch exceeds index bounds
            if i + batch_size >= max_index:
                # Return i to beginning
                i = min_index + lookback
            # Select next valid row range
            rows = np.arange(i, min(i + batch_size, max_index))
            # Increment i
            i += len(rows)
        # Initialize sample and target arrays
        samples = np.zeros((len(rows),
                            lookback // step,
                            np.shape(data)[-1]))
        targets = np.zeros((len(rows),))
        # Generate samples, targets
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][target_col]
        yield samples, targets