#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:18:06 2020

@author: peter
"""
# Imports plt + pd, and also changes settings for those libs
from modules.Settings import * 
import numpy as np
sensor = pd.read_csv("data/RWS01_MONIBAS_0021hrl0414ra_speed_flow.csv")



sensor.loc[sensor.flow < 0] = np.nan
sensor.loc[sensor.speed == -2] = np.nan

## gonna set speed at 0 flow to speed limit
sensor.loc[sensor.speed == -1] = #120 or 100
# test2