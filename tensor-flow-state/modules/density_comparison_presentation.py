# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:38:28 2020

@author: peterpiontek
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/final_data.csv", index_col = 0, parse_dates = True)

df = df[(df.index.year > 2017) & (df.index.year < 2019)]

df = df[["speed", "flow"]]
df["density"] =  (df.flow * 60) / df.speed
df2 = df.resample("W").agg({"speed": np.mean, "flow": np.sum, "density": np.mean})[:-1]

df2.speed.mean()
df2.flow.mean()

df2.flow.sort_values()
df2.speed.sort_values()

x = df2.index
y1 = df2.speed
y2 = df2.flow
y3 = df2.density

# create plot
fig, ax1 = plt.subplots()

ax1.set_title("Speed and flow, 2018")
ax1.plot(x, y2, color = 'r', label = 'flow')
ax1.set_ylabel("flow (no. of vehicles)")

ax2 = ax1.twinx()
ax2.plot(x, y1, 'b-', label = 'speed')
ax2.set_ylabel("speed (kph)")


h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = ax.twinx()
par2 = ax.twinx()
par2.spines["right"].set_position(("axes", 1.2))
make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)

p1, = ax.plot(x, y2, color = 'r', alpha = 0.3, label = 'Flow')
p2, = par1.plot(x, y1, 'b-', alpha = 0.3, label = 'Speed')
p3, = par2.plot(x, y3, 'g-', label = "Density")


ax.set_xlabel("Time")
ax.set_ylabel("Flow")
par1.set_ylabel("Speed")
par2.set_ylabel("Density")

ax.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

ax.legend(lines, [l.get_label() for l in lines])

plt.show()