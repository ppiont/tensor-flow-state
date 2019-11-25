# import necessary libs
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point #, Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
# from shapely.wkt import loads
import numpy as np
import owslib
import seaborn as sns

# set working dir
os.chdir("C:/Users/peterpiontek/Google Drive/tensor-flow-state/tensor-flow-state")

# define directories
datadir = "./data/"
plotdir = './plots/'

# import homebrew
from modules.ndw_get import ndw_get

one_north,two_north = 'RWS01_MONIBAS_0021hrl0339ra','RWS01_MONIBAS_0021hrl0336ra'
from_date, to_date = '06-01-2019','06-07-2019'

# fetch traffic data
sensor1_df = ndwGet(from_date, to_date, one_north)[['datetime','date','flow','speed','weekday','weekend','holiday','sensor_id','lon', 'lat']]
sensor2_df = ndwGet(from_date, to_date, two_north)[['datetime','date','flow','speed','weekday','weekend','holiday','sensor_id','lon', 'lat']]

# normalize flow col
sensor1_df['flow'] = sensor1_df['flow']/60
sensor2_df['flow'] = sensor2_df['flow']/60

print(sensor1_df.groupby(['date'], as_index=False)['flow'].mean())

print(38.03 * 60 * 24)

diff_df = pd.DataFrame()
diff_df['date'] = sensor1_df.date.unique()
diff_df['weekend'] = sensor1_df.groupby(['date'])['weekend'].max().ravel()
diff_df['holiday'] = sensor1_df.groupby(['date'])['holiday'].max().ravel()
diff_df['flow_diff'] = sensor2_df.groupby(['date'])['flow'].sum().ravel() - sensor1_df.groupby(['date'])['flow'].sum().ravel()
diff_df['speed_avg'] = ((sensor2_df.groupby(['date'])['speed'].mean().ravel()) + (sensor1_df.groupby(['date'])['speed'].mean().ravel()))/2
# diff_df['speed_avg'] = statistics.mean(((sensor2_df.groupby(['date'])['speed'].mean()), (sensor1_df.groupby(['date'])['speed'].mean())))


# calculate correlation for between flow difference and the average of the two average speeds :^)
pearson_correlation_coef = ((np.cov(diff_df.flow_diff, diff_df.speed_avg))[0,1] / np.std(diff_df.flow_diff) * np.std(diff_df.speed_avg))


##############################################################################
#################################### PLOT ####################################
# set figure dpi
mpl.rcParams['figure.dpi']= 300

# set seaborn format and style
sns.set(context='paper', style='white')

# define x, y1, y2 plot data
x, y1, y2 = diff_df.date,  diff_df.flow_diff, diff_df.speed_avg

# create figure, ax
fig, ax1 = plt.subplots()
# set title
ax1.set_title("Daily measurement discrepancies Jun-Aug '19\nbetween 2 adjacent NDW sensors on A2 north by ArenA")
ax1.plot(x, y1, 'b-', label = r'$volume\; divergence\;$')
ax1.set_ylabel('$\Delta\; Volume\;  [V_2 - V_1]$', color='b')
ax1.set_ylim([-4200,6000])
ax1.axvline(x='2019-06-24', color = 'y', lw = '12', alpha = 0.3, zorder=1)
ax1.axvline(x='2019-06-01', color = 'y', lw = '4', alpha = 0.3, zorder=1)
ax1.axvline(x='2019-07-11', color = 'y', lw = '5', alpha = 0.3, zorder=1)
ax1.axvline(x='2019-08-31', color = 'y', lw = '3', alpha = 0.3, zorder=1)
# ax.axhline(y=1200, color='b', lw=2, alpha=0.5, zorder=10)
ax1.axhline(y=0, color='black', lw=1, alpha=1, zorder=0)
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
for tick in ax1.get_yticklabels():
    tick.set_color('b')
# create second ax in same fig, sharing x-axis with ax1
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-', label = r'$mean\; velocity$')
ax2.set_ylabel(r'$Velocity\; [\frac{\bar v_1 + \bar v_2}{2}]$', color='r')
ax2.set_ylim([-84, 120])
for tl in ax2.get_yticklabels():
    tl.set_color('r')
# ax2.text('2019-08-29', -62, f'Mean flow difference: {diff_df.flow_diff.mean():.2f}',
#         horizontalalignment='right',
#         verticalalignment='bottom', color='b', fontsize=8)
# ax2.text('2019-08-29', -72, f'Mean speed: {diff_df.speed_avg.mean():.2f}',
#         horizontalalignment='right',
#         verticalalignment='bottom', color='r', fontsize=8)
ax2.text('2019-08-30', -82, f'Correlation: {pearson_correlation_coef:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom', fontsize=9)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=3)


# save figure in plot directory, defining dpi and bbox (tight avoids cropping)
fig.savefig(plotdir + "diff_plot.png", dpi=600, bbox_inches = 'tight')

#################################### PLOT ###################################
#############################################################################


# for x in sensor2_df.groupby(['date'])['flow'].sum().ravel():
#     print(f"{x:f}")
















# wgs84 = {'init': 'epsg:4326'}
# wgs84utm31n = {'init': 'epsg:32631'}
# rdnew = {'init': 'epsg:28992'}

# # create GeoDataFrames initialized with crs=WGS84 and sensor data + geom points
# geom1 = [Point(xy) for xy in zip(sensor1_df.X, sensor1_df.Y)]
# gdf1 = gpd.GeoDataFrame(sensor1_df, geometry=geom1, crs=wgs84)
# geom2 = [Point(xy) for xy in zip(sensor2_df.X, sensor2_df.Y)]
# gdf2 = gpd.GeoDataFrame(sensor1_df, geometry=geom2, crs=wgs84)

# print(gdf1.crs)
# gdf1.to_crs(wgs84utm31n, inplace=True)
# gdf2.to_crs(wgs84utm31n, inplace=True)


# gdf1.plot(marker='*', color='green', markersize=50)


# gdf1.to_file(filename='./data/wageningenPOI.geojson', driver='GeoJSON')



# # l=gdf1.distance(gdf.shift())




# from owslib.wms import WebMapService
# wmsUrl = 'https://www.osm-wms.de/'
# basemap = WebMapService(url=wmsUrl, version='1.1.1')
# basemap.identification.title
# print(list(basemap.contents))








# for x, y in zip(gdf1['X'], gdf1['Y']):
#     folium.CircleMarker(
#         [x, y],
#         radius=5,
#         # popup = ('City: ' + str(city).capitalize() + '<br>'
#         #          'Bike score: ' + str(bike) + '<br>'
#         #          'Traffic level: ' + str(traffic) +'%'
#         #         ),
#         color='b',
#         # key_on = traffic_q,
#         threshold_scale=[0,1,2,3],
#         # fill_color=colordict[traffic_q],
#         fill=True,
#         fill_opacity=0.7
#         ).add_to(m)
# m



# # 52.354559, 4.896297 ## sarphati

# # point = Point(52.354559, 4.89629)



# points = [[float(sensor1_df.X.unique()), float(sensor1_df.Y.unique())],[float(sensor2_df.X.unique()), float(sensor2_df.Y.unique())]]


# import folium
# m = folium.Map(location=[52.354559, 4.896297], zoom_start = 13)
# folium.Marker([52.30576, 4.9306], popup='Sensor 1').add_to(m)
# folium.Marker([52.30772, 4.92854], popup='Sensor 2').add_to(m)


# for point in range(0, len(points)):
#     folium.Marker(points[point]).add_to(m)
# m.save('foliumtest1.html')

# # how the pyproj error was fixed. Normal directory is
# #'~\AppData\Local\Continuum\anaconda3\envs\{ENV NAME}\Lib\site-packages\pyproj\proj_dir\share\proj'
# # i think. For some reason pyproj hasn't fixed this error in months, and it is in the wrong dir, which
# # i changed the env path to below
# # import os
# # os.environ['PROJ_LIB'] =r'C:\Users\peterpiontek\AppData\Local\Continuum\anaconda3\envs\test\Library\share'; <-------
# # os.environ['PROJ_LIB'] =r'C:\Users\peterpiontek\AppData\Local\Continuum\anaconda3\envs\test\Lib\site-packages\pyproj\proj-dir\share\proj';
# # os.environ.get('PROJ_LIB')
# # import pyproj, os.path
# # os.path.exists(pyproj.pyproj_datadir + '/epsg')




# ## test
# # array = np.ones((2,2))
# # df = pd.DataFrame(array)
# # df.drop(['geom'], axis=1, inplace=True)
# # df.columns = [['test1', 'test2']]
# # df['X'] = 4.896297
# # df['Y'] = 52.354559
# # geom = [Point(4.896297, 52.354559), Point(4.896297, 52.354559)]

# # testlist =  zip(df['X'], df['Y'])

# # df.dtypes
# # gdf = gpd.GeoDataFrame(array, geometry=geom, crs={'init': 'epsg:4326'})

# # gdf.to_crs(epsg=32631, inplace=True)