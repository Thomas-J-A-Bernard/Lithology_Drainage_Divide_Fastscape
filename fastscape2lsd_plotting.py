"""
Created on Wed Feb 12 12:04:03 2020
@author: thomas
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import lsdtopytools as lsd

path = '/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/lsdtopytools_model_ouputs/'
DataFrame = [pd.DataFrame() for i in range(330)]
for i in range(100, 330):
    DataFrame[i] = pd.read_csv(path + 'model_export_time_' + str(i*1e5) + '_theta_0.4.csv')

#%%=========================================================================%%#
#--------------------- KSN VS ELEVATION VS FLOW DISTANCE PLOT ----------------#

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(7.125, 6))

time_list = [147, 150, 153, 156, 159]
i = 0
for ax in axes.flat:
    
    print(i)
    df_b175 = DataFrame[time_list[i]][DataFrame[time_list[i]]['basin_key']==168]
    df_b38 = DataFrame[time_list[i]][DataFrame[time_list[i]]['basin_key']==30]
    fd_tot = max(df_b175['flow_distance']) + max(df_b38['flow_distance'])
    df_b38['flow_distance'] = fd_tot - df_b38['flow_distance']
    df_custom = pd.concat([df_b175, df_b38])
    
    s = ax.scatter(x=df_custom['flow_distance'], y=df_custom['elevation'], c=df_custom['k_sn'], vmin=0, vmax=125, cmap='jet')
    t = ax.text(0.025, 0.870, s='time: '+str(time_list[i]/10)+' Ma', bbox=dict(boxstyle='square', ec='k', fc='white'), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    
    ax.axis([0, fd_tot, ax.axis()[2], ax.axis()[3]])
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
    ax.xaxis.set_major_formatter(ticks_x)
    
    if i == 2:
        ax.set_ylabel('Elevation (m)')
    if i == 4:
        ax.set_xlabel('Flow distance (km)')
    
    i+=1

cax = fig.add_axes([0.875, 0.08, 0.02, 0.9])
fig.colorbar(s, label='k$_{sn}$', cax=cax)
fig.subplots_adjust(top=0.98, bottom=0.08, right=0.84, hspace=0.33)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/ksn_flow-distance_elevation_evolution_1.png', dpi=720)

#%%=========================================================================%%#
#---------------------------- KSN + HILLSHADE PLOT----------- ----------------#

df_tifs = pd.read_csv('/home/thomas/Documents/research/fastscape/script/' + "mountain_model_drainage_divide_migration_alltif.csv")

time = 160
file_name = df_tifs['name_of_file'].iloc[time]
path = df_tifs['path'].iloc[time]

# loading and preparing DEM
my_dem = lsd.LSDDEM(path='/home/thomas/Documents/research/fastscape/script/'+path, file_name=file_name)
my_dem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001)
my_dem.CommonFlowRoutines()
my_dem.save_array_to_raster_extent(my_dem.cppdem.get_DA_raster(), name = "drainage_area")
my_dem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 5)
my_dem.DefineCatchment(method="from_range", test_edges = False, min_area = 1e8, max_area = 1e24)

# get DEM hillshade
hillshade = my_dem.get_hillshade(); hillshade[hillshade == -9999] = np.nan
hillshade = hillshade[:, ~np.isnan(hillshade).all(0)]
hillshade = hillshade[~np.isnan(hillshade).all(1)]

# get lsdtopotool dataframe
df = DataFrame[time]

# get catchment outlines
outlines = my_dem.cppdem.get_catchment_perimeter()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.0, 3.0))

im = ax.imshow(hillshade, extent=my_dem.extent, cmap='gray')
sc = ax.scatter(x=df['x'], y=df['y'], c=df['k_sn'], s=2.5, cmap='jet', vmin=0, vmax=120)
for key, val in outlines.items():
    ax.scatter(val["x"], val["y"], c='k', s=0.5)

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
ax.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*1e-3))
ax.yaxis.set_major_formatter(ticks_y)

ax.axis(my_dem.extent)
ax.set_xlabel('Distance (km)'); ax.set_ylabel('Distance (km)')
ax.set_title('time: ' + str(time/10) + 'Ma')

cax = fig.add_axes([0.875, 0.23, 0.015, 0.67])
fig.colorbar(sc, label='k$_{sn}$', cax=cax)
fig.subplots_adjust(left=0.1, right=0.85, bottom=0.125, top=1)

# plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/model_export_figure/ksn+hillshade_plot_t00'+str(i)+'.png', dpi=360)

#%%=========================================================================%%#
#----------------------------- PUBLICATION FIGURE 3 --------------------------#
font = {'family':'arial', 'size':8}
matplotlib.rc('font',**font)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.125, 4.0))

time_list = [149, 160, 175, 200, 225, 299]
i = 0
for ax in axes.flat:
    
    print(i)
    df_b175 = DataFrame[time_list[i]][DataFrame[time_list[i]]['basin_key']==168]
    df_b38 = DataFrame[time_list[i]][DataFrame[time_list[i]]['basin_key']==30]
    fd_tot = max(df_b175['flow_distance']) + max(df_b38['flow_distance'])
    df_b38['flow_distance'] = fd_tot - df_b38['flow_distance']
    df_custom = pd.concat([df_b175, df_b38])
    
    s = ax.scatter(x=df_custom['flow_distance'], y=df_custom['elevation'], c=df_custom['k_sn'], s=10, vmin=0, vmax=125, cmap='jet', zorder=2)
    t = ax.text(0.035, 0.870, s='time: '+str(time_list[i]/10)+' Myrs', bbox=dict(boxstyle='square', ec='k', fc='white', lw=0.75), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, zorder=3)
    
    Rec = patches.Rectangle((65000, -1000), 35000, 3500, ls='--', lw=1, ec='k', fc='#D3E7FA', alpha=1, zorder=1)
    ax.add_patch(Rec)
    
    ax.axis([0, fd_tot, np.min(df_custom['elevation'])-50, np.max(df_custom['elevation'])+50])
    loc = ticker.MultipleLocator(base=500.0)
    ax.yaxis.set_major_locator(loc)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*1e-3))
    ax.xaxis.set_major_formatter(ticks_x)
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*1e-3))
    ax.yaxis.set_major_formatter(ticks_y)
    
    if i in [0,2,4]:
        ax.set_ylabel('Elevation (m)')
    if i in [4,5]:
        ax.set_xlabel('Flow distance (km)')
    
    i+=1

cax = fig.add_axes([0.075, 0.115, 0.91, 0.025])
fig.colorbar(s, label='k$_{sn}$', orientation='horizontal', cax=cax)
fig.subplots_adjust(top=0.975, bottom=0.25, left=0.075, right=0.985, wspace=0.15, hspace=0.3)
plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE3_ksn_plot_vs_diastance+elevation.png', dpi=1200)
plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE3_ksn_plot_vs_diastance+elevation.pdf', dpi=1200)