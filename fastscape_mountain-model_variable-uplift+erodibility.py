"""
Created on Wed Jan  8 10:01:40 2020
@author: Thomas Bernard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xsimlab
import xarray
import fastscape
import fastscape_plotting_functions
import datetime
import sys
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from fastscape2lsd import export_times_to_tif, fix_chi_and_recalculate_ksn
import pandas as pd
import lsdtopytools as lsd
from pathlib import Path

print('xarray version: ', xarray.__version__)
print('xarray-simlab version: ', xsimlab.__version__)
print('fastscape version: ', fastscape.__version__)

#%%=========================================================================%%#
#------------------ IMPORT MODELS AND PROCESSES FROM FASTSCAPE ---------------#

from fastscape.models import basic_model
from fastscape.processes import TotalErosion
from fastscape_custom_functions import (SquareBasementIntrusion, CircleBasementIntrusion, VariableUplift)

#%%=========================================================================%%#
#------------------------ SETUP CONDITIONS FOR THE MODEL ---------------------#

tstart = 0                                          # model time start
tt = 33e6                                           # model total duration
dt = 1e5                                            # model timestep duration
nt = int(tt/dt)                                     # number of iteration through the model
time = np.arange(tstart, tt, dt)

nx = 301; ny = 101                                  # grid shape of the model
lenghtx = 4.5e5; lenghty = 1.5e5                    # total lenght of the model
spacex = lenghtx/(nx-1); spacey = lenghty/(ny-1)    # grid node distance           

T1 = 15.0e6; T1_time_step = int(round(T1/dt))
T2 = 30.0e6; T2_time_step = int(round(T2/dt))
T3 = tt; T3_time_step = int(round(T3/dt))

U = np.zeros(nt)
U[tstart:T2_time_step] = 1e-3
U[T2_time_step:T3_time_step] = 0e-3

K_basement = np.zeros(nt)
K_basement[tstart:T1_time_step] = 2e-6
K_basement[T1_time_step:T3_time_step] = 0.5e-6

#%%=========================================================================%%#
#-------------------------------- INITIALIZE MODEL ---------------------------#

custom_model = basic_model.update_processes({'square_basement': SquareBasementIntrusion, 'uplift_func': VariableUplift})

in_ds = xsimlab.create_setup(
    model=custom_model,
    clocks={
        'time': time,
        'out': time,
    },
    master_clock='time',
    input_vars={
        'grid__shape': ('shape_yx', [ny, nx]),
        'grid__length': ('shape_yx', [lenghty, lenghtx]),
        'boundary__status': ('border', ['fixed_value', 'fixed_value', 'fixed_value', 'fixed_value']),        
        'uplift_func': {
            'uplift_rate': ('time', U),
            'coeff': 1,
            'axis': 2
        },
        'square_basement': {
            'x_origin_position': 80,
            'y_origin_position': 45,
            'x_lenght': 100,
            'y_lenght': 20,
            'basement_k_coef': ('time', K_basement),
            'rock_k_coef': 2e-6,
            'basement_diff': 0.1,
            'rock_diff': 0.1
        },
#        'circle_basement': {
#                'x_origin_position': 100,
#                'y_origin_position': 60,
#                'radius': 25,
#                'basement_k_coef': ('time', K_basement),
#                'rock_k_coef': 2e-6,
#                'basement_diff': 0.1,
#                'rock_diff': 0.1
#        },
        'spl': {
            'area_exp': 0.6,
            'slope_exp': 1.5
        },
    },
    output_vars={
        'out': ['topography__elevation',
                'drainage__area',
                'flow__basin',
                'terrain__slope',
                'spl__erosion'],
        None: ['boundary__border',
               'grid__x',
               'grid__y',
               'grid__spacing'],
    }
)

#%%=========================================================================%%#
#--------------------------------- RUN MODEL ---------------------------------#

start_time = datetime.datetime.now()
print('running model ...')
out_ds = in_ds.xsimlab.run(model=custom_model)
out_ds = out_ds.set_index(x='grid__x', y='grid__y')
end_time = datetime.datetime.now()
print('Model duration time: {}'.format(end_time-start_time))

sys.exit('...')

#%%=========================================================================%%#
#-------------------- SAVING MODEL OUTPUTS AS TIF ----------------------------#

print("saving outputs as tif ...")
prefix = "mountain_model_drainage_divide_migration"
directory = "mountain_model_drainage_divide_migration/"
export_times_to_tif(out_ds, directory = directory, prefix = prefix, epsg_code = 32635, resolution = spacex, X_min = 0, Y_min = 0, time_list = time)

#sys.exit('...')

#%%=========================================================================%%#
#-------------------------- TOPOGRAPHIC ANALYSES -----------------------------#
plt.ioff()
print('runing topographic analyses ...')

# Contains all the file of the analysis
df_tifs = pd.read_csv(prefix + "_alltif.csv")
theta_m_over_n = 0.4
minimum_time = 10e6

Path("./" + directory + "/%s_figure"%(prefix)).mkdir(parents=True, exist_ok=True)

# Iterate through all the files (could be done manually)
name_cpt = 0
for i in range(df_tifs.shape[0]):
    # name and path
    this_dem = df_tifs["name_of_file"].iloc[i]
    path = df_tifs["path"].iloc[i]
    # this skips while the time is not above a certain threshold (otherwise lsdtopytools crashes when the basin are too small with the random noise at the start)
    if(df_tifs["time"].iloc[i] < minimum_time):
        continue
    # quit()

    # Loading the DEM and creating a LSDDEM object
    mydem = lsd.LSDDEM(file_name = this_dem, path = path)
    # OPTIONAL preprocessing: fastscape has a slightly different flow routing, here you need to breach the depressions
    mydem.PreProcessing(filling = True, carving = True, minimum_slope_for_filling = 0.0001) 
    #Need to pregenerate a number of routines, it calculates flow direction, flow accumulation, drainage area , ...
    mydem.CommonFlowRoutines()
    A = mydem.cppdem.get_DA_raster()
    mydem.save_array_to_raster_extent( A, name = "drainage_area")
    # Here you will want to adapt the threshold (number of pixels): smaller = more rivers, bigger = less rivers
    mydem.ExtractRiverNetwork( method = "area_threshold", area_threshold_min = 5)
    # Now you need to define the catchments, this can be done with several methods, I prefer the range one, it;s probably the most accurate:
    mydem.DefineCatchment(method="from_range", test_edges = False, min_area = 1e7, max_area = 1e24)

    # Calculates chi coordinate with an according theta
    mydem.GenerateChi(theta = theta_m_over_n, A_0 = 1)
    df = mydem.df_base_river
    rec_df = pd.DataFrame(mydem.cppdem.get_receiver_data(df["x"].values, df["y"].values))
    chi = mydem.cppdem.get_chi_raster_all(theta_m_over_n, 1, 0)
    fix_chi_and_recalculate_ksn(df, rec_df, chi, theta = 0.4, A0 = 1)
    df.to_csv(directory + "lsdtopytools_model_ouputs/" + prefix + "_time_" + str(df_tifs["time"].iloc[i]) + "_theta_" + str(theta_m_over_n) + ".csv", index = False)

    fig, ax = lsd.quickplot.get_basemap(mydem , figsize = (6,6), cmap = "gist_earth", hillshade = True, 
                                        alpha_hillshade = 1, cmin = None, cmax = None,hillshade_cmin = None, hillshade_cmax = None, colorbar = False, 
                                        fig = None, ax = None, colorbar_label = None, colorbar_ax = None, fontsize_ticks = 8)
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    cb = ax.scatter(df["x"],df["y"], c = df["k_sn"], s = 5, lw = 0, zorder = 5, cmap = "magma", vmin = df["k_sn"].quantile(0.05), vmax = df["k_sn"].quantile(0.95))
    plt.colorbar(cb, label = r"$k_{sn}$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    name = str(name_cpt)
    while(len(name)<4):
        name = "0"+name

    plt.tight_layout()
    plt.savefig("./" + directory + "/%s_figure/"%(prefix) + prefix + name + "_" + str(df_tifs["time"].iloc[i]) + ".png" )
    plt.close(fig)
    name_cpt += 1

sys.exit('...')

#%%=========================================================================%%#
#-------------------------------- MODEL OUTPUTS ------------------------------#
plt.ion()

#fastscape_plotting_functions.topography_elevation_plot(out_ds, 180, dt, nx, ny, lenghtx, lenghty, (9, 3), 'gist_earth', 0.99, 15, "test", "test", False, 360)
#fastscape_plotting_functions.topography_swathprofile_x_plot(out_ds, 300, 1e5, nx, ny, lenghty, spacex, 110, 150, (4.5, 3), "/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/", "transversal_swath-profile_t30Ma.png", True, 360)
#fastscape_plotting_functions.topography_swathprofile_y_plot(out_ds, 120, 1e5, nx, ny, lenghtx, spacey, 34, 42, (4.5, 3), "/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/", "longitudinal_swath-profile_t8Ma.png", False, 360)

#%%=========================================================================%%#
#---------- PLOT TRANSVERSALE SLOPE AND ELEVATION THROUGH KEY TIMES ----------#

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,5))

time_list = [15e6, 17e6, 18e6, 19e6, 22e6, 30e6]
topo = [np.empty(()) for i in range(6)]
slope = [np.empty(()) for i in range(6)]

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(ny-1)*lenghty*1e-3))
    
x = np.linspace(0, ny-1, ny)
for i in range(0,6):
    topo[i] = out_ds.sel(out=time_list[i]).topography__elevation[:,110:150].mean(axis=1)
    slope[i] = out_ds.sel(out=time_list[i]).terrain__slope[:,110:150].mean(axis=1)

j=0    
for ax in axes.flat:
    s1 = ax.scatter(x=x, y=topo[j], c=slope[j], vmin=0, vmax=8, cmap='rainbow', zorder=2)
    ax.text(0.96, 0.9, s='time: '+str(time_list[j]*1e-6)+' Ma', bbox=dict(boxstyle='square', ec='k', fc='white'), verticalalignment='top', horizontalalignment='right', transform=ax.transAxes)
    ax.xaxis.grid(which='major', linestyle='--', color='darkred', alpha=0.33, zorder=1); 
    ax.yaxis.grid(which='major', linestyle='--', color='darkred', alpha=0.33, zorder=1)
    ax.xaxis.set_major_formatter(ticks_x)
    if j in [0,2,4]:
        ax.set_ylabel('Elevation (m)')
    if j in [4, 5]:
        ax.set_xlabel('Distance (km)')
    j+=1

cax = fig.add_axes([0.925, 0.11, 0.02, 0.865])
fig.colorbar(s1, label='Slope (°)', cax=cax)
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.975)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/slope+elevation_evolution.png', dpi=360)

#%%=========================================================================%%#
#----------- PLOT LONGITUDINAL SLOPE AND ELEVATION THROUGH KEY TIMES ---------#

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,5))

time_list = [15e6, 17e6, 19e6, 21e6, 23e6, 30e6]
topo = [np.empty(()) for i in range(6)]
slope = [np.empty(()) for i in range(6)]

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
    
x = np.linspace(0, nx-1, nx)
for i in range(0,6):
    topo[i] = out_ds.sel(out=time_list[i]).topography__elevation[34:42,:].mean(axis=0)
    slope[i] = out_ds.sel(out=time_list[i]).terrain__slope[34:42,:].mean(axis=0)

j=0    
for ax in axes.flat:
    s1 = ax.scatter(x=x, y=topo[j], c=slope[j], vmin=0, vmax=8, cmap='rainbow', zorder=2)
    ax.text(0.5, 0.075, s='time: '+str(time_list[j]*1e-6)+' Ma', bbox=dict(boxstyle='square', ec='k', fc='white'), verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.xaxis.grid(which='major', linestyle='--', color='darkred', alpha=0.33, zorder=1)
    ax.yaxis.grid(which='major', linestyle='--', color='darkred', alpha=0.33, zorder=1)
    ax.xaxis.set_major_formatter(ticks_x)
    if j in [0,2,4]:
        ax.set_ylabel('Elevation (m)')
    if j in [4, 5]:
        ax.set_xlabel('Distance (km)')
    j+=1

cax = fig.add_axes([0.925, 0.11, 0.02, 0.865])
fig.colorbar(s1, label='Slope (°)', cax=cax)
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.975)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/longitudinal_slope+elevation_evolution.png', dpi=360)

#%%=========================================================================%%#
#------------------- TRACK THE DRAINAGE DIVIDE THROUGH TIME ------------------#

drainage_position = np.zeros((nx))
drainage_position_time = [np.zeros((nx)) for i in range(33)]

for i in range(0, 33):
    
    basin_index = out_ds.sel(out=(i)*1e6).flow__basin[0,:]
    
    for j in range(0, nx):
    
        basin = out_ds.sel(out=(i)*1e6).flow__basin[:,j]
        mask = np.in1d(basin, basin_index, invert=True)
        drainage_index = np.where(mask == True)[0][0]    
        drainage_position_time[i][j] = drainage_index

#%%=========================================================================%%#
#------------------- PLOT THE DRAINAGE DIVIDE THROUGH TIME -------------------#

color_list = ['black', 'cyan', 'darkorange', 'darkgreen', 'darkred', 'yellow', 'pink', 'mediumblue', 'darkviolet', 'saddlebrown', 'olive', 'magenta', 'grey', 'gold', 'lime', 'steelblue']

fig = plt.figure(figsize = (7.5,3))
ax1 = fig.add_subplot(111)

for i in range(0, 16):
    ax1.plot(drainage_position_time[i+15], '--', color=color_list[i], label='t: '+str(round((i+15)))+' Ma')
    ax1.legend(loc='lower right', ncol=4, fancybox=False, framealpha=1, edgecolor='k')

Rec = patches.Rectangle((80, 45), 100, 20, ls='--', lw=1, ec='k', fc='#D3E7FA', alpha=1)
ax1.add_patch(Rec)

ax1.axis([0, nx, 0, ny])
ax1.invert_yaxis()
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
ax1.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
ax1.yaxis.set_major_formatter(ticks_y)

ax1.set_xlabel('Distance (km)')

fig.tight_layout()
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/drainage_divide_tracking.png', dpi=360)

#%%=========================================================================%%#
#-------------- PLOT THE DRAINAGE DIVIDE CHANGE THROUGH TIME -----------------#

drainage_change = [np.zeros((nx)) for i in range(33)]
drainage_change_m = np.zeros(33)

drainage_basement_change = [np.zeros((100)) for i in range(33)]
drainage_basement_change_m = np.zeros(33)
drainage_basement_change_std = np.zeros(33)

for i in range(1, 33):
    drainage_change[i] = abs(drainage_position_time[i]-drainage_position_time[i-1])
    drainage_change_m[i] = np.mean(drainage_change[i])
    
    drainage_basement_change[i] = drainage_change[i][80:180]
    drainage_basement_change_m[i] = np.mean(drainage_basement_change[i])
    drainage_basement_change_std[i] = np.std(drainage_basement_change[i])
    
plt.plot(drainage_basement_change_m)
plt.plot(drainage_change_m)
plt.plot(drainage_basement_change_m-drainage_change_m)
plt.axis([1,32,-1,7])


#%%=========================================================================%%#
#--------------------- CATCHMENT SIZE THROUGH TIME ---------------------------#

id_list_cc = ([13285, 13635])
id_count_cc = np.zeros((np.shape(id_list_cc)[0],nt))

id_list_nc = np.unique(out_ds.sel(out=30e6).flow__basin[25, 100:150])
id_count_nc = np.zeros((np.shape(id_list_nc)[0],nt))

id_list_sc = np.unique(out_ds.sel(out=30e6).flow__basin[60, 90:165])
id_count_sc = np.zeros((np.shape(id_list_sc)[0],nt))

for i in range(0, nt):
    for j in range(0, np.shape(id_list_cc)[0]):
        id_count_cc[j][i] = (out_ds.sel(out=i*dt).flow__basin == id_list_cc[j]).sum()
    for k in range(0, np.shape(id_list_nc)[0]):
        id_count_nc[k][i] = (out_ds.sel(out=i*dt).flow__basin == id_list_nc[k]).sum()
    for l in range(0, np.shape(id_list_sc)[0]):
        id_count_sc[l][i] = (out_ds.sel(out=i*dt).flow__basin == id_list_sc[l]).sum()

id_count_sc = np.delete(id_count_sc, (4), axis=0)

#%%=========================================================================%%#
#--------------------- PLOT CATCHMENT SIZE THROUGH TIME ----------------------#

fig = plt.figure(figsize=(3.5,3.5))
ax1 = fig.add_subplot(111)

L1 = ax1.plot(id_count_cc.T[0:330], '-', color='steelblue')
L2 = ax1.plot(id_count_nc.T[0:330], '-.', color='darkred')
L3 = ax1.plot(id_count_sc.T[0:330], '--', color='darkgoldenrod')

custom_lines = [Line2D([0], [0], color='steelblue', ls='-'),
                Line2D([0], [0], color='darkred', ls='-.'),
                Line2D([0], [0], color='darkgoldenrod', ls='--')]

ax1.legend(custom_lines, ['close by catchments', 'northern catchments', 'southern catchments'], bbox_to_anchor=(-0.02, 1., 1.04, 0.), loc='lower left', mode='expand', framealpha=1, edgecolor='k')

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*dt*1e-6))
ax1.xaxis.set_major_formatter(ticks_x)
ax1.axis([0, 300 , 0, ax1.axis()[3]])
ax1.set_xlabel('Time (Ma)')
ax1.set_ylabel('Catchment sizes (pixel)')

fig.tight_layout()
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/catchment_size_evolution.png', dpi=360)

#%%=========================================================================%%#
#--------------------- PLOT EROSION PATTERN THROUGH TIME ---------------------#

fig = plt.figure(figsize=(5,3.85))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

U = (np.ones((ny, nx))*1e-3*0.2e6) * (np.linspace(0, 1, ny)[::-1])[:,None]

E1 = out_ds.sel(out=15.4e6).topography__elevation + U - out_ds.sel(out=15.6e6).topography__elevation
I1 = ax1.imshow(out_ds.sel(out=16.0e6).spl__erosion[20:70, 130:190], vmin=0, vmax=125, cmap='jet')
D1 = ax1.plot(drainage_position_time[16][130:190]-20, '--', c='k')

E2 = out_ds.sel(out=17.9e6).topography__elevation + U - out_ds.sel(out=18.1e6).topography__elevation
I2 = ax2.imshow(out_ds.sel(out=18.0e6).spl__erosion[20:70, 130:190], vmin=0, vmax=125, cmap='jet')
D2 = ax2.plot(drainage_position_time[18][130:190]-20, '--', c='k')

E3 = out_ds.sel(out=19.9e6).topography__elevation + U - out_ds.sel(out=20.1e6).topography__elevation
I3 = ax3.imshow(out_ds.sel(out=20.0e6).spl__erosion[20:70, 130:190], vmin=0, vmax=125, cmap='jet')
D3 = ax3.plot(drainage_position_time[20][130:190]-20, '--', c='k')

E4 = out_ds.sel(out=24.9e6).topography__elevation + U - out_ds.sel(out=25.1e6).topography__elevation
I4 = ax4.imshow(out_ds.sel(out=29.9e6).spl__erosion[20:70, 130:190], vmin=0, vmax=125, cmap='jet')
D4 = ax4.plot(drainage_position_time[30][130:190]-20, '--', c='k')

ax1.set_title('Time: 16 Ma', fontsize=10); ax2.set_title('Time: 18 Ma', fontsize=10); ax3.set_title('Time: 20 Ma', fontsize=10); ax4.set_title('Time: 30 Ma', fontsize=10)

ax1.set_ylabel('Distance (km)'); ax3.set_ylabel('Distance (km)')
ax3.set_xlabel('Distance (km)'); ax4.set_xlabel('Distance (km)')

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
ax1.xaxis.set_major_formatter(ticks_x)
ax2.xaxis.set_major_formatter(ticks_x)
ax3.xaxis.set_major_formatter(ticks_x)
ax4.xaxis.set_major_formatter(ticks_x)

ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
ax1.yaxis.set_major_formatter(ticks_y)
ax2.yaxis.set_major_formatter(ticks_y)
ax3.yaxis.set_major_formatter(ticks_y)
ax4.yaxis.set_major_formatter(ticks_y)

cax = fig.add_axes([0.82, 0.14, 0.02, 0.775])
fig.colorbar(I2, label='Erosion rate', cax=cax)
fig.subplots_adjust(right=0.77, bottom=0.065, top=0.99, wspace=0.25, hspace=0.0)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/erosive_pattern_through_time_1.png', dpi=720)

#%%=========================================================================%%#
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 4.5))
time_list = [14.9e6, 17.5e6, 20.0e6, 22.5e6, 29.9e6]
basin1_id = 13586; basin2_id = 22169

i=0
for ax in axes.flat:
    
    print(i)
    time = time_list[i]
    
    mask = (out_ds.sel(out=time).flow__basin == basin1_id) | (out_ds.sel(out=time).flow__basin == basin2_id)
    result = out_ds.sel(out=time).spl__erosion
    result_mask = np.where(mask, result, np.nan)
    
    result_basin = result_mask[:, ~np.isnan(result_mask).all(0)]
    result_basin = result_basin[~np.isnan(result_basin).all(1)]
    
    # dd = drainage_position_time[int(time/1e6)]
    
    # for j in range(0,nx):
    #     if any(mask[:,j]) == True:
    #         index1 = j
    #         break
    # for j in range(nx-1, -1, -1):
    #     if any(mask[:,j]) == True:
    #         index2 = j
    #         break
    
    I = ax.imshow(result_basin, vmin=0, vmax=125, cmap='jet')
    # P = ax.plot(dd[index1:index2+1], c='k', linestyle='--')
    ax.set_title('Time: '+str(round(time_list[i]/1e6, 1))+' Ma', fontsize=10)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
    ax.xaxis.set_major_formatter(ticks_x)
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
    ax.yaxis.set_major_formatter(ticks_y)
        
    i+=1
    
cax = fig.add_axes([0.885, 0.11, 0.02, 0.775])
fig.colorbar(I, label='Erosion rate', cax=cax)
fig.subplots_adjust(left=0.025, right=0.87, wspace=0)
# plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/erosive_pattern_through_time_3.png', dpi=720)

#%%=========================================================================%%#
#----------------------------- PUBLICATION FIGURE 2 --------------------------#

font = {'family':'arial', 'size':8}
matplotlib.rc('font',**font)

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(7.125, 4.5))
time_list = [14.9e6, 16.0e6, 17.5e6, 20.0e6, 22.5e6, 29.9e6]
basin1_id = 28383; basin2_id = 27048

i=0
for ax in axes.flat:
    
    print(i)
    time = time_list[i]
    
    mask = (out_ds.sel(out=time).flow__basin == basin1_id) | (out_ds.sel(out=time).flow__basin == basin2_id)
    result = out_ds.sel(out=time).spl__erosion
    result_mask = np.where(mask, result, np.nan)
    
    result_basin = result_mask[:, ~np.isnan(result_mask).all(0)]
    result_basin = result_basin[~np.isnan(result_basin).all(1)]
    
    dd = drainage_position_time[int(time/1e6)]
    
    for j in range(0,nx):
        if any(mask[:,j]) == True:
            index1 = j
            break
    for j in range(nx-1, -1, -1):
        if any(mask[:,j]) == True:
            index2 = j
            break
    
    I = ax.imshow(result_basin, vmin=0, vmax=125, cmap='jet', zorder=2)
    P = ax.plot(dd[index1:index2+1], c='k', linestyle='--', zorder=3)
    
    Rec = patches.Rectangle((-5, 45), 40, 20, ls='--', lw=1, ec='k', fc='#D3E7FA', alpha=1, zorder=1)
    ax.add_patch(Rec)
    
    ax.set_title('Time: '+str(round(time_list[i]/1e6, 1))+' Myrs', fontsize=8)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
    ax.xaxis.set_major_formatter(ticks_x)
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
    ax.yaxis.set_major_formatter(ticks_y)
    
    if i in [0]:
        ax.set_ylabel('Distance (km)')
    
    i+=1

ax.text(x=-95, y=112.0, s='Distance (km)')
cax = fig.add_axes([0.0825, 0.115, 0.89, 0.025])
fig.colorbar(I, label='Erosion rate (10$^{1}$ m.Ma$^{-1}$)', cax=cax, orientation='horizontal')
fig.subplots_adjust(left=0.06, right=0.985, bottom=0.25, top=0.935, wspace=0.25)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE2_erosive_pattern.png', dpi=1200)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE2_erosive_pattern.pdf', dpi=1200)

#%%=========================================================================%%#
#----------------------------- PUBLICATION FIGURE 1 --------------------------#

color_list = ['black', 'cyan', 'darkorange', 'darkgreen', 'darkred', 'yellow', 'pink', 'mediumblue', 'darkviolet', 'saddlebrown', 'olive', 'magenta', 'grey', 'gold', 'lime', 'steelblue']

fig = plt.figure(figsize = (7.125, 2.70))
ax1 = fig.add_subplot(111)

for i in range(0, 16):
    ax1.plot(drainage_position_time[i+15], '--', color=color_list[i], label='t: '+str(round((i+15)))+' Myrs')
    #ax1.legend(bbox_to_anchor=(0., 1.1, 1., 0.), loc='lower left', mode='expand', ncol=4, fancybox=False, framealpha=1, edgecolor='k', borderaxespad=0.)
    ax1.legend(loc='lower right', ncol=4, fancybox=False, framealpha=1, edgecolor='k')

Rec = patches.Rectangle((80, 45), 100, 20, ls='--', lw=1, ec='k', fc='#D3E7FA', alpha=1)
ax1.add_patch(Rec)

ax1.axis([0, nx, 0, ny])
ax1.invert_yaxis()
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
ax1.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
ax1.yaxis.set_major_formatter(ticks_y)

ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Distance (km)')

fig.tight_layout()
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE1_drainage_divide_tracking.png', dpi=1200)
#plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE1_drainage_divide_tracking.pdf', dpi=1200)

#%%=========================================================================%%#
#----------------------------- PUBLICATION FIGURE 4 --------------------------#

fig = plt.figure(figsize=(6.75, 4.1))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

color_list = ['black', 'cyan', 'darkorange', 'darkgreen', 'darkred', 'yellow', 'pink', 'mediumblue', 'darkviolet', 'saddlebrown', 'olive', 'magenta', 'grey', 'gold', 'lime', 'steelblue']
for i in range(0, 16):
    ax1.plot(drainage_position_time[i+15], '--', color=color_list[i], label=str(round((i+15)))+' Myrs')
    ax1.legend(loc=2, ncol=1, fancybox=False, framealpha=1, edgecolor='k', bbox_to_anchor=(1.05,1.05))

Rec = patches.Rectangle((80, 45), 100, 20, ls='--', lw=1, ec='k', fc='#D3E7FA', alpha=1)
ax1.add_patch(Rec)

ax2.plot(drainage_basement_change_m, color='#9B3535', zorder=3)
ax2.plot(drainage_basement_change_m+drainage_basement_change_std, color='#DA1D88', zorder=2)
ax2.plot(drainage_basement_change_m-drainage_basement_change_std, color='#DA1D88', zorder=2)
ax2.fill_between(np.arange(0,33), drainage_basement_change_m+drainage_basement_change_std, drainage_basement_change_m-drainage_basement_change_std, color='#9B3535', alpha=0.33, zorder=1)

ax1.axis([0, nx, 0, ny])
ax2.axis([1, 32, ax2.axis()[2], ax2.axis()[3]])

ax1.invert_yaxis()
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(nx-1)*lenghtx*1e-3))
ax1.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/(ny-1)*lenghty*1e-3))
ax1.yaxis.set_major_formatter(ticks_y)

ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Distance (km)')

ax2.set_xlabel('Time (Myrs)')
ax2.set_ylabel('Drainage displacement (km)')

fig.subplots_adjust(left=0.125, right=0.775, bottom=0.125, top=0.95, hspace=0.35)
plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE4_displacement+change_drainage_divide.png', dpi=1200)
plt.savefig('/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/fastscape_model_outputs/FIGURE4_displacement+change_drainage_divide.pdf', dpi=1200)

#%%=========================================================================%%#
#---------------------------- SUPPLEMENTARY VIDEO 1 --------------------------#
plt.ioff()

for i in range(0, nt):
    if i<10:
        fastscape_plotting_functions.topography_elevation_plot(out_ds, i, dt, nx, ny, lenghtx, lenghty, (9, 3), 'gist_earth', 0.99, 15, "/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/topo_export_figure/", "topo_model_t00"+str(i), True, 720)
    if i>=10 and i<100:
        fastscape_plotting_functions.topography_elevation_plot(out_ds, i, dt, nx, ny, lenghtx, lenghty, (9, 3), 'gist_earth', 0.99, 15, "/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/topo_export_figure/", "topo_model_t0"+str(i), True, 720)
    if i>=100:
        fastscape_plotting_functions.topography_elevation_plot(out_ds, i, dt, nx, ny, lenghtx, lenghty, (9, 3), 'gist_earth', 0.99, 15, "/home/thomas/Documents/research/fastscape/script/mountain_model_drainage_divide_migration/topo_export_figure/", "topo_model_t"+str(i), True, 720)

#%%=========================================================================%%#

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(out_ds.sel(out=15e6).flow__basin)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(out_ds.sel(out=30e6).flow__basin)