"""
This file contains a set of function to quickly convert fastscape outputs to lsdtopytools
B.G.
"""

import lsdtopytools as lsd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xsimlab
import xarray
import fastscape
import datetime
import sys
import os
from rasterio.crs import CRS
import numpy as np
import numba as nb


from pathlib import Path

# @nb.jit(nopython = True)
# def recalculate_chi_through_inverted_stack(fd,receiver_fd,da,receiver_da, ID, recID, theta, A0):
# 	node_to_index = nb.typed.Dict.empty(key_type=nb.types.int32,value_type=nb.types.int32,)
# 	recnode_to_index = nb.typed.Dict.empty(key_type=nb.types.int32,value_type=nb.types.int32,)

# 	for i in range(ID.shape[0]):
# 		node_to_index[ID[i]] = i
# 		recnode_to_index[recID[i]] = i

# 	chi = np.zeros(fd.shape[0])
# 	recchi = np.zeros(fd.shape[0])
# 	i = chi.shape[0] - 1
# 	# going through the stack from baselevel
# 	while(i>=0):
# 		previous_chi = recchi[i]
# 		this_chi = previous_chi + abs(receiver_fd[i] - fd[i]) * (A0/da[i] + A0/receiver_da[i])/2
# 		# print(this_chi)
# 		recchi[recnode_to_index[ID[i]]] = this_chi
# 		i -= 1

# 	return chi


def export_times_to_tif(output_model, directory = "tif_outputs", prefix = "test_export", epsg_code = 32635, resolution = 30, X_min = 0, Y_min = 0, time_list = []):
	"""
		This function exports all the model outputs as tif files in a direcory.
		Parameters:
			output_model: the fastscape model object after running
			directory: the name of the directory where the tif files will be placed
			prefix: a prefix starting the name of each files of that run (makes automation clearer in case)
			epsg_code: The saving simulates a georeferencing for visualisation purposes, you can leave default (UTM zone 35N -> Romania RPZ)
			resolution: resolution in XY
			X_min: X at the bottom lsft corner
			Y_min: Y at the bottom right corner
			time_list: the list of times at which the topo will be saved
		returns: the name of the csv file containing all the infos
		B.G.
	"""
	# Creating the folder if not done yet
	Path("./" + directory + "fastscape_tif_outputs").mkdir(parents=True, exist_ok=True)

	# output dataframe
	df = {"name_of_file":[], "path": [], "full_path": [], "time": []}
	# Going through every timestep and saving the output
	for i in range(len(time_list)):
		topo = np.array(output_model.sel(out=time_list[i]).topography__elevation)
		name = str(i)
		while(len(name)<4):
			name = "0" + name
		lsd.raster_loader.save_raster(topo,X_min,X_min + resolution * (topo.shape[1] + 1),  Y_min + resolution * (topo.shape[0] + 1),Y_min,   resolution, CRS.from_epsg(epsg_code), os.path.join(directory, prefix +"_"+ name + "_" + str(time_list[i]) + ".tif") , fmt = 'GTIFF')
		df["name_of_file"].append(prefix +"_"+ name + "_" + str(time_list[i]) + ".tif")
		df["path"].append(directory + os.sep)
		df["full_path"].append(os.path.join(directory, prefix +"_"+ name + "_" + str(time_list[i]) ) )
		df["time"].append(time_list[i])
	# Saving the file
	pd.DataFrame(df).to_csv(prefix + "_alltif.csv", index = False)
	
	return prefix + "_alltif.csv"


def fix_chi_and_recalculate_ksn(df, rec_df, chi,theta = 0.4, A0 = 1):

	# df["chi"] = chi[df["row"],df["col"]].ravel()
	df["k_sn"] = (df["elevation"] - rec_df["receiver_elevation"])/(df["chi"] - rec_df["receiver_chi"])	

	




if(__name__ == "__main__"):
	pass