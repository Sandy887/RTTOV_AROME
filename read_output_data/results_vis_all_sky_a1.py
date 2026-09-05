#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

Contributions:
- Created in mid 2024
- Modified in mid 2025

| Script workflow      | • Reads results for all-sky simulation experiments at specific datetimes and hours configured by the user                
|                      | • Loads data via `reflectance_out` object defined in `class_reflectance_out`                                             
|                      | • Creates a dictionary of `reflectance_out` objects for each experiment 
|                      | • Further manipulations of the data can be modified according to users' needs
"""

import sys
# Paths to change
sys.path.append("/perm/km4c/RTTOV_AROME/")
#sys.path.append("/perm/km4c/RTTOV_AROME/visualize_results-visible/")
sys.path.append("/etc/ecmwf/nfs/dh2_home_b/km4c/.local/lib/python3.8/site-packages")

import xarray as xr
import numpy as np
from class_reflectance_out import *


M1 = np.load("/perm/km4c/RTTOV_AROME/masks/AROME_Mask.npy") # 
M = np.load("/perm/km4c/RTTOV_AROME/masks/clr_sky_filter.npy")

paths_1 = "/hpcperm/km4c/experiment_results" #"/etc/ecmwf/scratch/km4c/ml_simulations/visible/"
paths_2 = "/etc/ecmwf/scratch/km4c/ml_simulations/visible/Results_tmp_all_datetimes"  #"/etc/ecmwf/scratch/km4c/visop_runs/summer_period/"


visop = {'default':{}, 'v1':{}, 'v2':{}, 'v3':{}, 'v4':{}}
rttov132 = {'default':{}, 'v1':{}, 'v1.2':{}}
rttov122 = {'default':{}, 'v1':{}, 'v1.1':{}, 'v1.2':{}}

# Define hours
hours = ["060000","090000","120000", "150000"]

datetime_slice = list(f"202308{x:02d}_{hours[2]}" for x in range(1,32)) #FIXME make a dictionary of dates for each hour (key element)
datetime_slice_all = [f"202308{day:02d}_{hour}" for day in range(1, 32) for hour in hours]

time_slice = list(f"{x:02d}0000" for x in [6,9,12,15])
#time_slice = ["120000"]

visop['default'] = reflectance_out("visop", paths_1, "/visop_A_1.nc" , all_sky_ref_period=datetime_slice_all)
visop['v2'] = reflectance_out("visop", paths_1, "/visop_v2_2.nc" , all_sky_ref_period=datetime_slice)
rttov132['default'] = reflectance_out("rttov132", paths_1, "/rttov_v132_A_1.nc" , all_sky_ref_period=datetime_slice_all)
rttov122['default'] = reflectance_out("rttov122", paths_1, "/rttov_v122_A_1.nc", clear_ref_period=time_slice , all_sky_ref_period=datetime_slice_all, albedo=True) # rttov_v122_A_1.nc
rttov132['v1'] = reflectance_out("rttov132", paths_1, "/rttov_v132_v1_18.nc" , all_sky_ref_period=datetime_slice)
rttov122['v1'] = reflectance_out("rttov122", paths_1, "/rttov_v122_v1_4.nc" , all_sky_ref_period=datetime_slice)
rttov132['v1.2'] = reflectance_out("rttov132", paths_1, "/rttov_v132_v1.2.nc" , all_sky_ref_period=datetime_slice)
rttov122['v1.1'] = reflectance_out("rttov122", paths_1, "/rttov_v122_v1.1_5.nc" , all_sky_ref_period=datetime_slice)
rttov122['v1.2'] = reflectance_out("rttov122", paths_1, "/rttov_v122_A_1.2.nc" , all_sky_ref_period=datetime_slice)

rttov122['v3.1'] = reflectance_out("rttov122", paths_2, "/rttov_v122_v3.1.nc" , all_sky_ref_period=datetime_slice)
rttov122['v3.2'] = reflectance_out("rttov122", paths_2, "/rttov_v122_v3.2.nc" , all_sky_ref_period=datetime_slice)
rttov122['v3.3'] = reflectance_out("rttov122", paths_2, "/rttov_v122_v3.3.nc" , all_sky_ref_period=datetime_slice)

rttov122['v4'] = reflectance_out("rttov122", paths_2, "/rttov_v122_v4.nc" , all_sky_ref_period=datetime_slice)

rttov132['v3.1'] = reflectance_out("rttov132", paths_2, "/rttov_v132_v3.1.nc" , all_sky_ref_period=datetime_slice)
rttov132['v3.2'] = reflectance_out("rttov132", paths_2, "/rttov_v132_v3.2.nc" , all_sky_ref_period=datetime_slice)
rttov132['v3.3'] = reflectance_out("rttov132", paths_2, "/rttov_v132_v3.3.nc" , all_sky_ref_period=datetime_slice)

# Initialize results dictionaries
exp_results = {'default':{}, 'v1':{}, 'v2':{}, 'v3':{}, 'v4':{}}
exp_results['default'] = {"O": {}, "B-visop": {}, "B-rttov12": {}, "B-rttov13": {}}  # Reference experiments
exp_results['v1'] = {"B-visop": {}, "B-rttov12": {}, "B-rttov13": {}}  # experiments _v1_ 10% of snow
exp_results['v1.1'] = {"B-visop": {}, "B-rttov12": {}, "B-rttov13": {}}  # experiments _v1.1_ 20% of snow
exp_results['v1.2'] = {"B-visop": {}, "B-rttov12": {}, "B-rttov13": {}}  # experiments _v1.2_ 100% of snow
exp_results['v2'] = {"B-visop": {}}  # experiments _v2_ psf on
exp_results['v3.1'] = {"B-rttov12": {}, "B-rttov13": {}}  # experiments _v3_ NC tests
exp_results['v3.2'] = {"B-rttov12": {}, "B-rttov13": {}}  # experiments _v3_ NC tests
exp_results['v3.3'] = {"B-rttov12": {}, "B-rttov13": {}}  # experiments _v3_ NC tests
exp_results['v4'] = {"B-rttov12": {}, "B-rttov13": {}}  # experiments _v4_ OPAC scheme
ref = {"O": [], "B-rttov12": [], "B-rttov13": [], "OmB-12":[], "OmB-13":[], "13-12":[], "albedo":[], "SZA":[]} # all_days_all_times only used now for results with default settings
Albedo = {}
angle = {"SZA":{}}

# Populate the results
datetime_slice = list(f"202308{x:02d}_{hours[2]}" for x in range(1,32))
datetime_slice_all = [f"202308{day:02d}_{hour}" for day in range(1, 32) for hour in hours]

for date_time in datetime_slice:
    exp_results['v2']["B-visop"][date_time] = visop['v2'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time).values * M
    exp_results['v1']["B-rttov12"][date_time] = rttov122['v1'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v1']["B-rttov13"][date_time] = rttov132['v1'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v1.1']["B-rttov12"][date_time] = rttov122['v1.1'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v1.2']["B-rttov12"][date_time] = rttov122['v1.2'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v1.2']["B-rttov13"][date_time] = rttov132['v1.2'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.1']["B-rttov12"][date_time] = rttov122['v3.1'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.2']["B-rttov12"][date_time] = rttov122['v3.2'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.3']["B-rttov12"][date_time] = rttov122['v3.3'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.1']["B-rttov13"][date_time] = rttov132['v3.1'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.2']["B-rttov13"][date_time] = rttov132['v3.2'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v3.3']["B-rttov13"][date_time] = rttov132['v3.3'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['v4']["B-rttov12"][date_time] = rttov122['v4'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M


for date_time in datetime_slice_all:
    # Calculate values for each date-time 
    exp_results['default']["O"][date_time] = visop['default'].data.observed_reflectance_VIS006.sel(datetimes=date_time).values * M
    exp_results['default']["B-visop"][date_time] = visop['default'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time).values * M
    exp_results['default']["B-rttov12"][date_time] = rttov122['default'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    exp_results['default']["B-rttov13"][date_time] = rttov132['default'].data.synthetic_reflectance_VIS006.sel(datetimes=date_time, sky_condition='all_sky').values * M
    #Albedo[date_time[-6:]] = rttov122['default'].albedo_from_atlas[date_time[-6:]].values * M # Get albedo at each hour
    
    Albedo[date_time] = rttov122['default'].data.synthetic_albedo.sel(datetimes=date_time).values * M
    angle["SZA"][date_time] = rttov122['default'].data.SZA.sel(datetimes=date_time).values * M
    
    ref["O"].append(exp_results['default']["O"][date_time].flatten())
    ref["B-rttov12"].append(exp_results['default']["B-rttov12"][date_time].flatten())
    ref["B-rttov13"].append(exp_results['default']["B-rttov13"][date_time].flatten())
    ref["OmB-12"].append(exp_results['default']["O"][date_time].flatten() - exp_results['default']["B-rttov12"][date_time].flatten())
    ref["OmB-13"].append(exp_results['default']["O"][date_time].flatten() - exp_results['default']["B-rttov13"][date_time].flatten())
    ref["13-12"].append(exp_results['default']["B-rttov13"][date_time].flatten() - exp_results['default']["B-rttov12"][date_time].flatten())
    ref["albedo"].append(Albedo[date_time].flatten())
    ref["SZA"].append(angle["SZA"][date_time].flatten())

for key in ref:
    ref[key] = np.concatenate(ref[key])

     


