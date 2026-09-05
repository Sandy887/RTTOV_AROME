"""
This script consolidates multiple NetCDF (.nc) files into a single file. 
It supports merging files across different datetimes as well as across different experiments.
    
Author
------
Ms. Sandy Chkeir |  sandychkeir96@gmail.com

Created on the 5th of November 2024

Purpose:
    - Combine .nc files by datetime into one unified file.
    - Optionally combine .nc files from multiple experiments into one.
    - Validate that xarray datasets are concatenated correctly.
    - Enable execution over a defined time period and specific list of experiments for comparison.

Implementation Notes:
    - Currently, a single function performs the core merging task.
    - Latitude and longitude coordinates (for both observation and model grids) are excluded from concatenation.
    
Operators and Compatibility:
    - The existing script for RTTOV versions is built for the visible spectrum and can be extended for IR channels.
    - RTTOV v13.2 .nc files contain: reflectances, albedo, Experiment, and valid_time.
        - Other variables are identical to RTTOV v12.2.
        - Note: Although albedo should be the same in both versions, it has been retained separately in case modification is needed.
"""

from datetime import datetime as dt
import xarray as xr
import glob
import os, sys


def concatenate_netcdf_outputs(out_path, in_path, var_coordinate, op_version):
     """
     Inputs: 
             - input path to a group .nc files
             - the variable coordinate dimension over which we want to concatenate the .nc files
             - op_version is the operator version
     Outputs:
             - a single netcdf file to save the specified output path
             - output path
     """
     startTime = dt.now()

     if var_coordinate == "datetimes":
         exclude_vars = ['lat_obs_map', 'lon_obs_map', 'lat_arome_map', 'lon_arome_map', 'experiment']
     else:
         exclude_vars = ['lat_obs_map', 'lon_obs_map', 'lat_arome_map', 'lon_arome_map','valid_datetime']

     if op_version == "132":
         exclude_vars = ['experiment']
     elif op_version == "None":
         exclude_vars = []

     file_paths = sorted(glob.glob(f"{in_path}/*.nc"))  # Adjust path and extension

     datasets = [xr.open_dataset(fp).drop_vars(exclude_vars, errors='ignore') for fp in file_paths]

     # Concatenate along a new datetime dimension based on valid_datetime in each file
     combined_dataset = xr.concat(datasets, dim=var_coordinate)

     # Assign datetime values to the new dimension from valid_datetime in each file
     if var_coordinate == "datetimes":
         combined_dataset = combined_dataset.assign_coords(datetimes=[ds.valid_datetime.item() for ds in datasets])
     else:
         combined_dataset = combined_dataset.assign_coords(experiments=[ds.experiment.item() for ds in datasets])
     
     with xr.open_dataset(file_paths[0]) as ds:
         excluded_data = ds[exclude_vars]

     combined_dataset = combined_dataset.merge(excluded_data)
     print(combined_dataset)

     # Close individual datasets to free up memory
     for ds in datasets:
         ds.close()

     if var_coordinate == "datetimes":
         if op_version!="None":
             filename = f"{out_path}/{excluded_data.experiment.values[0]}.nc"
         else:
             filename = f"{out_path}/Hydrometeors.nc"
     else:
         filename = f"{out_path}/All_data_{op_version}.nc"

     if os.path.exists(filename):
         os.remove(filename)
         print(f"Deleted existing file: {filename}")

     if op_version!="None":
         combined_dataset['experiment'] = combined_dataset['experiment'].astype(str)

     combined_dataset['valid_datetime'] = combined_dataset['valid_datetime'].astype(str)

     combined_dataset.to_netcdf(filename)

     print(f"The data are concatenated over {var_coordinate} dimension. It tooks ", dt.now() - startTime)

# To change this
concatenate_netcdf_outputs(out_path="/path/to//ml_simulations/visible/Results_tmp_all_datetimes", in_path= "/path/to/ml_simulations/visible/rttov_v132_v1.2/",var_coordinate='datetimes', op_version='v132')

