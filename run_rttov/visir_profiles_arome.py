"""
A collection of utility functions for performing miscellaneous tasks on AROME model data 
during RTTOV operator execution for VIS and IR channels.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

Created in 2023

Purpose:
    - Support the preparation and transformation of AROME profiles and fields.
    - Assist in data handling tasks required during RTTOV simulations.

Notes:
    - Functions in this module are context-specific and tailored to AROME + RTTOV integration.
    - Extend functions here if new preprocessing steps are needed.
"""


import glob 
import sys
import os
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import pandas as pd

from pysolar.solar import get_altitude, get_azimuth
import pytz 

from scipy.interpolate import griddata
import seaborn as sns

#import import_ipynb
# import rttov_op_paths
# path_rttov = rttov_op_paths.RTTOV
# sys.path.append(path_rttov+'compilation_dir/lib')
# sys.path.append(path_rttov+'/wrapper')

# from rttov_wrapper_f2py import *
# import pyrttov

# -----------------------------------------------------------------------------
# Meteorogical Class
# -----------------------------------------------------------------------------

class meteo_var():
    pass

def validate_grb_datetime(files_check, datetime_check):
     forecast_datetime_list = get_date_time(files_check)  # forecast datetimes from the gribs
     print(len(forecast_datetime_list))
     # yyyy = forecast_datetime_list[0].year
     # mm = forecast_datetime_list[0].month
     # dd = forecast_datetime_list[0].day
     # hh = forecast_datetime_list[0].hour
     print("Date-time validation   ")
     if datetime_check == forecast_datetime_list[0]:
         print(f"The date and time {datetime_check} are valid")
     else:
         print(f"Error: The date and time {datetime_check} are valid")
         sys.exit()
     print("Date-time validation is done")

def get_profile(grb_list, time_step, par_ID, level_type, plot_par= False):
    
    """ 
    this function perform several tasks:
    - takes as input a grib list of several files, where each file corresponds to a single forecast time step and the time_step is the index of the target datetime chosen. 
    - Based on the specified par_ID and level_type, it filters all messages in the grib file and select only the messages corresponding to the parameter variable and it's level type specified.
    - The data selected could be 2D array (surface variable) or 3D array (vertical profile).
    - if plot_par=True, it plots the data on map and save these plots in a directory named plots.
    - if the parameter_name is not known in the grib, we assign it a name accounting for its parameter ID (AROME code).
    - it lats and lons coordinates are not saved for the selected case study, then we save them seperatly in numpy arrays.
    - check the units
    - return the arome_variable with: data values, parameter unit, name, and optionally lats and lons information as well.
    """
    arome_variable = meteo_var()
    # filter over a list of messages that match our field
    messages = grb_list[time_step].select(indicatorOfParameter=par_ID, typeOfLevel=level_type)
    
    # Count the number of messages
    num_messages = len(messages)
    
    parameter_name = messages[0].name
    arome_variable.parameter_name = parameter_name
    parameter_unit = messages[0].units
    arome_variable.parameter_unit = parameter_unit
    lats, lons = messages[0].latlons()

    message_selected = messages[0].values # aim to get data.shape for lat and lons size
    num_lats, num_lons = message_selected.shape
    # print(num_lats, num_lons)
    if True:
        arome_variable.projparams = messages[0].projparams
        arome_variable.lat, arome_variable.lon = messages[0].latlons()

    # Initialize an empty 3D array to store the data
    data = np.zeros((num_messages, num_lats, num_lons))
    
    if num_messages == 1:
        # 2D array
        data[0,:,:] = message_selected
    else:
        # Loop over the messages and populate the 3D array
        for i, message in enumerate(messages):
                value = message.values
                data[i, :, :] = value
                
    arome_variable.data = data
    
    if arome_variable.parameter_name == 'unknown':
        arome_variable.parameter_name, arome_variable.parameter_unit = get_parameter_info(par_ID)
    
    # if 'surface pressure' in arome_variable.parameter_name.lower():
    #     arome_variable.data = np.exp(arome_variable.data)
        
    if 'pressure' in arome_variable.parameter_name.lower():
        if parameter_unit == 'Pa':
            arome_variable.data = arome_variable.data / 100
            arome_variable.parameter_unit = 'hPa'

    return arome_variable

import numpy as np

def get_modelstate(grbs, index):
    """
    Read profiles from GRIB messages and assemble the model state dict.
    
    Parameters
    ----------
    grbs : <pygrib.GribMessages>
        Opened GRIB messages object.
    index : int
        Time step or message index to read from.
    
    Returns
    -------
    modelstate : dict
        Dictionary with keys:
        - 'T'      : Temperature on model levels [K]
        - 'QV'     : Specific humidity on model levels [kg/kg]
        - 'U10M'   : 10 m U wind component [m/s]
        - 'V10M'   : 10 m V wind component [m/s]
        - 'PSFC'   : Surface pressure [Pa]
        - 'P'      : Pressure on model levels [Pa]
        - 'QC_DIA' : Liquid water mixing ratio [kg/kg]
        - 'QI_DIA' : Cloud ice mixing ratio [kg/kg]
        - 'QG'     : Graupel mixing ratio [kg/kg]
        - 'QS'     : Snow mixing ratio [kg/kg]
        - 'QR'     : Rain mixing ratio [kg/kg]
        - 'CLC'    : Cloud fraction [unitless]
        - 'HHL'    : Hybrid level heights [m]
        - 'HML'    : Model level geopotential heights [m]
    """
    print("get_modelstate - Reading the GRIB files")
    
    # %Temeprature in Kelvin
    T_surface = get_profile(grbs, index, 11, "surface", plot_par=False)
    T_surface.parameter_name = 'surface_temp'
    T2m       = get_profile(grbs, index, 11, "heightAboveGround", plot_par=False)
    T_hybrid  = get_profile(grbs, index, 11, "hybrid", plot_par=False)

    # Winds at 10m
    U10m = get_profile(grbs, index, 33, "heightAboveGround", plot_par=False)
    V10m = get_profile(grbs, index, 34, "heightAboveGround", plot_par=False)

    # Pressure in hPa
    P_surface = get_profile(grbs, index, 1, "surface", plot_par=False)
    Pressure  = get_profile(grbs, index, 1, "hybrid", plot_par=False)
    # P_aboveGround = get_profile(grbs, index, 1, "heightAboveGround", plot_par = False)
    
    # Specific humidity in [kg kg**-1]
    H = get_profile(grbs, index, 51, "hybrid", plot_par=False)

    # Geopotential & surface elevation
    z_surface    = get_profile(grbs, index, 8,  "surface", plot_par=False)
    geopotential = get_profile(grbs, index, 6,  "hybrid",  plot_par=False)
    
    # %Geopotential in m**2 s**-2 or gpm
    #geopotential_height = get_profile(gribfile, 8, "surface", plot_par = False)

    # height above ground = (gribnumber=6+level=modellevel/9.80665)-gribnumber=8 (height of surface)
    
    # Hydrometeors (ensure non-negative)
    def clean(var):
        var.data = np.where(var.data < 0, 0, var.data)
        return var
    # LQW: LIQUID WATER, CLIC: CLOUD ICE, GRAUPOL, SNOW, RAIN
    
    LQW     = clean(get_profile(grbs, index, 188, 'hybrid', plot_par=False))
    CLIC    = clean(get_profile(grbs, index, 189, 'hybrid', plot_par=False))
    GRAUPOL = clean(get_profile(grbs, index, 190, 'hybrid', plot_par=False))
    SNOW    = clean(get_profile(grbs, index, 240, 'hybrid', plot_par=False))
    RAIN    = clean(get_profile(grbs, index, 241, 'hybrid', plot_par=False))

    # Cloud fraction
    Cloud_fraction = get_profile(grbs, index, 36, 'hybrid', plot_par=False)
    Cloud_fraction.parameter_name = 'Cloud fraction'
    Cloud_fraction.parameter_unit = 'Unitless'

    print("get_modelstate - GRIB files successfully read")

    # Compute heights
    HML = geopotential.data / 9.80665 
    nz = HML.shape[0]
    HHL = np.zeros( (nz+1,HML.shape[1],HML.shape[2]), dtype=HML.dtype )
    HHL[1:-1,:] = 0.5*( HML[:-1,:] + HML[1:,:] )
    HHL[ 0,:] = HML[ 0,:] + 0.5*(HML[ 0,:]-HML[ 1,:])
    HHL[-1,:] = z_surface.data[0,:,:] #HML[-1,:] - 0.5*(HML[-2,:]-HML[-1,:])

    # Assemble modelstate dictionary
    modelstate = {
        'T'      : T_hybrid.data,
        'T0m'    : T_surface.data,
        'T2m'    : T2m.data,
        'QV'     : H.data,
        'U10M'   : U10m.data, 
        'V10M'   : V10m.data,
        'PSFC'   : P_surface.data, #[0,:,:]
        'P'      : Pressure.data * 100,  # convert hPa to Pa if needed
        'QC_DIA' : LQW.data,
        'QI_DIA' : CLIC.data,
        'QG'     : GRAUPOL.data,
        'QS'     : SNOW.data,
        'QR'     : RAIN.data,
        'CLC'    : Cloud_fraction.data,
        'geopot' : geopotential.data,
        'HHL'    : HHL,
        'HML'    : HML,
        '2d-lat' : LQW.lat,
        '2d-lon' : LQW.lon,
        'Z'      : z_surface.data
    }

    # --- gather parameter metadata/stats ---
    parameters = [
        T_hybrid, T_surface, T2m, H,
        U10m, V10m, P_surface, Pressure,
        LQW, CLIC, GRAUPOL, SNOW, RAIN,
        z_surface, geopotential, Cloud_fraction
    ]
    parameters_info = []
    for var in parameters:
        info = {
            "Name": var.parameter_name,
            "Unit": var.parameter_unit,
            "Shape [dim1,dim2,dim3]": var.data.shape,
            "hybrid levels - dim1": var.data.shape[0],
            "Model grid - dim2,dim3": [var.data.shape[1], var.data.shape[2]],
        }
        parameters_info.append(info)
        print("Model data gathered:---------------------------")
        print(var.parameter_name,
              var.parameter_unit,
              var.data.shape)

    return modelstate

def return_grb_list(grb_path):
    path_ = grb_path + "/*.grb"
    list_ = glob.glob(path_)
    grb_names = [os.path.basename(path_) for path_ in list_]
    return grb_names

# def reformat_profile(arr,nlevels):
#     # reshape to out-dims: (nprofiles, nlevels); convention from TOA to ground
#     return np.flip(arr.reshape((nlevels, -1)).transpose(), axis=1)

def reformat_profile(arr, nprofiles):
    # Get the original shape of arr
    orig_shape = arr.shape

    # Reshape the array to have dimensions (nlevels, -1)
    reshaped_arr = arr.reshape(orig_shape[0], -1)

    # Transpose the reshaped array
    transposed_arr = np.transpose(reshaped_arr)

    # Reshape the transposed array to have dimensions (nprofiles, nlevels)
    reformatted_arr = np.reshape(transposed_arr, (nprofiles, orig_shape[0]))

    return reformatted_arr

def retrieve_profile(arr, orig_shape):
    # Get the number of levels, latitudes, and longitudes from the original shape
    nlevels, lat, lon = orig_shape

    # Reshape the reformatted array to have dimensions (lat * lon, nprofiles, nlevels)
    reshaped_arr = np.reshape(arr, (lat * lon, -1, nlevels))

    # Transpose the reshaped array
    transposed_arr = np.transpose(reshaped_arr, axes=(1, 0, 2))

    # Reshape the transposed array to have dimensions (nlevels, lat, lon)
    retrieved_arr = np.reshape(transposed_arr, orig_shape)

    return retrieved_arr

def expand2nprofiles(n, nprof):
        # Transform 1D array to a [nprof, nlevels] array
        outp = np.empty((nprof, len(n)), dtype=n.dtype)
        for i in range(nprof):
            outp[i, :] = n[:]
        return outp

# def save_dataset_to_netcdf(dataset, index, channel, final_directory, simulation_output_dir):
#     if channel == "visible":
#          os.makedirs(f'{simulation_output_dir}/visible/{final_directory}/', exist_ok=True)
#          filename = f'{simulation_output_dir}/visible/{final_directory}/out_{index}.nc'
#     if channel == "infrared":
#          os.makedirs(f'{simulation_output_dir}/infrared/{final_directory}/', exist_ok=True)
#          filename = f'{simulation_output_dir}/infrared/{final_directory}/out_{index}.nc'
#     if os.path.exists(filename):
#          os.remove(filename)
#          print(f"Deleted existing file: {filename}")
#     dataset.to_netcdf(filename)
#     print(f"Saved data to {filename}")

def save_dataset_to_netcdf(
    xr_dataset,
    time_index,
    band_name,
    run_label,
    output_root
):
    """
    Save an xarray Dataset to NetCDF under a directory structure
    organized by band and run label.
    
    Args:
        xr_dataset (xarray.Dataset): Dataset to save.
        time_index (int): Index or timestep identifier.
        band_name (str): e.g. "visible" or "infrared".
        run_label (str): A label for this simulation run (used as subdirectory).
        output_root (str): Base directory for all outputs.
    """
    target_dir = os.path.join(output_root, band_name, run_label)
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.join(target_dir, f"output_{time_index}.nc")

    if os.path.exists(filename):
        os.remove(filename)
        print(f"Deleted existing file: {filename}")

    xr_dataset.to_netcdf(filename)
    print(f"Saved data to {filename}")

def get_parameter_info(parameter_id):
    parameter_info = {
        188: ('Liquid Water', 'Kg / Kg'),
        189: ('Cloud Ice', 'Kg / Kg'),
        190: ('Graupol', 'Kg / Kg'),
        241: ('Rain', 'Kg / Kg'),
        240: ('Snow', 'Kg / Kg'),
        # Add more parameter IDs and corresponding names and units as needed
    }
    return parameter_info.get(parameter_id, ('Unknown', 'Unknown'))

def geopotential_to_geo_height(geopotential):
    """
    Input: geopotential height is in gpm.
    Output: surface elevation_level
    """
    g = 9.80665  # Acceleration due to gravity on Earth in m/s²
    # geo_height = geopotential/g
    surface_elevation_level = geopotential / 1000.0
    return surface_elevation_level
    
def get_forecast_datetime_index(forecast_datetime_list, target_datetime):
    try:
        index = forecast_datetime_list.index(target_datetime)
        return index
    except ValueError:
        return -1  # If the target datetime is not found in the list

def make_dt_timezone_utc(dt):
    timezone = pytz.utc
    dt_aware = timezone.localize(dt)
    return dt_aware

def get_sun_zenith(lat, lon, datetime_dt):
    return 90.0 - np.abs(get_altitude(lat, lon, datetime_dt))

def get_sun_azimuth(lat, lon, datetime_dt):
    return get_azimuth(lat, lon, datetime_dt)

def get_date_time(grb_list):
    # get all time steps
    valid_datetime_list = []
    forecast_datetime_list = []
    
    for step in range(0,len(grb_list)):
        message = grb_list[step].select(indicatorOfParameter=1, typeOfLevel="surface")[0] # we don't need a specific level type
        
        data_date = message.dataDate
        data_time = message.dataTime
        validity_date = message.validityDate
        validity_time = str(message.validityTime)
        # step_type = message.stepType
        # step_units = message.stepUnits
        # forecast_time = message.forecastTime
        
        # Calculate the hour and minute based on validity_time
        hour = int(validity_time) // 100
        minute = (int(validity_time) % 100) * 60

        # Convert to datetime.datetime format
        forecast_datetime = datetime.datetime.strptime(str(validity_date), "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)

        # Print the converted datetime values
        # print("Validity Date & Time:", data_date, " ",data_time)
        # print("Forecast Date & Time:", forecast_datetime)
        # print("---")
        
        
        forecast_datetime_list.append(forecast_datetime)
        
    return forecast_datetime_list
