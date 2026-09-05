"""
This script contains utility functions for handling observation-related workflows 
in VIS/IR simulations. It supports both pre-processing of input observation files 
and post-processing of RTTOV output results.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

Created in 2023
"""

import sys

import glob
import os
import time
# import datetime
# import mplhep as hep

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
import numpy as np
import xarray as xr
import matplotlib.path as mpath

import os
import shutil
import pickle

def delete_files_with_index(directory, index):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file
    for file in files:
        # Check if the file is a NetCDF file and contains the index in its name
        if file.endswith('.nc') and str(index) in file:
            # Construct the absolute path to the file
            file_path = os.path.join(directory, file)
            
            # Delete the file
            os.remove(file_path)
            print(f"Deleted file: {file}")

def obs_mask3(obs_lon, obs_lat):
    """
    Create a mask to filter observations within a trapezoidal domain.

    Inputs:
    1. latlons: 2D array where latlons[:,0] is the flattened lats array and latlons[:,1] is the flattened lons array of the model domain D2
    2. obs: xarray dataset of observations

    Outputs:
    1. mask matrix with the same shape as obs.lon and obs.lat
    """

    # Extract the coordinates of the smaller domain (trapezoid)
    min_lon_small = 3 #5
    max_lon_small = 24  #22
    min_lat_small = 41  #43.5
    max_lat_small = 53  #51

    #  mask for points within the trapezoidal domain
    rect_mask_lon = (obs_lon >= min_lon_small) & (obs_lon <= max_lon_small)
    rect_mask_lat = (obs_lat >= min_lat_small) & (obs_lat <= max_lat_small)
    rect_mask = rect_mask_lon & rect_mask_lat
    
    return rect_mask

def apply_mask(obs_lon, obs_lat, mask):
    """
    The observation grid domain is very large, and we're only interested to use the points leing in the smallest domain containing the model grid domain D2.
    Inputs:
    1. latlons: 2d array where latlons[:,0] is the flattened lats array and latlons[:,1] is the flattened lons array of the model domain D2
    2. obs: xarray dataset of observations
    Outputs:
    1. mask matrix
    """
    
    filtered_var1 = np.full_like(obs_lon, np.nan)
    filtered_var2 = np.full_like(obs_lat, np.nan)

    filtered_var1[mask] = obs_lon[mask]
    filtered_var2[mask] = obs_lat[mask]
    
    row, col = np.where(mask)

    new_dim_r = len(np.unique(row))
    new_dim_c = len(np.unique(col))
    
    new_filtered_var1 = np.zeros((new_dim_r,new_dim_c))
    new_filtered_var2 = np.zeros((new_dim_r,new_dim_c))
    
    row_indices = np.searchsorted(np.unique(row), row)
    col_indices = np.searchsorted(np.unique(col), col)

    # Use np.add.at to accumulate values in f2_v1 and f2_v2
    np.add.at(new_filtered_var1, (row_indices, col_indices), filtered_var1[mask])
    np.add.at(new_filtered_var2, (row_indices, col_indices), filtered_var2[mask])

    return new_filtered_var1, new_filtered_var2


# time.clock is not available any more in Python3.8+
try :
    from time import perf_counter as time_clock
except ImportError :
    from time import clock as time_clock

def map_to_sat_grid_edited(reflectance_sd, channel='', varname='refl', plot_only_latlon=False, vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path='./') :
     """Map data on model grid to satellite grid
     """

     with open(repo_path+'/RTTOV_AROME/obs_grid/grid_sd_rttov_map.pkl', 'rb') as f:
         grid_sd = pickle.load(f)
     with open(repo_path+'/RTTOV_AROME/obs_grid/grid_sat_rttov_map.pkl', 'rb') as f:
         sat_grid = pickle.load(f)
     with open(repo_path+'/RTTOV_AROME/obs_grid/tile_rttov_map.pkl', 'rb') as f:
         tile = pickle.load(f)

    #elif args.gridtype == 'lamcoco' : # Arome
     if True:
         sys.path.append("/etc/ecmwf/nfs/dh2_perm_b/km4c/visop_operator/vis_op/")

         from vo2_utilities import print_dict, changes_with_dim

         from vo2_icon_map import Latlon2triMapper, convert_quad_to_tri_grid
         from vo2_lamcoco_subdomain import lamcoco_latlon2xy

        # Arome has a regular grid in lambert conformal conic coordinates x, y
        # pretend x is lon, y is lat --> we can use the Mapper for regular lat-lon grids
        
        # definition of regular grid in lamcoco coordinates
         ll_grid = { 'lat_min':grid_sd['y0'], 'dlat':grid_sd['dy'], 'nlat':grid_sd['ny'],
                    'lon_min':grid_sd['x0'], 'dlon':grid_sd['dx'], 'nlon':grid_sd['nx'] }
         #print('lamcoco coordinates for model grid: ', grid_sd['y0'], ' < y < ', grid_sd['y0'] + grid_sd['dy']*grid_sd['ny'])
         #print('                                    ', grid_sd['x0'], ' < x < ', grid_sd['x0'] + grid_sd['dx']*grid_sd['nx'])

        # compute pixel center coordinates
         #print('sat grid coordinates : lat ', sat_grid['lat'].shape, ' changes with dim ', changes_with_dim(sat_grid['lat']))
         #print('                       lon ', sat_grid['lon'].shape, ' changes with dim ', changes_with_dim(sat_grid['lon']))
         qcx, qcy = lamcoco_latlon2xy( sat_grid['lat']*180/3.141592, sat_grid['lon']*180/3.141592,
                                      lon_0=grid_sd['projparams']['lon_0'], lat_0=grid_sd['projparams']['lat_0'],
                                      lat_1=grid_sd['projparams']['lat_1'], lat_2=grid_sd['projparams']['lat_2'] )
         #print('lamcoco coordinates for sat grid: ', qcy.min(), ' < y < ', qcy.max(), ' shape ', qcy.shape, ' changes with dim ', changes_with_dim(qcy) )
         #print('                                  ', qcx.min(), ' < x < ', qcx.max(), ' shape ', qcx.shape, ' changes with dim ', changes_with_dim(qcx) )

        # generate quad grid for satellite pixels
         #print('*** generating triangle grid'); starttime = time_clock()
         import vo2_icon_map as icon_map
         quad_grid = icon_map.quads_from_regular_grid( qcy, qcx )#, clockwise=False)
        # (expects 2d lat, 2d lon array where lat changes with dimension 0 and lon with 1)
       
        # create triangle grid from quad grid
         tri_grid = convert_quad_to_tri_grid(quad_grid, area=True)
         #print('*** generating triangle grid took ', time_clock()-starttime, ' seconds...')

        # map reflectance from model grid to triangle grid
         #print('*** initializing lamcoco mapper'); starttime = time_clock()
         mapper = Latlon2triMapper( ll_grid, tri_grid, method='exact', nqmax=32 )
         #print('*** initializing lamcoco mapper took ', time_clock()-starttime, ' seconds...')

         #print('*** executing lamcoco mapper'); starttime = time_clock()
         image_tri = mapper.map( reflectance_sd )
         #print('*** executing lamcoco mapper took ', time_clock()-starttime, ' seconds...')
                
        # two triangles = one pixel...
         image_sd = ( (image_tri[0::2]*tri_grid['area'][0::2] + image_tri[1::2]*tri_grid['area'][1::2]) \
                   / (                tri_grid['area'][0::2] +                 tri_grid['area'][1::2]) ).reshape(sat_grid['lat'].T.shape).T
        #image_sd = ((image_tri[0::2] + image_tri[1::2])/2).reshape(sat_grid['lat'].T.shape).T
        #image_sd = ((image_tri[0::2] + image_tri[1::2])/2).reshape(sat_grid['lat'].shape)

        # copied from rotlatlon case
         indices_sd = np.where((sat_grid['lat'] >= tile['clat_min']) & (sat_grid['lat'] <= tile['clat_max']) & \
                           (sat_grid['lon'] >= tile['clon_min']) & (sat_grid['lon'] <= tile['clon_max']))
     else :
         raise ValueError('Unknown grid type ')
    
     return image_sd, indices_sd

def smooth_data(data, window_size):
    smoothed_data = uniform_filter(data.astype(float), size=window_size, mode='constant', cval=0.0)
    return smoothed_data
