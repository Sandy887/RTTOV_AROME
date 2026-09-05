# this script is written by Sandy Chkeir

# it reads the HRIT obs data for the visible 0.6 and water vapour infrared 6.2 and 7.3 channels.
# loads these channels, read into lons and lats as well as the data
# mask over the selected region and save these data
# the save information is useful for interpolating the model data onto observation grid resolution
# smoothening is then required
# plots the observations
# the histogram statistics should be later used IR vs VIS observed.

#from obs_funct import *
import xarray as xr

import os
import zipfile
import tarfile
import glob
from satpy import Scene
from satpy.scene import Scene
from satpy.resample import get_area_def
from satpy import find_files_and_readers
import warnings

from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")


# Path to the zip file containing HRIT data

# print('Step 1: Start the extraction of compressed HRIT files')
# files = os.listdir(directory)
# # print(len(time_slot))
# for filename in files:
    
#     year =filename[24:28]
#     month =filename[28:30]
#     day =filename[30:32]
#     hour =filename[32:34]
#     minute = filename[34:36]
#     print(f'date:{year}/{month}/{day} at {hour}:{minute}')
    
#     # Path to the directory where you want to extract the contents
#     # extracted_dir = zip_file_path + 'ext_obs_' + current_yyyy_mm_dd + '/' + time_slot[idx] + '/'
#     destination = '/path/to/obs_archive'
#     extracted_dir = f'{destination}/ext_obs_{year}{month}{day}/{hour}{minute}/'
    
#     if not os.path.exists(extracted_dir):
#         os.makedirs(extracted_dir)
#         # Unzip the files
#         path = directory+filename
#         with tarfile.open(path, 'r:gz') as tar_ref:
#             tar_ref.extractall(extracted_dir)
#             print(f'file for the date {year}{month}{day} at {hour}{minute} is now extracted')
#     else:
#         print(f'This file for the date {year}{month}{day} at {hour}{minute} is already extracted')

# print('Step 1: Finish the extraction of compressed HRIT files')
##### H-000-MSG3__-MSG3________-WV_073___-000008___-202308261145-__

year = '2023'
month ='08'
#days = '26'
days = [f'{i:02}' for i in range(1, 18)]
hours = ['0845','0945','1045','1145','1245','1345','1445','1545',
         '1645','1745','1845','1945','2045','2145','2245','2345']
#hours = ['1145']
for day in days:
    for hour in hours:
        print(f"For {day} at {hour}: started")
        destination_final = f'/etc/ecmwf/scratch/kaan/obs_archive/{year}{month}{day}{hour}.nc'
        main_path = '/ec/res4/hpcperm/kaan/external/sierra-charlie/seviri_2023_08_dd_hh_45_00/'
#        main_path = '/ec/res4/scratch/kaan/convert_hrit_nc/'
        if os.path.exists(destination_final):
            print(f'The Channels for the date {year}{month}{day} at {hour} is already prepared and saved')
        else:
            # load the channels in a defined Scene named scv
            # fnames = glob.glob(extracted_dir+'H*'+ current_yyyy_mm_dd + time_slot[idx] + '*__')
            fnames = glob.glob(f'{main_path}H*{year}{month}{day}{hour}*__')
            print('fnames: ',fnames)
            scn = Scene(reader='seviri_l1b_hrit', filenames=fnames)
            scn.load(['VIS006'])
#            scn.load(['WV_062'])
#            scn.load(['WV_073'])
        
            vis006 = scn['VIS006']
            vis006_data_values = vis006.values / 100
            scn_lon, scn_lat = vis006.attrs['area'].get_lonlats()
        
#            ir0062 = scn['WV_062']
#            ir0073 = scn['WV_073']
#            ir0062_data_values = ir0062.values
#            ir0073_data_values = ir0073.values
        
            # generate the mask and later save it 
#            mask_a = obs_mask3(scn_lon,scn_lat)
#            min_x=np.min(np.where(mask_a==1)[0])
#            max_x=np.max(np.where(mask_a==1)[0])
#            min_y=np.min(np.where(mask_a==1)[1])
#            max_y=np.max(np.where(mask_a==1)[1])
            scene_llbox = scn.crop(ll_bbox=(5.498,22.102, 42.981,51.819)) #AROME-Austria domain
            vis006_llbox = scene_llbox['VIS006']
            cutout_refl = vis006_llbox.values
            cutout_lon,cutout_lat= vis006_llbox.attrs['area'].get_lonlats()
           
#            cutout_lon = scn_lon[min_x:max_x,min_y:max_y]
#            cutout_lat = scn_lat[min_x:max_x,min_y:max_y]
#            cutout_refl = vis006_data_values[min_x:max_x,min_y:max_y]
#            cutout_ir1 = ir0062_data_values[min_x:max_x,min_y:max_y]
#            cutout_ir2 = ir0073_data_values[min_x:max_x,min_y:max_y]
        
            dim1 = cutout_lon.shape[0]
            dim2 = cutout_lon.shape[1]
        
            dt = f'{year}{month}{day}{hour}00'
            dt = str(dt)
            dt = datetime.strptime(dt, '%Y%m%d%H%M%S')
            datetime_info = np.full((dim1, dim2), dt)
        
            masked_data = xr.Dataset(
                {
                    'Reft_VIS06': (('dim1', 'dim2'), cutout_refl),
#                    'BT_WV62': (('dim1', 'dim2'), cutout_ir1),
#                    'BT_WV73': (('dim1', 'dim2'), cutout_ir2),
                    'lon': (('dim1', 'dim2'), cutout_lon),
                    'lat': (('dim1', 'dim2'), cutout_lat),
                    'time': (('dim1', 'dim2'), datetime_info),
                }, coords={'dim1': range(dim1), 'dim2': range(dim2)}
            )
        
            # Define the xarray Dataset


            # Set attributes for variables using .attrs on each variable
            masked_data['Reft_VIS06'].attrs = {'units': 'unitless', 'long_name': 'Observed Reflectance (0.6 μm)'}
#            masked_data['BT_WV62'].attrs = {'units': 'Kelvin', 'long_name': 'Observed Brightness Temperature (6.2 μm)'}
#            masked_data['BT_WV73'].attrs = {'units': 'Kelvin', 'long_name': 'Observed Brightness Temperature (7.3 μm)'}
            masked_data['lon'].attrs = {'units': 'degree north', 'long_name': 'Longitude'}
            masked_data['lat'].attrs = {'units': 'degree south', 'long_name': 'Latitude'}

        
            # Save the xarray Dataset to a NetCDF file
            name = f'{year}{month}{day}{hour}'
            # filename = '/path/to/seviri_obs_dat/masked_obs/monitoring_obs/'+str(name)+'.nc'
            filename = '/etc/ecmwf/scratch/kaan/obs_archive/'+str(name)+'.nc'
        
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted existing file: {filename}")
            # print(masked_data)
        
            masked_data.to_netcdf(filename, engine='netcdf4')
            # print(f'Saved data to {filename}')
            print(f'file for the date {year}{month}{day} at {hour} is now saved')
print('All day saved.')
