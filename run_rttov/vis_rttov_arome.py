"""
This script generates a single visible‑channel scene for a specified date by running RTTOV on AROME‑Austria forecast data. By default it processes the SEVIRI MSG‑3 0.6 µm channel, but it’s designed to be easily extended to additional channels or other sensors.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com


Revision history
----------------
* 2023‑08‑16 – initial commit (new_runs)
* 2024‑07‑09 – edits
* 2024‑10‑09 – major changes
* 2024‑10‑14 – added interpolation

------------------------------------------------------------------------
The usual steps to take when running RTTOV are as follows:
------------------------------------------------------------------------
  1. Specify required RTTOV options
  2. Read coefficients
  3. Allocate RTTOV input and output structures
  4. Set up the chanprof array with the channels/profiles to simulate
  5. Read input profile(s)
  6. Set up surface emissivity and/or reflectance
  7. Call rttov_direct and store results
  8. Deallocate all structures and arrays

  ! If nthreads is greater than 1 the parallel RTTOV interface is used.
  ! To take advantage of multi-threaded execution you must have compiled
  ! RTTOV with openmp enabled. See the user guide and the compiler flags.
"""

import sys, os, importlib
import glob, time, datetime, timeit, pytz

import xarray as xr
import numpy as np
import pandas as pd

import rttov_op_paths

from global_land_mask import globe
from pysolar.solar import get_altitude, get_azimuth

from vis_sat_ang import *
from visir_obs_utils import map_to_sat_grid_edited
from visir_profiles_arome import *
from cloud_hydro_proc import *
from visir_rttov_switches import update_surface_emissivity

def run_rttov(grbs, index, dt,config, run, channel_used, settings_dict, output_root, rttov_op= "v122", repo_path='./',specific_output_name = ''):
    """
    Is the main funtion to run RTTOV. It generates a single visible‑channel scene using RTTOV and AROME‑Austria forecasts.

    Args:
        List ([grbs]):
            List of GRIB files (from pygrib) containing model forecasts.
        index (int):
            Index into `grbs` for the desired forecast field. It's 0 by default.
        dt (datetime.datetime):
            Target date/time for the simulation.
        config (dict):
            RTTOV configuration settings (e.g., coefficient paths, options).
        run (Bool):
            Switch to run the RTTOV model after reading and processing the forecast data.
        channel_used (int):
            SEVIRI channel number to simulate (e.g., 1 for the 0.6 µm visible band).
        settings_dict (dict):
            RTOV_OP switches and other simulation flags.
        rttov_op (str, optional):
            RTTOV version identifier (default: "v122").
        repo_path (str, optional):
            Base path to the cloned RTTOV_AROME repository (default: "./").
        specific_output_name (str, optional):
            Custom filename (or suffix) for the output file (default: "").
        output_root (str):
            Directory in which to write all simulation results.

    Returns:
    bool: 
        True if RTTOV was run (based on the `run` argument), 
        False otherwise.

    """
    
    '''
    ------------------------------------------------------------------------
     Import pyrttov library corresponding to RTTOV version chosen by the user
    ------------------------------------------------------------------------
    '''
    # base path for the requested version
    if rttov_op == "v122":
        path_rttov = rttov_op_paths.RTTOV
    elif rttov_op == "v132":
        path_rttov = rttov_op_paths.RTTOV132
    else:
        raise ValueError(f"Unsupported RTTOV version: {rttov_op!r}")
    
    # library and wrapper directories at the front of sys.path
    lib_dir     = os.path.join(path_rttov, "compilation_dir", "lib")
    wrapper_dir = os.path.join(path_rttov, "wrapper")
    sys.path.insert(0, lib_dir)
    sys.path.insert(0, wrapper_dir)

    # dynamic import of the Fortran wrapper module and pyrttov library
    wrapper = importlib.import_module("rttov_wrapper_f2py")
    pyrttov = importlib.import_module("pyrttov")

    rttov_installdir = path_rttov
    
    '''
    ------------------------------------------------------------------------
     Get inputs from grib file and assemble the modelstate dict
    ------------------------------------------------------------------------
    '''
    
    modelstate = get_modelstate(grbs, index)
    
    print("run_rttov - read the nprofiles and levels")
    nlevels = modelstate['T'].shape[0]
    dim1, dim2 = modelstate['T'][0,:,:].shape
    nprofiles = dim1 * dim2

    print(f'run_rttov - total profiles are {nprofiles} for each hybrid level (in total {nlevels} levels)')
    print("run_rttov - read the latlon coordinates ")
    
    # Reshape lats and lons into 1D arrays of nprofiles size
    latitude = modelstate['2d-lat'].flatten()
    longitude = modelstate['2d-lon'].flatten()
    print(f"latitude min/max/size: {latitude.min()}/{latitude.max()}/{latitude.shape}")
    print(f"longitude min/max/size: {longitude.min()}/{longitude.max()}/{longitude.shape}")

    print("=== processing channel VIS006 ===")
    
    '''
    ---------------------------------------------------------------------------------
     Compute Liquid water content in g/m3 and liquid water path in Kg/m2: LWC & LWP
     Compute Ice water content in g/m3 and Ice water path in Kg/m2: IWC & IWP
     Computation is done prior to reformating the profiles
    ---------------------------------------------------------------------------------
    '''
    rho_rttov = get_density_rttov(modelstate)
    print("min/mean/max rho_rttov in Kg/m3: ", rho_rttov.min(), rho_rttov.mean(), rho_rttov.max())
    
    z = modelstate['HHL'] if 'HHL' in modelstate else None
    qc_tot = modelstate['QC_DIA']
    qi_tot = modelstate['QI_DIA'] + settings_dict['snow2ice']*modelstate['QS']
    temp = modelstate['T']
    clc = modelstate['CLC']
    NC = settings_dict["NC"] # Cloud condensation nuclei CCN, another name: total droplet concetration Ntot from (Martin et al, 1994)
    hydros = {"Ri": None, "Rw": None, "LWC": None, "IWC": None, 'LWP':None, 'IWP':None}
    hydros = get_cloud_mircophysics(qc_tot, qi_tot, rho_rttov, clc, temp, z, NC, adjust_by_cloud_fraction=True)

    '''
    ---------------------------------------------------------------------------------
     Reformat the variables into nlevels profiles
    ---------------------------------------------------------------------------------
    '''

    for key, arr in modelstate.items():
        if (key != "2d-lat") and (key != "2d-lon"):
            modelstate[key] = reformat_profile(arr, nprofiles)

    for key, arr in hydros.items():
        if (key == "Ri") or (key == "Rw"):
            hydros[key] = reformat_profile(arr, nprofiles)

    '''
    ---------------------------------------------------------------------------------
     Calculate position of the sun (solar angles) and the satellite  (sat angles)
    ---------------------------------------------------------------------------------
    '''
    
    # Make the datetime object timezone-aware
    datetime_dt = make_dt_timezone_utc(dt)

    if settings_dict["fixed_time"]:
        # aims to use the atmospheric output fields at different hours of the day
        new_dt = datetime_dt.replace(hour=settings_dict["cte_hour"])
        datetime_dt = new_dt
    print(f"run_rttov - the datetime in UTC    {datetime_dt}")
    print('run_rttov [sat_ang_vis] - calculating solar angles using pyorbital')
    sunzen, sunazi, satzen, satazi = sat_ang_vis(True, datetime_dt, longitude, latitude)

    sza, saa, vza, vaa = np.asfortranarray(sunzen), np.asfortranarray(sunazi), \
         np.asfortranarray(satzen), np.asfortranarray(satazi)
    phi = vaa - saa
    idcs = np.where( phi < 0.0 )
    phi[idcs] += 360.0

    inval = np.where( (sza < 0) | (sza >= 90) | (vza < 0) | (vza >= 90) )
    valid = np.where( (sza >= 0) & (sza < 90) & (vza >= 0) & (vza < 90) )

    if len(inval[0]) > 0 :
        print('WARNING: THERE ARE {} PIXELS WITH INVALID ANGLES!'.format(len(inval[0])))
        
    alpha = np.asfortranarray( 180. - dphi_to_alpha( vza*np.pi/180, sza*np.pi/180, phi*np.pi/180 )*180./pi ) # alpha=0 <-> backscattering

    scatter_angle= np.reshape(alpha, (dim1, dim2))
    if True:
        print('    angles used to get scattering angle alpha:')
        print('      sza    ', sza.shape, sza.min(), sza.mean(), sza.max())
        print('      saa    ', saa.shape, saa.min(), saa.mean(), saa.max())
        print('      vza    ', vza.shape, vza.min(), vza.mean(), vza.max())
        print('      vaa    ', vaa.shape, vaa.min(), vaa.mean(), vaa.max())
        print('      alpha  ', alpha.shape, alpha.min(), alpha.mean(), alpha.max())

    satazi = satazi - sunazi
    sunazi = 0*sunazi - 180
    print('    angles used as inputs to rttov_object:')
    print('                vza min/mean/max = ', satzen.min(), satzen.mean(), satzen.max())
    print('                vaa min/mean/max = ', satazi.min(), satazi.mean(), satazi.max())
    print('                sza min/mean/max = ', sunzen.min(), sunzen.mean(), sunzen.max())
    print('                saa min/mean/max = ', sunazi.min(), sunazi.mean(), sunazi.max())

    # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
    angles = np.array([[0, 0, 0, 0] for i in range(nprofiles)], dtype=np.float64)
    angles[:,0] = satzen
    angles[:,1] = satazi
    angles[:,2] = sunzen
    angles[:,3] = sunazi
    
    '''
    ---------------------------------------------------------------------------------
     Get surface (type, geometery), fetch, s2m, skin, and datetimes
    ---------------------------------------------------------------------------------
    '''
    
    surftype = np.zeros((nprofiles,2))
    surfgeom = np.zeros((nprofiles,3))
        
    surfgeom[:,0] = latitude
    surfgeom[:,1] = longitude
    surfgeom[:,2] = modelstate['Z'][:,0]/1000

    is_on_land = globe.is_land(latitude, longitude)
    land_array = 1 - is_on_land

    surftype[:,0] = land_array.astype(int)  # 0 if land, 1 if sea, 2 if seaice
    surftype[:,1] = np.where(surftype[:,0] == 1, 1, 0) # 0 if fresh water, 1 if ocean

    idcs_l = np.where( surftype[:,0] == 0 )
    idcs_s = np.where( surftype[:,0] >  0 )
    print('land pixels: {}, non-land pixels: {}'.format( len(idcs_l[0]), len(idcs_s[0]) ))
    
    # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
    skin  = np.array([[270., 35.0, 0.0, 0.0, 3.0, 5.0, 15.0, 0.1, 0.3] for i in range(nprofiles)], dtype=np.float64)
    
    skin[:,0] = modelstate['T'][:,-1]
    print("The skin temperature min/mean/max: ", skin[:,0].min(), skin[:,0].mean(), skin[:,0].max())
    
    # surface data = lowest model half level data
    fetch = 1e5*np.ones(nprofiles)  # default
    s2m = np.stack([modelstate['P'][:,-1]/100, modelstate['T'][:,-1], 
                    modelstate['QV'][:,-1], modelstate['U10M'][:,0], modelstate['V10M'][:,0],
                    fetch],axis=1).astype(np.float64)
    
    # Datetime information
    datetime_var = np.full((dim1, dim2), dt)
    
    utc_time = datetime.datetime(datetime_dt.year, datetime_dt.month, 
                                 datetime_dt.day, datetime_dt.hour, datetime_dt.minute)
    valid_date =  np.array([utc_time.year, utc_time.month, 
                            utc_time.day, utc_time.hour, utc_time.minute, utc_time.second])
    utc_time -= datetime.timedelta(minutes=15) # e.g. 11 minute delay from south pole to Germany
    valid_date_profiles =  np.array([utc_time.year, utc_time.month, utc_time.day, utc_time.hour, 
                                     utc_time.minute, utc_time.second])
    datetimes = np.array([valid_date_profiles for i in range(nprofiles)], dtype=np.int32)
    print("The valid datetime is ", valid_date)
    
    '''
    ---------------------------------------------------------------------------------
     Set up the profile data
    ---------------------------------------------------------------------------------
    '''
    
    rttov_object = config.rttov_object
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    # Gas units
    gas_units = 1  # 1 for kg/kg over dry air, 2 for ppmv over moist air
    mmr_cldaer = 1   # kg/kg (cld+aer) units
    
    myProfiles.GasUnits = gas_units
    myProfiles.P = modelstate['P']/100 # [hPa],   [n_profiles,n_levels]
    myProfiles.T = modelstate['T']
    myProfiles.Q = np.where(modelstate['QV'] < 0.1000E-10, 0.1000E-10, modelstate['QV']) # this is set like that in visop
    myProfiles.Angles = angles
    myProfiles.S2m = s2m
    myProfiles.Skin = skin  # Specification of the surface pressure
    myProfiles.SurfType = surftype
    myProfiles.SurfGeom = surfgeom
    myProfiles.DateTimes = datetimes
    print("The valid datetime used in rttov122 is ", myProfiles.DateTimes)
    myProfiles.Cfrac = modelstate['CLC']
    
    # WATER CLOUDS
    cloudscheme = np.ones((nprofiles, 2), dtype=np.int32)
    cloudscheme[:,0] = rttov_object.Options.CLWScheme  # clw_scheme : (1) OPAC or (2) Deff scheme
    cloudscheme[:,1] = 1
    if config.operator_version=="13.2":
        cloudscheme[:,0] = 2
        cloudscheme[:,1] = 1
        myProfiles.ClwScheme = cloudscheme
    if config.operator_version=="12.2":
        myProfiles.ClwScheme = cloudscheme[:,0]

    # effective diameter: minimum is 2 microns
    myProfiles.Clwde = settings_dict['lwc'] * hydros['Rw'] * 2
    
    # Cloud types - concentrations in kg/kg
    myProfiles.Stco = settings_dict['lwc'] * modelstate['QC_DIA']  # Stratus Continental STCO
    myProfiles.Stma =  0 * modelstate['QC_DIA'] # Stratus Maritime STMA
    myProfiles.Cucc = 0 * modelstate['QC_DIA']  # Cumulus Continental Clean CUCC
    myProfiles.Cucp = 0 * modelstate['QC_DIA']  # Cumulus Continental Polluted CUCP
    myProfiles.Cuma = 0 * modelstate['QC_DIA']  # Cumulus Maritime CUMA

    # icecloud[2][nprofiles]: ice_scheme, idg
    icecloud = np.array([[1,2] for i in range(nprofiles)], dtype=np.int32)
    myProfiles.IceCloud = icecloud
    # effective diamete minimum 10 microns
    myProfiles.Icede = hydros['Ri']*2
    myProfiles.Cirr = modelstate['QI_DIA'] + settings_dict['snow2ice'] * modelstate['QS'] # all ice clouds CIRR
    rttov_object.Profiles = myProfiles

    '''
    ---------------------------------------------------------------------------------
     Determine albedo / emissivity
    ---------------------------------------------------------------------------------
    '''

    print('Setting up emissivity/albedo array...')

    if config.operator_version=="13.2":
        #Learnt from VISOP code. Not sure what are the other 3 dimenions
        surfemisrefl_seviri = np.zeros((5,nprofiles,config.nchan), dtype=np.float64)
    else:
        surfemisrefl_seviri = np.zeros((2,nprofiles,config.nchan), dtype=np.float64)
    
    verbose = dict(albedo=None, emissivity=None, both = None, 
                   thermal = True, benchmark=True,grid = None, sea_brdf=True)
    
    rttov_object.SurfEmisRefl = update_surface_emissivity(rttov_object,
                                             surfemisrefl_seviri, surftype,
                                             dt,
                                             config,
                                             verbose
                                            )
    '''
    ---------------------------------------------------------------------------------
     Call RTTOV
    ---------------------------------------------------------------------------------
    '''
    
    if run == False:
        print("Visualizations purposes when run = False")
        sys.exit(1)
    config.rttov_object.printOptions()

    # call RTTOV

    if verbose["both"] is None: print('Calling runDirect on RTTOV object...')

    if verbose["benchmark"]:
        t0 = timeit.default_timer()

     # Call the RTTOV direct model for each instrument:
     # no arguments are supplied to runDirect so all loaded channels are
     # simulated
    try:
        rttov_object.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)

    if verbose["benchmark"]:
        t1 = timeit.default_timer()
        #print('<BENCHMARK> call_rttov took {}s, runDirect took {}s.'.format(t1-t_call_rttov, t1-t0))

    if verbose["both"] is None:
        print('negative albedo values: {} of {}'.format( np.count_nonzero( surfemisrefl_seviri[1,:,0] < 0.0 ),
                                                         surfemisrefl_seviri[1,:,0].size ))


    computed_emissivity = rttov_object.SurfEmisRefl[0, :, :]
    computed_brdf = rttov_object.SurfEmisRefl[1, :, :]

    # Reft_VIS06: Simulated visible channel reflectances
    
    sky_conditions = ['all_sky', 'clear']
    datetime_var = np.full((dim1, dim2), dt)

    refl_out = np.zeros((dim1,dim2, 2))
    albedo = np.zeros((dim1,dim2))
    scatter_angle = np.zeros((dim1,dim2))

    # Reflectances in all- and clear- skies
    refl_out[:,:,0] = np.reshape(rttov_object.Refl[:, 0],(dim1,dim2)) # All-sky Refl
    refl_out[:,:,1] = np.reshape(rttov_object.ReflClear[:, 0],(dim1,dim2)) # Clear Refl

    # Albedo and scattering angle
    albedo = np.reshape(np.pi*computed_brdf[:,0], (dim1,dim2))
    tmp_angle = scat_ang(sunzen,satzen,sunazi,satazi)
    scatter_angle = np.reshape(tmp_angle, (dim1, dim2))

    # interpolate the outputs onto observation grid saved by visop operator
    grid_obs = {"lon":np.load(repo_path+"/RTTOV_AROME/obs_grid/O_grid_lon.npy"),"lat":np.load(repo_path+"/RTTOV_AROME/obs_grid/O_grid_lat.npy")} # observation grid from visop results
    grid_model = np.stack([latitude, longitude], axis=1)
    dim3, dim4 = grid_obs["lon"].shape
    print(f"[run_rttov]     new dimensions of interpolated outputs are ({dim3},{dim4})")

    refl_out_interpolated = np.zeros((dim3, dim4, 2))
    albedo_interpolated = np.zeros((dim3, dim4))
    scatter_angle_interpolated = np.zeros((dim3, dim4))
    sunzen_interpolated = np.zeros((dim3, dim4))
    sunazi_interpolated = np.zeros((dim3, dim4))
    satzen_interpolated = np.zeros((dim3, dim4))
    satazi_interpolated = np.zeros((dim3, dim4))
    iwp_interpolated = np.zeros((dim3, dim4))
    lwp_interpolated = np.zeros((dim3, dim4))

    refl_out_interpolated[:,:,0], _ = map_to_sat_grid_edited(refl_out[:,:,0], channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False , repo_path=repo_path )
    refl_out_interpolated[:,:,1], _ = map_to_sat_grid_edited(refl_out[:,:,1], channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    albedo_interpolated, _          = map_to_sat_grid_edited(albedo, channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False,repo_path=repo_path )
    scatter_angle_interpolated, _   = map_to_sat_grid_edited(scatter_angle, channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    sunzen_interpolated, _          = map_to_sat_grid_edited(np.reshape(sunzen, (dim1, dim2)), channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    satzen_interpolated, _          = map_to_sat_grid_edited(np.reshape(satzen, (dim1, dim2)), channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    sunazi_interpolated, _          = map_to_sat_grid_edited(np.reshape(sunazi, (dim1, dim2)), channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    satazi_interpolated, _          = map_to_sat_grid_edited(np.reshape(satazi, (dim1, dim2)), channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    iwp_interpolated, _             = map_to_sat_grid_edited(hydros['IWP'], channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)
    lwp_interpolated, _             = map_to_sat_grid_edited(hydros['LWP'], channel='', varname='refl', plot_only_latlon=False, 
                                                             vmin=None, vmax=None, cmap=None, save_tile=False, dbg=False ,repo_path=repo_path)

    dt_string = f'{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
    final_name = dt_string
    valid_datetime = dt_string
    
    sim_out = xr.Dataset(
    {
        'synthetic_reflectance_VIS006': (('lat_obs', 'lon_obs', 'sky_condition'), refl_out_interpolated),
        'synthetic_albedo':(('lat_obs', 'lon_obs'), albedo_interpolated),
        'scatter_angle':(('lat_obs', 'lon_obs'),scatter_angle_interpolated),
        'SZA':(('lat_obs', 'lon_obs'), sunzen_interpolated),
        'SAA':(('lat_obs', 'lon_obs'), sunazi_interpolated),
        'VZA':(('lat_obs', 'lon_obs'), satzen_interpolated),
        'VAA':(('lat_obs', 'lon_obs'), satazi_interpolated),
        'IWP':(('lat_obs', 'lon_obs'), iwp_interpolated),
        'LWP':(('lat_obs', 'lon_obs'), lwp_interpolated),
        'lat_obs_map':(('lat_obs', 'lon_obs'), grid_obs["lat"]),
        'lon_obs_map':(('lat_obs', 'lon_obs'), grid_obs["lon"]),
        'lat_arome_map':(('dim1', 'dim2'), latitude.reshape(dim1,dim2)),
        'lon_arome_map':(('dim1', 'dim2'), longitude.reshape(dim1,dim2)),
        'valid_datetime': (('k'), [valid_datetime]),
        'experiment': (('k'), [config.exp_name])
    }, coords={'dim1': range(dim1), 'dim2': range(dim2), 'lat_obs': range(dim3), 'lon_obs': range(dim4), 'sky_condition': sky_conditions, 'k':range(1)})

    variable_attrs = {
    'synthetic_reflectance_VIS006': {'units': 'unitless', 'long_name': 'synthetic visible reflectance (0.6 μm)'},
    'synthetic_albedo': {'units': 'unitless', 'long_name': 'surface albedo = RTTOV_BRDF * pi'},
    'scatter_angle': {'units': 'degree', 'long_name': 'Scattering angle'},
    'SZA': {'units': 'degree', 'long_name': 'sun zenith angle'}, 
    'SAA': {'units': 'degree', 'long_name': 'sun azimuth angle'},
    'VZA': {'units': 'degree', 'long_name': 'sat zenith angle'},
    'VAA': {'units': 'degree', 'long_name': 'sat azimuth angle'},
    'IWP': {'units': 'Kg/m2', 'long_name': 'integrated ice water path'},
    'LWP': {'units': 'Kg/m2', 'long_name': 'integrated liquid water path'},
    'lat_obs_map': {'units': 'degree north', 'long_name': 'latitude of observations map saved by visop operator'},
    'lon_obs_map': {'units': 'degree east', 'long_name': 'longitude of observations map saved by visop operator'},
    'lat_arome_map': {'units': 'degree north', 'long_name': 'latitude arome model'},
    'lon_arome_map': {'units': 'degree east', 'long_name': 'longitude arome model'},
    'valid_datetime': {'units': 'UTC', 'long_name': 'valid forecast time'},
    'experiment': {'long_name': 'experiment name'}}

    for var_name, attrs in variable_attrs.items():
        sim_out[var_name].attrs = attrs

    # Save the xarray Dataset to a NetCDF file
    if len(specific_output_name)==0:
        save_dataset_to_netcdf(sim_out, final_name, channel_used, config.exp_name, output_root)
    else: 
        sim_out.to_netcdf(specific_output_name)

    print('[run_rttov]. Done')

    # ------------------------------------------------------------------------
    # Deallocate memory
    # ------------------------------------------------------------------------

    # Because of Python's garbage collector, there should be no need to
    # explicitly deallocate memory

    return run

