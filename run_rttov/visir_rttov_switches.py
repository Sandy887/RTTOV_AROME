"""
This script defines RTTOV switches and configuration settings for VIS and IR channel simulations.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

History:
Created in 2023
Modified on June 2024, 23.05.2025
Reorganised in July 2025 (latest date)

Purpose:
    - Provide a centralized place to manage RTTOV options (e.g., sensor settings, channel selection, simulation flags).
    - Enable easy adjustment of switches used across multiple radiative transfer simulations.

Notes:
    - Tailored for RTTOV version compatibility with visible and infrared channel processing.
"""


import glob, sys, os, importlib, timeit
import rttov_op_paths
import numpy as np


class RadiativeTransfer():
    def __init__(self, rttov_op):
        # class attribute
        self.operator = "RTTOV"
        
        if rttov_op=="v122":
            self.operator_version = "12.2"
            self.rttov_installdir = rttov_op_paths.RTTOV # pick the installdir path for the requested version
            print("The user is running RTTOV operator v12.2.")
        elif rttov_op=="v132":
            self.operator_version = "13.2"
            self.rttov_installdir = rttov_op_paths.RTTOV132
            print("The user is running RTTOV operator v13.2.")
        else:
            raise ValueError(f"Unsupported RTTOV version: {rttov_op!r}")

        # Getting pyrttov lib: 1. insert the library and wrapper directories at the front of sys.path
        lib_dir     = os.path.join(self.rttov_installdir, "compilation_dir", "lib")
        wrapper_dir = os.path.join(self.rttov_installdir, "wrapper")
        sys.path.insert(0, lib_dir)
        sys.path.insert(0, wrapper_dir)

        # Getting pyrttov lib: 2. dynamically import the Fortran wrapper module and pyrttov itself
        wrapper = importlib.import_module("rttov_wrapper_f2py")
        pyrttov = importlib.import_module("pyrttov")

        self.pyrttov_lib = pyrttov

def sim_setup_visible(exp_name, rttov_op = 'v122',settings_dict={"clw_scheme":2}, satellite = "SEVIRI MSG-3", chan_list = (1,), chan_names = ('VIS06'), atlas_dir = '/perm/km4c/rttov-13.2/'):
    """
    RTTOV Guide comments:
    
    * Set the options for each Rttov instance:
    - the path to the coefficient file must always be specified
    - turn RTTOV interpolation on (because input pressure levels differ from
      coefficient file levels)
    - set the verbose_wrapper flag to true so the wrapper provides more
      information
    - enable solar simulations for SEVIRI
    - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
      the SEVIRI and MHS simulations)
    
    # Read optical depth and cloud coefficient files together
    # we can set options to ensure that options and coefficients are consistent

    Contributers comments:
    SC-comments:
    * For the simulation setup, set the RTTOV options parameters.
    * This method shall be a class with two different setups for IR/VIS simulations: an easy way to run different simulations.
    * It can be configured for different operators versions, PS: Make sure that the common switchs can work for all added versions :)

    * Expanded by Philipp to have a switch in settings_dict for clw_scheme 2025-04-20. If nothing is defined it defaults to clw_scheme = 2 (deff)
    """
    
    # ------------------------------------------------------------------------
    # Set up Rttov instances for each instrument
    # ------------------------------------------------------------------------
    
    config = RadiativeTransfer(rttov_op=rttov_op)
    config.instrument = satellite
    
    rttov_object = config.pyrttov_lib.Rttov()

    # select channels, can be configured outside the function
    # https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov12/ir_srf/rtcoef_msg_4_seviri_srf.html
    config.chan_list = chan_list
    config.chan_names = chan_names
    config.exp_name = exp_name
    config.nchan = len(chan_list)

    # CLOUD COEFFICIENT files and MFASIS LOOKUPTABLE 
    if rttov_op=='v122':
        rttov_object.FileCoef = '{}/{}'.format(config.rttov_installdir,
                                          "rtcoef_rttov12/rttov9pred54L/rtcoef_msg_3_seviri.dat")
        rttov_object.FileSccld = '{}/{}'.format(config.rttov_installdir,
                            "/rtcoef_rttov12/cldaer_visir/sccldcoef_msg_3_seviri.dat")
        
        # PG: Setting this later on does not work, not sure why.
        if settings_dict['clw_scheme']==1:
            rttov_object.FileMfasisCld = '{}/{}'.format(config.rttov_installdir,"/rtcoef_rttov12/mfasis_lut/rttov_mfasis_cld_msg_3_seviri_opac_v12.2.H5")
            rttov_object.Options.CLWScheme = 1
        if settings_dict['clw_scheme']==2:
            rttov_object.FileMfasisCld = '{}/{}'.format(config.rttov_installdir,"/rtcoef_rttov12/mfasis_lut/rttov_mfasis_cld_msg_3_seviri_deff_v12.2.H5")
            rttov_object.Options.CLWScheme = 2

    elif rttov_op=='v132':
        rttov_object.FileCoef = '{}/{}'.format(config.rttov_installdir,
                                          "rtcoef_rttov13/rttov13pred54L/rtcoef_msg_3_seviri_7gas.dat")
        rttov_object.FileSccld = '{}/{}'.format(config.rttov_installdir,
                            "/rtcoef_rttov13/cldaer_visir/sccldcoef_msg_3_seviri.dat")
        rttov_object.FileMfasisCld = '{}/{}'.format(config.rttov_installdir,"/rtcoef_rttov13/mfasis_lut/rttov_mfasis_cld_msg_3_seviri_deff.H5")

    ### 1. Initialise RTTOV options structure
    rttov_object.Options.AddInterp = True
    rttov_object.Options.InterpMode = 1 # interpolation method
    rttov_object.Options.RegLimitExtrap = True

    rttov_object.Options.Switchrad = False
    rttov_object.Options.DoLambertian = False
    rttov_object.Options.AddRefrac = True
    rttov_object.Options.DtauTest = False
    
    rttov_object.Options.AddSolar = True
    rttov_object.Options.AddClouds = True
    # rttov_object.Options.GridBoxAvgCloud = True
    rttov_object.Options.UserCldOptParam = False
    rttov_object.Options.VisScattModel = 3  # Scattering model for solar source, DOM=1, MFASIS=3
    rttov_object.Options.IrScattModel  = 1 # (Chou scaling)
    rttov_object.Options.CO2Data = False #(default = false)
    rttov_object.Options.OzoneData = False #(default = false)
    rttov_object.Options.UseQ2m = True
    if rttov_op == 'v132': # These are set as in visop
            rttov_object.Options.DomRayleigh = True
            rttov_object.Options.PlaneParallel = False
    # test example options
    rttov_object.Options.VerboseWrapper = True  # Turn on verbose wrapper output
    rttov_object.Options.DomNstreams = 16        # Number of DOM streams to use (8 or 12 are also fine)
    rttov_object.Options.Nthreads = 8            # Take advantage of multiple threads if RTTOV was   compiled with OpenMP
    rttov_object.Options.StoreRad = True         # Store all radiance outputs
    rttov_object.Options.DoCheckinput = True
    rttov_object.Options.Verbose = False # False: do not print warnings
    rttov_object.Options.FixHgpl = True # If true the surface elevation is assigned to the
    # specified surface pressure (default = false)
    
    # for IR and BRDF atlases
    rttov_object.Options.SolarSeaBrdfModel = 2
    rttov_object.Options.IrSeaEmisModel = 2
    
    # ApplyRegLimits=True: Input profiles can be clipped to the regression limits when the limits are exceeded
    rttov_object.Options.ApplyRegLimits = True
    
    # Load the instruments: for HIRS and MHS do not supply a channel list and
    try:
        rttov_object.loadInst(chan_list)
    except config.pyrttov_lib.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Load the emissivity and BRDF atlases
    # ------------------------------------------------------------------------

    # - load data for the month in the profile data
    # - load the IR emissivity atlas data for multiple instruments so it can be used for SEVIRI and HIRS
    # - SEVIRI is the only VIS/NIR instrument we can use the single-instrument initialisation for the BRDF atlas

    irAtlas = config.pyrttov_lib.Atlas()
    #irAtlas.AtlasPath = '{}/{}'.format(config.rttov_installdir, "emis_data")
    irAtlas.AtlasPath = '{}/{}'.format(atlas_dir, "emis_data")
    
    brdfAtlas = config.pyrttov_lib.Atlas()
    brdfAtlas.AtlasPath = '{}/{}'.format(atlas_dir, "brdf_data")
    
    config.rttov_object = rttov_object
    config.irAtlas = irAtlas
    config.brdfAtlas = brdfAtlas
    
    return config

def update_surface_emissivity(
    rttov_object,
    surfemisrefl_seviri,
    surftype,
    dt,
    config,
    verbose
):
    """
    Originally developed by Leonhard Scheck and later adapted by Sandy Chkeir to integrate with this workflow.
    
    Update the RTTOV object's SurfEmisRefl field by filling missing
    emissivity and albedo values, associating with the object, and
    optionally loading and applying IR/BRDF atlases.

    Parameters:
        rttov_object:  An instance of an RTTOV object with a SurfEmisRefl attribute.
        surfemisrefl_seviri:  A 2×N×M array for emissivity (0) and albedo (1).
        surftype:  Surface type array (used for later filling operations).
        dt:  datetime object (used for month extraction).
        config:  Configuration object exposing .irAtlas and .brdfAtlas.
        verbose:  Dict with keys ["emissivity", "albedo", "both", "benchmark", "grid", "sea_brdf"].

    Returns:
        rttov_object with updated SurfEmisRefl.

    Be mindfull that:
    - Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
    - Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
      calcemis/calcrefl TRUE - see RTTOV user guide)
    """
    # 1) Initialize missing user fields, elements < 0 will still be filled by RTTOV
    if verbose.get("emissivity") is None:
        surfemisrefl_seviri[0, :, :] = -1
    else:
        print('There is no user-specified emissivity...')

    if verbose.get("albedo") is None:
        surfemisrefl_seviri[1, :, :] = -1
    else:
        print('There is no user-specified albedo...')

    # 2) Associate with RTTOV object
    if verbose.get("both") is None:
        print('Associating emissivity/albedo with RTTOV object...')
    rttov_object.SurfEmisRefl = surfemisrefl_seviri

    # 3) Fill missing values via atlases if any remain
    if (np.any(surfemisrefl_seviri[1,...] < 0)) or (np.any(surfemisrefl_seviri[0,...] < 0)):
        print('There are still negative albedo/emissivity values -> fill with values from atlas...')
        # Load IR emissivity atlas
        print('loading emissivity atlas...')
        irAtlas = config.irAtlas
        irAtlas.loadIrEmisAtlas(dt.month, ang_corr=True, atlas_id=2)
        irAtlas.IncSea = False # if True use IR atlas for sea surface types
        irAtlas.IncLand = True

        # Load BRDF atlas
        print('loading brdf atlas...')
        brdfAtlas = config.brdfAtlas
        if verbose.get("benchmark"):
            t0 = timeit.default_timer()
        brdfAtlas.loadBrdfAtlas(dt.month, rttov_object)
        if verbose.get("benchmark"):
            print(f'<BENCHMARK> loading BRDF atlas took {timeit.default_timer() - t0:.3f} sec')
        brdfAtlas.IncLand = True # True --> use BRDF atlas for this surface type
        brdfAtlas.IncSea = True
        brdfAtlas.IncSeaIce = True

        if verbose.get("both") is None:
            flags = []
            if brdfAtlas.IncSea: flags.append('sea')
            if brdfAtlas.IncLand: flags.append('land')
            if brdfAtlas.IncSeaIce: flags.append('seaice')
            print('included in BRDF:', ' '.join(flags))
            neg = np.count_nonzero(surfemisrefl_seviri[1, :, 0] < 0)
            # The 2rd dimenion is instrumnet channel in the order 1-12 (0-11 in array order)
            total = surfemisrefl_seviri[1, :, 0].size
            print(f'negative albedo values: {neg} of {total}')

        # Apply atlases
        if verbose.get("benchmark"):
            t1 = timeit.default_timer()
        try:
            # Do not supply a channel list for SEVIRI: 
            #this returns emissivity/BRDF values for all
            # *loaded* channels which is what is required
            if verbose.get("both") is None:
                print('calling getEmisBrdf...')
            surfemisrefl_seviri[0,...] = irAtlas.getEmisBrdf(rttov_object)
            surfemisrefl_seviri[1,...] = brdfAtlas.getEmisBrdf(rttov_object)
        except pyrttov.RttovError as e:
            # If there was an error the emissivities/BRDFs 
            # will not have been modified so it
            # is OK to continue and call RTTOV with 
            # calcemis/calcrefl set to TRUE everywhere
            sys.stderr.write(f"Error calling atlas: {e}\n")
        if verbose.get("benchmark"):
            print(f'<BENCHMARK> getting albedo/emissivity took {timeit.default_timer() - t1:.3f} sec')
        if verbose["both"] == None:
             print('albedo histogram: ', np.histogram( surfemisrefl_seviri[1,...].ravel(), np.arange(0,0.31,0.01) ) )

             print('negative albedo values: {} of {}'.format( np.count_nonzero( surfemisrefl_seviri[1,:,0] < 0.0 ),
                                                             surfemisrefl_seviri[1,:,0].size ))
             idcs_l = np.where( (surfemisrefl_seviri[1,:,0] < 0.0) & (surftype[:,0] == 0) )
             idcs_s = np.where( (surfemisrefl_seviri[1,:,0] < 0.0) & (surftype[:,0] >  0) )
             print('({} over land, {} over sea)'.format( len(idcs_l[0]), len(idcs_s[0]) ))
    else:
        if verbose.get("both") is None:
            print('no need to load emissivity or BRDF atlas...')

    if np.any(surfemisrefl_seviri[1,:,0] < 0):
         if verbose["grid"] is None :
             idcs_nl = np.where( (surfemisrefl_seviri[1,:,0] <= 0.0) & (surftype[:,0] == 0) )
             if len(idcs_nl[0]) > 0 :
                 idcs_l = np.where( (surfemisrefl_seviri[1,:,0] > 0.0) & (surftype[:,0] == 0) )
                 alb_l = surfemisrefl_seviri[1,:,0][idcs_l].mean()
                 if len(idcs_l[0]) > 0 :
                     alb_l = surfemisrefl_seviri[1,:,0][idcs_l].mean()
                 else :
                     alb_l = 0.0
                     print('WARNING: No land pixels -> cannot compute average land albedo value')
                 #print("Line 2405")
                 surfemisrefl_seviri[1,:,0][idcs_nl] = alb_l
                 print('WARNING: Replaced {} negative land albedo values with average land albedo value {}'.format( len(idcs_nl[0]), alb_l*np.pi ))
         else:
             print('overwriting negative albedo values with data from neighbour pixels...')
             alb = surfemisrefl_seviri[1,:,0] + 0.0
             surfemisrefl_seviri[1,:,0] = fill_negative_from_neighbours( alb, verbose["grid"] ) # to be implemented

         if verbose["both"] is None:
             print('negative albedo values: {} of {}'.format( np.count_nonzero( surfemisrefl_seviri[1,:,0] < 0.0 ),surfemisrefl_seviri[1,:,0].size ))
             idcs_l = np.where( (surfemisrefl_seviri[1,:,0] < 0.0) & (surftype[:,0] == 0) )
             idcs_s = np.where( (surfemisrefl_seviri[1,:,0] < 0.0) & (surftype[:,0] >  0) )
             print('({} over land, {} over sea)'.format( len(idcs_l[0]), len(idcs_s[0]) ))
    if verbose["sea_brdf"]:
         if verbose["both"] is None: print('setting sea albedo values to -1 --> will be replaced by sea brdf model values...')
         idcs_s = np.where( surftype[:,0] == 1 )
         surfemisrefl_seviri[1,:,0][idcs_s] = -1.0
         print('negative albedo values: {} of {}'.format( np.count_nonzero( surfemisrefl_seviri[1,:,0] < 0.0 ),surfemisrefl_seviri[1,:,0].size ))

    return surfemisrefl_seviri

def sim_setup_infrared(exp_name, satellite = "SEVIRI MSG-3", atlas_dir = '/perm/km4c/rttov-13.2/'):
    """
    SC-comments:
    Similar to the visible setup but for IR instead. Currently configured for rttov 12.2
    """
    
    # ------------------------------------------------------------------------
    # Set up Rttov instances for each instrument
    # ------------------------------------------------------------------------
    
    config = RadiativeTransfer(rttov_op='v122')
    config.Instrument = satellite
    
    rttov_object = config.pyrttov_lib.Rttov()

    # select channels
    chan_list = (5, 6)
    config.chan_list = chan_list
    config.nchan_seviri = len(chan_list)
    config.chan_seviri_names = ('WV62', 'WV73')
    config.exp_name = exp_name
    
    rttov_object.FileCoef = '{}/{}'.format(config.rttov_installdir,
                                          "rtcoef_rttov12/rttov9pred54L/rtcoef_msg_3_seviri.dat")
#     CLOUD COEFFICIENT files
    rttov_object.FileSccld = '{}/{}'.format(config.rttov_installdir,
                            "/rtcoef_rttov12/cldaer_visir/sccldcoef_msg_3_seviri.dat")
    
    ### 1. Initialise RTTOV options structure
    rttov_object.Options.AddInterp = True
    rttov_object.Options.InterpMode = 1 # interpolation method
    
    rttov_object.Options.Switchrad = True
    rttov_object.Options.DoLambertian = False
    rttov_object.Options.AddRefrac = True
    rttov_object.Options.DtauTest = False
    
    rttov_object.Options.AddSolar = False
    rttov_object.Options.AddClouds = True
    # rttov_object.Options.GridBoxAvgCloud = True
    rttov_object.Options.UserCldOptParam = False
    rttov_object.Options.VisScattModel = 1  # Scattering model for solar source, DOM=1, MFASIS=3
    rttov_object.Options.IrScattModel  = 2 # (Chou scaling)
    rttov_object.Options.CO2Data = False #(default = false)
    rttov_object.Options.OzoneData = False #(default = false)
    rttov_object.Options.UseQ2m = False
    # test example options
    rttov_object.Options.VerboseWrapper = True  # Turn on verbose wrapper output
    rttov_object.Options.DomNstreams = 16        # Number of DOM streams to use (8 or 12 are also fine)
    rttov_object.Options.Nthreads = 8            # Take advantage of multiple threads if RTTOV was   compiled with OpenMP
    rttov_object.Options.StoreRad = True         # Store all radiance outputs
    rttov_object.Options.DoCheckinput = True
    rttov_object.Options.Verbose = False # False: do not print warnings
    rttov_object.Options.FixHgpl = False # If true the surface elevation is assigned to the
    # specified surface pressure (default = false)
    
    # for IR and BRDF atlases
    rttov_object.Options.SolarSeaBrdfModel = 2
    rttov_object.Options.IrSeaEmisModel = 1
    
    # ApplyRegLimits=True: Input profiles can be clipped to the regression limits when the limits are exceeded
    rttov_object.Options.ApplyRegLimits = True
    
    # Load the instruments: for HIRS and MHS do not supply a channel list and
    try:
        rttov_object.loadInst(chan_list)
    except config.pyrttov_lib.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Load the emissivity and BRDF atlases
    # ------------------------------------------------------------------------

    # - load data for the month in the profile data
    # - load the IR emissivity atlas data for multiple instruments so it can be used for SEVIRI and HIRS
    # - SEVIRI is the only VIS/NIR instrument we can use the single-instrument initialisation for the BRDF atlas

    irAtlas = config.pyrttov_lib.Atlas()
    #irAtlas.AtlasPath = '{}/{}'.format(config.rttov_installdir, "emis_data")
    irAtlas.AtlasPath = '{}/{}'.format(atlas_dir, "emis_data") 
    
    
    brdfAtlas = config.pyrttov_lib.Atlas()
    brdfAtlas.AtlasPath = '{}/{}'.format(atlas_dir, "brdf_data")
    
    config.rttov_object = rttov_object
    config.irAtlas = irAtlas
    config.brdfAtlas = brdfAtlas
    
    return config