"""
This script provides functions for processing hydrometeors in RTTOV models. 

Author
------
Ms. Sandy Chkeir |  sandychkeir96@gmail.com

Created in 2024

It includes calculations of cloud microphysical properties such as:
        - Liquid Water Content (LWC)
        - Ice Water Content (IWC)
        - Effective radius of liquid droplets (Rw)
        - Effective radius of ice particles (Ri)
        - Integrated water paths for ice and liquid clouds

These calculations are intended to support radiative transfer modeling and diagnostics in RTTOV-based workflows.
"""

import xarray as xr
from numpy import *
import numpy as np
import sys
import pandas as pd
from numpy import pi, maximum, zeros

def get_density_rttov(modelvars):
    """
    from rttov_gas_cloud_aerosols_units.dvi
    The unit conversion for clouds: 
    The optical properties of ice and water clouds in RTTOV are parameterized from ice water content (IWC) and liquid water content (LW C) in g m−3
    , respectively. However, NWP models provide cloud information in units of mass mixing ratio (or specific cloud ice or liquid water content) in kg kg−1
    , i.e. ratio between the mass of ice/liquid water and the mass of moist air. If we consider that the air follows the perfect gas law, then the conversion for ice cloud is:
    
    IWC/LWC = q_ice/q_water * density 
    density = 10e2 * P / (R_moist * T) # 10e2 is used to convert from hPa to Pa
    R_moist = ( R / M_dry ) * (1 + (M_dry/M_h2o - 1)*q_h2O)
    
    """
    
    pres = modelvars['P'] if 'P' in modelvars else None
    temp = modelvars['T']   if 'T'   in modelvars else None
    qc   = modelvars['QC']  if 'QC'  in modelvars else None
    qi   = modelvars['QI']  if 'QI'  in modelvars else None
    qv   = modelvars['QV']  if 'QV'  in modelvars else None # qc or q_h2O
    
    # Physical constants and their values in rttov const.F9
    M_dry = 28.9644 # Molar mass of dry air in g/mol (mair in rttov const.F90)
    M_h2o = 18.01528 # Molar mass of water vapor in g/mol (mh2o in rttov const.F90)
    R =  8.3144598 # Ideal gas constant (=NAkB; rgc in rttov const.F90) in J mol−1 K−1
    
    # Calculate the density
    R_moist = ( R / M_dry ) * (1 + (M_dry/M_h2o - 1)*qv)
    rho = pres / (R_moist * temp) # pres in Pa
    rho = rho/1000 # in kg/m3
    # print("R_moist in J.g-1.K-1", R_moist[0,0,0])
    # print("temp in K", temp[0,0,0])
    # print("pres in Pa", pres[0,0,0])
    # print("rho in Kg/m3", rho[0,0,0])
    # sys.exit()
    return rho

def calculate_lwc_iwc(qc_tot, qi_tot, rho_rttov):
    """Calculates Liquid Water Content (LWC) and Ice Water Content (IWC)."""
    lwc = 1000.0 * maximum(qc_tot, 0) * rho_rttov  # g/m3
    iwc = 1000.0 * maximum(qi_tot, 0) * rho_rttov  # g/m3
    return lwc, iwc

def adjust_lwc_iwc_by_cloud_fraction(lwc, iwc, clc):
    """Adjusts LWC and IWC based on cloud fraction."""
    
    # Apply adjustment where cloud fraction (clc) is significant
    cloud_indices = np.where(clc > 1.e-6)  
    lwc[cloud_indices] = lwc[cloud_indices] / clc[cloud_indices]  
    iwc[cloud_indices] = iwc[cloud_indices] / clc[cloud_indices]  
    return lwc, iwc

def calculate_effective_radius_liquid(lwc, NC, rw_min=2.5, rw_max=26.0):
    """Calculates effective radius of liquid droplets (Rw)."""
    rw_tmp = 1.0e+6 * (0.75 * lwc / (pi * 0.67 * NC * 1.0e+6)) ** (1.0 / 3.0)
    rw_tmp = np.where(rw_tmp > rw_max, rw_max, rw_tmp)  # Limit maximum to 26 micrometers
    rw = np.where(rw_tmp < rw_min, rw_min, rw_tmp)  # Limit minimum to 2.5 micrometers
    return rw

def calculate_effective_radius_ice(iwc, temp, ri_min=5.0, ri_max=60.0):
    """Calculates effective radius of ice crystals (Ri) using the Wyser (1998) method."""
    ri_tmp = zeros(iwc.shape)
    ri = zeros(iwc.shape)
    b = zeros(iwc.shape)
    
    # Apply Wyser method where conditions are met
    ice_indices = np.where((iwc > 1e-8) & (temp < 273.0))  
    b[ice_indices] = -2.0 + np.log10(iwc[ice_indices] / 50.0) * ((273.0 - temp[ice_indices])**1.5) * 1e-3
    ri_tmp[ice_indices] = ((3.0*np.sqrt(3.0))/(np.sqrt(3.0)+4.0)) * ( 377.4 + 203.3*b[ice_indices] + 37.91*b[ice_indices]**2 + 2.3696*b[ice_indices]**3 )
    ri_tmp = np.where(ri_tmp > ri_max, ri_max, ri_tmp) # Limit maximum to 60 micrometers
    ri = np.where(ri_tmp < ri_min, ri_min, ri_tmp)  # Limit minimum to 5 micrometers
    return ri

def get_cloud_mircophysics(qc_tot, qi_tot, rho_rttov, clc, temp, z, NC, adjust_by_cloud_fraction=True):
    """Calculates cloud microphysics properties including LWC, IWC, Rw, and Ri.

    Args:
        qc_tot: Total liquid water content.
        qi_tot: Total ice water content.
        z: height (m)
        NC: cloud condensation nuclei, name taken from visop code
        rho_rttov: Air density from RTTOV.
        clc: Cloud cover/fraction.
        temp: Temperature.
        adjust_by_cloud_fraction: Boolean to control whether to adjust water content 
                                  by cloud fraction (default: True).

    Returns:
        A dictionary containing Ri, Rw, LWC, and IWC.
    """
    
    lwc, iwc = calculate_lwc_iwc(qc_tot, qi_tot, rho_rttov)
    print("Using rttov ...")
    print('    min/mean/max IWC ', iwc.min(), iwc.mean(), iwc.max())
    print('    min/mean/max LWC ', lwc.min(), lwc.mean(), lwc.max())

    lwp, iwp = calculate_water_paths(lwc, iwc, z) # Before adjustment
    
    if adjust_by_cloud_fraction:
        lwc, iwc = adjust_lwc_iwc_by_cloud_fraction(lwc, iwc, clc)
        print('    min/mean/max IWC ', iwc.min(), iwc.mean(), iwc.max())
        print('    min/mean/max LWC ', lwc.min(), lwc.mean(), lwc.max())
    
    rw = calculate_effective_radius_liquid(lwc, NC, rw_min=2.5, rw_max=26.0)   
    print('    min/mean/max Rw ', rw.min(), rw.mean(), rw.max())
    
    print('    r_i,eff : using Wyser (1998)')
    ri = calculate_effective_radius_ice(iwc, temp, ri_min=5.0, ri_max=60.0)
    print('    min/mean/max Ri ', ri.min(), ri.mean(), ri.max())
    
    return {"Ri": ri, "Rw": rw, "LWC": lwc, "IWC": iwc, 'LWP':lwp, 'IWP':iwp}

def calculate_water_paths(lwc, iwc, z):
    """Calculates integrated water paths for ice and water clouds.

    Args:
        lwc: Liquid Water Content (LWC) [g/m3].
        iwc: Ice Water Content (IWC) [g/m3].
        z: height (m)

    Returns:
        A dictionary containing LWP and IWP.
    """
    dz = z[:-1,...] - z[1:,...] # Layer thickness [m]
    lwp = (1e-3 * lwc * dz).sum(axis=0)  # kg/m2
    iwp = (1e-3 * iwc * dz).sum(axis=0)  # kg/m2
    print('    min/mean/max LWP ', lwp.min(), lwp.mean(), lwp.max())
    print('    min/mean/max IWP ', iwp.min(), iwp.mean(), iwp.max())
    return lwp, iwp
