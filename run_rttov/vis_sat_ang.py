"""
This script provides functions for handling solar and satellite angles, specifically for the VIS06 SEVIRI solar channel.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

History:
    - Original code by Leonhard Scheck, June 2024.
    - Edited and adapted by Sandy Chkeir, June 2024.

Purpose:
    - Calculate and manage solar/satellite geometry for radiative transfer simulations in the visible spectrum.

Notes:
    - This version is focused on compatibility with RTTOV 12.2.
    - Currently marked for cleanup (see FIXME comment).

TODO:
    - Refactor and streamline code for clarity and maintainability.
"""

angles_pyorbital_flag = True
from numpy import *
from math import *
import numpy as np

def scat_ang(sza_,vza_,saa_,vaa_):
    print(r"sat_ang_vis [scat_ang] - Calculating the scattering angle $\alpha$")
    rel_azi = vza_ - vaa_
    d = pi/180
    cosine_tmp =  np.cos(np.array(d*vza_))*np.cos(np.array(d*sza_)) +np.sin(np.array(d*vza_))*np.sin(np.array(d*sza_))*np.cos(np.array(d*rel_azi))
    alpha = np.arccos(np.array(cosine_tmp))*(180/pi)
    if alpha.min() <= 9 or alpha.max() >= 140:
         print(r"Warning: scattering angle $\alpha$ is out of bounds")
         sys.exit(1)
    else:
         print(r" Done with no warnings - scattering angle $\alpha$ is within the MFASIS limits")
    return alpha

def dphi_to_alpha( theta, theta0, dphi_in, cosines=False, lrt=True, deg=False ) :
    """Compute scattering angle from zenith angles and difference in azimuthal angles of satellite and sun.
       Scattering angle convention (like in libRadtran):
       Backward scattering <-> alpha=pi, Forward scattering <-> alpha=0"""

    if deg:
         f_in = np.pi/180
         f_out = 180/np.pi
    else:
         f_in = 1.0
         f_out = 1.0

    if lrt: # backscattering -> dphi = 180deg
         dphi = np.pi - dphi_in*f_in
    else:    # backscattering -> dphi = 0
         dphi = dphi_in*f_in

    if cosines:
         mu = theta*f_in
         mu0 = theta0*f_in
    else:
         mu    = np.cos(theta*f_in)
         mu0   = np.cos(theta0*f_in)

    alpha = np.arccos(mu*mu0 + np.sqrt(1-mu**2)*np.sqrt(1-mu0**2)*np.cos(dphi))

    if lrt: # backscattering -> alpha = 180deg
         alpha_out = np.pi - alpha
    else:   # backscattering -> alpha = 0
         alpha_out = alpha

    return alpha_out * f_out

def sat_ang_vis(angles_pyorbital, utc_time_var, lon_v1,lat_v1):
    if angles_pyorbital : # pyorbital -- see https://pyorbital.readthedocs.io/en/latest/
        print("Line 712 from vo2_sat_grid.py /vis_op/")
        print('[read_sat_grid] computing sun/sat angles using pyorbital (no scan line dependency)...')
        import pyorbital
        from pyorbital.astronomy import get_alt_az
        from pyorbital.orbital import get_observer_look
        from datetime import datetime, timedelta
        utc_time = datetime(utc_time_var.year, utc_time_var.month, utc_time_var.day, utc_time_var.hour, utc_time_var.minute)
        utc_time -= timedelta(minutes=4) # e.g. 11 minute delay from south pole to Germany
        print('                for time ', utc_time)
        # print("Line: 721 check args.sza_max",args.sza_max)
        #valid = np.where( (sza >= 0) & (sza <= 85.0) )
        sza_po = zeros(lon_v1.shape)
        saa_po = zeros(lon_v1.shape)
        #sza_po[valid], saa_po[valid] = pyorbital.astronomy.get_alt_az(utc_time, lon_v1[valid], lat_v1[valid])
        sza_po, saa_po = pyorbital.astronomy.get_alt_az(utc_time, lon_v1, lat_v1)
        print("[read_sat_grid] Checkinputs - sza should be [0 - 85]")
        sza_po *= 180/pi
        saa_po *= 180/pi
        sza_po = 90.0 - sza_po
        saa_po +=180
        if sza_po.any() < 0:
             print("Warning: sza is negative somewhere!")
        elif sza_po.any() > 85:
             print("Warning: sza exceeds 85 degrees!")
        else:
             print("Solar zenith angle is within MFASIS limits [0 - 85]")
        # pyorbital.astronomy.get_alt_az(utc_time, lon, lat)
        # Return sun altitude and azimuth from utc_time, lon, and lat. [return values seem to be in rad]
        # lon,lat in degrees
        # print("Line: 733 check args.sat_lon",0.0)
        azi = zeros(lon_v1.shape)
        ele = zeros(lon_v1.shape)
        #azi[valid], ele[valid] = pyorbital.orbital.get_observer_look(0.0, 0.0, 36e3, utc_time, lon_v1[valid], lat_v1[valid], 0.0 )
        azi, ele = pyorbital.orbital.get_observer_look(0.0, 0.0, 36e3, utc_time, lon_v1, lat_v1, 0.0)
        # ele may be NaN where it should be 0
        nanidcs =np. where(np.isnan(np.array(ele)))
        ele[nanidcs] = 1e-5
        #print 'ELE is NAN at ', argwhere(isnan(ele))

        # azimuth clockwise from north, elevation = 90deg - zenith angle
        vza_po, vaa_po = 90.0 - ele, azi
        # pyorbital.orbital.get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt)
        # Calculate observers look angle to a satellite. http://celestrak.com/columns/v02n02/
        # utc_time: Observation time (datetime object)
        # lon: Longitude of observer position on ground in degrees east
        # lat: Latitude of observer position on ground in degrees north
        # alt: Altitude above sea-level (geoid) of observer position on ground in km
        # Return: (Azimuth, Elevation)

        return sza_po, saa_po, vza_po, vaa_po

