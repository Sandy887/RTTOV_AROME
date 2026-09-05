#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author
------
Ms. Sandy Chkeir |  sandychkeir96@gmail.com

Contributions:
- Created in mid 2024
- Modified in mid 2025

| Script workflow      | • Reads results for clear-sky simulation experiments at specific datetimes and hours configured by the user                
|                      | • Loads data via `reflectance_out` object defined in `class_reflectance_out`                                              
"""

import numpy as np
import xarray as xr
import warnings, sys
warnings.filterwarnings('ignore')
from class_reflectance_out import *

paths_1 = "/hpcperm/km4c/experiment_results" #"/etc/ecmwf/scratch/km4c/ml_simulations/visible/"
paths_2 = paths_1 #"/etc/ecmwf/scratch/km4c/visop_runs/summer_period/"

# -------------------------------------------------------------------------------------------------------------------------------------------
# Class objects for operators versions
# -------------------------------------------------------------------------------------------------------------------------------------------

#datetime_slice = list(f"202308{x:02d}_120000" for x in range(1,32))
time_slice = list(f"{x:02d}0000" for x in [6,9,12,15])
visop = reflectance_out("visop", paths_2, "/visop_A_1.nc", clear_ref_period=time_slice)
rttov132 = reflectance_out("rttov132", paths_1, "/rttov_v132_A_1.nc", clear_ref_period=time_slice)
rttov122 = reflectance_out("rttov122", paths_1, "/rttov_v122_A_1.nc", clear_ref_period=time_slice, albedo=True, angle=True)
