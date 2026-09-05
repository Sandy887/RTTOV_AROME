"""
Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com

Contributions:
- Created in mid 2024
- Modified in mid 2025

In this script, the code:
- Derives observed clear-sky reflectance from xarray datasets using a darkest-pixel approach.
- Derives synthetic clear-sky reflectance (darkest-pixel or mean-pixel) based on operator version.
- Extracts albedo atlases, scattering, solar zenith, and viewing zenith angles at darkest-pixel locations.
- Encapsulates workflows in a `reflectance_out` class for organized data processing.
- Includes an `AromeHydrometeors` loader class for AROME hydrometeor datasets.

PS:
- Loads observed data exclusively from the “visop” operator version.
- Supports synthetic data from any operator version (tested with rttov122 and rttov132).
- Derives albedo atlases and angle products only from rttov122 outputs
  (initially generated with rttov122, whose inputs match those of rttov132 and visop).

Data input summary
| Category             | Details                                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------------------|
| Dataset variables    | • `observed_reflectance_VIS006` (visop only)                                                                              |
|                      | • `synthetic_reflectance_VIS006`                                                                                          |
|                      | • `clearsky_reflectance_VIS006` (for visop clearsky synthetic reflectance)                                                |
|                      | • `scatter_angle`, `SZA`, `VZA`                                                                                           |
| Operator versions    | • `"visop"` for observed data                                                                                             |
|                      | • any `"rttovxxx"` (e.g. `rttov122`, `rttov132`) for synthetic data                                                       |
|                      | • only `"rttov122"` supports `albedo_atlas` and angle extraction                                                          |
| Datetime formatting  | • Format: `YYYYMMDD_HHMMSS` (e.g. `20230817_120000`)                                                                      |
|                      | • Note: `20230817_180000` is skipped due to missing data                                                                  |
| Mask file            | • Path: `/path/to/RTTOV_AROME/Masks_visible/clr_sky_filter.npy`                                                         |
|                      | • Must exist and match the spatial dimensions of your dataset                                                             |
| Path handling        | • Provide `dirpath` and `filepath` separately (or use `pathlib.Path`)                                                     |
|                      | • They're concatenated to locate the NetCDF file for xarray                                                               |
| Key naming           | • Uses the last 6 characters of the datetime string (i.e.`"HHMMSS"`) as the lookup key in output dictionaries             |
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
from scipy.signal import convolve2d

import numpy as np
import xarray as xr


def observed_albedo(
    data: xr.Dataset,
    datetimes: List[str],
    *,
    method: Optional[str] = None
) -> xr.DataArray:
    """
    Derive observed clear‐sky reflectance by darkest‐pixel over a given period.

    Args:
        data: xarray Dataset with variable 'observed_reflectance_VIS006'
        datetimes: list of datetime strings to select
        method: reserved for future methods (currently only darkest‐pixel)

    Returns:
        2D DataArray of the minimum reflectance
    """
    arr = data.observed_reflectance_VIS006.sel(datetimes=datetimes)
    return arr.min(dim="datetimes")


from typing import List, Optional, Tuple
import xarray as xr

def synthetic_clr_ref(
    data: xr.Dataset,
    op_version: str,
    datetimes: List[str],
    *,
    method: Optional[str] = None
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """
    Derive synthetic clear-sky reflectance by darkest-pixel or mean over period.

    For 'visop', when method is None, uses synthetic_reflectance_VIS006 for darkest-pixel;
    otherwise uses clearsky_reflectance_VIS006.

    Args:
        data: xarray Dataset with reflectance variables
        op_version: operator version string, e.g. 'visop' or 'rttov122'
        datetimes: list of datetime strings
        method: 'mean' for average, None for darkest-pixel

    Returns:
        data_out: 2D DataArray of reflectance
        idx_out: indices of darkest pixel (only when method=None), else None
    """
    sel = {"datetimes": datetimes}

    if "visop" in op_version:
        if method is None:
            arr = data.synthetic_reflectance_VIS006.sel(**sel)
            data_out = arr.min(dim="datetimes")
            idx_out = arr.argmin(dim="datetimes", skipna=False)
        else:
            arr = data.clearsky_reflectance_VIS006.sel(**sel)
            data_out = arr.mean(dim="datetimes")
            idx_out = None
    else:
        # RTTOV case
        sel["sky_condition"] = "all_sky" if method is None else "clear"
        arr = data.synthetic_reflectance_VIS006.sel(**sel)
        if method is None:
            data_out = arr.min(dim="datetimes")
            idx_out = arr.argmin(dim="datetimes", skipna=False)
        else:
            data_out = arr.mean(dim="datetimes")
            idx_out = None

    return data_out, idx_out

def albedo_atlas(
    data: xr.Dataset,
    datetimes: List[str],
    mask_clr: np.ndarray
) -> xr.DataArray:
    """
    Extract atlas albedo by indexing synthetic_clr_ref at darkest‐pixel mask (mask_clr).

    Args:
        data: xarray Dataset with 'synthetic_clr_ref' variable
        datetimes: list of datetime strings
        mask_clr: 2D array of indices for darkest pixels

    Returns:
        2D DataArray of albedo atlas
    """
    arr = data.synthetic_albedo.sel(datetimes=datetimes)
    return arr.isel(datetimes=mask_clr)


def synthetic_angles(
    data: xr.Dataset,
    datetimes: List[str],
    mask_clr: np.ndarray
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Get scattering, solar, and viewing angles at darkest‐pixel mask (mask_clr).

    Args:
        data: xarray Dataset with scatter_angle, SZA, VZA
        datetimes: list of datetime strings
        mask_clr: 2D array of indices for darkest pixels

    Returns:
        Tuple of (SCA, SZA, VZA) DataArrays
    """
    def _pick(var_name: str) -> xr.DataArray:
        return getattr(data, var_name).sel(datetimes=datetimes).isel(datetimes=mask_clr)

    return _pick("scatter_angle"), _pick("SZA"), _pick("VZA")

def smooth_data(data, window_size):
    smoothed_data = uniform_filter(data.astype(float), size=window_size, mode='constant', cval=0.0)
    return smoothed_data

class reflectance_out:
    def __init__(
        self,
        operator: str,
        dirpath: Path,
        filepath: Path,
        *,
        clear_ref_period: Optional[List[str]] = None,
        all_sky_ref_period: Optional[List[str]] = None,
        albedo: bool = False,
        angle: bool = False,
        mask_path = Path("/path/to/RTTOV_AROME/masks/clr_sky_filter.npy")
    ):
        """
        Load dataset and precompute observed/synthetic reflectances.

        Args:
            operator: e.g. 'visop' or 'rttov122'
            dirpath: base directory as Path
            filepath: file name as Path
            clear_ref_period: list of datetime strings for clear‐sky computations
            all_sky_ref_period: list of datetime strings for all‐sky statistics
            compute_albedo_atlas: whether to compute atlas albedo
            compute_angles: whether to compute scattering/viewing angles
        """
        self.operator = operator
        self.data = xr.open_dataset(dirpath+filepath)

        # Containers
        self.observed_clear: Dict[str, xr.DataArray] = {}
        self.synthetic_clear: Dict[str, Tuple[xr.DataArray, Optional[xr.DataArray]]] = {}
        self.albedo_atlas: Dict[str, xr.DataArray] = {}
        self.angles: Dict[str, Dict[str, xr.DataArray]] = {}

        if clear_ref_period:
            self._compute_clear_sky(clear_ref_period, albedo, angle)

        if all_sky_ref_period:
            self._compute_all_sky(all_sky_ref_period, mask_path)

    def _compute_clear_sky(
        self,
        clear_ref_period: List[str],
        albedo: bool,
        angle: bool
    ):
        # build list of datetime slices (same as in original __init__)
        datetime_slices = [
            [
                f"202308{x:02d}_{t}"
                for x in range(1, 32)
                if not (t == "180000" and x == 17)
            ]
            for t in clear_ref_period
        ]

        # initialize containers
        self.clear_synthetic = {"darkest-pixel": {}, "mean-pixel": {}, "mask_clr": {}}
        self.albedo_from_atlas = {}
        self.angles = {"SCA": {}, "SZA": {}, "VZA": {}}
        if self.operator == "visop":
            self.clear_observed = {}

        # loop exactly as before
        for dt_list in datetime_slices:
            key = dt_list[0][-6:]
            if self.operator == "visop":
                self.clear_observed[key] = observed_albedo(self.data, dt_list, method=None)

            dark, mask_clr = synthetic_clr_ref(self.data, self.operator, dt_list, method=None)
            self.clear_synthetic["darkest-pixel"][key] = dark
            self.clear_synthetic["mask_clr"][key] = mask_clr

            mean_pixel, _ = synthetic_clr_ref(self.data, self.operator, dt_list, method="mean")
            self.clear_synthetic["mean-pixel"][key] = mean_pixel

            if albedo and self.operator == "rttov122":
                self.albedo_from_atlas[key] = albedo_atlas(self.data, dt_list, mask_clr)

            if angle and self.operator == "rttov122":
                sca, sza, vza = synthetic_angles(self.data, dt_list, mask_clr)
                self.angles["SCA"][key] = sca
                self.angles["SZA"][key] = sza
                self.angles["VZA"][key] = vza

    def _compute_all_sky(self, datetimes: List[str], mask_path):
        """
        Compute all-sky observed and synthetic arrays, masked by clear-sky filter.
        Supports both 'visop' and 'rttov122' operators.
        """
        mask = np.load(mask_path)

        if self.operator == "visop":
            obs = (
                self.data.observed_reflectance_VIS006
                .sel(datetimes=datetimes)
                .values * mask
            )
            syn = (
                self.data.synthetic_reflectance_VIS006
                .sel(datetimes=datetimes)
                .values * mask
            )
        else:
            obs = None
            syn = (
                self.data.synthetic_reflectance_VIS006
                .sel(datetimes=datetimes, sky_condition="all_sky")
                .values * mask
            )

        self.all_sky_observed = obs.flatten() if obs is not None else None
        self.all_sky_synthetic = syn.flatten()

class AromeHydrometeors:
    def __init__(self, dirpath: Path, filepath: Path):
        """Simple loader for AROME hydrometeor dataset."""
        self.path = dirpath+filepath
        self.data = xr.open_dataset(self.path)
def superobbing(data, k, m):
    """ 
    function superobbing(data, k, m) performs averaging of data for every k*m pixels block
    inputs: 
           - 2d data array (usually a reflectance or ir images)
           - k, and m are the block dimenions
    output:
           - 2d superobbing data with reduced resolution
    """
    kernel = np.ones((k, m)) / (k * m) # Averaging kernel
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)
    smoothed_data = convolve2d(data, kernel, mode='valid') # Apply 2D convolution
    superobbed = smoothed_data[::k, ::m] # Downsample to get representative pixels
    return superobbed