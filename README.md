# VISSAT-research

This repository simulates satellite visible reflectances using AROME-Austria model forecasts and the RTTOV radiative transfer model.


## Features

- Compatible with RTTOV v12.2 and v13.2
- Simulates the SEVIRI visible (VIS) 0.6 µm channel
- Complete pipeline: data preprocessing → RTTOV simulation → postprocessing
- Modular codebase for clarity and easy extension
- Reproduces all figures and plots of most presented work


## Runing simulations

Inside `run_rttov/` directory, you find:

- **Quick single simulation + plots**:  
  Use the Jupyter notebook  
  `run_rttov/run_rttov_notebook.ipynb`

- **Batch simulations over a date range**:  
  Run  
  `python3 run_rttov/visir_run_main.py`

- Supporting scripts in `run_rttov/`, which are used internally by these main interfaces.
- optional: you can configure the python scripts and set various assumptions for radiative transfer.

#### Combining multiple time steps

To merge outputs from several time steps into a single NetCDF file:

1. Change into the `data_processing/` directory.
2. Edit `concat_output_data.py` as needed (e.g., input folder, which operator you're using).
3. Run:
   ```bash
   python3 concat_output_data.py
   ```
   
## Reproducing results

The `all_sky_results/` and `clear_sky_results/` directories contain notebooks that reproduce the results presented in the manuscript.


## Installation

### 1. Compile RTTOV

Download and compile RTTOV from the [NWP-SAF website](https://www.nwpsaf.eu/site/software/rttov/).  
Be sure to compile both the core library and the Python interface (`pyrttov`).


### 2. Clone this repository

```bash
git clone https://github.com/Sandy887/RTTOV_AROME.git
cd RTTOV_AROME
```


### 3. Configure paths

Edit the following file to point to your local RTTOV and pyrttov installations: `run_rttov/rttov_op_paths.py`

Ensure `pyrttov` is accessible via `PYTHONPATH`.

### 4. Set up the environment

It's dependent on your server environment. In my case, the following modules are required on the ECMWF virtual desktop:

`gcc/8.5.0`, `prgenv/gnu`, `python3/3.8.8-01`, `hdf5/1.10.6`, `netcdf4/4.7.4`

Use the following commands to configure your environment:

```bash
# Reset environment
module purge
unset LD_LIBRARY_PATH

# Load required modules
module load gcc/8.5.0 prgenv/gnu python3/3.8.8-01 hdf5/1.10.6 netcdf4/4.7.4

# Set library paths
export LD_LIBRARY_PATH=$HDF5_DIR/lib:$NETCDF4_DIR/lib:$PYTHON3_DIR/lib:$LD_LIBRARY_PATH
```

To run the code successfully, make sure:

- RTTOV is compiled (including `pyrttov`)
- `PYTHONPATH` includes `pyrttov` (set in `rttov_op_paths.py`)
- This repository is cloned and paths configured
- All required modules are loaded for both RTTOV versions

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
   
## Citation

If you use this code or its results in your research, please cite:

```bibtex
@misc{SATVIS-manuscript-2025,
  author    = {Chkeir, Sandy},
  title     = {CloudyVisibleRadiances: RTTOV-AROME Cloudy Radiance Simulations},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Sandy887/RTTOV_AROME},
  note      = {Accessed 2025-08-30}
}
```

---

## Acknowledgements

- This work would not have been possible without the supervision and valuable discussions with **Philipp Griewank**, particularly regarding figure production and visualization.
- Many thanks to **Leonhard Scheck** (DWD) for his support and insights on visible-channel simulations.
- Special thanks to **Florian Meier** (Geosphere Austria) for his guidance on handling AROME model output and related processes.