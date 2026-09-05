"""
Utility functions for loading GRIB data and running RTTOV experiments.

Author
------
Ms. Sandy Chkeir |  sandychkeir96@gmail.com
"""

from pathlib import Path
#from datetime import datetime
from typing import Dict, Any
import pygrib
import logging
from vis_rttov_arome import *
from visir_rttov_switches import *
from visir_profiles_arome import *

# Constants for repository path, if needed
REPO_PATH = Path('/path/to/repo/')


def load_grib(file_path: Path) -> list:
    """
    Open a GRIB file and return a list containing the GRIB message.

    Args:
        file_path: Path to the GRIB file.
    Returns:
        List containing the GRIB message object.
    """
    grb = pygrib.open(str(file_path))
    return [grb]


def process_entry(
    entry: Dict[str, Any],
    label: str,
    rttov_op: str,
    channel: str,
    test_name: str,
    settings: Dict[str, Any],
    output_root: str
) -> bool:
    """
    Load GRIB, validate datetime, configure and run RTTOV for a single entry.

    Args:
        entry: Dictionary with file info and date/time keys.
        label: Label for logging.
        channel: 'visible' or 'infrared'.
        test_name: Identifier for the test.
        settings: Settings dict for visible channel.
    Returns:
        Success flag from the runner function.
    """
    file_path = Path(entry['path_to_file']) / entry['file_name'].lstrip('/')
    grbs = load_grib(file_path)
    target_dt = datetime.datetime(entry['year'], entry['month'], entry['day'], entry['hour'])
    validate_grb_datetime(grbs, target_dt)
    logging.info(f"{label}: target datetime is {target_dt}")

    # Choose setup and runner functions
    if channel == 'visible':
        config = sim_setup_visible(test_name, rttov_op, settings)
        runner = run_rttov
        runner_kwargs = {
            'settings_dict': settings,
            'rttov_op': rttov_op,
            'repo_path': str(REPO_PATH),
            'output_root': output_root
        }
    else:
        config = sim_setup_infrared(test_name)
        runner = run_rttov_ir
        runner_kwargs = {}

    success = runner(
        grbs=grbs,
        index=0,
        dt=target_dt,
        config=config,
        run=True,
        channel_used=channel,
        **runner_kwargs
    )
    return success


def run_experiments(
    date_dict: Dict[str, Dict[str, Any]],
    time_dict: Dict[str, Dict[str, Any]],
    loop_dates: bool,
    loop_hours: bool,
    test_name: str,
    rttov_op: str,
    channel: str,
    settings: Dict[str, Any],
    output_root: str
) -> None:
    """
    Iterate over date or time dictionaries and process each entry.
    """
    if loop_dates:
        logging.info("Running experiments for selected days")
        for label, entry in date_dict.items():
            process_entry(entry, label, rttov_op, channel, test_name, settings, output_root)

    if loop_hours:
        logging.info("Running experiments for selected hours")
        for label, entry in time_dict.items():
            process_entry(entry, label, rttov_op, channel, test_name, settings, output_root)

    logging.info("All experiments completed.")