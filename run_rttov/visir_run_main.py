"""
Main script to run RTTOV simulations using utility functions.

Author
------
Ms. Sandy Chkeir  |  sandychkeir96@gmail.com
"""

import logging
from pathlib import Path
from typing import Dict, Any
from log_tools import update_count, log_experiment
from visir_utils import *
from datetime import date
from datetime import datetime

# ----------------------------------------------------------------------------
# Constants & Switching versions
# ----------------------------------------------------------------------------
LOG_PATH = Path('/path/to/log/experiments_log_visible.txt')
COUNT_FILE = Path('/path/to/log/count_rttov132.text') # count_rttov132 / count_rttov122

# Configure operator
# 1. change COUNT_FILE (line 24)
# 2. change exp dictionary (line 40)
# 3. change run_experiments argument (line 80)
# 4. change REPO_PATH in visir_utils.py (line 19)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    """Entry point for script execution."""
    # Update run counter and log experiment metadata
    count = update_count(str(COUNT_FILE))
    exp = {
        'test': 'rttov_v132_v1.2',
        'purpose': 'test updated code with default settings, 31 days, 12 UTC, rttov v132',
        'date': datetime.now()
    }
    log_experiment(exp['test'], exp['purpose'], exp['date'], str(LOG_PATH))

    # Simulation settings for visible channel
    settings: Dict[str, Any] = {
        'snow2ice': 0,
        'lwc': 1,
        'NC': 2.0e8,
        'clw_scheme':1,
        'fixed_time': None,
        'cte_hour': None
    }

    # Build dictionaries for dates and hours
    date_dict = {
        f"Day {i}": {
            'path_to_file': f"/path/to/arome_forecasts/august23/day_{i:02}/",
            'file_name': f"/202308{i:02}_09_0003.grb",
            'year': 2023,
            'month': 8,
            'day': i,
            'hour': 12
        } for i in range(1, 3)
    }

    time_dict = {
        f"{hour}UTC": {
            'path_to_file': "/path/to/grbs_tmp/may23/day_08/",
            'file_name': f"/20230508_09_000{idx}.grb",
            'year': 2023,
            'month': 5,
            'day': 8,
            'hour': hour
        } for idx, hour in enumerate([12, 11, 10], start=1)
    }

    # Execute experiments
    start_time = datetime.now()
    run_experiments(
        date_dict=date_dict,
        time_dict=time_dict,
        loop_dates=True,
        loop_hours=False,
        test_name=exp['test'],
        rttov_op="v132",
        channel='visible',
        settings=settings,
        output_root = '/path/to/ml_simulations'
    )
    elapsed = datetime.now() - start_time
    logging.info(f"Total runtime: {elapsed}")


if __name__ == "__main__":
    main()
