import argparse
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from one.api import ONE

from psyfun import io
from psyfun.config import paths

parser = argparse.ArgumentParser(
    description="Download and process IBL data flexibly with command-line flags."
    )
# Data fetching options
parser.add_argument(
    '-s', '--fetch_sessions', action='store_true',
    help='Fetch session metadata from server'
    )
parser.add_argument(
    '-i', '--fetch_insertions', action='store_true',
    help='Fetch insertion metadata from server'
    )
parser.add_argument(
    '-u', '--fetch_uinfo', action='store_true',
    help='Fetch unit/cluster metadata from server (no spike times)'
    )
parser.add_argument(
    '-t', '--fetch_spikes', action='store_true',
    help='Fetch spike time data from server (requires uinfo)'
    )
parser.add_argument('-a', '--all', action='store_true',
help='Fetch everything: sessions, insertions, uinfo, spikes (not BWM)'
)
args = parser.parse_args()

# --all sets all relevant flags True
if args.all:
    args.fetch_sessions = True
    args.fetch_insertions = True
    args.fetch_uinfo = True
    args.fetch_spikes = True

# Instantiate database connection
one = ONE()

# Fetch session metadata
if args.fetch_sessions:
    print("Fetching sessions...")
    df_sessions = io.fetch_sessions(one, qc=True, save=True)

# Fetch probe insertion metadata
df_insertions = None
if args.fetch_insertions:
    print("Fetching probe insertions...")
    df_insertions = io.fetch_insertions(one, save=True)

# Fetch unit info metadtada and spike time data (optional)
if args.fetch_uinfo or args.fetch_spikes:
    # Fetch or load insertions as needed
    if df_insertions is None:
        print(f"Loading insertions from {paths['insertions']}")
        try:
            df_insertions = pd.read_parquet(paths['insertions'])
        except FileNotFoundError:
            raise RuntimeError(
                "Must run 'fetch_data.py --fetch_insertions' before trying to fetch units."
            )
    print(
        "Fetching unit info and spike times..." if args.fetch_spikes else
        "Fetching unit info..."
        )
    spike_file = paths['spikes'] if args.fetch_spikes else ''
    df_units = io.fetch_unit_info(
        one,
        df_insertions,
        uinfo_file=paths['units'],
        spike_file=spike_file,
        histology='traced'
        )


# ~# BWM options (no shortcut for all)
# ~parser.add_argument('--fetch_bwm_insertions', action='store_true', help='Fetch BWM probe insertions from server')
# ~parser.add_argument('--fetch_bwm_uinfo', action='store_true', help='Fetch BWM unit/cluster info from server (no spike times)')
# ~parser.add_argument('--fetch_bwm_spikes', action='store_true', help='Fetch BWM spike times from server (requires uinfo)')
# ~# BWM Insertions
# ~df_insertions_bwm = None
# ~if args.fetch_bwm_insertions:
    # ~# BWM task starts is always loaded from file
    # ~print("Loading BWM task starts from file...")
    # ~df_bwm = pd.read_csv('metadata/BWM_task_starts.csv')
    # ~# Always apply BWM cutoff from config
    # ~print(f"Filtering BWM session with >= {bwm_pretask_length} seconds pre-task")
    # ~df_controls = df_bwm.query('task_start > @bwm_pretask_length')
    # ~print("Fetching BWM insertions...")
    # ~df_insertions_bwm = io.fetch_BWM_insertions(one, df_controls)
# ~else:
    # ~print(f"Loading BWM insertions from {paths['BWM_insertions']}")
    # ~df_insertions_bwm = pd.read_parquet(paths['BWM_insertions'
# ~# BWM Units (cluster info) and Spikes (optional)
# ~if args.fetch_bwm_uinfo or args.fetch_bwm_spikes:
    # ~if df_insertions_bwm is None:
        # ~try:
            # ~df_insertions_bwm = pd.read_parquet(paths['BWM_insertions'])
        # ~except FileNotFoundError:
            # ~raise RuntimeError("Run 'fetch_data.py --fetch_bwm_insertions' before trying to fetch units.")
    # ~print("Fetching BWM unit info and spike times..." if args.fetch_spikes else "Fetching unit info...")
    # ~spike_file = paths['BWM_spikes'] if args.fetch_bwm_spikes else ''
    # ~df_uinfo = io.fetch_unit_info(one, df_insertions_bwm, uinfo_file=paths['BWM_units'], spike_file=spike_file)
