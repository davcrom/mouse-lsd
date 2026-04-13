import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.ion()
from matplotlib import colors

from iblatlas.atlas import AllenAtlas
import iblatlas.plots as anatomyplots

from psyfun.config import *
from psyfun.atlas import coarse_regions
from psyfun.plots import cm2in, qc_grid, set_plotsize, clip_axes_to_ticks

df_sessions = pd.read_parquet(paths['sessions'])
from psyfun.io import get_trajectory_labels
df_sessions = get_trajectory_labels(df_sessions)

# Check that necessary task data is present
# Note: rather than checking the raw datasets, we just check if we were able to
# find protocol timings with io._fetch_protocol_timings
task_timings = [
    'task00_spontaneous_start', 'task00_spontaneous_stop',
    'task00_rfm_start', 'task00_rfm_stop',
    'task00_replay_start', 'task00_replay_start',
    'LSD_admin',
    'task01_spontaneous_start', 'task01_spontaneous_stop',
    'task01_rfm_start', 'task01_rfm_stop',
    'task01_replay_start', 'task01_replay_start',
    ]
df_sessions['task_ok'] = df_sessions.apply(
    lambda x: all([not np.isnan(x[time]) for time in task_timings]),
    axis='columns'
    )

# Check for spike sorted data set for each probe
ephys_datasets = [
    'alf/probe00/pykilosort/spikes.times.npy',
    'alf/probe01/pykilosort/spikes.times.npy',
    'alf/probe00/iblsorter/spikes.times.npy',
    'alf/probe01/iblsorter/spikes.times.npy'
    ]
df_sessions['ephys_ok'] = df_sessions.apply(
    lambda x: any([x[dset] for dset in ephys_datasets]),
    axis='columns'
    )  # note: here we use 'any' rather than 'all', we just need some neurons

# Pivot sessions table (filtered by checks above) to sessions x subjects matrix
sessions_pivot = df_sessions.query(
    'task_ok == True and ephys_ok == True'
    ).pivot(
        columns='trajectory_label', index='subject', values='control_recording'
        )

# Make a custom color map
# Plot the sessions x subjects grid
# ~fig, ax = plt.subplots(figsize=(cm2in(32), cm2in(18)))
fig, ax = plt.subplots()
ax = qc_grid(
    sessions_pivot,
    qcval2num={np.nan: 0., False: 0.01, True: 1.},
    cmap=CMAPS['recording_type'],
    ax=ax,
    xticklabels=sessions_pivot.index,
    legend=False
    )
ax.set_xlabel('Mouse')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Insertion trajectory')
set_plotsize(w=12, h=8, ax=ax)
fig.savefig(paths['figures'] / 'insertions_grid.svg')


# Load cluster info from file, keep only good units
df_units = pd.read_parquet(paths['units'])
# Merge session info into units dataframe
df_units = pd.merge(
    df_units.query('ks2_label =="good"'),
    df_sessions.query('task_ok == True and ephys_ok == True'),
    on=['subject', 'eid', 'session_n'],
    how='left'
    )
# Clean up columns
df_units = df_units[[
    col for col in df_units.columns if
    not col.endswith('_x') or col.endswith('_y')
    ]]
df_units['coarse_region'] = coarse_regions(df_units['region'])


# Get number of unit per coarse region recorded in the LSD condition (sorted)
region_counts = df_units.query(
    'control_recording == False'
    ).groupby('coarse_region').apply(len).sort_values(ascending=False)
n_regions = len(region_counts)
# Bar plot number of units per coarse region in LSD condition
fig, ax = plt.subplots()
ax.bar(
    np.arange(len(region_counts)) - 0.2,
    region_counts,
    width=0.3,
    fc=LSDCOLOR,
    ec='gray'
    )
# Loop over sorted regions and plot number of units recorded in control condition
for xpos, region in enumerate(region_counts.index):
    n_units = df_units.query(
        '(control_recording == True) & (coarse_region == @region)'
        ).apply(len)
    ax.bar(xpos + 0.2, n_units, width=0.3, fc=CONTROLCOLOR, ec='gray')
# Add a label for total number of units in LSD condition
n_lsd = len(df_units.query('control_recording == False'))
ax.text(
    0.98, 0.98, f'N LSD = {n_lsd}',
    fontsize=LABELFONTSIZE, color=LSDCOLOR, ha='right', va='top',
    transform=ax.transAxes
    )
# Add a label for total number of units in control condition
n_saline = len(df_units.query('control_recording == True'))
ax.text(
    0.98, 0.88, f'N Saline = {n_saline}',
    fontsize=LABELFONTSIZE, color=CONTROLCOLOR, ha='right', va='top',
    transform=ax.transAxes
    )
ax.set_xticks(np.arange(n_regions))
ax.set_xticklabels(region_counts.index)
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel('Number of units')
set_plotsize(w=12, h=6, ax=ax)
fig.savefig('figures/N_neurons.svg')

# Plot distribution of units across brain regions
for control in [False, True]:
    cmap = CMAPS['control'] if control else CMAPS['LSD']
    for ap_coord in BRAINSLICEAPCOORDS:
        fig, ax = plt.subplots()
        region_counts = df_units.query(
            'control_recording == @control'
            ).groupby('region').apply(
                len, include_groups=False
                ).sort_values(ascending=False)
        fig, ax, cbar = anatomyplots.plot_scalar_on_slice(
            region_counts.index,
            np.log10(region_counts.values),
            coord=ap_coord * 1000,  # in um
            background='boundary',
            atlas=AllenAtlas(res_um=50),
            clevels=[0, 3],
            cmap=CMAPS['control'] if control else CMAPS['LSD'],
            show_cbar=True,
            ax=ax
        )
        ax.text(
            0.01, 1.02, f'AP: {ap_coord}mm',
            fontsize=LABELFONTSIZE,
            transform=ax.transAxes
            )
        ax_pos = ax.get_position()  # returns [left, bottom, width, height]
        cbar_pos = cbar.ax.get_position()
        cbar.ax.set_position(
            [cbar_pos.x0, ax_pos.y0, cbar_pos.width, ax_pos.height]
            )
        cbar.set_ticks(np.linspace(0, 3, 4))
        cbar.set_ticklabels(
            ['$10^{%d}$' % tick for tick in np.linspace(0, 3, 4)]
            )
        cbar.set_label('N units')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        set_plotsize(w=4, h=4, ax=ax)
        clip_axes_to_ticks(ax=ax, spines=[])
        fig.savefig(
            f'figures/N_neurons_{"saline" if control else "LSD"}_AP{ap_coord}.svg'
            )

# Plot mean number of simultaneously recorded pairs of neurons from each
# pair of regions
exclude_regions = ['None', 'Fiber tract', 'Auditory Ctx', 'Visual Ctx', 'Somatosens. Ctx',
                    'Motor Ctx']
for control in [False, True]:
    # Group by subject, eid, and coarse_region to count neurons per region per recording
    region_counts = df_units.query('control_recording == @control').query(
        'coarse_region not in @exclude_regions'
        ).groupby(
            ['subject', 'eid', 'coarse_region']
        ).size().reset_index(name='count')
    # Pivot to get regions as columns for each recording
    pivot = region_counts.pivot_table(
        index=['subject', 'eid'],
        columns='coarse_region',
        values='count',
        fill_value=0
        )
    # Compute mean co-occurrence matrix
    # Note: dot product gives total pairwise neuron counts across all recordings
    mean_cooccurrence = cooccurrence = pivot.T.dot(pivot) / len(pivot)
    # Remove diagonal (mean number of units per region per recording)
    np.fill_diagonal(mean_cooccurrence.values, np.nan)

    fig, ax = plt.subplots()
    im = ax.matshow(
        mean_cooccurrence.values,
        norm=colors.LogNorm(
            vmin=100,
            vmax=10000
            ),
        cmap=CMAPS['control'] if control else CMAPS['LSD']
        )
    # Add ticklabels
    ax.set_xticks(range(len(mean_cooccurrence.columns)))
    ax.set_yticks(range(len(mean_cooccurrence.index)))
    ax.set_xticklabels(mean_cooccurrence.columns, rotation=90)
    ax.set_yticklabels(mean_cooccurrence.index)
    set_plotsize(w=6, h=6, ax=ax)
    # Add colorbar with same height as plot
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 0.02, bbox.height])
    plt.colorbar(im, cax=cax)
    cax.set_ylabel('N pairs')
    fig.savefig(
        f'figures/N_pairs_{"saline" if control else "LSD"}.svg'
            )
