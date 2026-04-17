from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl

ibl_project = {
    'name': 'psychedelics',
    'protocol': 'passiveChoiceWorld'
}

from pathlib import Path

# If data is in project root, not in psyfun/
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

paths = {
    'metadata': PROJECT_ROOT / Path('metadata/metadata.csv'),
    'sessions': PROJECT_ROOT / Path('metadata/sessions.pqt'),
    'insertions': PROJECT_ROOT / Path('metadata/insertions.pqt'),
    'trajectories': PROJECT_ROOT / Path('metadata/trajectories.csv'),
    'units': PROJECT_ROOT / Path('data/units.pqt'),
    'bombcell': PROJECT_ROOT / Path('data/bombcell.pqt'),
    'spikes': PROJECT_ROOT / Path('data/spikes.h5'),
    'BWM_insertions': PROJECT_ROOT / Path('metadata/BWM_insertions.pqt'),
    'BWM_units': PROJECT_ROOT / Path('data/BWM_units.pqt'),
    'BWM_spikes': PROJECT_ROOT / Path('data/BWM_spikes.h5'),
    'figures': PROJECT_ROOT / Path('figures')
}

qc_datasets = {
    'task': [
        'raw_task_data_00/_iblrig_taskSettings.raw.json',
        'raw_task_data_00/_iblrig_stimPositionScreen.raw.csv'
        'alf/task_00/_ibl_passivePeriods.intervalsTable.csv',
        'raw_task_data_01/_iblrig_taskSettings.raw.json',
        'raw_task_data_01/_iblrig_stimPositionScreen.raw.csv',
        'alf/task_01/_ibl_passivePeriods.intervalsTable.csv',
        'raw_task_data_02/_iblrig_taskSettings.raw.json',
        'raw_task_data_02/_iblrig_stimPositionScreen.raw.csv'
        'alf/task_02/_ibl_passivePeriods.intervalsTable.csv',
        ],
    'ephys': [
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.sync.npy',
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec00.sync.npy',
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec00.ap.cbin',
        'alf/probe00/pykilosort/spikes.times.npy',
        'alf/probe00/iblsorter/spikes.times.npy',
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.sync.npy',
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec01.sync.npy',
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec01.ap.cbin',
        'alf/probe01/pykilosort/spikes.times.npy',
        'alf/probe01/iblsorter/spikes.times.npy'
        ],
    'video': [
        'raw_video_data/_iblrig_bodyCamera.raw.mp4',
        'raw_video_data/_iblrig_bodyCamera.frameData.bin',
        'raw_video_data/_iblrig_leftCamera.raw.mp4',
        'raw_video_data/_iblrig_leftCamera.frameData.bin',
        'raw_video_data/_iblrig_rightCamera.raw.mp4',
        'raw_video_data/_iblrig_rightCamera.frameData.bin',
        ]
}

control_recordings = [
    ['ZFM-08631', '2025-03-21', 'cdc1965e-4324-422a-a9d4-86b4e5d0de92'],
    ['ZFM-08584', '2025-03-19','5c28961e-1183-43f9-86a8-9c1c9d8ed743'],
    ['ZFM-08457', '2025-03-18', '878b86b7-9245-40a3-843e-9ebf0a8251db'],
    ['ZFM-08631', '2025-03-18', '58c61f29-d4e6-4ca7-b6de-bd860e83fe4e'],
    ['ZFM-08458', '2025-03-14', '55f3644c-3f86-423e-9beb-6604b5aa4e2c'],
    ['ZFM-08631', '2025-03-12', 'bb0e0ad2-4c98-4c98-b92f-559683e8a6f3'],
    ['ZFM-08584', '2025-03-12', 'e8128c55-b322-438a-9498-edd47ae4b794'],
    ['ZFM-08457', '2025-03-11', 'c7cf8e25-1e2c-4b03-a5f5-5a049f1cd228'],
    ['ZFM-08458', '2025-03-11', '3e9d9490-7fb2-4aa3-b8cd-83f216ad8cde']
]
df_controls = pd.DataFrame(data=control_recordings, columns=['subject', 'date', 'eid'])

TASKTIMINGS = [
    'task00_spontaneous_start', 'task00_spontaneous_stop',
    'task00_rfm_start', 'task00_rfm_stop',
    'task00_replay_start', 'task00_replay_stop',
    'LSD_admin',
    'task01_spontaneous_start', 'task01_spontaneous_stop',
    'task01_rfm_start', 'task01_rfm_stop',
    'task01_replay_start', 'task01_replay_stop'
    ]

postLSD_epoch_length = 5 * 60  # seconds
postLSD_epoch_starts = np.arange(0, 30, 10) * 60  # seconds

FIGUREDPI = 180
LABELFONTSIZE = 8
LSDCOLOR = 'mediumorchid'
CONTROLCOLOR = 'mediumseagreen'
CMAPS = {
    'recording_type': mpl.colors.LinearSegmentedColormap.from_list(
        'rec_type', [(0., 'white'), (0.01, LSDCOLOR), (1., CONTROLCOLOR)]
        ),
    'n_units': mpl.colormaps['Reds'],
    'LSD': mpl.colors.LinearSegmentedColormap.from_list("LSD_cmap", ['white', LSDCOLOR]),
    'control': mpl.colors.LinearSegmentedColormap.from_list("control_cmap", ['white', CONTROLCOLOR]),
    'qc': mpl.colors.LinearSegmentedColormap.from_list(
        'qc_cmap',
        [(0., 'white'), (0.01, 'gray'), (0.1, 'palevioletred'), (0.33, 'violet'), (0.66, 'orange'), (1., 'limegreen')],
    )
}

EPOCHLABELS = {
    'spontaneous': 'Spontaneous',
    'replay': 'Visual+Auditory\nStimulation',
    'rfm': 'Receptive Field Mapping',
    }
METRICLABELS = {
    'coefficient_of_variation': 'CV',
    'mean_rate': 'Mean Rate (Hz)',
    'fano_factor_dt100': 'Fano Factor (dt=0.1s)',
    'fano_factor_dt1000': 'Fano Factor (dt=1s)',
    'fano_factor_dt10000': 'Fano Factor (dt=10s)',
    'lempel_ziv_complexity_dt100': 'LZ Complexity (dt=0.1s)',
    'lempel_ziv_complexity_dt1000': 'LZ Complexity (dt=1s)',
    'lempel_ziv_complexity_dt10000': 'LZ Complexity (dt=10s)',
    }
METRICMEANTICKS = {
    'coefficient_of_variation': [0, 2.5, 5],
    'mean_rate': [0, 5, 10, 15],
    'fano_factor_dt100': [0, 1.5, 3],
    'fano_factor_dt1000': [0, 5, 10],
    'fano_factor_dt10000': [0, 10, 20, 30],
    }
POPMETRICLABELS = {
    'pc1': 'PC1',
    'n80': 'N80',
    'powlaw': 'Slope',
    'ngsc': 'NGSC'
    }
POPMETRICTICKS = {
    'pc1': [0, 0.15],
    'n80': [50, 250, 450],
    'powlaw': [-0.75, -0.5, -0.25],
    'ngsc': [0.56, 0.63, 0.7],
    }


# Map QC flags to numbers for plotting
QCVAL2NUM = {
    np.nan: 0.,
    'NOT SET': 0.01,
    'NOT_SET': 0.01,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.1
}

BRAINSLICEAPCOORDS = [2, 1, 0, -1, -2, -3, -4]  # mm
