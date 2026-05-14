# mouse-lsd — project reference

Structural knowledge for navigating this project. Workflow, communication,
and skill preferences live in the user-level `~/.claude/CLAUDE.md`.

## Overview

Analysis code for a Neuropixels experiment recording mouse cortical and
subcortical activity under LSD versus control, using the IBL
`passiveChoiceWorld` passive protocol. Registered in IBL Alyx under the
`psychedelics` project name.

## Environment

- Python 3.13, virtualenv at `~/.venv/ibl`, managed by `uv`.
- The `psyfun` package is installed editable (`uv pip install -e .`) so
  notebooks and scripts can import it.

## Repository layout

- `psyfun/` — importable package, shared analysis code (see below).
- `scripts/` — top-level runnable data-fetching and analysis scripts.
- `tests/` — pytest tests for `psyfun`.
- `specs/` — design specs for non-trivial changes.
- `metadata/` — session and insertion tables, plus CSV metadata.
- `data/` — unit tables and spike-time stores.
- `notebooks/` — exploratory notebooks (legacy, not maintained).
- `figures/` — generated figures.
- `histology/`, `openfield/`, `video/`, `stats/` — modality-specific analyses.
- `docs/`, `archive/`, `davide/` — documentation and older work.

## The `psyfun` package

- `config.py` — project constants: `ibl_project` (Alyx project and
  protocol names), `paths` (absolute paths to every generated table),
  `df_controls` (control-session eids), `TASKTIMINGS` / `PASSIVE_SLOTS`
  (task-epoch column names), plotting constants.
- `io.py` — fetching from Alyx and loading generated tables (entry points
  below).
- `histology.py` — predicates over Alyx insertion records
  (`insertion_picks`, `insertion_alignment_uploaded`,
  `insertion_alignment_resolved`) and `list_histology_tifs`.
- `atlas.py` — Allen atlas region remapping (`region_parcellation`,
  `combine_regions`, `remap_names`).
- `spikes.py` — single-unit spike-train metrics (coefficient of
  variation, Fano factor, Lempel-Ziv complexity, modulation index).
- `population.py` — population-level dimensionality (PCA helpers).
- `spike_sorting.py` — `SpikeSortingQC` class.
- `plots.py` — figure helpers.
- `util.py` — statistics and dataframe helpers (bootstrap CI, ICC,
  sliding epochs, decay fits).

### Key `io.py` entry points

```python
from psyfun.io import fetch_sessions, load_sessions, load_units, load_session_spikes
```

- `fetch_sessions(one, save=True)` — queries Alyx for `psychedelics`
  sessions, checks dataset-extraction status, adds task-epoch timings;
  writes `metadata/sessions.pqt`. Returns the dataframe.
- `fetch_insertions(one, save=True)` — one row per probe insertion in the
  project; writes `metadata/insertions.pqt`.
- `fetch_unit_info(one, df_insertions, uinfo_file='', spike_file='')` —
  per-unit cluster table across probes; optionally writes a units parquet
  and a spike-times HDF5.
- `load_sessions(drop_if_nan=TASKTIMINGS, drop_extra_columns=True)` —
  reads `sessions.pqt`, drops sessions missing task timings.
- `load_units(eids=None, unit_filter=None, add_coarse_regions=True)` —
  reads `units.pqt`, with optional session and quality filtering.
- `load_spikes(uuids, remove_duplicates=True)` — per-uuid spike times
  from `data/spikes.h5`, returned as a dataframe indexed by uuid.
- `load_session_spikes(unit_filter=None, remove_duplicate_spikes=True)` —
  joins sessions, units, and spike times into one dataframe.
- `load_metadata()` — reads `metadata/metadata.csv` (per-recording
  experimental metadata, including LSD administration time).

## Data access

Recordings live in IBL infrastructure across three layers:

- **Alyx** (`https://alyx.internationalbrainlab.org`) — relational
  metadata: sessions, subjects, insertions, dataset records, QC,
  alignment. Accessed via REST.
- **Flatiron** — the data blobs (raw recordings, alf-extracted
  derivatives), downloaded on demand and cached locally.
- **Local cache** — what has been downloaded so far.

All access goes through the `one` library:

```python
from one.api import ONE
one = ONE()
```

### Identifiers

- `eid` (UUID) — a recording session.
- `pid` (UUID) — a probe insertion. `pid` ↔ `eid + probe_name` via
  `one.eid2pid(eid)` / `one.pid2eid(pid)`.
- subject — a name string, e.g. `ZFM-08631`.

### Per-session Alyx calls

```python
one.alyx.rest('sessions', 'read', id=eid)
# session dict; useful fields: 'extended_qc' and
# 'data_dataset_session_related' (every registered dataset, each entry
# carrying 'collection', 'name', 'data_url', 'revision', 'version',
# 'default_revision')

one.alyx.rest('insertions', 'list', session=eid)   # probe insertion records
one.alyx.rest('insertions', 'list', id=pid)[0]     # one insertion; histology under ['json']

one.load_dataset(eid, name, collection)            # fetch actual file contents
```

### Collections

Datasets are organised into collections matching folder paths under the
session:

| collection | content |
|---|---|
| `raw_task_data_NN/` | iblrig outputs for the Nth task run: `_iblrig_taskSettings.raw.json`, stim and frame data |
| `alf/task_NN/` | IBL-extracted task data in the session/FPGA clock: `_ibl_passivePeriods.intervalsTable.csv`, `_ibl_passiveStims.table.csv`, `_ibl_passiveGabor.table.csv` |
| `raw_ephys_data/probeNN/` | raw SpikeGLX per probe: `.ap.cbin`, `.sync.npy`, `.ap.meta` |
| `alf/probeNN/<sorter>/` | spike-sorted output: `spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv`, cluster metrics. `<sorter>` is one of `iblsorter`, `pykilosort` |
| `raw_video_data/` | raw video per camera: `_iblrig_<C>Camera.raw.mp4` for `<C>` in `{left, right, body}` |
| `alf/` | per-camera extracted: `_ibl_<C>Camera.times.npy`, `_ibl_<C>Camera.dlc.pqt`, `_ibl_<C>Camera.lightningPose.pqt` |

## Generated tables

Built by `psyfun` and stored under `metadata/` and `data/` — local
rebuilds, not Alyx state. Absolute paths are in `psyfun.config.paths`.

- `metadata/sessions.pqt` — one row per session: metadata,
  dataset-extraction status, task-epoch timings. Built by `fetch_sessions`.
- `metadata/insertions.pqt` — one row per probe insertion. Built by
  `fetch_insertions`.
- `metadata/metadata.csv` — hand-maintained per-recording experimental
  metadata, including LSD administration time.
- `metadata/histology_status.pqt` — built by
  `scripts/check_histology_status.py`.
- `data/units.pqt` — per-unit cluster table joined across sessions. Built
  by `fetch_unit_info`.
- `data/spikes.h5` — spike times keyed by unit UUID.
- `data/bombcell.pqt` — bombcell unit-QC outputs. Built by
  `scripts/unit_qc.py`.

## Dataset peculiarities

- **Passive run ordering.** A session has two passive `passiveChoiceWorld`
  runs (`task_pre`, `task_post`) and may have a spontaneous-only LSD-filler
  run between them. Slots are named by passive-run order, *not* by
  `raw_task_data_NN` number — the filler can be `raw_task_data_01` with
  passive runs at `00` and `02`.
- **Extractor merge bug.** On at least one session (`c7cf8e25`, `task_00`)
  the IBL extractor merged the LSD-filler block into the task, producing a
  600 s spontaneous interval instead of 300 s. Epoch boundaries are
  clipped to protocol-design durations to compensate.
- **SpikeGLX filename forms.** iblrig writes `imec0` or `imec00` (and
  `imec1` / `imec01`) inconsistently — check both forms when testing
  dataset presence.
- **Dataset revisions.** Alyx supports revisions (`#YYYY-MM-DD#` between
  dataset name and extension). Today every dataset has revision `''`;
  re-sortings or re-extractions will land under dated revisions. Pick the
  registry entry whose `default_revision` is the string `'True'`.
- **Pose tracking.** `_ibl_<C>Camera.dlc.pqt` (DeepLabCut, older) and
  `_ibl_<C>Camera.lightningPose.pqt` (Lightning Pose, newer) are separate
  per-camera files. This project treats only Lightning Pose as valid pose
  output.
- **Bombcell output is not on Alyx.** It lives on the local filesystem
  under `one.eid2path(eid) / 'spike_sorters' / <sorter> / <probe> / 'bombcell'`.
- **`one.list_datasets` is unreliable** — its results do not always
  correspond to what actually loads. Use the `data_dataset_session_related`
  registry field or `one.load_dataset` instead.
