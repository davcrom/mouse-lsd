# psychedelics

## Installation

This project uses functions from the `psyfun` package, which should be installed to allow relative imports in notebooks. From the project root directory, run:
```bash
pip install -e .
```

## Structure

- `psyfun/` — importable package with shared analysis functions
- `scripts/` — top-level analysis scripts (`fetch_data.py`, `dataset_overview.py`, `population_dimensionality.py`, `single_unit.py`)
- `notebooks/` — exploratory and analysis notebooks
- `data/`, `metadata/` — datasets and session metadata
- `figures/` — generated figures
- `histology/`, `openfield/`, `video/`, `stats/` — task- and modality-specific analyses
- `docs/`, `archive/` — documentation and older work
- `specs/` — implementation specs for non-trivial changes

## Data sources

Recordings live in the IBL infrastructure. There are three layers:

- **Alyx** (`https://alyx.internationalbrainlab.org`) — relational metadata: sessions, subjects, insertions (probes), dataset records, QC, alignment. Accessed via REST.
- **Flatiron S3** — actual data blobs (raw recordings, alf-extracted derivatives). Files are downloaded on demand and cached locally under `~/Downloads/ONE/...`.
- **Local cache** — what's been downloaded so far. Determines what loads instantly versus what hits the network.

All access goes through the `one` Python library:

```python
from one.api import ONE
one = ONE()  # uses the default Alyx connection
```

### Sessions and identifiers

- A recording session is identified by an **`eid`** (UUID).
- A probe insertion is identified by a **`pid`** (UUID). `pid` ↔ `eid + probe_name` via `one.eid2pid(eid)` or `one.pid2eid(pid)`.
- A subject is identified by its name (e.g. `ZFM-06048`).

### Collections

Datasets are organised into collections matching folder paths under the session:

| collection | content |
|---|---|
| `raw_task_data_NN/` | iblrig outputs for the Nth task collection: `_iblrig_taskSettings.raw.json`, `_iblrig_stimPositionScreen.raw.csv`, frame data |
| `alf/task_NN/` | IBL-extracted task-level data in the session/FPGA clock: `_ibl_passivePeriods.intervalsTable.csv`, `_ibl_passiveStims.table.csv`, `_ibl_passiveGabor.table.csv` |
| `raw_ephys_data/probeNN/` | raw SpikeGLX data per probe: `.ap.cbin`, `.sync.npy` (probe-to-session-clock map), `.ap.meta` |
| `alf/probeNN/<sorter>/` | spike-sorted output: `spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv`, plus cluster metrics. `<sorter>` is one of `iblsorter`, `pykilosort` |
| `raw_video_data/` | raw video per camera: `_iblrig_<C>Camera.raw.mp4`, `frameData.bin` for `<C>` in `{left, right, body}` |
| `alf/` (top level) | per-camera extracted: `_ibl_<C>Camera.times.npy` (frame times in session clock), `_ibl_<C>Camera.dlc.pqt` (DeepLabCut tracking), `_ibl_<C>Camera.lightningPose.pqt` (Lightning Pose tracking — newer replacement) |

Iblrig uses inconsistent naming for SpikeGLX probe files: `imec0` vs `imec00`, `imec1` vs `imec01`. Try both when checking presence.

### Pose tracking (DeepLabCut vs Lightning Pose)

Both are registered as one file per camera with the same `_ibl_<C>Camera.` prefix and different extensions:
- `_ibl_<C>Camera.dlc.pqt` — DeepLabCut output. Older pipeline; what most existing sessions have.
- `_ibl_<C>Camera.lightningPose.pqt` — Lightning Pose output. Newer pipeline replacing DLC.

There is no single dataset that covers all three cameras at once.

### Insertion records (histology / trajectory)

Per-probe trajectory and histology pipeline state lives in Alyx's `insertions` table, not in any loadable dataset file. Accessed via:

```python
ins = one.alyx.rest('insertions', 'list', id=pid, no_cache=True)[0]
```

Useful fields:
- `ins['json']['xyz_picks']` — list of manual probe-tip picks on the histology stack.
- `ins['json']['extended_qc']['alignment_count']` — how many alignments have been uploaded.
- `ins['json']['extended_qc']['alignment_resolved']` — whether the trajectory is finalised.

Histology TIF stacks for a subject are not on Alyx; they live in a flatiron HTTP directory under the subject's path. Listing requires an authenticated HTTP request (see `scripts/check_histology_status.py`).

### Session table

`fetch_sessions` in `psyfun/io.py` (run by `scripts/fetch_data.py`) builds
`metadata/sessions.pqt`, one row per session. Alongside session metadata
and task-epoch timings, each row records which datasets IBL has extracted:

- Task alf datasets, per passive slot (`task_pre`, `task_post`).
- Ephys, per probe: raw AP, sync, spike sorting, bombcell. `probeNN_sorter` also records the chosen sorter name, revision, and version.
- Video, per camera: raw video, pose tracking, three video-QC outcomes.
- Histology: image stacks, plus per-probe tracing and alignment state.

Most dataset columns hold a three-state value: `extraction complete`, `extraction error` (raw data present but the derived dataset is missing), or `raw data missing`. See `specs/check_dataset_extraction.md` for the full column list and rules.

### Local cached outputs

Some derivative tables aggregated by `psyfun` are stored under `data/` and `metadata/` rather than on Alyx:

- `metadata/sessions.pqt` — one row per session, with metadata, dataset-presence status, and task-epoch timings.
- `metadata/insertions.pqt` — one row per probe insertion in the project.
- `metadata/histology_status.pqt` — output of `scripts/check_histology_status.py`.
- `data/units.pqt` — unit-level table joined across sessions.
- `data/spikes.h5` — spike times keyed by unit UUID.
- `data/bombcell.pqt` — bombcell QC outputs.

These are local rebuilds, not Alyx state. Regenerate by running the relevant script in `scripts/`.
