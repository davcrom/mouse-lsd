# Spec: registry-based dataset check in `sessions.pqt`

## Status

Implemented (2026-05-20) — bombcell-proportion + histology-state revision of
the 2026-05-15 schema. Prior schema implemented 2026-05-15.

Supersedes `specs/check_dataset_extraction.md` (implemented 2026-05-14,
commits `e808962`..`7b8b3e0`). The 2026-05-15 revision kept the
registry-based mechanism introduced there and changed the column schema:
dropped Alyx columns the project does not use, renamed task timing and
dataset columns to a uniform shorthand convention, simplified the
file-status vocabulary to two values, and replaced the four histology
columns per probe with a single ordered-state column.

This revision changes two of those columns: `probeNN_bombcell` (a
present/missing flag) becomes `probeNN_bombcell_GOOD`, a float fraction
of GOOD units; the session-level `image_stacks` column is removed and
its signal absorbed into `probeNN_histology` as a new `'no-stacks'`
state, with `'missing'` renamed `'no-insertion'`.

## Problem

`sessions.pqt` needs one set of columns that says, per session, which
data files exist — task alf, raw ephys, spike sorting, raw video, pose —
plus a per-probe bombcell GOOD-unit fraction and per-probe histology
pipeline state, so the table is a single source of truth for what data is
available. The columns must be queryable by
dataset name with a `*<shorthand>*` glob so a single column name maps
unambiguously to the underlying dataset.

The existing implementation reports a three-state vocabulary
(`'extraction complete'` / `'extraction error'` / `'raw data missing'`)
that conflates file presence with extraction-pipeline state, and uses
ad-hoc column names (`probe00_spikes`, `task_pre_intervalsTable`,
`left_camera_pose`) that do not match the on-disk dataset names. Several
Alyx columns surfaced into `sessions.pqt` (`project`, `lab`, `number`,
the derived `tasks` list) are unused downstream.

## Inputs

### Per session, from Alyx

- `one.alyx.rest('sessions', 'list', project='psychedelics', task_protocol='passiveChoiceWorld')`
  — initial session enumeration. Verified return fields (8): `id`,
  `subject`, `start_time`, `number`, `lab`, `projects` (list), `url`,
  `task_protocol`.

- `one.alyx.rest('sessions', 'read', id=eid)` — one call per session.
  Used for two fields:
  - `extended_qc` : dict or None. Source of per-camera video-QC values.
  - `data_dataset_session_related` : list of dicts, one per registered
    dataset. Verified against session `8dfd9963-25e5-4f63-8f91-5b27a5852628`
    (145 entries). Keys per entry: `name`, `collection`, `dataset_type`,
    `data_url` (str or None), `url`, `id`, `hash`, `file_size`,
    `version`, `revision`, `default_revision` (string `'True'`/`'False'`),
    `qc`.

- `one.alyx.rest('insertions', 'list', session=eid, no_cache=True)` —
  one call per session. List of insertion records with `name`
  (`probe00`/`probe01`) and `id` (pid). Used to enumerate probe slots.

- `one.alyx.rest('insertions', 'list', id=pid, no_cache=True)[0]` —
  one call per probe. Full insertion record with `json`. Source of the
  histology-pipeline state for that probe.

### Per session, from a settings file

- `_iblrig_taskSettings.raw.json` in each `raw_task_data_NN` collection,
  read by `_load_passive_run` (already in `psyfun/io.py`) to classify a
  run as `passive` vs `spontaneous`-filler via `PYBPOD_PROTOCOL`.

### Local filesystem

- Bombcell output: `one.eid2path(eid) / 'spike_sorters' / <sorter> / <probe> / 'bombcell' / 'templates._bc_qMetrics.parquet'`.
  Read as a parquet. Verified to contain a `bc_unitType` column with
  string values in `{'GOOD', 'NON-SOMA', 'MUA', 'NOISE', ''}`, one row
  per detected unit.

### Data server

- `psyfun.histology.list_histology_tifs(subject, lab, par)` — HTTP GET
  of `<HTTP_DATA_SERVER>/histology/<lab>/<subject>/downsampledStacks_25/sample2ARA/`,
  parses `href="*.tif"` entries, returns the filename list. Cached
  per `(subject, lab)` across the run.

## Outputs

`sessions.pqt` columns, exhaustive list. Every column is either listed
in the "Retained / derived non-check columns" subsection below or in
one of the per-check subsections that follow.

### Columns dropped from the Alyx-native set

Of the 8 fields returned by `sessions/list`, three are not written to
`sessions.pqt`: `projects` (list of project names — constant for this
experiment), `lab` (constant `mainenlab`), `number` (within-subject
session index, not read by any in-tree consumer). The derived `tasks`
list (`task_protocol.split('/')`) is also not written.

### Retained / derived non-check columns

| column | type | source | content |
|---|---|---|---|
| `eid` | str (UUID) | `sessions/list` `id`, renamed | session identifier |
| `subject` | str | `sessions/list` `subject` | animal name (e.g. `ZFM-08631`) |
| `start_time` | str (ISO 8601) | `sessions/list` `start_time` | session start in local time |
| `url` | str | `sessions/list` `url` | Alyx REST URL for the session record |
| `task_protocol` | str | `sessions/list` `task_protocol` | full IBL protocol string (slash-joined task list, e.g. `passiveChoiceWorld/_iblrig_misc_LSD_filler/passiveChoiceWorld`) |
| `n_probes` | int | `_count_probes` | number of probe insertions registered in Alyx for this session |
| `n_tasks` | int | derived from `task_protocol` | count of tokens in `task_protocol` containing `passive` (case-insensitive) |
| `control_recording` | bool | `_label_controls` | `True` if `eid` is in `psyfun.config.df_controls`, else `False` |
| `session_n` | int | dense rank of `start_time` within `subject` | within-subject session index, starting at 1 |

### File-status vocabulary

All file-status cells take one of two strings: `'present'` or
`'missing'`. No reference to extraction.

### Column-name rule

`<slot>_<shorthand>` for task and ephys columns (the same dataset name
appears under multiple collections, so the slot disambiguates). Raw
filenames and raw Alyx QC keys for video and video-QC columns (the
camera name is already in the filename / key). Every file-status column
is queryable by `*<shorthand>*` against the underlying dataset name.

### Task (6 columns)

Slots `pre` and `post`, by passive-run order — not by `raw_task_data_NN`
number (a spontaneous-only LSD-filler run can sit between the two
passive runs).

| column | matches dataset |
|---|---|
| `pre_intervalsTable` | `_ibl_passivePeriods.intervalsTable.csv` |
| `pre_passiveStims` | `_ibl_passiveStims.table.csv` |
| `pre_passiveGabor` | `_ibl_passiveGabor.table.csv` |
| `post_intervalsTable` | `_ibl_passivePeriods.intervalsTable.csv` |
| `post_passiveStims` | `_ibl_passiveStims.table.csv` |
| `post_passiveGabor` | `_ibl_passiveGabor.table.csv` |

### Task timing columns (13 columns)

Renamed from `task_pre_*` / `task_post_*` to `pre_*` / `post_*`:

```
pre_spontaneous_start, pre_spontaneous_stop,
pre_rfm_start,         pre_rfm_stop,
pre_replay_start,      pre_replay_stop,
LSD_admin,
post_spontaneous_start, post_spontaneous_stop,
post_rfm_start,         post_rfm_stop,
post_replay_start,      post_replay_stop,
```

`PASSIVE_SLOTS` becomes `('pre', 'post')`; `TASKTIMINGS` is renamed
element-wise to match. Written by `_fetch_protocol_timings`, not by the
dataset check, but updated in the same pipeline pass.

### Ephys (8 columns per probe × 2 slots = 16 columns)

Slots `probe00`, `probe01`.

| column | matches / content |
|---|---|
| `probeNN_ap.cbin` | `_spikeglx_ephysData_g0_t0.imec<N>.ap.cbin` or `…imec<NN>.ap.cbin` in `raw_ephys_data/probeNN` |
| `probeNN_sync.npy` | `_spikeglx_ephysData_g0_t0.imec<N>.sync.npy` or `…imec<NN>.sync.npy` in `raw_ephys_data/probeNN` |
| `probeNN_spikes.times` | `spikes.times.npy` in `alf/probeNN/<sorter>` |
| `probeNN_spikes.clusters` | `spikes.clusters.npy` in `alf/probeNN/<sorter>` |
| `probeNN_clusters.uuids` | `clusters.uuids.csv` in `alf/probeNN/<sorter>` |
| `probeNN_sorter` | registry `version` string verbatim (e.g. `pykilosort_ibl_1.4.1`); `''` if no sorting registered |
| `probeNN_bombcell_GOOD` | float in `[0, 1]`: fraction of detected units labelled `GOOD` in `templates._bc_qMetrics.parquet`; `NaN` if the file is absent or the slot has no insertion |
| `probeNN_histology` | one of `'resolved'`, `'aligned'`, `'traced'`, `'no-tracing'`, `'no-stacks'`, `'no-insertion'` |

`probeNN_ap.cbin`, `probeNN_sync.npy`, `probeNN_spikes.times`,
`probeNN_spikes.clusters`, `probeNN_clusters.uuids` are
`'present'`/`'missing'`. `probeNN_bombcell_GOOD` is a float column.
When the probe slot has no insertion, the five file-status columns are
`'missing'`, `probeNN_sorter` is `''`, `probeNN_bombcell_GOOD` is `NaN`,
and `probeNN_histology` is `'no-insertion'`.

### Video (6 columns)

No slot prefix (the camera name is already in the dataset name).

| column | matches dataset |
|---|---|
| `leftCamera.raw` | `_iblrig_leftCamera.raw.mp4` in `raw_video_data` |
| `leftCamera.lightningPose` | `_ibl_leftCamera.lightningPose.pqt` in `alf` |
| `rightCamera.raw` | `_iblrig_rightCamera.raw.mp4` |
| `rightCamera.lightningPose` | `_ibl_rightCamera.lightningPose.pqt` |
| `bodyCamera.raw` | `_iblrig_bodyCamera.raw.mp4` |
| `bodyCamera.lightningPose` | `_ibl_bodyCamera.lightningPose.pqt` |

All six are `'present'`/`'missing'`.

### Video QC (9 columns)

Verbatim Alyx `extended_qc` keys (leading underscore retained):

```
_videoLeft_dropped_frames, _videoLeft_timestamps, _videoLeft_pin_state,
_videoRight_dropped_frames, _videoRight_timestamps, _videoRight_pin_state,
_videoBody_dropped_frames, _videoBody_timestamps, _videoBody_pin_state,
```

Value is the QC outcome string as written by Alyx, after a
whitespace-to-underscore normalisation (`'NOT SET'` → `'NOT_SET'`,
applied uniformly to any outcome). Any outcome string present in the
registry is allowed; no enumeration is imposed by this check. The
`_dropped_frames` and `_pin_state` Alyx values are lists whose first
element is the outcome; `_timestamps` holds the outcome string
directly. Empty string when `extended_qc` is absent or has no entry
for the key.

### Session-level histology

No session-level histology column. Image-stack presence is no longer its
own column; it gates the per-probe `probeNN_histology` state (`'no-stacks'`
vs `'no-tracing'`). The `_check_image_stacks` helper survives and feeds
that gating.

## Behavior

### Presence rule

Build, once per session, from `data_dataset_session_related`:

```python
present = {
    (d['collection'], d['name'])
    for d in data_dataset_session_related
    if d['data_url']
}
```

A dataset is present if its `(collection, name)` pair is in `present`.
The full entries are also keyed by `(collection, name)` so the ephys
check can read `version`, `revision`, and `default_revision`. Every
file-status cell is `'present'` if its dataset is present, `'missing'`
otherwise. There is no raw-prerequisite distinction.
`probeNN_bombcell_GOOD` is the one ephys column read from the local
filesystem rather than the registry (see Ephys per probe).

### Task slots

Enumerate the passive slots with `_list_passive_raw_collections(eid, one)`:
derive the `raw_task_data_NN` collection names from
`data_dataset_session_related` (collections matching `^raw_task_data_\d+$`,
sorted by `NN`), classify each with `_load_passive_run`, keep the runs
tagged `kind='passive'` in run order. Element 0 → `pre` slot; element 1
→ `post`. For each passive slot, the raw collection (e.g.
`raw_task_data_02`) maps to its alf collection by
`raw_col.replace('raw_task_data_', 'alf/task_')`.

For each of the three task datasets, status is `'present'` if its
`(alf_collection, name)` is in `present`, else `'missing'`. When a slot
has no passive run, its three columns are `'missing'`.

### Ephys per probe

Enumerate probe slots from `one.alyx.rest('insertions', 'list', session=eid)`,
sorted by `name`; slot 0 / 1 → column prefix `probe00` / `probe01`. When
a slot has no insertion, write all file-status columns `'missing'`,
`sorter = ''`, `bombcell_GOOD = NaN`, `histology = 'no-insertion'`.

For a probe with insertion record `ins`:

- `probeNN_ap.cbin` : `'present'` if either `…imec<N>.ap.cbin` or
  `…imec<NN>.ap.cbin` is in collection `raw_ephys_data/<ins['name']>`.
- `probeNN_sync.npy` : same two-form check for `…imec<N>.sync.npy` /
  `…imec<NN>.sync.npy`.
- Choose sorter and revision (see below). With `<sorter>` resolved,
  check the three files in `alf/<probe>/<sorter>`:
  `spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv`.
  Each is its own column.
- `probeNN_bombcell_GOOD` : build the path
  `one.eid2path(eid) / 'spike_sorters' / <sorter> / <probe> / 'bombcell' / 'templates._bc_qMetrics.parquet'`.
  If it does not exist (including when `<sorter> = ''`), the value is
  `NaN`. If it exists, read the parquet and compute
  `(df['bc_unitType'] == 'GOOD').mean()` over all rows — the denominator
  is the total number of detected units, including rows whose
  `bc_unitType` is the empty string. No `bombcell` import is needed; the
  `bc_unitType` column is already in the parquet.

**Sorter selection.**

1. For each sorter in `('iblsorter', 'pykilosort', 'ks2')` (priority
   order), look in `data_dataset_session_related` for entries with
   `name == 'spikes.times.npy'` and
   `collection == f'alf/{probe}/{sorter}'`. Take the first sorter in
   that order that has any such entry.
2. If the sorter has more than one `spikes.times.npy` entry
   (multiple revisions), pick the one whose `default_revision` is the
   string `'True'`. If only one entry exists, take it regardless of the
   `default_revision` value. The selected entry's `version` is the
   sorter version string.
3. `probeNN_sorter = version` (e.g. `'pykilosort_ibl_1.4.1'`). If `version`
   is empty or no sorter has a registered `spikes.times.npy`,
   `probeNN_sorter = ''`.

**Histology state.** `_check_histology_probe` takes a `stacks_present:
bool` argument (the subject's image-stack presence, computed once per
session by `_check_datasets`). Evaluation order, first match wins:

| state | predicate |
|---|---|
| `'no-insertion'` | no insertion in this slot (`ins is None`) |
| `'resolved'` | `insertion_alignment_resolved(full)` |
| `'aligned'`  | `insertion_alignment_uploaded(full)` |
| `'traced'`   | `insertion_picks(full)` |
| `'no-tracing'` | insertion exists, no picks, `stacks_present` is True |
| `'no-stacks'` | insertion exists, no picks, `stacks_present` is False |

`no-insertion` is checked first, so it supersedes `no-stacks` when a slot
has neither insertion nor stacks. The Alyx record `full` is fetched
(`one.alyx.rest('insertions', 'list', id=ins['id'], no_cache=True)[0]`)
only when `ins is not None`. Predicates are
`psyfun.histology.insertion_picks`, `insertion_alignment_uploaded`,
`insertion_alignment_resolved`.

### Video per camera

`<camera>Camera.raw` is `'present'` if `_iblrig_<camera>Camera.raw.mp4`
is in `raw_video_data`. `<camera>Camera.lightningPose` is `'present'`
if `_ibl_<camera>Camera.lightningPose.pqt` is in `alf`. DLC output is
not accepted as pose tracking; a camera with only DLC has
`<camera>Camera.lightningPose = 'missing'`.

Video QC cells come from `extended_qc[key]`:
`_dropped_frames` and `_pin_state` values are lists whose first element
is the outcome string; `_timestamps` values are the outcome string
directly. Missing key or missing `extended_qc` → empty string.

### Image-stack gating

`_check_image_stacks(subject, lab)` returns `'present'` iff
`list_histology_tifs(subject, lab, par)` returns at least one filename
matching `*_RD.tif` and at least one matching `*_GR.tif`; else
`'missing'`. Cached per `(subject, lab)`. `_check_datasets` computes
`stacks_present = _check_image_stacks(subject, lab) == 'present'` once per
session and passes it into each `_check_histology_probe` call. No
`image_stacks` column is written.

### Producer pipeline

`fetch_sessions` in `psyfun/io.py` produces `sessions.pqt`:

```
query Alyx for project+protocol
  → drop project, lab, number columns; keep task_protocol, derive n_tasks
  → _count_probes
  → _check_datasets       # writes the columns in Outputs above
  → _label_controls
  → _fetch_protocol_timings   # writes pre_*/post_* timing columns
  → _insert_LSD_admin_time
  → rank session_n
  → save parquet
```

`_check_datasets` calls `sessions/read` once per session and uses both
`extended_qc` and `data_dataset_session_related`. Its helpers
(`_check_task_alf`, `_check_probe`, `_check_camera`,
`_check_image_stacks`, `_check_histology_probe`) realise the per-section
behavior above; `_check_image_stacks` now feeds the per-probe histology
gating instead of writing its own column. `_list_passive_raw_collections`
derives `raw_task_data_NN` collection names from
`data_dataset_session_related` (not `one.list_datasets`).

### Consumer updates

Every in-tree reader of the renamed columns or constants is updated to
the new schema in the same change:

- `psyfun/config.py` — `PASSIVE_SLOTS = ('pre', 'post')`; `TASKTIMINGS`
  renamed element-wise (`task_pre_*` → `pre_*`, `task_post_*` →
  `post_*`; `LSD_admin` unchanged).
- `psyfun/io.py` — `load_sessions` continues to use `TASKTIMINGS` as its
  NaN-drop filter (now the renamed list). Module docstring reference to
  `specs/check_dataset_extraction.md` → `specs/check_session_datasets.md`.
- `scripts/dataset_overview.py` — reads columns by their new names;
  renders the two-state vocabulary, the `probeNN_sorter` version
  string, the `probeNN_histology` states, and the verbatim
  `_video*_*` QC keys.
- `scripts/fetch_data.py` — reads the new file-status columns when
  deciding what to download.
- `tests/test_check_datasets.py` — assertions on the new column names,
  two-state vocabulary, `probeNN_histology` states, and `image_stacks`
  RD+GR rule.
- `tests/test_fetch_protocol_timings.py` — assertions on the renamed
  timing columns (`pre_*` / `post_*`).

Bombcell-proportion + histology-state revision (scope: `psyfun/io.py`
and `tests/test_check_datasets.py` only — grep confirms no script reads
`probeNN_bombcell`, `image_stacks`, or the `probeNN_histology` state
values):

- `psyfun/io.py` — `_check_probe` emits `probeNN_bombcell_GOOD` (the
  proportion / `NaN`) in place of `probeNN_bombcell`;
  `_check_histology_probe` gains the `stacks_present` argument and the
  six-state ladder; `_check_datasets` computes `stacks_present` once and
  drops the `image_stacks` output assignment.
- `tests/test_check_datasets.py` — `_check_probe` tests write a real
  parquet with a `bc_unitType` column (the prior `bombcell.write_text("x")`
  fixture is not a parquet) and assert the GOOD fraction / `NaN`;
  `_check_histology_probe` tests adopt the new signature, rename the
  no-insertion expectation to `'no-insertion'`, and add a `'no-stacks'`
  case; the `_check_datasets` test drops any `image_stacks` output
  assertion (its `_check_image_stacks` monkeypatch stays); the
  `_check_image_stacks` unit tests are unchanged.

## Out of scope

Functional gaps:

- Post-registration file drift (truncated / hash-mismatched files on
  the server whose registry record still says they exist).
- Semantic correctness of extracted files (e.g. an extractor merging
  two protocol blocks).
- Re-running IBL extractions.

Adjacent artefacts not modified:

- The aggregated `data/bombcell.pqt` output of `scripts/unit_qc.py`.
  `probeNN_bombcell_GOOD` reads the per-probe bombcell parquet on disk,
  not this aggregated table; there is no fallback to `data/units.pqt`.
- `scripts/check_histology_status.py` and its output
  `metadata/histology_status.pqt`. The script calls `load_sessions`
  but does not read any of the columns renamed by this spec.

In-tree readers of these names that are NOT updated by this change:

- `scripts/single_unit.py`, `scripts/population_dimensionality.py` —
  separate planned refactor; same exclusion as the prior spec.
- `scripts/validate_gabor_alignment.py` — reads `sessions.pqt`
  task-epoch timings under the old `task_pre_*`/`task_post_*` names.
  Validation utility, not on the analysis path; flagged here so it
  will fail loudly after the rename and can be migrated when needed.
- `scripts/dump_xyz_picks.py` — calls `load_sessions` but does not
  read any renamed columns; no change needed.
- `notebooks/` — legacy, not maintained.
- `archive/`, `davide/` — not maintained.

Spec file lifecycle:

- The prior spec `specs/check_dataset_extraction.md` is left in place
  with its `Status: Implemented` header intact; this file supersedes
  it. Deleting the prior spec is the user's call, not this change.

## Implementation plan

Bombcell-proportion + histology-state revision:

- `tickets/10-bombcell-good-proportion-column.md` — `_check_probe` emits
  `probeNN_bombcell_GOOD` (float fraction of GOOD units / `NaN`).
- `tickets/11-histology-six-states-drop-image-stacks.md` —
  `_check_histology_probe` six-state ladder with `stacks_present` arg;
  `_check_datasets` drops the `image_stacks` column.

(The 01–09 tickets implemented the prior 2026-05-15 schema.)

## Decisions

- **Two-state file vocabulary.** Source: user, this conversation. The
  three-state vocabulary mixed file presence with extraction-pipeline
  state; the rare "extraction error" cell (raw present, derived absent)
  duplicates what a missing-derived-file already conveys, and a clean
  load did not catch the semantic errors that actually affected this
  project. Reducing to `'present'`/`'missing'` matches what the
  downstream code uses the column for.
- **Drop `projects`, `lab`, `number`, `tasks`.** Source: user, this
  conversation. The Alyx field is `projects` (plural, list-typed) —
  constant for this experiment (`['psychedelics']`); `lab` is constant
  (`mainenlab`); `number` is the within-subject session index, never
  read by `psyfun` or `scripts/` (`session_n` is the derived
  replacement); `tasks` (the `task_protocol.split('/')` list) is no
  longer read after `_fetch_protocol_timings` was rewritten to derive
  collections from the registry.
- **Keep `task_protocol` + `n_tasks`.** Source: user. `task_protocol`
  is the Alyx-native string and is read by `fetch_insertions`.
  `n_tasks` (count of `passive` tokens) is a useful convenience
  derivable from `task_protocol`.
- **Drop the `task_` prefix from timing columns.** Source: user.
  Columns become `pre_spontaneous_start`, etc. `PASSIVE_SLOTS` becomes
  `('pre', 'post')`, `TASKTIMINGS` is renamed element-wise.
- **Column-name rule `<slot>_<shorthand>`.** Source: user. Slot
  disambiguates the same dataset name across multiple collections
  (`probe00` vs `probe01`, `pre` vs `post`). Shorthand retains enough
  of the original dataset name that `*<shorthand>*` globs the file
  unambiguously.
- **No slot prefix on video columns.** Source: user. The camera name
  is already in the filename (`leftCamera.raw`), so a slot prefix would
  be redundant. Same for the Alyx QC keys (`_videoLeft_…`). The schema
  isn't perfectly uniform, but consistent `*<shorthand>*` queryability
  is preserved.
- **One column per spike-sorter dataset file.** Source: user — "someone
  should be able to use `*<shorthand>*` to query for/load it." Lumping
  `spikes.times.npy` + `spikes.clusters.npy` + `clusters.uuids.csv`
  into one `probeNN_spikes` cell would break that.
- **`probeNN_sorter` carries the registry `version` string verbatim.**
  Source: user. Empty string when no sorter is registered. No
  parenthesised sorter-name + version format.
- **Single `probeNN_histology` ordered-state column.** Source: user.
  The boolean predicates are sequential and blocking: stacks gate
  tracing, alignment cannot be uploaded without tracing, cannot be
  resolved without alignment. Compressing them to one ordered-state
  column captures every distinction the booleans did. States, highest
  reached wins: `'resolved'` > `'aligned'` > `'traced'` >
  `'no-tracing'` (insertion + stacks, no picks) > `'no-stacks'`
  (insertion, no stacks) > `'no-insertion'` (no insertion).
- **Image-stack state folded into `probeNN_histology`, `image_stacks`
  column removed.** Source: user, this revision. Stacks are the first
  histology-pipeline step (they gate tracing), so the per-subject
  signal belongs in the per-probe state ladder as `'no-stacks'` rather
  than a separate session-level column. No consumer reads `image_stacks`.
- **`'no-tracing'` kept as a distinct rung.** Source: user. "Insertion +
  stacks present, not yet traced" is the actionable TRACE-PICKS step and
  is distinct from `'no-stacks'` (cannot trace) and `'traced'`; the user
  chose to keep it rather than collapse.
- **`'no-insertion'` checked first.** Source: user. The insertion
  precedes stacks in the pipeline, so a missing insertion supersedes a
  missing stack; renamed from the prior `'missing'` for clarity.
- **`image_stacks` rule (both `*_RD.tif` and `*_GR.tif`) unchanged.**
  Source: user. The previous `>= 2` threshold was a proxy for the same
  intent; checking the colour-channel suffixes is direct. The helper
  survives to gate the histology state.
- **`probeNN_bombcell_GOOD` = fraction of GOOD units.** Source: user,
  this revision. Numerator is `bc_unitType == 'GOOD'` only (excludes
  `NON-SOMA`); denominator is the total number of detected units (every
  row of the qMetrics parquet, including empty-label rows). Replaces the
  present/missing flag, whose information (file exists) is subsumed by
  a non-`NaN` value.
- **Read the proportion from the already-checked qMetrics parquet.**
  Source: verified — `templates._bc_qMetrics.parquet` carries the
  `bc_unitType` column, so the proportion is a plain parquet read and
  avoids adding a `bombcell` dependency to session building.
- **`NaN`, not a sentinel string, for absent bombcell / no insertion.**
  Source: user — "NaN is fine, clean float column." Keeps the column a
  clean float dtype.
- **No fallback to `units.pqt` for the proportion.** Source: user — the
  ONE cache is ground truth. The current `units.pqt` has no `bc_label`
  column anyway, so a fallback would be ill-defined.
- **Sorter priority `iblsorter > pykilosort > ks2`, then
  `default_revision == 'True'`.** Carried from the prior spec
  (`check_dataset_extraction.md`). Most sessions have a single sorter.
- **`default_revision == 'True'` filter only between multiple
  revisions.** Source: user, this conversation. When a sorter has a
  single registered `spikes.times.npy`, take it regardless of the
  `default_revision` value, so a non-default-flagged single revision
  is not silently treated as "no sorting".
- **QC outcomes are allowed verbatim with whitespace normalisation.**
  Source: user, this conversation. Replace internal whitespace with
  underscore (`'NOT SET'` → `'NOT_SET'`) so trivial Alyx variants
  collapse to a single token; do not otherwise enumerate or restrict
  the allowed set.
- **`data_url` non-null is the presence test.** Carried from the prior
  spec. Verified: every entry in the 145-dataset reference session has
  non-null `data_url`.
- **Spec file renamed** `check_dataset_extraction.md` →
  `check_session_datasets.md`. The new vocabulary no longer references
  "extraction". The `psyfun/io.py` docstring reference to the prior
  spec path is updated as part of Consumer updates.
- **Post-registration drift remains accepted as a gap.** Carried from
  the prior spec.
- **`data_url` host after IBL's S3 migration.** Carried from the prior
  spec. The non-null presence test still holds if `data_url` is later
  served from S3; any flatiron-specific assumption does not.

## Open questions

(none)
