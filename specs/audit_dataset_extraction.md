# Replace `sessions.pqt` with a load-based dataset audit

## Status

Implemented (2026-05-14)

Note: the script callers under `scripts/` (`dataset_overview.py`,
`single_unit.py`, `population_dimensionality.py`) were only partially
migrated — `dataset_overview.py` is done, the other two are deferred to
a planned larger refactor. `scripts/fetch_data.py` (the data-fetch path)
is fully migrated. Notebooks were left untouched (legacy).

## Goal

Replace the current `metadata/sessions.pqt` with a table whose dataset-presence columns come from actually loading each relevant dataset via `one.load_dataset`. The current table mixes Alyx metadata, a list-based dataset check (`_check_datasets`), parsed extended QC (`_unpack_session_dict`), and task-epoch timings (`_fetch_protocol_timings`). The list-based check is unreliable (some datasets exist in `one.list_datasets` but fail to load due to checksum or file-record issues), the extended QC adds dozens of low-signal video columns, and the dataset list hard-coded in `config.qc_datasets` is incomplete (no spike sorting, no video alf, lightning pose missing entirely).

The new table is the single source of truth for what data exists per session, validated by load.

## Datasets the audit must cover

Three categories. A **slot** is a position within a modality that can hold data for a session: the two passive-task positions (`task_pre`, `task_post`), the two probe positions (`probe00`, `probe01`), and the three camera positions (`left`, `right`, `body`). A session has one cell per `(slot, file)` pair even when the slot is empty (status `raw data missing`).

Status per `(session, slot, file)` cell is one of:

- `extraction complete`: `one.load_dataset(eid, dataset, collection)` returns without raising.
- `extraction error`: the prerequisite raw upstream is present (taskSettings loads for passive tasks; raw `.cbin` / `.sync.npy` for probes; raw `.mp4` for cameras) but the alf dataset raises `ALFObjectNotFound`.
- `raw data missing`: no raw upstream at this slot.

### Task (passive protocol)

Two slots: `task_pre` (first passive run) and `task_post` (second passive run). The slot names are `task_pre` / `task_post` — deliberately not `task00` / `task01` — to avoid confusion with the IBL `raw_task_data_NN` collections and their `alf/task_NN` counterparts, which are numbered by physical run order (including spontaneous-filler runs) and so do not line up with the passive-run index.

**Enumerating the two passive slots:** `_fetch_protocol_timings` already does this work via `_list_raw_task_collections` and `_load_passive_run` in `psyfun/io.py`. `_load_passive_run` loads `_iblrig_taskSettings.raw.json` for each `raw_task_data_NN`, reads `PYBPOD_PROTOCOL`, and tags the run `kind='passive'` or `kind='spontaneous'`. Add `_list_passive_raw_collections(eid, one)` that reuses that classification and returns the raw collection names of the `passive` runs in run order; element 0 is the `task_pre` slot, element 1 is `task_post`. Do not re-implement the regex/settings-loading logic.

For each passive slot, the raw collection name (e.g. `raw_task_data_01`) maps to its alf collection by `raw_col.replace("raw_task_data_", "alf/task_")`. The audit loads three files from that alf collection:

| file | dataset name |
|---|---|
| intervalsTable | `_ibl_passivePeriods.intervalsTable.csv` |
| passiveStims | `_ibl_passiveStims.table.csv` |
| passiveGabor | `_ibl_passiveGabor.table.csv` |

Prerequisite raw: `_iblrig_taskSettings.raw.json` in `raw_task_data_NN` with `PYBPOD_PROTOCOL` containing `passive`. (A run whose protocol contains `spontaneous` but not `passive` is the LSD-filler run; it fills no passive slot but is still used by `_fetch_protocol_timings` to set `LSD_admin`.)

### Ephys (per probe, slots `probe00`, `probe01`)

Probe slots are read from the session's registered insertions: `one.alyx.rest('insertions', 'list', session=eid)`. Each insertion record has a `name` field (e.g. `probe00`, `probe01`) and an `id` field (the `pid`). Iterate insertions sorted by `name` and emit one column-set per probe slot found. If a session has only one probe, columns for the missing slot are written with status `raw data missing` and `probeNN_sorter = ''`.

Five columns per probe:

| column | meaning | source |
|---|---|---|
| `probeNN_raw_ap` | status of the raw SpikeGLX AP recording | `_spikeglx_ephysData_g0_t0.imecN.ap.cbin` in `raw_ephys_data/probeNN/` |
| `probeNN_sync` | status of the probe-to-session-clock map | `_spikeglx_ephysData_g0_t0.imecN.sync.npy` in `raw_ephys_data/probeNN/` |
| `probeNN_sorter` | name of the sorter whose output is being used, with revision tag if non-empty | see below |
| `probeNN_spikes` | status of the three extracted spike-data files used downstream | `spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv` in `alf/probeNN/<sorter>/` |
| `probeNN_bombcell` | status of the bombcell unit-quality output | local file presence under `one.eid2path(eid) / 'spike_sorters' / <sorter> / probeNN / 'bombcell'` — see note below |

Plain-language note on `sync`: SpikeGLX records each probe on its own clock. The `.sync.npy` file is the time-base map IBL extracts so spike times can be expressed in the session clock used by everything else (task, video, behaviour). Required for any cross-modal alignment.

The IBL SpikeGLX filename inconsistently uses `imec0` / `imec1` versus `imec00` / `imec01`. Try both forms; status = `extraction complete` if either loads.

**Choosing the sorter and revision per probe:**

1. Query Alyx with `one.alyx.rest('datasets', 'list', session=eid, name='spikes.times.npy', collection='alf/probeNN/__sorter__')` (template — substitute each known sorter from `{iblsorter, pykilosort, ks2}`) and read each record's `revision` and `date_created`.
2. Among the records returned, pick the one with the most recent `date_created`. That record's collection determines the sorter name; its `revision` is the revision tag.
3. Try to load `spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv` from that exact (collection, revision) pair. If all three load, `probeNN_spikes = 'extraction complete'` and `probeNN_sorter = "<sorter>"` (plus `"#<revision>#"` suffix when revision is non-empty).
4. If no sorter dataset is registered at all on Alyx, `probeNN_sorter = ''` and `probeNN_spikes` follows the three-state rule based on whether the raw prerequisite exists.

Today every dataset in this project has `revision=''`, so `probeNN_sorter` will currently be one of `''`, `iblsorter`, `pykilosort`. The revision-tag suffix is reserved for future re-sortings, which IBL registers under `#YYYY-MM-DD#` revisions.

Prerequisite raw for `probeNN_spikes` and `probeNN_bombcell`: `raw_ap` present.

**Bombcell is not an Alyx-registered dataset.** It is produced by this repo's own script (`scripts/unit_qc.py` via `psyfun/spike_sorting.py`), which writes its output next to the spike-sorting data at `one.eid2path(eid) / 'spike_sorters' / <sorter> / probeNN / 'bombcell'`. So `probeNN_bombcell` is checked by **local file presence**, not `one.load_dataset`: resolve the session directory with `one.eid2path(eid)` (this respects whatever ONE cache directory is configured on the machine running the audit — never hard-code a cache path), build the bombcell directory path with the same `<sorter>` chosen above, and check whether the bombcell output file is present there (e.g. `templates._bc_qMetrics.parquet`). Status: `extraction complete` if present, `extraction error` if `raw_ap` is present but the bombcell output is not, `raw data missing` otherwise. Bombcell can be absent even when `probeNN_spikes` is `extraction complete` — some sessions have the sorter alf output but bombcell has not been run.

### Video (per camera, slots `left`, `right`, `body`)

Lightning Pose is registered per camera on Alyx (`_ibl_<C>Camera.lightningPose.pqt`), not as a single dataset across the three videos — the dataset-type name `camera.lightningPose` is instantiated with the `_ibl_<C>Camera` prefix per camera. None of our sessions have lightning pose yet; the column is forward-looking. DLC output is intentionally not checked: the project standardises on Lightning Pose, so a camera with only DLC tracking counts as `extraction error`, not as tracked.

Five columns per camera. Two are load-based status cells; three carry video QC values pulled from the session extended QC.

| column | meaning | source |
|---|---|---|
| `<C>_camera_raw_video` | status of raw video | `_iblrig_<C>Camera.raw.mp4` in `raw_video_data/` |
| `<C>_camera_pose` | status of Lightning Pose tracking | `_ibl_<C>Camera.lightningPose.pqt` in `alf/` |
| `<C>_camera_dropped_frames` | video QC: dropped-frame check outcome | session extended QC key `_video<Cam>_dropped_frames` |
| `<C>_camera_timestamps` | video QC: timestamp check outcome | session extended QC key `_video<Cam>_timestamps` |
| `<C>_camera_pin_state` | video QC: pin-state check outcome | session extended QC key `_video<Cam>_pin_state` |

The extended-QC keys use the camera name capitalised, e.g. `_videoLeft_dropped_frames`, `_videoRight_timestamps`, `_videoBody_pin_state`. The audit stores the QC outcome string only: `_video<Cam>_dropped_frames` and `_video<Cam>_pin_state` hold a list whose first element is the outcome (e.g. `['PASS', 14, 0]`) — take element 0; `_video<Cam>_timestamps` holds the outcome string directly (e.g. `'PASS'`). When extended QC is absent or has no entry for a key, the column is the empty string.

Prerequisite raw for `pose`: `raw_video` present. The three QC columns are not load-based status cells; they carry the extended-QC outcome only.

### Histology / trajectory (from Alyx insertion records, not from `load_dataset`)

Fields per insertion (pid) come from `one.alyx.rest('insertions', 'list', id=pid, no_cache=True)[0]`. Logic and definitions match `scripts/check_histology_status.py`, which the new code imports from rather than duplicates.

| column | scope | meaning | source |
|---|---|---|---|
| `image_stacks` | session-level (per subject; same for all probes of a session) | histology TIF stacks uploaded for this subject | `scripts.check_histology_status.list_histology_tifs(subject, lab, par)` returns >= 2 entries; the result is cached per `(subject, lab)` within a single `fetch_sessions` run |
| `probeNN_traced` | per probe | probe track traced in the histology volume (Alyx `xyz_picks` recorded) | `bool((insertion.get('json') or {}).get('xyz_picks') or [])` |
| `probeNN_alignment_uploaded` | per probe | at least one alignment uploaded | `((insertion.get('json') or {}).get('extended_qc') or {}).get('alignment_count', 0) > 0` |
| `probeNN_alignment_resolved` | per probe | trajectory finalized | `((insertion.get('json') or {}).get('extended_qc') or {}).get('alignment_resolved') is True` |

These are booleans, not the three-state vocabulary used elsewhere — they describe pipeline progress, not file presence. `image_stacks` is repeated once per session (same value for all probes within a session). When a probe slot has no registered insertion, the three per-probe histology columns are `False`.

The new code should reuse the three predicate functions already defined in `scripts/check_histology_status.py` (`insertion_picks` — produces the `probeNN_traced` column, `insertion_alignment_uploaded`, `insertion_alignment_resolved`) and the `list_histology_tifs` helper, rather than reimplementing them. If those helpers stay in `scripts/`, import them; if cleaner, move them into a new `psyfun/histology.py` module.

## Columns of the new `sessions.pqt`

### Keep (from current schema)

Identification and grouping:
- `eid`, `subject`, `lab`, `start_time`, `projects`, `task_protocol`, `number`, `url`
- `n_probes`, `n_tasks`, `tasks` (list of protocol strings)
- `session_n`, `control_recording`

Task-epoch timings (output of `_fetch_protocol_timings`), renamed from the current `task00` / `task01` prefix to `task_pre` / `task_post`:
- `task_pre_spontaneous_start`, `task_pre_spontaneous_stop`
- `task_pre_rfm_start`, `task_pre_rfm_stop`
- `task_pre_replay_start`, `task_pre_replay_stop`
- `task_post_spontaneous_start`, `task_post_spontaneous_stop`
- `task_post_rfm_start`, `task_post_rfm_stop`
- `task_post_replay_start`, `task_post_replay_stop`
- `LSD_admin`

The rename is in scope: `_fetch_protocol_timings` currently writes `f'task{passive_idx:02d}_{epoch}_{endpoint}'`; change it to map `passive_idx` 0 → `task_pre`, 1 → `task_post`. The `TASKTIMINGS` constant in `psyfun/config.py` and every caller of these columns must be updated to match.

### Add (new audit columns)

Task (6 columns):
- `task_pre_intervalsTable`, `task_pre_passiveStims`, `task_pre_passiveGabor`
- `task_post_intervalsTable`, `task_post_passiveStims`, `task_post_passiveGabor`

Ephys (10 columns):
- `probe00_raw_ap`, `probe00_sync`, `probe00_sorter`, `probe00_spikes`, `probe00_bombcell`
- `probe01_raw_ap`, `probe01_sync`, `probe01_sorter`, `probe01_spikes`, `probe01_bombcell`

Video (15 columns):
- `left_camera_raw_video`, `left_camera_pose`, `left_camera_dropped_frames`, `left_camera_timestamps`, `left_camera_pin_state`
- `right_camera_raw_video`, `right_camera_pose`, `right_camera_dropped_frames`, `right_camera_timestamps`, `right_camera_pin_state`
- `body_camera_raw_video`, `body_camera_pose`, `body_camera_dropped_frames`, `body_camera_timestamps`, `body_camera_pin_state`

Histology / trajectory (7 columns):
- `image_stacks`
- `probe00_traced`, `probe00_alignment_uploaded`, `probe00_alignment_resolved`
- `probe01_traced`, `probe01_alignment_uploaded`, `probe01_alignment_resolved`

### Drop (from current schema)

- All hard-coded dataset-path boolean columns written by `_check_datasets`: every column whose name starts with `alf/`, `raw_ephys_data/`, `raw_task_data_`, or `raw_video_data/`. Replaced by the audit columns above.
- All extended-QC video columns written by `_unpack_session_dict`: `videoBody_*`, `videoLeft_*`, `videoRight_*` (24 columns). The three per-camera QC values kept (dropped frames, timestamps, pin state) are pulled into the named video columns above; the rest are not used downstream and can be re-derived from Alyx if needed.
- `session_qc`: the Alyx aggregate session QC value is not kept.
- `new_recording`: the post-2025 start-time flag is not kept.

Net column delta: drop ~60, add 38, keep 21. Result: ~59 columns vs. current ~80.

## Integration with existing fetch logic

Current `fetch_sessions` in `psyfun/io.py`:

```
query Alyx for project+protocol
  → _unpack_session_dict (optional, default off)
  → _count_probes
  → split task_protocol into 'tasks'
  → _check_datasets               # list-based, REPLACE
  → _label_controls
  → new_recording flag            # REMOVE
  → _fetch_protocol_timings       # KEEP
  → _insert_LSD_admin_time        # KEEP
  → rank session_n
  → save parquet
```

Replacement plan:

1. Remove `_check_datasets` and the `qc_datasets` constant in `psyfun/config.py`.
2. Add `_audit_datasets(series, one)` in `psyfun/io.py` that returns the 38 new audit columns for one session. Reuse `_list_raw_task_collections` already present in `psyfun/io.py`; add helpers `_list_passive_raw_collections`, `_audit_task_alf`, `_audit_probe`, `_audit_camera`, `_audit_histology`. Pseudocode skeleton:

   ```python
   def _audit_datasets(series, one):
       eid = series['eid']
       out = {}
       # Task
       passives = _list_passive_raw_collections(eid, one)
       for slot in (0, 1):
           raw_col = passives[slot] if slot < len(passives) else None
           out.update(_audit_task_alf(eid, slot, raw_col, one))
       # Ephys
       insertions = sorted(
           one.alyx.rest('insertions', 'list', session=eid, no_cache=True),
           key=lambda ins: ins['name'],
       )
       for slot in (0, 1):
           ins = insertions[slot] if slot < len(insertions) else None
           out.update(_audit_probe(eid, slot, ins, one))
           out.update(_audit_histology_probe(eid, slot, ins, one))
       # Video
       for cam in ('left', 'right', 'body'):
           out.update(_audit_camera(eid, cam, one))
       # Session-level histology
       out['image_stacks'] = _check_image_stacks(series['subject'], series['lab'])
       for key, val in out.items():
           series[key] = val
       return series
   ```

3. Wire `_audit_datasets` into `fetch_sessions` in place of `_check_datasets` (apply per row via `df_sessions.progress_apply(_audit_datasets, one=one, axis='columns')`). Remove the `new_recording` flag line.
4. Drop the default-on `_unpack_session_dict` call from `fetch_sessions`. Keep the function in the module for ad-hoc use but stop populating `videoBody_*` / `videoLeft_*` / `videoRight_*` and `session_qc` in the default fetch path. The three per-camera QC values still needed (dropped frames, timestamps, pin state) are read inside `_audit_camera` instead.
5. Delete `scripts/audit_passive_extraction.py` and `metadata/passive_extraction_audit.csv` once the new sessions table is generated; the audit columns now live in `metadata/sessions.pqt`.

Each `_audit_*` helper returns a dict whose keys are the column names listed in the schema. Three-state cells use the string vocabulary `'extraction complete'`, `'extraction error'`, `'raw data missing'`. Histology cells use Python `bool`.

Performance: each session needs ~35 load attempts (3 task files × 2 tasks + 3 ephys × 2 probes + ~8 sorter/bombcell files for up to 2 sorters per probe + 2 video × 3 cameras ≈ 35). At ~50 ms per load for already-cached files and a few seconds for cache misses, the full audit over 53 sessions is bounded by network for first-time runs and dominated by file I/O on rerun. Run with `tqdm.pandas()` progress (already in the codebase).

## Migration

1. Land the spec and review.
2. Implement `_audit_datasets` and the trimmed `fetch_sessions`. Write `tests/test_audit_datasets.py` mirroring `tests/test_fetch_protocol_timings.py`. Tests should mock `one.load_dataset` to raise `ALFObjectNotFound` for chosen datasets and assert the resulting status strings; `one.alyx.rest` calls (insertions, sessions) should be similarly mocked.
3. Regenerate `metadata/sessions.pqt` from scratch by running `scripts/fetch_data.py` end-to-end.
4. Rename the timing columns: change `_fetch_protocol_timings` to write `task_pre` / `task_post` prefixes, update `TASKTIMINGS` in `psyfun/config.py`, and migrate every caller of `task00_*` / `task01_*` timing columns.
5. Grep the codebase for usages of dropped or renamed columns:
   ```bash
   grep -rn "alf/probe\|raw_ephys_data/\|raw_task_data_\|raw_video_data/\|videoBody_\|videoLeft_\|videoRight_\|task00_\|task01_" \
       psyfun/ scripts/ notebooks/
   ```
   Migrate or delete each caller.
6. Delete `psyfun.config.qc_datasets` and the now-orphan `_check_datasets`. Keep `_unpack_session_dict` available for ad-hoc use; remove its call site in `fetch_sessions`.
7. Delete `scripts/audit_passive_extraction.py` and `metadata/passive_extraction_audit.csv`.
8. Commit and update issue #1 to reference this spec as the source of the resolved audit.

## Out of scope

- The aggregated `data/bombcell.pqt` cached output (written by `scripts/unit_qc.py`). Bombcell presence is now in scope, but `probeNN_bombcell` checks for the per-probe bombcell output directory on disk under `one.eid2path(eid)`, not by reading `data/bombcell.pqt`.
- Re-running IBL extractions (issue #1).
- The fallback's correctness on individual sessions (issue #1, still unverifiable).
- The standalone `scripts/check_histology_status.py` and `metadata/histology_status.pqt`. The four histology columns added here are the same data joined into the session-level table; the standalone script and its detailed to-do-list output are kept for histology-pipeline triage.

## Affected files

- `psyfun/io.py`: replace `_check_datasets`, drop default `_unpack_session_dict` call, remove the `new_recording` flag line, rename `_fetch_protocol_timings` output prefixes to `task_pre` / `task_post`, add `_audit_datasets` and helpers (`_list_passive_raw_collections`, `_audit_task_alf`, `_audit_probe`, `_audit_camera`, `_audit_histology_probe`, `_check_image_stacks`, `_pick_latest_sorter`).
- `psyfun/config.py`: remove `qc_datasets`; update `TASKTIMINGS` to the `task_pre` / `task_post` column names.
- `psyfun/histology.py` (optional new module): destination for `insertion_picks`, `insertion_alignment_uploaded`, `insertion_alignment_resolved`, `list_histology_tifs` if moved out of `scripts/check_histology_status.py`. If kept in scripts, import from there.
- `tests/test_audit_datasets.py` (new): mock-based unit tests for each `_audit_*` helper.
- `scripts/audit_passive_extraction.py`: delete after `fetch_sessions` subsumes it.
- `metadata/sessions.pqt`: regenerate (separate commit).
- `metadata/passive_extraction_audit.csv`: delete.

## Open questions

None. The bombcell output layout, video QC key names, and the timing-column rename were all confirmed by inspecting a local ONE cache. No code may reference that cache location directly — the session directory is always resolved at run time via `one.eid2path(eid)`.
