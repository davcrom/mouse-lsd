# Spec: registry-based dataset-extraction check in `sessions.pqt`

## Status

Implemented (2026-05-14)

Supersedes the load-based version of this spec, which was implemented in
commits `e808962`..`7b8b3e0` (function `_audit_datasets` and helpers in
`psyfun/io.py`, tests in `tests/test_audit_datasets.py`). This revision
replaces the load-based check with a registry-based one; the columns in
`sessions.pqt` are almost unchanged, but the mechanism and the
`probe00_sorter` / `probe01_sorter` contents change.

## Problem

`sessions.pqt` needs one set of columns that says, per session, which
datasets IBL has extracted — task alf, spike sorting, raw ephys, video
pose — so the table is a single source of truth for what data exists.

The implemented version checks each dataset by calling `one.load_dataset`
and catching `ALFObjectNotFound`. That works but is expensive: ~35 load
attempts per session, and checking raw video means downloading the full
`.mp4` files on a cache miss (so raw-video checking was made opt-in via a
`check_raw_video` flag). It also does not catch the failure mode that
actually matters for this project — IBL's extractor producing a
structurally valid but semantically wrong file (see
`_fpga_timings_from_alf` in `psyfun/io.py`, which works around a session
where the extractor merged the LSD-filler block into `task_00`); a clean
`load_dataset` says nothing about that.

The Alyx `sessions/read` response carries a `data_dataset_session_related`
field: the full list of datasets registered for the session, with a
`data_url` per entry. `_audit_datasets` already calls `sessions/read`
(for `extended_qc`) and discards this field. Checking dataset presence
against that list — `data_url` non-null means the file is registered on
the public data server — replaces all ~35 loads with filters over data
already fetched, removes the raw-video download problem entirely, and is
strictly more informative than the old `list_datasets` membership check
this project's `_check_datasets` once used (the registry entry also
carries the sorter `version` and `revision`).

What this check does not catch: post-registration file drift (a
truncated or hash-mismatched file on the server whose registry record
still says it exists). That is rare, is an infrastructure problem fixed
by re-syncing from the data server, and is out of scope (see Decisions).

## Inputs

### Per session, from Alyx

- `one.alyx.rest('sessions', 'read', id=eid)` — one call per session.
  Used for two fields:
  - `extended_qc` : dict or None. Source of the three per-camera video-QC
    values (unchanged from the implemented version).
  - `data_dataset_session_related` : list of dicts, one per registered
    dataset. Verified against session `8dfd9963-25e5-4f63-8f91-5b27a5852628`
    (145 entries). Each dict has these keys:

    | key | type | notes |
    |---|---|---|
    | `name` | str | e.g. `spikes.times.npy` |
    | `collection` | str | e.g. `alf/probe01/pykilosort`, `raw_ephys_data/probe00`, `alf/task_00`, `raw_video_data`, `alf` |
    | `dataset_type` | str | e.g. `spikes.times` |
    | `data_url` | str or None | public-server URL of the file; non-null for all 145 entries in the verified session, including raw `.mp4` and `.cbin` |
    | `url` | str | Alyx dataset record URL |
    | `id` | str | dataset UUID |
    | `hash` | str | md5 |
    | `file_size` | int | bytes |
    | `version` | str | extractor/sorter version, e.g. `pykilosort_ibl_1.4.1`, `2.21.2`, `iblvideo_2.2.0` |
    | `revision` | str | revision tag; `''` for every dataset in this project today |
    | `default_revision` | str | the string `'True'` or `'False'` (not a bool) |
    | `qc` | str | per-dataset QC; `NOT_SET` for every entry in the verified session |

- `one.alyx.rest('insertions', 'list', session=eid, no_cache=True)` — one
  call per session. List of insertion records; each has `name`
  (`probe00` / `probe01`) and `id` (the pid). Used to enumerate probe
  slots. Unchanged from the implemented version.

- `one.alyx.rest('insertions', 'list', id=pid, no_cache=True)[0]` — one
  call per probe. Full insertion record with `json`. Source of the four
  histology columns. Unchanged from the implemented version.

### Per session, from a settings file

- `_iblrig_taskSettings.raw.json` in each `raw_task_data_NN` collection.
  Read by `_load_passive_run` (already in `psyfun/io.py`) to classify a
  run as `passive` vs `spontaneous`-filler via `PYBPOD_PROTOCOL`. This is
  the only file content the check reads — a small settings JSON, not bulk
  data — and `_fetch_protocol_timings` already reads it in the same
  `fetch_sessions` pass (see Decisions).

### Local filesystem

- Bombcell output directory: `one.eid2path(eid) / 'spike_sorters' / <sorter> / <probe> / 'bombcell'`.
  Checked for the file `templates._bc_qMetrics.parquet`. Bombcell is not
  an Alyx-registered dataset — confirmed: the only `spike_sorters/...`
  entry in the verified session's registry is `_kilosort_raw.output.tar`,
  not bombcell output. `one.eid2path` resolves whatever ONE cache
  directory is configured on the machine running the check; no cache path
  is hard-coded.

### Histology image stacks, from the data server

- `psyfun.histology.list_histology_tifs(subject, lab, par)` — HTTP
  directory listing. Unchanged from the implemented version.

## Outputs

The check adds 38 columns to each session row. The column names and the
three-state vocabulary are unchanged from the implemented version; only
`probe00_sorter` / `probe01_sorter` contents change (see Behavior →
Ephys).

Three-state cells use the strings `'extraction complete'`,
`'extraction error'`, `'raw data missing'`. Histology cells use Python
`bool`. The sorter cells and the three per-camera video-QC cells are
strings (possibly empty).

### Task (6 columns)

`task_pre_intervalsTable`, `task_pre_passiveStims`, `task_pre_passiveGabor`,
`task_post_intervalsTable`, `task_post_passiveStims`, `task_post_passiveGabor`

### Ephys (10 columns)

`probe00_raw_ap`, `probe00_sync`, `probe00_sorter`, `probe00_spikes`, `probe00_bombcell`,
`probe01_raw_ap`, `probe01_sync`, `probe01_sorter`, `probe01_spikes`, `probe01_bombcell`

### Video (15 columns)

`left_camera_raw_video`, `left_camera_pose`, `left_camera_dropped_frames`, `left_camera_timestamps`, `left_camera_pin_state`,
`right_camera_raw_video`, `right_camera_pose`, `right_camera_dropped_frames`, `right_camera_timestamps`, `right_camera_pin_state`,
`body_camera_raw_video`, `body_camera_pose`, `body_camera_dropped_frames`, `body_camera_timestamps`, `body_camera_pin_state`

### Histology (7 columns)

`image_stacks`,
`probe00_traced`, `probe00_alignment_uploaded`, `probe00_alignment_resolved`,
`probe01_traced`, `probe01_alignment_uploaded`, `probe01_alignment_resolved`

## Behavior

### Presence rule

Build, once per session, the set of registered datasets from
`data_dataset_session_related`:

```python
present = {
    (d['collection'], d['name'])
    for d in data_dataset_session_related
    if d['data_url']
}
```

A dataset "is present" if its `(collection, name)` pair is in `present`.
Also keep the full entries keyed by `(collection, name)` so the ephys
check can read `version` / `revision` / `default_revision`.

Three-state status for an extracted dataset, given whether it is present
and whether its raw prerequisite is present:

- present → `'extraction complete'`
- absent, raw prerequisite present → `'extraction error'`
- absent, raw prerequisite absent → `'raw data missing'`

Raw datasets (`raw_ap`, `raw_video`) have no upstream prerequisite: they
are `'extraction complete'` if present, `'raw data missing'` if not.

### Task (passive protocol)

Two slots, `task_pre` and `task_post`, named by passive-run order — not
by `raw_task_data_NN` number, because a spontaneous-only LSD-filler run
can sit between the two passive runs (e.g. `00` passive, `01` filler,
`02` passive).

Enumerate the passive slots with `_list_passive_raw_collections(eid, one)`:
derive the `raw_task_data_NN` collection names from
`data_dataset_session_related` (collections matching `^raw_task_data_\d+$`,
sorted by `NN`), then classify each with `_load_passive_run`, keeping the
runs tagged `kind='passive'` in run order. Element 0 is the `task_pre`
slot, element 1 is `task_post`.

For each passive slot, the raw collection (e.g. `raw_task_data_02`) maps
to its alf collection by `raw_col.replace('raw_task_data_', 'alf/task_')`.
Check three files in that alf collection:

| column suffix | dataset name |
|---|---|
| `intervalsTable` | `_ibl_passivePeriods.intervalsTable.csv` |
| `passiveStims` | `_ibl_passiveStims.table.csv` |
| `passiveGabor` | `_ibl_passiveGabor.table.csv` |

Raw prerequisite: the slot's `raw_task_data_NN` collection exists (it
does, since the slot was enumerated from it). So each task cell is
`'extraction complete'` if the alf dataset is present, else
`'extraction error'`. When a slot has no passive run (element missing
from `_list_passive_raw_collections`), all three of its columns are
`'raw data missing'`.

### Ephys (per probe, slots `probe00`, `probe01`)

Enumerate probe slots from `one.alyx.rest('insertions', 'list', session=eid)`,
sorted by `name`; slot 0 / 1 → column prefix `probe00` / `probe01`. When
a slot has no insertion, write `raw_ap = sync = spikes = bombcell = 'raw data missing'`
and `sorter = ''`.

For a probe with insertion record `ins` (collection base
`raw_ephys_data/<ins['name']>`):

- `probeNN_raw_ap` : present if `_spikeglx_ephysData_g0_t0.imec<N>.ap.cbin`
  or `_spikeglx_ephysData_g0_t0.imec<NN>.ap.cbin` is in that collection
  (IBL uses both the single- and double-digit `imec` forms; the verified
  session uses single-digit). `'extraction complete'` / `'raw data missing'`.
- `probeNN_sync` : same two-form check for `...imec<N>.sync.npy` /
  `...imec<NN>.sync.npy`. Three-state, raw prerequisite = `raw_ap` present.
- `probeNN_sorter`, `probeNN_spikes` : choose a sorter and revision (see
  below), then check the three spike files
  (`spikes.times.npy`, `spikes.clusters.npy`, `clusters.uuids.csv`) in
  `alf/<probe>/<sorter>`. `probeNN_spikes` is three-state with raw
  prerequisite = `raw_ap` present (present only if all three files are
  present).
- `probeNN_bombcell` : three-state by local file presence of
  `templates._bc_qMetrics.parquet` under
  `one.eid2path(eid) / 'spike_sorters' / <sorter> / <probe> / 'bombcell'`,
  raw prerequisite = `raw_ap` present. Bombcell can be absent even when
  `probeNN_spikes` is `'extraction complete'`.

**Choosing the sorter and revision:**

1. For each sorter in `('iblsorter', 'pykilosort', 'ks2')` (priority
   order), look for `data_dataset_session_related` entries with
   `name == 'spikes.times.npy'` and `collection == f'alf/{probe}/{sorter}'`.
   Take the first sorter in that order that has any such entry.
2. Among that sorter's `spikes.times.npy` entries, pick the one whose
   `default_revision` is `'True'` (string compare; the field is
   serialized as a string). Its `revision` is the revision tag; its
   `version` is the sorter version string.
3. `probeNN_sorter` = `'<sorter>'`, plus `'#<revision>#'` when `revision`
   is non-empty, plus `' (<version>)'` when `version` is non-empty.
   Example today (empty revision): `'pykilosort (pykilosort_ibl_1.4.1)'`.
   Example with a future re-sorting:
   `'iblsorter#2025-06-01# (iblsorter_1.9.0)'`.
4. If no sorter has a registered `spikes.times.npy`, `probeNN_sorter = ''`,
   `probeNN_spikes` is three-state by the raw prerequisite alone
   (`'extraction error'` if `raw_ap` present, else `'raw data missing'`),
   and `probeNN_bombcell` uses `<sorter> = ''` for its path (the bombcell
   file will not be found, so the cell follows the same three-state rule).

### Video (per camera, slots `left`, `right`, `body`)

Five columns per camera. Two are registry-based status cells; three carry
video-QC outcomes from `extended_qc`.

- `<C>_camera_raw_video` : present if `_iblrig_<C>Camera.raw.mp4` is in
  `raw_video_data`. `'extraction complete'` / `'raw data missing'`.
  Always checked — registry membership is free, so there is no
  `check_raw_video` flag.
- `<C>_camera_pose` : three-state for `_ibl_<C>Camera.lightningPose.pqt`
  in `alf`, raw prerequisite = `raw_video` present. DLC output is
  intentionally not accepted as pose tracking; a camera with only DLC
  counts as `'extraction error'`. No session has lightning pose today.
- `<C>_camera_dropped_frames`, `<C>_camera_timestamps`,
  `<C>_camera_pin_state` : the QC outcome string from `extended_qc` keys
  `_video<Cam>_dropped_frames`, `_video<Cam>_timestamps`,
  `_video<Cam>_pin_state` (`<Cam>` = camera name capitalised). The
  `dropped_frames` and `pin_state` keys hold a list whose first element
  is the outcome; `timestamps` holds the outcome string directly. Empty
  string when `extended_qc` is absent or has no entry for the key.
  Unchanged from the implemented version.

### Histology (from Alyx insertion records)

Unchanged from the implemented version.

- `image_stacks` : session-level; `True` when
  `psyfun.histology.list_histology_tifs(subject, lab, par)` returns >= 2
  entries. Cached per `(subject, lab)` across the run.
- `probeNN_traced`, `probeNN_alignment_uploaded`,
  `probeNN_alignment_resolved` : per probe, from
  `one.alyx.rest('insertions', 'list', id=pid, no_cache=True)[0]` via
  `psyfun.histology.insertion_picks`, `insertion_alignment_uploaded`,
  `insertion_alignment_resolved`. `False` for a probe slot with no
  insertion.

### Integration with `fetch_sessions`

Current `fetch_sessions` pipeline (after the implemented load-based
version):

```
query Alyx for project+protocol
  → _count_probes
  → split task_protocol into 'tasks'
  → _audit_datasets               # load-based, REPLACE
  → _label_controls
  → _fetch_protocol_timings       # KEEP
  → _insert_LSD_admin_time        # KEEP
  → rank session_n
  → save parquet
```

Replacement:

1. Rename `_audit_datasets` → `_check_datasets` and its helpers
   (`_audit_task_alf` → `_check_task_alf`, `_audit_probe` → `_check_probe`,
   `_audit_camera` → `_check_camera`, `_audit_histology_probe` →
   `_check_histology_probe`; `_check_image_stacks` keeps its name). The
   word "audit" is replaced by "check" throughout the module, the spec,
   and the test file.
2. Rewrite the renamed helpers to take the per-session `present` set (and
   the keyed entries for the ephys version/revision lookup) instead of
   calling `one.load_dataset`. `_check_datasets` calls `sessions/read`
   once and uses both `extended_qc` and `data_dataset_session_related`.
3. Delete `_pick_latest_sorter` (the sorter and revision now come from
   `data_dataset_session_related`), `_loads`, and the `check_raw_video`
   parameter on `_check_datasets` and `fetch_sessions`.
4. Change `_list_passive_raw_collections` to derive the
   `raw_task_data_NN` collection names from
   `data_dataset_session_related` rather than `one.list_datasets`.
5. `fetch_sessions` calls `_check_datasets` in place of `_audit_datasets`;
   the per-row `progress_apply` no longer passes `check_raw_video`.

## Out of scope

- Post-registration file drift (truncated / hash-mismatched files on the
  server). The check trusts a non-null `data_url`; it does not download
  or re-hash. See Decisions.
- Semantic correctness of extracted files (e.g. an extractor merging two
  protocol blocks). Neither this check nor the load-based one detects it.
- Re-running IBL extractions (issue #1).
- The aggregated `data/bombcell.pqt` output of `scripts/unit_qc.py`.
  `probeNN_bombcell` checks the per-probe bombcell directory on disk.
- The standalone `scripts/check_histology_status.py` and
  `metadata/histology_status.pqt`.
- Migrating the downstream script callers `scripts/single_unit.py` and
  `scripts/population_dimensionality.py` (a separate planned refactor;
  `scripts/dataset_overview.py` and `scripts/fetch_data.py` are already
  migrated). Notebooks are legacy and left untouched.

## Decisions

- **Registry check replaces load check.** Source: user, this
  conversation. `_audit_datasets` already fetches `sessions/read`; the
  `data_dataset_session_related` field replaces ~35 `load_dataset`
  attempts per session with filters over data already in hand, and
  removes the raw-video download cost that forced `check_raw_video` to be
  opt-in.
- **`data_url` non-null is the presence test**, not bare row presence.
  Verified: all 145 entries in session `8dfd9963…` have non-null
  `data_url`, including raw `.mp4` and `.cbin`. This is stronger than the
  old `list_datasets` membership check (which the original spec rejected)
  because it reflects a file record on the public data server, not just a
  `Dataset` row.
- **Post-registration drift is accepted as a gap.** Source: user, this
  conversation. It is rare, is an infrastructure problem (re-sync from the
  data server), and a clean `load_dataset` did not catch the semantic
  errors that actually affected this project, so the validation it
  uniquely provided was low-value here.
- **The check still reads `_iblrig_taskSettings.raw.json`.** The registry
  exposes the file inventory, not file contents, so passive-vs-filler
  classification still needs `PYBPOD_PROTOCOL`. This is a small settings
  JSON, not bulk data, and `_fetch_protocol_timings` already reads it for
  every `raw_task_data_NN` in the same `fetch_sessions` pass, so the
  check adds no new bulk downloads. Reusing `_load_passive_run` keeps one
  classification path (DRY) rather than inventing a registry-only
  heuristic.
- **Sorter priority `iblsorter` > `pykilosort` > `ks2`.** The entry dict
  has no `date_created`, so the implemented version's "most recent
  `date_created`" rule cannot be applied without the extra
  `datasets/list` query that this revision removes. A fixed priority
  order makes the choice deterministic; `iblsorter` is IBL's current
  sorter. Most sessions have only one sorter, so this rarely matters.
- **Within a sorter, take the `default_revision == 'True'` entry.**
  Source: user, this conversation.
- **`probeNN_sorter` carries name + revision + version.** Source: user
  ("keep as much information about the spike sorter as possible"). The
  `version` string (e.g. `pykilosort_ibl_1.4.1`) is free in the registry
  entry; format is `'<sorter>'` + `'#<revision>#'` (when revision
  non-empty) + `' (<version>)'` (when version non-empty).
- **Spec file renamed** `audit_dataset_extraction.md` →
  `check_dataset_extraction.md` as part of the audit→check terminology
  pass. The `psyfun/io.py` docstring reference to the old path must be
  updated during implementation.
- `data_url` host after IBL's S3 migration. Every `data_url` in the
  verified session points at `ibl.flatironinstitute.org`. IBL is
  migrating datasets to AWS S3; if `data_url` is later served from S3, or
  becomes null for S3-only datasets, the non-null presence test still
  holds but any flatiron-specific assumption does not. The user is
  gathering details on the migration and expects a small URL-level fix,
  not a redesign — recorded here so it is not lost.
