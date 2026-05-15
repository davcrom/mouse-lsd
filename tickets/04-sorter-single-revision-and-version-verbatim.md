# Ticket: Sorter selection — single-revision passthrough, version string verbatim

Status: Done
Blocked by: 03

## Goal

`_pick_sorter` selects the registered sorter via the existing
`SPIKE_SORTERS` priority order, but the `default_revision == 'True'`
filter is only applied when more than one `spikes.times.npy` entry
exists for the chosen sorter. With a single entry, that entry is
taken regardless of its `default_revision` value, so a non-default-flagged
sole revision is not treated as "no sorting". `probeNN_sorter` is the
sorter's registered `version` string verbatim (e.g.
`pykilosort_ibl_1.4.1`); empty string when no sorter or no version.
The `_format_sorter` helper and its parenthesised
`name#revision# (version)` output are removed.

## Touches

- `psyfun/io.py:366-399` — rewrite `_pick_sorter`:
  - Return shape: 2-tuple `(sorter_name, version)`. The sorter name is
    still needed by `_check_probe` for the `alf/<probe>/<sorter>`
    collection key and the on-disk bombcell path; `revision` is no
    longer written to the output schema and is dropped from the return.
  - For each `sorter` in `SPIKE_SORTERS`, gather entries with
    `name == 'spikes.times.npy'` and `collection == f'alf/{probe}/{sorter}'`.
    If `len(entries) == 1`, take it regardless of its
    `default_revision` value. If `> 1`, take the first whose
    `default_revision == 'True'`; fall back to `entries[0]` if none
    matches. Return `(sorter, entry.get('version') or '')`.
  - Returns `('', '')` when no sorter has any registered entry.
- `psyfun/io.py:390-399` — delete `_format_sorter`.
- `psyfun/io.py:419-434` — `_check_probe` unpacks the new 2-tuple;
  `probeNN_sorter` is the version string directly.
- `tests/test_check_datasets.py:107-143` — rewrite `_pick_sorter` /
  `_format_sorter` tests:
  - `test_pick_sorter_priority_order` — expects `version` string of the
    higher-priority sorter.
  - `test_pick_sorter_none_registered` — expects `''`.
  - `test_pick_sorter_prefers_default_revision` — multi-revision case;
    expects the `'True'` entry's `version`.
  - **new** `test_pick_sorter_single_non_default_revision` — one entry
    with `default_revision='False'`; expects that entry's `version`.
  - Remove `test_format_sorter_variants` and any assertion on
    `_format_sorter` output.

## Approach

The only behavioural change is the single-revision rule; the
priority-order walk is unchanged. Drop the now-unused `revision` field
from `_pick_sorter`'s return because the verbatim version string
already disambiguates and the spec does not place `revision` anywhere
in the output schema.

## Acceptance

- `pytest tests/test_check_datasets.py -k "pick_sorter or check_probe"`
  passes.
- For a registry with one `pykilosort` entry, `default_revision='False'`,
  `version='pykilosort_ibl_1.4.1'`, `_check_probe` writes
  `probeNN_sorter == 'pykilosort_ibl_1.4.1'`.
- `grep -n "_format_sorter\| (\(pykilosort\|iblsorter\|ks2\)" psyfun/ tests/`
  returns no hits.
