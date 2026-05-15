# Ticket: Rename `PASSIVE_SLOTS` to `('pre','post')` and `TASKTIMINGS` to `pre_*`/`post_*`

Status: Done
Blocked by: none

## Goal

`sessions.pqt` carries timing columns under the new shorthand
(`pre_spontaneous_start` … `post_replay_stop`, with `LSD_admin`
unchanged), and `PASSIVE_SLOTS == ('pre', 'post')`. `_fetch_protocol_timings`
writes the renamed columns; `load_sessions` filters on the renamed list.

## Touches

- `psyfun/config.py:43-53` — `TASKTIMINGS` element-wise rename
  (`task_pre_*` → `pre_*`, `task_post_*` → `post_*`); `PASSIVE_SLOTS`
  becomes `('pre', 'post')`.
- `psyfun/io.py:249-286` — `_fetch_protocol_timings` continues to use
  `TASKTIMINGS` / `PASSIVE_SLOTS`; behaviour unchanged, column names
  follow the rename automatically because slot prefixes are derived
  from `PASSIVE_SLOTS`.
- `psyfun/io.py:769-803` — `load_sessions`: default `drop_if_nan` and
  `columns_to_keep` continue to import `TASKTIMINGS`; no code change
  required beyond the import.
- `tests/test_fetch_protocol_timings.py` — every `task_pre_*` /
  `task_post_*` assertion renamed.

## Approach

Pure rename. `PASSIVE_SLOTS` is the only place the slot prefix is
generated for timings (`io.py:279`) and `TASKTIMINGS` is the
write-then-set list (`io.py:264`). Updating the two constants
propagates through both call sites. Tests provide the acceptance
check.

## Acceptance

- `pytest tests/test_fetch_protocol_timings.py` passes with assertions
  reading `pre_spontaneous_start`, `post_replay_stop`, etc.
- `from psyfun.config import PASSIVE_SLOTS; assert PASSIVE_SLOTS == ('pre','post')`.
- `from psyfun.config import TASKTIMINGS` lists 13 entries, all starting
  with `pre_`/`post_` or equal to `LSD_admin`.
