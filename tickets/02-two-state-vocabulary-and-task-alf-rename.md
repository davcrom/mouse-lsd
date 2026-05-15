# Ticket: Two-state file vocabulary; task-alf columns become `<slot>_<shorthand>`

Status: Done
Blocked by: 01

## Goal

Every file-status cell is one of two strings, `'present'` or `'missing'`
— no `'extraction complete'` / `'extraction error'` / `'raw data missing'`
anywhere in the produced `sessions.pqt`. For the passive task alf
datasets, the six columns are renamed to
`pre_intervalsTable`, `pre_passiveStims`, `pre_passiveGabor`,
`post_intervalsTable`, `post_passiveStims`, `post_passiveGabor`.

## Touches

- `psyfun/io.py:31-48` — drop the three-state constants
  (`EXTRACTION_COMPLETE`, `EXTRACTION_ERROR`, `RAW_DATA_MISSING`);
  add module-level `PRESENT = 'present'`, `MISSING = 'missing'`. Keep
  `TASK_ALF_FILES`, `SPIKE_SORTERS`, `BOMBCELL_OUTPUT_FILE`. Drop
  `SPIKE_ALF_FILES` (replaced in a later ticket).
- `psyfun/io.py:311-315` — delete `_three_state`; replace its callers
  with a direct ternary `PRESENT if x else MISSING`.
- `psyfun/io.py:344-356` — `_check_task_alf` writes the six renamed
  columns, dropping the `task_` prefix. When the slot has no passive
  run, all three of its columns are `MISSING` (not `RAW_DATA_MISSING`).
- `tests/test_check_datasets.py:87-103` — the `_check_task_alf` tests
  use the new column names and the two-state vocabulary.

## Approach

Slot prefix comes from `PASSIVE_SLOTS` (already updated in ticket 01).
`_check_task_alf` builds `prefix = PASSIVE_SLOTS[slot]`; the column
keys become `f'{prefix}_{short}'` and produce e.g. `pre_intervalsTable`.
Other check helpers continue to emit three-state strings *for now* —
they are rewritten in later tickets. Keep `io.EXTRACTION_*` removed so
nothing in-tree still references the old vocabulary; if any later
ticket still depends on it, that is a discovery to surface.

## Acceptance

- `pytest tests/test_check_datasets.py -k "task_alf"` passes; assertions
  read columns `pre_intervalsTable` / `post_intervalsTable` etc. and
  values `io.PRESENT` / `io.MISSING`.
- `python -c "from psyfun import io; assert io.PRESENT == 'present' and io.MISSING == 'missing'"` succeeds.
- `grep -n "extraction complete\|extraction error\|raw data missing\|EXTRACTION_COMPLETE\|EXTRACTION_ERROR\|RAW_DATA_MISSING" psyfun/ tests/` returns no hits (consumers in `scripts/` are migrated by ticket 09).
