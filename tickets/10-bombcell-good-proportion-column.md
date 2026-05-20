# Ticket: `probeNN_bombcell_GOOD` proportion column

Spec: check_session_datasets
Status: Done
Blocked by: none

## Goal

The per-probe ephys column reports the fraction of bombcell `GOOD` units
as a float instead of a present/missing flag. `_check_probe` emits
`probe{NN}_bombcell_GOOD` = `(bc_unitType == 'GOOD').mean()` read from the
cached qMetrics parquet, `NaN` when the parquet is absent or the slot has
no insertion.

## Touches

- `psyfun/io.py` — `_check_probe` (currently io.py:389-425).
  - No-insertion branch: replace `f'{prefix}_bombcell': MISSING` with
    `f'{prefix}_bombcell_GOOD': np.nan`.
  - Insertion branch: build `bombcell_path` as today
    (`session_path / 'spike_sorters' / sorter / probe / 'bombcell' /
    BOMBCELL_OUTPUT_FILE`). If the file does not exist, value is
    `np.nan`. If it exists, `pd.read_parquet(bombcell_path)` and value is
    `(df['bc_unitType'] == 'GOOD').mean()`. Emit under key
    `f'{prefix}_bombcell_GOOD'`.
- `tests/test_check_datasets.py` — `_check_probe` tests
  (lines ~169, 185, 196, 206).

## Approach

- `np` is already imported in `psyfun/io.py` (io.py:4); `pd` too.
- `BOMBCELL_OUTPUT_FILE = 'templates._bc_qMetrics.parquet'` (io.py:51) is
  unchanged; reuse it for the path.
- The qMetrics parquet already carries the `bc_unitType` column (verified
  against real bombcell output) — no `bombcell` import, just
  `pd.read_parquet`.
- Denominator is all rows including empty-string-label rows; `.mean()` of
  the boolean mask gives exactly that.
- In the test fixture, the complete-probe case currently does
  `bombcell.write_text("x")` (test line ~152) — that is not a parquet.
  Replace with writing a real parquet:
  `pd.DataFrame({'bc_unitType': [...]}).to_parquet(bombcell_path)` with a
  known mix (e.g. 1 GOOD, 1 MUA, 1 NOISE → 1/3). Assert the float; use
  `math.isnan` for the `NaN` cases (file-absent and no-insertion).

## Acceptance

- `pytest tests/test_check_datasets.py -k "check_probe"` passes.
- Complete probe with a parquet of known labels →
  `probe00_bombcell_GOOD` equals the known GOOD fraction.
- Probe with insertion but no bombcell file → `probe00_bombcell_GOOD` is
  `NaN`.
- Slot with no insertion → `probe01_bombcell_GOOD` is `NaN`.
- No `probe00_bombcell` / `probe01_bombcell` key is emitted.
