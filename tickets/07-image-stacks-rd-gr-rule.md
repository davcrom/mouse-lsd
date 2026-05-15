# Ticket: `image_stacks` requires both `*_RD.tif` AND `*_GR.tif`

Status: Done
Blocked by: 02

## Goal

`image_stacks` is a per-subject column with value `'present'` iff
`list_histology_tifs(subject, lab, par)` returns at least one filename
matching `*_RD.tif` AND at least one matching `*_GR.tif`; otherwise
`'missing'`. Replaces the previous boolean derived from a `>= 2`
threshold over all tif filenames.

## Touches

- `psyfun/io.py:492-495` — rewrite `_check_image_stacks`:
  - Keep the `@lru_cache` decorator and `(subject, lab)` cache key.
  - Fetch the filename list once; check
    `any(fname.endswith('_RD.tif') for fname in tifs)` AND
    `any(fname.endswith('_GR.tif') for fname in tifs)`.
  - Return the string `'present'` or `'missing'`, not a bool. (The
    caller already writes the value verbatim into the dataframe;
    no further change needed there.)
- `tests/test_check_datasets.py` — add tests for the three cases:
  - Both `*_RD.tif` and `*_GR.tif` present → `'present'`.
  - Only one of the two suffixes present → `'missing'`.
  - Empty filename list → `'missing'`.
  Use `monkeypatch.setattr(io, 'list_histology_tifs', ...)` to return
  canned filename lists; clear the lru cache between cases with
  `io._check_image_stacks.cache_clear()`.

## Approach

`psyfun.histology.list_histology_tifs` returns the list of `.tif`
filenames (e.g. `'ZFM-08631_AP_RD.tif'`). The suffix check is a simple
string predicate over that list; no parsing is required.

## Acceptance

- `pytest tests/test_check_datasets.py -k "image_stacks"` passes.
- A canned `list_histology_tifs` returning
  `['x_RD.tif', 'y_GR.tif']` produces `'present'`; returning
  `['x_RD.tif', 'y_RD.tif']` produces `'missing'`.
