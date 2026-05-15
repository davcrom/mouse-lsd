# Ticket: Rename camera columns to filename-style; split video-QC into 9 verbatim-key columns

Status: Done
Blocked by: 02

## Goal

Per camera (`left`, `right`, `body`), two file-status columns under
the raw-filename style:

- `<camera>Camera.raw` — `_iblrig_<camera>Camera.raw.mp4` in
  `raw_video_data`.
- `<camera>Camera.lightningPose` — `_ibl_<camera>Camera.lightningPose.pqt`
  in `alf`. DLC output is not accepted as pose.

Both two-state (`'present'` / `'missing'`).

Plus the nine video-QC columns under the verbatim Alyx
`extended_qc` keys (leading underscore retained):
`_videoLeft_dropped_frames`, `_videoLeft_timestamps`,
`_videoLeft_pin_state`, and the `Right`/`Body` counterparts. Value is
the outcome string from Alyx (first element of the list for
`_dropped_frames` and `_pin_state`; string directly for
`_timestamps`), with whitespace replaced by underscore (`'NOT SET'` →
`'NOT_SET'`). Empty string when the key is missing or `extended_qc`
is absent. No enumeration of allowed values.

## Touches

- `psyfun/io.py:457-489` — rewrite `_check_camera`:
  - Output two file-status keys named `f'{cam}Camera.raw'` and
    `f'{cam}Camera.lightningPose'` (no `_camera_` infix, no slot
    prefix; the camera name is in the filename).
  - For each of `('dropped_frames', 'timestamps', 'pin_state')`,
    emit the column under key `f'_video{cam.capitalize()}_{suffix}'`
    (verbatim Alyx key). Value comes from `_qc_outcome`, then is
    normalised with `re.sub(r'\s+', '_', outcome)` so `'NOT SET'`
    becomes `'NOT_SET'`. Empty string when the key is absent.
- `psyfun/io.py:457-466` — keep `_qc_outcome` (returns `''` when
  absent; lists return first element). Apply whitespace normalisation
  in the caller, not inside `_qc_outcome`.
- `tests/test_check_datasets.py:216-235` — rewrite the camera tests
  for the new column names, two-state vocabulary, verbatim QC keys,
  and whitespace normalisation:
  - `leftCamera.raw == 'present'` with the matching raw mp4.
  - `leftCamera.lightningPose == 'missing'` with no alf entry.
  - `_videoLeft_dropped_frames == 'PASS'` from list-typed entry.
  - `_videoLeft_timestamps == 'WARNING'` from string-typed entry.
  - `_videoLeft_pin_state == ''` when absent.
  - **new** `_videoLeft_timestamps == 'NOT_SET'` when Alyx wrote
    `'NOT SET'` (whitespace normalisation).
  - The no-camera case has both file-status columns `'missing'` and
    QC columns `''`.

## Approach

The current implementation already reads list-vs-string QC entries via
`_qc_outcome`; the changes are pure naming + whitespace normalisation.
`re.sub` already used elsewhere in the module; no new import needed.

## Acceptance

- `pytest tests/test_check_datasets.py -k "camera"` passes.
- All twelve new column names (3 cameras × 2 file + 3 cameras × 3 QC =
  15 — wait, that's 6 file + 9 QC = 15 total) appear with the spec's
  exact names. A simple smoke: build `_check_camera` outputs for
  `left`, `right`, `body` and assert the union of keys equals the
  15-column spec.
