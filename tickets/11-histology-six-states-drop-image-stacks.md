# Ticket: histology six-state ladder, drop `image_stacks` column

Spec: check_session_datasets
Status: Done
Blocked by: none

## Goal

`probe{NN}_histology` carries six states gated by image-stack presence,
and the session-level `image_stacks` column is removed.
`_check_histology_probe` gains a `stacks_present` argument; `_check_datasets`
computes stack presence once per session and stops emitting `image_stacks`.

## Touches

- `psyfun/io.py`
  - `_check_histology_probe` (io.py:428-446): add `stacks_present: bool`
    parameter. Evaluation, first match wins:
    1. `ins is None` → `'no-insertion'`
    2. `insertion_alignment_resolved(full)` → `'resolved'`
    3. `insertion_alignment_uploaded(full)` → `'aligned'`
    4. `insertion_picks(full)` → `'traced'`
    5. `stacks_present` → `'no-tracing'`
    6. else → `'no-stacks'`
    Fetch `full` only when `ins is not None` (unchanged).
  - `_check_datasets` (io.py:492-529): compute
    `stacks_present = _check_image_stacks(series['subject'], lab) == PRESENT`
    once; pass it into both `_check_histology_probe(eid, slot, ins, one,
    stacks_present)` calls; remove the
    `out['image_stacks'] = _check_image_stacks(...)` line (io.py:526).
- `tests/test_check_datasets.py`
  - `_check_histology_probe` tests (lines ~279-321): new signature; rename
    no-insertion expectation `'missing'` → `'no-insertion'`; add a
    `'no-stacks'` case and make the `'no-tracing'` case pass
    `stacks_present=True`.
  - `_check_datasets` test (line ~326): drop any assertion on an
    `image_stacks` output key (the `_check_image_stacks` monkeypatch at
    line ~349 stays — it now feeds histology gating).
  - `_check_image_stacks` unit tests (lines ~365-377): unchanged.

## Approach

- `_check_image_stacks` (io.py:483-489) is unchanged: RD+GR rule, returns
  `PRESENT`/`MISSING`, `lru_cache`d per `(subject, lab)`. Compare its
  result to `PRESENT` to get the bool.
- Predicates `insertion_picks`, `insertion_alignment_uploaded`,
  `insertion_alignment_resolved` are already imported (io.py:21).
- Highest-reached-wins is an if/elif chain; `ins is None` short-circuits
  first so `'no-insertion'` supersedes `'no-stacks'`.
- Ephys column count per probe stays at 8 (`bombcell` slot now
  `bombcell_GOOD` from ticket 10 — independent; this ticket only touches
  histology and the dropped session column).

## Acceptance

- `pytest tests/test_check_datasets.py -k "histology_probe or image_stacks or check_datasets"`
  passes.
- All six states reachable:
  - `ins=None` → `'no-insertion'`.
  - insertion, no picks, `stacks_present=False` → `'no-stacks'`.
  - insertion, no picks, `stacks_present=True` → `'no-tracing'`.
  - insertion with picks only → `'traced'`.
  - `alignment_count > 0`, not resolved → `'aligned'`.
  - `alignment_resolved` → `'resolved'`.
- `_check_datasets` output has no `image_stacks` key.
