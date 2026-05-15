# Ticket: Replace per-probe histology booleans with `probeNN_histology` ordered state

Status: In progress
Blocked by: 03

## Goal

Per probe slot, a single column `probeNN_histology` carries one of
five strings: `'resolved'`, `'aligned'`, `'traced'`, `'no-tracing'`,
`'missing'`. The previous three boolean columns
(`probeNN_traced`, `probeNN_alignment_uploaded`,
`probeNN_alignment_resolved`) are removed. Evaluation order, highest
reached wins: `resolved` > `aligned` > `traced` > `no-tracing` >
`missing`. `'missing'` is used iff the probe slot has no insertion.

## Touches

- `psyfun/io.py:440-454` — rewrite `_check_histology_probe`:
  - When `ins is None`, return `{f'probe{slot:02d}_histology': 'missing'}`.
  - Otherwise call `one.alyx.rest('insertions', 'list', id=ins['id'], no_cache=True)[0]` (unchanged) and walk the
    three predicates in the order
    `insertion_alignment_resolved` → `'resolved'`,
    `insertion_alignment_uploaded` → `'aligned'`,
    `insertion_picks` → `'traced'`, fallthrough → `'no-tracing'`.
- `tests/test_check_datasets.py:240-265` — rewrite the two existing
  histology tests for the new column. Add tests covering each of the
  five states: insertion with resolved alignment → `'resolved'`;
  insertion with `alignment_count > 0` but no `alignment_resolved` →
  `'aligned'`; insertion with `xyz_picks` but no alignment count →
  `'traced'`; insertion with no `xyz_picks` → `'no-tracing'`;
  `ins=None` → `'missing'`.

## Approach

`psyfun.histology.insertion_picks` /
`insertion_alignment_uploaded` / `insertion_alignment_resolved` are
already the right predicates and are imported in `psyfun/io.py`. The
ordering is encoded as a short if-elif chain; the order matters because
`insertion_alignment_resolved` is the strictest state.

## Acceptance

- `pytest tests/test_check_datasets.py -k "histology_probe"` passes;
  assertions cover all five states.
- A probe with `xyz_picks=[]` and no alignment data produces
  `probe00_histology == 'no-tracing'` (insertion exists but no picks);
  `ins=None` produces `'missing'`. These two cases must be
  distinguishable.
