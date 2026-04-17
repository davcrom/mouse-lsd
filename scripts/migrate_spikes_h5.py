"""One-off migration: flat per-uuid datasets -> per-uuid groups with 'times'.

Old layout: h5file[uuid] is a (n_spikes,) dataset of spike times.
New layout: h5file[uuid] is a group containing 'times' (and optionally
'duplicates', future per-uuid datasets).

Idempotent: uuids already stored as groups are skipped.
"""
import h5py

from psyfun.config import paths


if __name__ == '__main__':
    n_migrated = 0
    n_skipped = 0
    with h5py.File(paths['spikes'], 'a') as h5file:
        for uuid in list(h5file.keys()):
            node = h5file[uuid]
            if isinstance(node, h5py.Group):
                n_skipped += 1
                continue
            times = node[:]
            del h5file[uuid]
            grp = h5file.create_group(uuid)
            grp.create_dataset('times', data=times)
            n_migrated += 1
    print(f"Migrated {n_migrated} uuids; {n_skipped} already in group layout.")
