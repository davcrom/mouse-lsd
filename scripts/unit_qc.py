import os
import shutil
import tarfile
from tqdm import tqdm
from matplotlib import pyplot as plt

import bombcell

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from psyfun.io import load_units

class SpikeSortingQC():

    def __init__(self, eid, probe, one):
        self.eid = eid
        self.probe = probe
        self.one = one

        # Try to see if KS output is already downloaded and extracted
        sorter = self._get_sorter()
        self.spike_sorting_dir = one.eid2path(self.eid) / f'spike_sorters/{probe}/{sorter}'
        # Pick a random KS output file to check for
        sorting_extracted = (self.spike_sorting_dir / 'amplitudes.npy').exists()

        if not sorting_extracted:
            self.spike_sorting_filepath = self.download_spike_sorting()
            # Overwrite dir with one that actually had the KS output archive
            self.spike_sorting_dir = self.spike_sorting_filepath.parent
            self.extract_spike_sorting()


    def _get_sorter(self):
        collections = self.one.list_collections(self.eid)
        has_pykilosort = any(
            ['pykilosort' in col for col in collections if self.probe in col]
        )
        has_iblsorter = any(
            ['iblsorter' in col for col in collections if self.probe in col]
            )
        if has_iblsorter:  # priority given to iblsorter
            sorter = 'iblsorter'
        elif has_pykilosort:
            sorter = 'pykilosort'
        else:
            sorter = None


    def download_spike_sorting(self):
        print("Loading spike sorting data...")
        try:
            spike_sorting_filepath = self.one.load_dataset(
                eid,
                '*_kilosort_raw.output.tar',
                collection=f'spike_sorters/pykilosort/{probe}'
            )
        except ALFObjectNotFound:
            spike_sorting_filepath = self.one.load_dataset(
                eid,
                '*_kilosort_raw.output.tar',
                collection=f'spike_sorters/iblsorter/{probe}'
            )
        return spike_sorting_filepath


    def extract_spike_sorting(self):
        with tarfile.open(self.spike_sorting_filepath, "r:*") as tar:
            self._safe_tar_extract(tar, self.spike_sorting_dir)


    def remove_spike_sorting(self):
        print("Deleting spike sorting archive...")
        spike_sorting_filepath = self.spike_sorting_dir / '_kilosort_raw.output.tar'
        try:
            os.remove(spike_sorting_filepath)
            print("    spike sorting archive deleted.")
        except FileNotFoundError:
            print(f"    spike sorting archive not found: {spike_sorting_filepath}")


    def _safe_tar_extract(self, tar, path):
        print("Extracting spike sorting data...")
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not os.path.realpath(member_path).startswith(os.path.realpath(path)):
                raise Exception("Unsafe path detected")
        tar.extractall(path)


    def load_waveforms(self):
        try:
            waveforms = self.one.load_dataset(
                self.eid,
                '*clusters.waveforms.py',
                collection=f'alf/{self.probe}/pykilosort'
            )
            channels = self.one.load_dataset(
                self.eid,
                'clusters.waveformsChannels.npy',
                collection=f'alf/{self.probe}/pykilosort'
                )
        except ALFObjectNotFound:
            waveforms = self.one.load_dataset(
                self.eid,
                '*clusters.waveforms.py',
                collection=f'alf/{self.probe}/iblsorter'
            )
            channels = self.one.load_dataset(
                self.eid,
                'clusters.waveformsChannels.npy',
                collection=f'alf/{self.probe}/iblsorter'
                )
        return waveforms, channels


    def run_bombcell(self):
        print("Preparing bombcell...")
        self.bombcell_dir = self.spike_sorting_dir / 'bombcell'
        if os.path.exists(self.bombcell_dir):
            print("Deleting previous bombcell output...")
            shutil.rmtree(self.bombcell_dir)
        os.mkdir(self.spike_sorting_dir / 'bombcell')

        param = bombcell.get_default_parameters(
            kilosort_path=self.spike_sorting_dir,
            kilosort_version=''  # '4' if KS4, else anything
        )

        print("Running bombcell...")
        quality_metrics, param, unit_type, unit_type_string = bombcell.run_bombcell(
            ks_dir=self.spike_sorting_dir,
            save_path=self.spike_sorting_dir / 'bombcell',
            param=param,
            save_figures=True
        )

        plt.close('all')
        print(f"    results saved to {self.bombcell_dir}")

        bombcell_results = pd.DataFrame.from_dict(quality_metrics)
        bombcell_results['bc_label'] = unit_type_string

        return bombcell_results


if __name__ == '__main__':
    one = ONE()

    # Load sessions
    df_units = load_units()  # session info

    probes = df_units.groupby(['eid', 'probe'])
    for (eid, probe), group in tqdm(probes, total=len(probes)):
        ssqc = SpikeSortingQC(eid, probe, one)
        bombcell_results = ssqc.run_bombcell()
        # TODO: merge bombcell results into units table
        ssqc.remove_spike_sorting()
