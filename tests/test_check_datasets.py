"""Tests for psyfun.io._check_datasets and its per-modality helpers."""
import pandas as pd
import pytest

from one.alf.exceptions import ALFObjectNotFound

from psyfun import io


def dsr_entry(name, collection, data_url="https://example/url",
              default_revision="True", revision="", version=""):
    """A `data_dataset_session_related` entry with the keys the check reads."""
    return {
        "name": name, "collection": collection, "data_url": data_url,
        "default_revision": default_revision, "revision": revision,
        "version": version,
    }


class FakeAlyx:
    """Minimal Alyx REST stub. `rest_map` is keyed by (endpoint, action)."""

    def __init__(self, rest_map: dict):
        self._rest_map = rest_map

    def rest(self, endpoint, action, **kwargs):
        handler = self._rest_map.get((endpoint, action), [])
        return handler(kwargs) if callable(handler) else handler


class FakeOne:
    """ONE stub. `datasets` keyed by (eid, filename, collection) for load_dataset."""

    def __init__(self, datasets=None, rest_map=None, path_map=None):
        self._datasets = datasets or {}
        self.alyx = FakeAlyx(rest_map or {})
        self._path_map = path_map or {}

    def load_dataset(self, eid, name, collection, revision=None):
        key = (eid, name, collection)
        if key not in self._datasets:
            raise ALFObjectNotFound(name)
        return self._datasets[key]

    def eid2path(self, eid):
        return self._path_map[eid]


# --- _raw_task_collections_from_registry ----------------------------------

def test_raw_task_collections_from_registry_sorted():
    dsr = [
        dsr_entry("x", "raw_task_data_02"),
        dsr_entry("y", "alf"),
        dsr_entry("z", "raw_task_data_00"),
    ]
    assert io._raw_task_collections_from_registry(dsr) == [
        "raw_task_data_00", "raw_task_data_02"
    ]


# --- _list_passive_raw_collections ----------------------------------------

def test_list_passive_raw_collections_skips_filler():
    eid = "E"
    dsr = [
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_00"),
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_01"),
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_02"),
    ]
    one = FakeOne(datasets={
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
             "SESSION_DATETIME": "2025-03-11T18:00:00"},
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_spontaneous",
             "SESSION_DATETIME": "2025-03-11T18:20:00"},
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_02"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
             "SESSION_DATETIME": "2025-03-11T18:50:00"},
    })
    assert io._list_passive_raw_collections(eid, dsr, one) == [
        "raw_task_data_00", "raw_task_data_02"
    ]


# --- _check_task_alf -------------------------------------------------------

def test_check_task_alf_complete_and_error():
    present = {("alf/task_00", "_ibl_passivePeriods.intervalsTable.csv")}
    out = io._check_task_alf(present, 0, "raw_task_data_00")
    assert out["task_pre_intervalsTable"] == io.EXTRACTION_COMPLETE
    assert out["task_pre_passiveStims"] == io.EXTRACTION_ERROR
    assert out["task_pre_passiveGabor"] == io.EXTRACTION_ERROR


def test_check_task_alf_raw_missing():
    out = io._check_task_alf(set(), 1, None)
    assert set(out) == {
        "task_post_intervalsTable", "task_post_passiveStims", "task_post_passiveGabor"
    }
    assert all(v == io.RAW_DATA_MISSING for v in out.values())


# --- _pick_sorter / _format_sorter ----------------------------------------

def test_pick_sorter_priority_order():
    dsr = [
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  version="pykilosort_ibl_1.4.1"),
        dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                  version="iblsorter_1.9.0"),
    ]
    assert io._pick_sorter(dsr, "probe00") == (
        "iblsorter", "", "iblsorter_1.9.0"
    )


def test_pick_sorter_none_registered():
    assert io._pick_sorter([], "probe00") == ("", "", "")


def test_pick_sorter_prefers_default_revision():
    dsr = [
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  default_revision="False", revision="2024-01-01"),
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  default_revision="True", revision="2025-06-01",
                  version="pykilosort_ibl_1.4.1"),
    ]
    assert io._pick_sorter(dsr, "probe00") == (
        "pykilosort", "2025-06-01", "pykilosort_ibl_1.4.1"
    )


def test_format_sorter_variants():
    assert io._format_sorter("", "", "") == ""
    assert io._format_sorter(
        "pykilosort", "", "pykilosort_ibl_1.4.1"
    ) == "pykilosort (pykilosort_ibl_1.4.1)"
    assert io._format_sorter(
        "iblsorter", "2025-06-01", "iblsorter_1.9.0"
    ) == "iblsorter#2025-06-01# (iblsorter_1.9.0)"


# --- _check_probe ----------------------------------------------------------

def test_check_probe_complete(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    bombcell = (tmp_path / "spike_sorters" / "iblsorter" / "probe00"
                / "bombcell" / io.BOMBCELL_OUTPUT_FILE)
    bombcell.parent.mkdir(parents=True)
    bombcell.write_text("x")
    present = {
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.ap.cbin"),
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.sync.npy"),
        ("alf/probe00/iblsorter", "spikes.times.npy"),
        ("alf/probe00/iblsorter", "spikes.clusters.npy"),
        ("alf/probe00/iblsorter", "clusters.uuids.csv"),
    }
    dsr = [dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                     version="iblsorter_1.9.0")]
    out = io._check_probe(present, dsr, 0, ins, tmp_path)
    assert out["probe00_raw_ap"] == io.EXTRACTION_COMPLETE
    assert out["probe00_sync"] == io.EXTRACTION_COMPLETE
    assert out["probe00_sorter"] == "iblsorter (iblsorter_1.9.0)"
    assert out["probe00_spikes"] == io.EXTRACTION_COMPLETE
    assert out["probe00_bombcell"] == io.EXTRACTION_COMPLETE


def test_check_probe_imec_double_digit_form(tmp_path):
    ins = {"name": "probe01", "id": "PID1"}
    present = {
        ("raw_ephys_data/probe01", "_spikeglx_ephysData_g0_t0.imec01.ap.cbin"),
    }
    out = io._check_probe(present, [], 1, ins, tmp_path)
    assert out["probe01_raw_ap"] == io.EXTRACTION_COMPLETE
    # raw present, no sync / sorter / bombcell -> extraction error
    assert out["probe01_sync"] == io.EXTRACTION_ERROR
    assert out["probe01_sorter"] == ""
    assert out["probe01_spikes"] == io.EXTRACTION_ERROR
    assert out["probe01_bombcell"] == io.EXTRACTION_ERROR


def test_check_probe_raw_missing(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    out = io._check_probe(set(), [], 0, ins, tmp_path)
    assert out["probe00_raw_ap"] == io.RAW_DATA_MISSING
    assert out["probe00_sync"] == io.RAW_DATA_MISSING
    assert out["probe00_spikes"] == io.RAW_DATA_MISSING
    assert out["probe00_bombcell"] == io.RAW_DATA_MISSING


def test_check_probe_no_insertion():
    out = io._check_probe(set(), [], 1, None, None)
    assert out["probe01_raw_ap"] == io.RAW_DATA_MISSING
    assert out["probe01_sorter"] == ""


def test_check_probe_revision_suffix(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    present = {
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.ap.cbin"),
        ("alf/probe00/iblsorter", "spikes.times.npy"),
        ("alf/probe00/iblsorter", "spikes.clusters.npy"),
        ("alf/probe00/iblsorter", "clusters.uuids.csv"),
    }
    dsr = [dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                     revision="2025-06-01", version="iblsorter_1.9.0")]
    out = io._check_probe(present, dsr, 0, ins, tmp_path)
    assert out["probe00_sorter"] == "iblsorter#2025-06-01# (iblsorter_1.9.0)"


# --- _check_camera ---------------------------------------------------------

def test_check_camera_qc_and_status():
    present = {("raw_video_data", "_iblrig_leftCamera.raw.mp4")}
    extended_qc = {
        "_videoLeft_dropped_frames": ["PASS", 14, 0],
        "_videoLeft_timestamps": "WARNING",
        # no pin_state entry
    }
    out = io._check_camera(present, "left", extended_qc)
    assert out["left_camera_raw_video"] == io.EXTRACTION_COMPLETE
    assert out["left_camera_pose"] == io.EXTRACTION_ERROR
    assert out["left_camera_dropped_frames"] == "PASS"
    assert out["left_camera_timestamps"] == "WARNING"
    assert out["left_camera_pin_state"] == ""


def test_check_camera_raw_missing():
    out = io._check_camera(set(), "body", {})
    assert out["body_camera_raw_video"] == io.RAW_DATA_MISSING
    assert out["body_camera_pose"] == io.RAW_DATA_MISSING
    assert out["body_camera_dropped_frames"] == ""


# --- _check_histology_probe ------------------------------------------------

def test_check_histology_probe_traced_and_resolved():
    eid = "E"
    ins = {"name": "probe00", "id": "PID0"}
    full = {
        "id": "PID0",
        "json": {
            "xyz_picks": [[1, 2, 3]],
            "extended_qc": {"alignment_count": 2, "alignment_resolved": True},
        },
    }
    one = FakeOne(rest_map={("insertions", "list"): [full]})
    out = io._check_histology_probe(eid, 0, ins, one)
    assert out == {
        "probe00_traced": True,
        "probe00_alignment_uploaded": True,
        "probe00_alignment_resolved": True,
    }


def test_check_histology_probe_no_insertion():
    out = io._check_histology_probe("E", 1, None, FakeOne())
    assert out == {
        "probe01_traced": False,
        "probe01_alignment_uploaded": False,
        "probe01_alignment_resolved": False,
    }


# --- _check_datasets -------------------------------------------------------

def test_check_datasets_data_url_none_is_absent(tmp_path, monkeypatch):
    """A registry entry with a null `data_url` does not count as present."""
    eid = "E"
    session = {
        "extended_qc": {},
        "data_dataset_session_related": [
            dsr_entry("_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"),
            dsr_entry("_ibl_passiveStims.table.csv", "alf/task_00", data_url=None),
            dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_00"),
        ],
    }
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"):
                {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                 "SESSION_DATETIME": "2025-03-11T18:00:00"},
        },
        rest_map={
            ("sessions", "read"): session,
            ("insertions", "list"): [],
        },
        path_map={eid: tmp_path},
    )
    monkeypatch.setattr(io, "_check_image_stacks", lambda subject, lab: False)
    series = pd.Series({"eid": eid, "subject": "S", "lab": "L"})
    out = io._check_datasets(series, one=one)
    assert out["task_pre_intervalsTable"] == io.EXTRACTION_COMPLETE
    assert out["task_pre_passiveStims"] == io.EXTRACTION_ERROR
    assert out["task_pre_passiveGabor"] == io.EXTRACTION_ERROR
    assert out["task_post_intervalsTable"] == io.RAW_DATA_MISSING
