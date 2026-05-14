"""Tests for psyfun.io._audit_datasets and its per-modality helpers."""
from pathlib import Path

import pandas as pd
import pytest

from one.alf.exceptions import ALFObjectNotFound

from psyfun import io


class FakeAlyx:
    """Minimal Alyx REST stub. `rest_map` is keyed by (endpoint, action)."""

    def __init__(self, rest_map: dict):
        self._rest_map = rest_map

    def rest(self, endpoint, action, **kwargs):
        handler = self._rest_map.get((endpoint, action), [])
        return handler(kwargs) if callable(handler) else handler


class FakeOne:
    """ONE stub. Datasets keyed by (eid, filename, collection); revision ignored."""

    def __init__(self, datasets=None, list_map=None, rest_map=None, path_map=None):
        self._datasets = datasets or {}
        self._list_map = list_map or {}
        self.alyx = FakeAlyx(rest_map or {})
        self._path_map = path_map or {}

    def load_dataset(self, eid, name, collection, revision=None):
        key = (eid, name, collection)
        if key not in self._datasets:
            raise ALFObjectNotFound(name)
        return self._datasets[key]

    def list_datasets(self, eid):
        return self._list_map.get(eid, [])

    def eid2path(self, eid):
        return self._path_map[eid]


# --- _audit_task_alf -------------------------------------------------------

def test_audit_task_alf_complete_and_error():
    eid = "E"
    one = FakeOne(datasets={
        (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"): "ok",
    })
    out = io._audit_task_alf(eid, 0, "raw_task_data_00", one)
    assert out["task_pre_intervalsTable"] == io.EXTRACTION_COMPLETE
    assert out["task_pre_passiveStims"] == io.EXTRACTION_ERROR
    assert out["task_pre_passiveGabor"] == io.EXTRACTION_ERROR


def test_audit_task_alf_raw_missing():
    out = io._audit_task_alf("E", 1, None, FakeOne())
    assert set(out) == {
        "task_post_intervalsTable", "task_post_passiveStims", "task_post_passiveGabor"
    }
    assert all(v == io.RAW_DATA_MISSING for v in out.values())


# --- _list_passive_raw_collections ----------------------------------------

def test_list_passive_raw_collections_skips_filler():
    eid = "E"
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"):
                {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                 "SESSION_DATETIME": "2025-03-11T18:00:00"},
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"):
                {"PYBPOD_PROTOCOL": "_iblrig_tasks_spontaneous",
                 "SESSION_DATETIME": "2025-03-11T18:20:00"},
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_02"):
                {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                 "SESSION_DATETIME": "2025-03-11T18:50:00"},
        },
        list_map={eid: [
            "raw_task_data_00/_iblrig_taskSettings.raw.json",
            "raw_task_data_01/_iblrig_taskSettings.raw.json",
            "raw_task_data_02/_iblrig_taskSettings.raw.json",
        ]},
    )
    assert io._list_passive_raw_collections(eid, one) == [
        "raw_task_data_00", "raw_task_data_02"
    ]


# --- _pick_latest_sorter ---------------------------------------------------

def test_pick_latest_sorter_newest_wins():
    eid = "E"

    def datasets_list(kwargs):
        registry = {
            "alf/probe00/pykilosort": [
                {"revision": "", "date_created": "2024-01-01T00:00:00"}],
            "alf/probe00/iblsorter": [
                {"revision": "", "date_created": "2025-06-01T00:00:00"}],
        }
        return registry.get(kwargs["collection"], [])

    one = FakeOne(rest_map={("datasets", "list"): datasets_list})
    assert io._pick_latest_sorter(eid, "probe00", one) == ("iblsorter", "")


def test_pick_latest_sorter_none_registered():
    one = FakeOne(rest_map={("datasets", "list"): []})
    assert io._pick_latest_sorter("E", "probe00", one) == ("", "")


# --- _audit_probe ----------------------------------------------------------

def _sorter_handler(registry):
    def handler(kwargs):
        return registry.get(kwargs["collection"], [])
    return handler


def test_audit_probe_complete(tmp_path):
    eid = "E"
    ins = {"name": "probe00", "id": "PID0"}
    bombcell = (tmp_path / "spike_sorters" / "iblsorter" / "probe00"
                / "bombcell" / io.BOMBCELL_OUTPUT_FILE)
    bombcell.parent.mkdir(parents=True)
    bombcell.write_text("x")
    one = FakeOne(
        datasets={
            (eid, "_spikeglx_ephysData_g0_t0.imec0.ap.cbin",
             "raw_ephys_data/probe00"): "ok",
            (eid, "_spikeglx_ephysData_g0_t0.imec0.sync.npy",
             "raw_ephys_data/probe00"): "ok",
            (eid, "spikes.times.npy", "alf/probe00/iblsorter"): "ok",
            (eid, "spikes.clusters.npy", "alf/probe00/iblsorter"): "ok",
            (eid, "clusters.uuids.csv", "alf/probe00/iblsorter"): "ok",
        },
        rest_map={("datasets", "list"): _sorter_handler({
            "alf/probe00/iblsorter": [
                {"revision": "", "date_created": "2025-06-01T00:00:00"}],
        })},
        path_map={eid: tmp_path},
    )
    out = io._audit_probe(eid, 0, ins, one)
    assert out["probe00_raw_ap"] == io.EXTRACTION_COMPLETE
    assert out["probe00_sync"] == io.EXTRACTION_COMPLETE
    assert out["probe00_sorter"] == "iblsorter"
    assert out["probe00_spikes"] == io.EXTRACTION_COMPLETE
    assert out["probe00_bombcell"] == io.EXTRACTION_COMPLETE


def test_audit_probe_imec_double_digit_form(tmp_path):
    eid = "E"
    ins = {"name": "probe01", "id": "PID1"}
    one = FakeOne(
        datasets={
            (eid, "_spikeglx_ephysData_g0_t0.imec01.ap.cbin",
             "raw_ephys_data/probe01"): "ok",
        },
        rest_map={("datasets", "list"): []},
        path_map={eid: tmp_path},
    )
    out = io._audit_probe(eid, 1, ins, one)
    assert out["probe01_raw_ap"] == io.EXTRACTION_COMPLETE
    # raw present, no sync / sorter / bombcell -> extraction error
    assert out["probe01_sync"] == io.EXTRACTION_ERROR
    assert out["probe01_sorter"] == ""
    assert out["probe01_spikes"] == io.EXTRACTION_ERROR
    assert out["probe01_bombcell"] == io.EXTRACTION_ERROR


def test_audit_probe_raw_missing(tmp_path):
    eid = "E"
    ins = {"name": "probe00", "id": "PID0"}
    one = FakeOne(
        rest_map={("datasets", "list"): []},
        path_map={eid: tmp_path},
    )
    out = io._audit_probe(eid, 0, ins, one)
    assert out["probe00_raw_ap"] == io.RAW_DATA_MISSING
    assert out["probe00_sync"] == io.RAW_DATA_MISSING
    assert out["probe00_spikes"] == io.RAW_DATA_MISSING
    assert out["probe00_bombcell"] == io.RAW_DATA_MISSING


def test_audit_probe_no_insertion():
    out = io._audit_probe("E", 1, None, FakeOne())
    assert out["probe01_raw_ap"] == io.RAW_DATA_MISSING
    assert out["probe01_sorter"] == ""


def test_audit_probe_revision_suffix(tmp_path):
    eid = "E"
    ins = {"name": "probe00", "id": "PID0"}
    one = FakeOne(
        datasets={
            (eid, "_spikeglx_ephysData_g0_t0.imec0.ap.cbin",
             "raw_ephys_data/probe00"): "ok",
            (eid, "spikes.times.npy", "alf/probe00/iblsorter"): "ok",
            (eid, "spikes.clusters.npy", "alf/probe00/iblsorter"): "ok",
            (eid, "clusters.uuids.csv", "alf/probe00/iblsorter"): "ok",
        },
        rest_map={("datasets", "list"): _sorter_handler({
            "alf/probe00/iblsorter": [
                {"revision": "2025-06-01", "date_created": "2025-06-01T00:00:00"}],
        })},
        path_map={eid: tmp_path},
    )
    out = io._audit_probe(eid, 0, ins, one)
    assert out["probe00_sorter"] == "iblsorter#2025-06-01#"


# --- _audit_camera ---------------------------------------------------------

def test_audit_camera_qc_and_status():
    eid = "E"
    extended_qc = {
        "_videoLeft_dropped_frames": ["PASS", 14, 0],
        "_videoLeft_timestamps": "WARNING",
        # no pin_state entry
    }
    one = FakeOne(datasets={
        (eid, "_iblrig_leftCamera.raw.mp4", "raw_video_data"): "ok",
    })
    out = io._audit_camera(eid, "left", extended_qc, one)
    assert out["left_camera_raw_video"] == io.EXTRACTION_COMPLETE
    assert out["left_camera_pose"] == io.EXTRACTION_ERROR
    assert out["left_camera_dropped_frames"] == "PASS"
    assert out["left_camera_timestamps"] == "WARNING"
    assert out["left_camera_pin_state"] == ""


def test_audit_camera_raw_missing():
    out = io._audit_camera("E", "body", {}, FakeOne())
    assert out["body_camera_raw_video"] == io.RAW_DATA_MISSING
    assert out["body_camera_pose"] == io.RAW_DATA_MISSING
    assert out["body_camera_dropped_frames"] == ""


# --- _audit_histology_probe ------------------------------------------------

def test_audit_histology_probe_traced_and_resolved():
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
    out = io._audit_histology_probe(eid, 0, ins, one)
    assert out == {
        "probe00_traced": True,
        "probe00_alignment_uploaded": True,
        "probe00_alignment_resolved": True,
    }


def test_audit_histology_probe_no_insertion():
    out = io._audit_histology_probe("E", 1, None, FakeOne())
    assert out == {
        "probe01_traced": False,
        "probe01_alignment_uploaded": False,
        "probe01_alignment_resolved": False,
    }
