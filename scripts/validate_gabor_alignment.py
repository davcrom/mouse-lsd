"""
Validate `metadata/sessions.pqt` task-epoch timings against local spike data.

For every session in `paths['sessions']` that has spike data in `paths['spikes']`,
build a depth-binned z-scored PSTH around each passive task's replay window.

Per task panel, gabor onsets come from one of two sources:

1. `alf/task_NN/_ibl_passiveGabor.table.csv` (FPGA-aligned, ground truth)
   when the IBL extractor produced it.

2. `raw_task_data_NN/_iblrig_stimPositionScreen.raw.csv` (rig wall-clock)
   converted to FPGA seconds via `anchor.spontaneous_start +
   (gabor_rig_clock - anchor.SESSION_DATETIME)`, where `anchor` is the
   first passive on the same session that *does* have alf intervals.
   This tests the fixed-delta fallback used in `psyfun.io._fetch_protocol_timings`.

Each PNG is one session. Per panel: heatmap (depth x time z-score), source
of timings in the title, response stats in a corner box, vertical line at
peak time, red ticks on the depth axis marking significant bins.

Usage:
    python scripts/validate_gabor_alignment.py
    python scripts/validate_gabor_alignment.py --baseline median
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.ioff()
from scipy.ndimage import median_filter

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from psyfun.config import paths


PRE = 1.0
POST = 1.0
T_BIN = 0.01
D_BIN = 20.0
Z_THRESH = 3.0
RESPONSE_WINDOW = (0.0, 0.6)  # frontal areas can respond on movement-elicited timescales
CONTRAST_THRESH = 0.1
MEDIAN_FILTER_BINS = 5  # 1-D median filter size along the time axis (set 0 to disable)
RAW_COLLECTION_RE = re.compile(r"^raw_task_data_(\d+)$")


def session_spikes_by_depth(
    df_units: pd.DataFrame, eid: str, h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray]:
    units = df_units[df_units["eid"] == eid]
    times_chunks, depths_chunks = [], []
    for _, row in units.iterrows():
        if row["uuid"] not in h5 or pd.isna(row["depth"]):
            continue
        ts = h5[row["uuid"]]["times"][:]
        if len(ts) == 0:
            continue
        times_chunks.append(ts)
        depths_chunks.append(np.full(len(ts), row["depth"], dtype=np.float32))
    if not times_chunks:
        return np.array([]), np.array([])
    return np.concatenate(times_chunks), np.concatenate(depths_chunks)


def bin_spikes_2d(
    spike_times: np.ndarray, spike_depths: np.ndarray,
    t_edges: np.ndarray, d_edges: np.ndarray,
) -> np.ndarray:
    counts, _, _ = np.histogram2d(spike_depths, spike_times, bins=[d_edges, t_edges])
    return counts


def _baseline_stats(base: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-depth (location, scale) of a baseline window. method ∈ {'mean','median'}."""
    if method == 'mean':
        loc = base.mean(axis=1, keepdims=True)
        scale = base.std(axis=1, keepdims=True)
    elif method == 'median':
        loc = np.median(base, axis=1, keepdims=True)
        scale = 1.4826 * np.median(np.abs(base - loc), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown baseline method: {method}")
    return loc, np.where(scale == 0, np.nan, scale)


def zscored_psth(
    binned: np.ndarray, t_edges: np.ndarray, events: np.ndarray,
    pre: float, post: float, t_bin: float, baseline: str = 'mean',
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each event, z-score its [-pre, +post] window per depth using that
    same trial's pre-stim baseline (mean+std or median+MAD). Then average
    across trials.
    """
    n_pre, n_post = int(pre / t_bin), int(post / t_bin)
    n_total = n_pre + n_post
    centers = np.arange(-n_pre, n_post) * t_bin + t_bin / 2
    valid = (events - pre >= t_edges[0]) & (events + post <= t_edges[-1])
    events = events[valid]
    n_depth = binned.shape[0]
    if len(events) == 0:
        return centers, np.full((n_depth, n_total), np.nan)
    idx_event = np.searchsorted(t_edges, events) - 1
    z_trials = np.full((n_depth, n_total, len(events)), np.nan)
    for i, idx in enumerate(idx_event):
        chunk = binned[:, idx - n_pre:idx + n_post]
        loc, scale = _baseline_stats(chunk[:, :n_pre], baseline)
        z_trials[:, :, i] = (chunk - loc) / scale
    psth = np.nanmean(z_trials, axis=2)
    if MEDIAN_FILTER_BINS > 1:
        nan_mask = np.isnan(psth)
        filtered = median_filter(np.nan_to_num(psth, nan=0.0),
                                 size=(1, MEDIAN_FILTER_BINS), mode='nearest')
        filtered[nan_mask] = np.nan
        psth = filtered
    return centers, psth


def response_stats(z: np.ndarray, centers: np.ndarray) -> dict:
    mask = (centers >= RESPONSE_WINDOW[0]) & (centers < RESPONSE_WINDOW[1])
    z_post = z[:, mask]
    finite = np.isfinite(z_post).all(axis=1)
    if not finite.any():
        return {"n_depth": 0, "n_resp": 0, "frac": 0.0,
                "peak_z": np.nan, "peak_t": np.nan, "resp_mask": np.zeros(z.shape[0], bool)}
    peak_per_bin_all = np.full(z.shape[0], np.nan)
    peak_per_bin_all[finite] = np.abs(z_post[finite]).max(axis=1)
    resp_mask = np.nan_to_num(peak_per_bin_all, nan=0.0) > Z_THRESH
    peak_idx = int(np.nanargmax(peak_per_bin_all))
    peak_t_idx = int(np.abs(z[peak_idx, mask]).argmax())
    return {
        "n_depth": int(finite.sum()),
        "n_resp": int(resp_mask.sum()),
        "frac": float(resp_mask.sum() / max(finite.sum(), 1)),
        "peak_z": float(np.nanmax(peak_per_bin_all)),
        "peak_t": float(centers[mask][peak_t_idx]),
        "resp_mask": resp_mask,
    }


def list_passive_collections(eid: str, one: ONE) -> list[dict]:
    """For each raw_task_data_NN, classify by PYBPOD_PROTOCOL and grab rig_t0."""
    raw_cols = sorted(
        {m.group(0): int(m.group(1))
         for d in one.list_datasets(eid)
         for m in [RAW_COLLECTION_RE.match(d.split("/")[0])] if m}.items(),
        key=lambda kv: kv[1],
    )
    out = []
    for raw_col, _ in raw_cols:
        try:
            settings = one.load_dataset(eid, "_iblrig_taskSettings.raw.json", raw_col)
        except ALFObjectNotFound:
            continue
        proto = settings.get("PYBPOD_PROTOCOL", "")
        rig_str = settings.get("SESSION_DATETIME") or settings.get("SESSION_START_TIME")
        if rig_str is None:
            continue
        rig_t0 = datetime.fromisoformat(rig_str)
        if "spontaneous" in proto and "passive" not in proto:
            kind = "spontaneous"
        elif "passive" in proto:
            kind = "passive"
        else:
            continue
        out.append({"raw": raw_col, "alf": raw_col.replace("raw_task_data_", "alf/task_"),
                    "kind": kind, "rig_t0": rig_t0})
    return out


def session_anchor(
    eid: str, collections: list[dict], one: ONE,
) -> tuple[float, datetime, str] | None:
    """First passive whose alf intervalsTable can be loaded. Returns (spont_start_FPGA, rig_t0, alf_col)."""
    for c in collections:
        if c["kind"] != "passive":
            continue
        try:
            intervals = one.load_dataset(eid, "_ibl_passivePeriods.intervalsTable.csv", c["alf"])
        except ALFObjectNotFound:
            continue
        if "Unnamed: 0" in intervals.columns:
            intervals = intervals.set_index("Unnamed: 0")
        spont_start = float(intervals["spontaneousActivity"].iloc[0])
        return spont_start, c["rig_t0"], c["alf"]
    return None


def _parse_iblrig_datetime(dt_str: str) -> datetime:
    """Tolerate >6 decimal microseconds in iblrig timestamps."""
    if "." in dt_str:
        main, decimals = dt_str.split(".")
        return datetime.fromisoformat(f"{main}.{decimals[:6]}")
    return datetime.fromisoformat(dt_str)


def events_for_task(
    eid: str, task_idx: int, collections: list[dict],
    anchor: tuple[float, datetime, str] | None, one: ONE,
) -> tuple[np.ndarray | None, str]:
    """Return (events_FPGA, source_label) for the i-th passive collection."""
    passives = [c for c in collections if c["kind"] == "passive"]
    if task_idx >= len(passives):
        return None, f"no passive #{task_idx}"
    target = passives[task_idx]
    # Prefer alf gabor table.
    try:
        gabor = one.load_dataset(eid, "_ibl_passiveGabor.table.csv", target["alf"])
        events = gabor.loc[gabor["contrast"] > CONTRAST_THRESH, "start"].to_numpy(dtype=np.float64)
        return events, f"alf ({target['alf']})"
    except ALFObjectNotFound:
        pass
    # Rig-clock fallback.
    if anchor is None:
        return None, "fallback: no anchor on session"
    spont_fpga, anchor_rig_t0, anchor_alf = anchor
    try:
        df = one.load_dataset(eid, "_iblrig_stimPositionScreen.raw.csv", target["raw"])
    except ALFObjectNotFound:
        return None, f"fallback: no stimPositionScreen in {target['raw']}"
    # iblrig writes the CSV without a header row; pd.read_csv promotes the first
    # data row to columns. Restore it.
    df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df], ignore_index=True)
    # Column 2 holds the start datetime; column 1 holds contrast (when present).
    try:
        contrast = df.iloc[:, 1].astype(float).to_numpy()
    except (ValueError, TypeError):
        contrast = np.full(len(df), np.nan)
    times_rig = []
    for s in df.iloc[:, 2]:
        try:
            times_rig.append(_parse_iblrig_datetime(str(s)))
        except ValueError:
            times_rig.append(None)
    valid = np.array([t is not None for t in times_rig])
    if np.isfinite(contrast).any():
        valid &= contrast > CONTRAST_THRESH
    deltas = np.array([(times_rig[i] - anchor_rig_t0).total_seconds() if valid[i] else np.nan
                       for i in range(len(times_rig))])
    events = spont_fpga + deltas[valid]
    return events, f"fallback: {target['raw']} via anchor {anchor_alf}"


def render_panel(
    ax: plt.Axes, task_idx: int, source: str, events: np.ndarray | None,
    binned: np.ndarray, t_edges: np.ndarray, d_edges: np.ndarray,
    saved_window: tuple[float, float] | None, baseline: str,
) -> None:
    title_top = f"task{task_idx:02d}"
    if events is None or len(events) == 0:
        ax.text(0.5, 0.5, f"{title_top}\n\nno events\n\n{source}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    centers, z = zscored_psth(binned, t_edges, events, PRE, POST, T_BIN, baseline=baseline)
    if not np.isfinite(z).any():
        ax.text(0.5, 0.5, f"{title_top}\n\nPSTH unstable\n\n{source}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    n_in_window = "?"
    if saved_window is not None and not any(pd.isna(v) for v in saved_window):
        n_in_window = int(((events >= saved_window[0]) & (events <= saved_window[1])).sum())
    stats = response_stats(z, centers)
    im = ax.imshow(
        z, aspect="auto", cmap="bwr", vmin=-3, vmax=3, origin="lower",
        extent=[centers[0], centers[-1], d_edges[0], d_edges[-1]],
    )
    ax.axvline(0, ls="--", color="k", lw=0.6)
    if np.isfinite(stats["peak_t"]):
        ax.axvline(stats["peak_t"], ls=":", color="green", lw=0.8)
    # Mark significant depth bins as ticks on the right edge.
    bin_centers_d = 0.5 * (d_edges[:-1] + d_edges[1:])
    sig_depths = bin_centers_d[stats["resp_mask"]]
    if len(sig_depths) > 0:
        ax.scatter(np.full(len(sig_depths), centers[-1]), sig_depths,
                   marker="|", s=18, color="red", clip_on=False)
    ax.set_title(f"{title_top}    source: {source}", fontsize=9, loc="left")
    ax.set_xlabel("time from gabor onset (s)")
    ax.set_ylabel("depth on probe (um)")
    # Stats annotation in upper-left.
    txt = (f"n events: {len(events)} (in saved window: {n_in_window})\n"
           f"peak |z|: {stats['peak_z']:.1f} @ {1000*stats['peak_t']:+.0f} ms\n"
           f"resp depth bins (|z|>{Z_THRESH:.0f} in [0,300]ms): "
           f"{stats['n_resp']}/{stats['n_depth']} ({100*stats['frac']:.1f}%)")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.6"))
    return im


def make_session_figure(
    label: str, eid: str, panels: list[dict], d_edges: np.ndarray, baseline: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), squeeze=False)
    last_im = None
    for j, p in enumerate(panels):
        im = render_panel(
            axes[0, j], p["task_idx"], p["source"], p["events"],
            p["binned"], p["t_edges"], d_edges, p["saved_window"], baseline,
        )
        if im is not None:
            last_im = im
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[0, -1], fraction=0.04, pad=0.02)
        cbar.set_label("z-score")
    fig.suptitle(f"{label} ({eid[:8]}) — baseline: {baseline}", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", choices=("mean", "median"), default="mean",
        help="per-trial baseline normalization: mean+std or median+MAD (default: mean)",
    )
    args = parser.parse_args()
    one = ONE()
    df_sessions = pd.read_parquet(paths["sessions"])
    df_units = pd.read_parquet(paths["units"])
    out_dir = Path(paths["figures"]) / "validate_gabor_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    skipped: list[tuple[str, str]] = []
    with h5py.File(paths["spikes"], "r") as h5:
        for _, row in df_sessions.iterrows():
            eid = row["eid"]
            label = f"{str(row.get('start_time',''))[:10]} {row.get('subject','?')}"
            spike_times, spike_depths = session_spikes_by_depth(df_units, eid, h5)
            if len(spike_times) == 0:
                skipped.append((eid, "no spike data"))
                continue
            t_edges = np.arange(0, spike_times.max() + T_BIN, T_BIN)
            d_edges = np.arange(0, np.ceil(spike_depths.max() / D_BIN) * D_BIN + D_BIN, D_BIN)
            binned = bin_spikes_2d(spike_times, spike_depths, t_edges, d_edges)

            collections = list_passive_collections(eid, one)
            anchor = session_anchor(eid, collections, one)

            panels = []
            for task_idx in (0, 1):
                events, source = events_for_task(eid, task_idx, collections, anchor, one)
                t0 = row.get(f"task{task_idx:02d}_replay_start")
                t1 = row.get(f"task{task_idx:02d}_replay_stop")
                saved = (t0, t1) if pd.notna(t0) and pd.notna(t1) else None
                panels.append({
                    "task_idx": task_idx, "source": source, "events": events,
                    "binned": binned, "t_edges": t_edges, "saved_window": saved,
                })
                # Stats for summary table.
                if events is not None and len(events) > 0:
                    centers, z = zscored_psth(binned, t_edges, events, PRE, POST, T_BIN,
                                              baseline=args.baseline)
                    if np.isfinite(z).any():
                        s = response_stats(z, centers)
                        summary_rows.append({
                            "eid": eid[:8], "date": str(row.get("start_time",""))[:10],
                            "subject": row.get("subject"),
                            "task": f"task{task_idx:02d}", "source": source,
                            "n_events": len(events),
                            "peak_z": s["peak_z"], "peak_t_ms": 1000*s["peak_t"],
                            "n_resp": s["n_resp"], "n_depth": s["n_depth"],
                        })

            fig = make_session_figure(label, eid, panels, d_edges, args.baseline)
            slug = label.replace(" ", "_").replace(":", "")
            out = out_dir / f"{slug}_{eid[:8]}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"saved {out}")

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv_out = out_dir / "summary.csv"
        summary.to_csv(csv_out, index=False)
        print(f"\nsaved {csv_out}")
        print(f"\n=== summary ({len(summary)} panels) ===")
        print(summary.to_string(index=False))
    if skipped:
        print(f"\n=== skipped: {len(skipped)} ===")
        for eid, reason in skipped:
            print(f"  {eid[:8]}: {reason}")


if __name__ == "__main__":
    main()
