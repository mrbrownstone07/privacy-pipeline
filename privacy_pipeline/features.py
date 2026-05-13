from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import pywt
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Dataset(NamedTuple):
    """Return type of load_dataset — supports both attribute and tuple unpacking."""
    X           : np.ndarray   # (n_samples, n_features)  float64
    y           : np.ndarray   # (n_samples,)             int — label-encoded
    target_names: np.ndarray   # (n_classes,)             str

    def __repr__(self) -> str:
        return (
            f"Dataset(samples={len(self.y)}, features={self.X.shape[1]}, "
            f"classes={list(self.target_names)})"
        )


# ── Signal / segmentation constants ──────────────────────────────────────────
FS        = 12_000
SEG_LEN   = 1_024
STRIDE    = 0           # 0 = no overlap
WPD_WAV   = "db4"
WPD_LEVEL = 4
N_BANDS   = 2 ** WPD_LEVEL   # 16
ACF_LAGS  = SEG_LEN // 2     # 512

# ── Column registries ─────────────────────────────────────────────────────────
TIME_COLS = ["mean", "std", "rms", "kurtosis", "skewness",
             "crest_factor", "impulse_factor", "shape_factor", "peak_to_peak"]
FREQ_COLS = ["dom_freq", "freq_center", "rms_freq", "freq_variance", "spec_entropy"]
_WPD_DESC = ["wpd_mean", "wpd_std", "wpd_rms", "wpd_energy",
             "wpd_kurtosis", "wpd_skewness", "wpd_variance"]
WPD_COLS  = [f"{d}_{j}" for d in _WPD_DESC for j in range(N_BANDS)]
META_COLS = ["file_id", "fault_type", "fault_size", "load"]

_GROUP_COLS: dict[str, list[str]] = {
    "time":      TIME_COLS,
    "frequency": FREQ_COLS,
    "wpd":       WPD_COLS,
    "acf":       ["acf_burst_spacing"],
}


# ── Feature implementations ───────────────────────────────────────────────────

def _time_features(x: np.ndarray) -> dict:
    mu, var = np.mean(x), np.var(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    p2p = float(np.max(x) - np.min(x))
    if var == 0:
        return dict(mean=float(mu), std=0.0, rms=rms,
                    kurtosis=np.nan, skewness=np.nan,
                    crest_factor=np.nan, impulse_factor=np.nan,
                    shape_factor=np.nan, peak_to_peak=p2p)
    sigma    = float(np.sqrt(var))
    mean_abs = float(np.mean(np.abs(x)))
    peak     = float(np.max(np.abs(x)))
    z        = (x - mu) / sigma
    return dict(mean=float(mu), std=sigma, rms=rms,
                kurtosis=float(np.mean(z ** 4)), skewness=float(np.mean(z ** 3)),
                crest_factor=peak / rms, impulse_factor=peak / mean_abs,
                shape_factor=rms / mean_abs, peak_to_peak=p2p)


def _freq_features(x: np.ndarray, fs: float | None = None) -> dict:
    fs = fs if fs is not None else FS
    fft_v  = np.fft.rfft(x)
    freqs  = np.fft.rfftfreq(len(x), d=1.0 / fs)
    amp    = np.abs(fft_v)
    amp_sq = amp ** 2
    s_amp  = amp.sum() + 1e-12
    s_sq   = amp_sq.sum() + 1e-12
    pk     = amp_sq / s_sq
    fc     = float(np.sum(freqs * amp) / s_amp)
    return dict(
        dom_freq     =float(freqs[np.argmax(amp)]),
        freq_center  =fc,
        rms_freq     =float(np.sqrt(np.sum(freqs ** 2 * amp_sq) / s_sq)),
        freq_variance=float(np.sum((freqs - fc) ** 2 * amp_sq) / s_sq),
        spec_entropy =float(-np.sum(pk * np.log(pk + 1e-12))),
    )


def _wpd_features(x: np.ndarray) -> dict:
    wp    = pywt.WaveletPacket(data=x, wavelet=WPD_WAV, mode="symmetric", maxlevel=WPD_LEVEL)
    nodes = [nd.path for nd in wp.get_level(WPD_LEVEL, "natural")]
    out   = {}
    for j, node in enumerate(nodes):
        c   = wp[node].data
        mu  = float(np.mean(c))
        var = float(np.var(c))
        rms = float(np.sqrt(np.mean(c ** 2)))
        eng = float(np.sum(c ** 2))
        if var == 0:
            out.update({f"wpd_mean_{j}": mu, f"wpd_std_{j}": 0.0,
                        f"wpd_rms_{j}": rms, f"wpd_energy_{j}": eng,
                        f"wpd_kurtosis_{j}": np.nan, f"wpd_skewness_{j}": np.nan,
                        f"wpd_variance_{j}": 0.0})
        else:
            sig = float(np.sqrt(var))
            z   = (c - mu) / sig
            out.update({f"wpd_mean_{j}": mu, f"wpd_std_{j}": sig,
                        f"wpd_rms_{j}": rms, f"wpd_energy_{j}": eng,
                        f"wpd_kurtosis_{j}": float(np.mean(z ** 4)),
                        f"wpd_skewness_{j}": float(np.mean(z ** 3)),
                        f"wpd_variance_{j}": var})
    return out


def _acf_burst_spacing(x: np.ndarray) -> float:
    x_c  = x - np.mean(x)
    corr = np.correlate(x_c, x_c, mode="full")
    z    = corr[len(x) - 1]
    if z == 0:
        return np.nan
    acf    = corr[len(x): len(x) + ACF_LAGS] / z
    peaks, _ = find_peaks(acf, prominence=0.1, distance=10)
    return float(np.median(np.diff(peaks))) if len(peaks) >= 2 else np.nan


# ── Public API ─────────────────────────────────────────────────────────────────

def feature_columns(groups: list[str] | None = None) -> list[str]:
    """Return ordered feature column names for the given groups."""
    return [col for g in (groups or ["wpd"]) for col in _GROUP_COLS.get(g, [])]


def extract_features(
    segments: list[np.ndarray],
    metadata: list[dict],
    groups: list[str] | None = None,
    fs: float | None = None,
) -> pd.DataFrame:
    """
    Extract features from pre-segmented signals.

    Parameters
    ----------
    segments : list of 1-D arrays, each length SEG_LEN
    metadata : list of dicts with keys: file_id, fault_type, fault_size, load
    groups   : feature groups — subset of {time, frequency, wpd, acf}; default ['wpd']
    fs       : sampling frequency in Hz; None uses the module default (12 000 Hz)

    Returns
    -------
    DataFrame  columns: META_COLS + selected feature columns
    """
    assert len(segments) == len(metadata)
    groups  = list(groups or ["wpd"])
    records = []
    for seg, meta in zip(segments, metadata):
        row = {k: meta[k] for k in META_COLS}
        if "time"      in groups: row.update(_time_features(seg))
        if "frequency" in groups: row.update(_freq_features(seg, fs=fs))
        if "wpd"       in groups: row.update(_wpd_features(seg))
        if "acf"       in groups: row["acf_burst_spacing"] = _acf_burst_spacing(seg)
        records.append(row)
    return pd.DataFrame(records, columns=META_COLS + feature_columns(groups))


def segment_signal(
    signal: np.ndarray,
    stride: int = STRIDE,
    seg_len: int | None = None,
) -> list[np.ndarray]:
    """
    Sliding-window segmentation. stride=0 means no overlap (step = seg_len).

    Parameters
    ----------
    signal  : 1-D time-series
    stride  : step size; 0 means no overlap (step = seg_len)
    seg_len : window length in samples; None uses the module default (1024)
    """
    seg_len = seg_len if seg_len is not None else SEG_LEN
    effective = seg_len if stride == 0 else stride
    if effective < 0:
        raise ValueError(f"stride must be >= 0, got {stride}")
    segs = [signal[s: s + seg_len]
            for s in range(0, len(signal) - seg_len + 1, effective)]
    overlap = f"{100*(1 - effective/seg_len):.0f}% overlap" if stride else "no overlap"
    print(f"  Segmentation: seg_len={seg_len}, stride={effective} ({overlap}), segments={len(segs)}")
    return segs


def load_dataset(
    file_path     : str | Path,
    meta_cols     : list[str] | None = None,
    label_col     : str              = "label",
    feature_groups: list[str] | None = None,
) -> Dataset:
    """
    Generic CSV loader — works for any dataset with a label column.

    Parameters
    ----------
    file_path      : path to CSV
    meta_cols      : columns to drop (non-feature metadata)
    label_col      : column with class labels
    feature_groups : keep only columns for these groups; None = all remaining columns

    Returns
    -------
    Dataset(X, y, target_names) — also unpackable as a 3-tuple
    """
    df        = pd.read_csv(file_path)
    drop_cols = list(meta_cols or []) + [label_col]

    if feature_groups is not None:
        wanted    = set(feature_columns(feature_groups))
        feat_cols = [c for c in df.columns if c not in drop_cols and c in wanted]
        X = df[feat_cols].values.astype(np.float64)
    else:
        X = df.drop(columns=drop_cols, errors="ignore").values.astype(np.float64)

    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    return Dataset(X=X, y=y, target_names=le.classes_)


def preprocess_features(X: np.ndarray) -> np.ndarray:
    """Impute NaN with column means, then StandardScaler fit_transform."""
    X = SimpleImputer(strategy="mean").fit_transform(X)
    return StandardScaler().fit_transform(X)
