#!/usr/bin/env python3
"""
extract_paderborn.py

Downloads the Paderborn Bearing Dataset from Zenodo, reads the vibration
channel from each .mat file, and produces a feature CSV using the same
privacy_pipeline API (segment_signal, extract_features) as the CWRU pipeline.

Usage
-----
    python extract_paderborn.py                          # all bearings
    python extract_paderborn.py --subset Artificial      # one experiment group
    python extract_paderborn.py --out my_features.csv    # custom output path
    python extract_paderborn.py --no-download            # skip download (files exist)
    python extract_paderborn.py --groups time frequency wpd  # feature groups

Notes on library constants
--------------------------
SEG_LEN = 1024 samples  (default in privacy_pipeline.features)
  After resampling from 64 kHz → 12 kHz each segment covers ~85 ms,
  identical to the CWRU pipeline.  Frequency-domain columns therefore have
  the same absolute Hz scale as CWRU.

FS = 12 000 Hz  (default in privacy_pipeline.features)
  Signals are resampled to 12 kHz before segmentation so all frequency
  features are directly comparable with the CWRU dataset.

Output CSV schema
-----------------
  file_id | experiment | fault_size | load | label | <feature columns>

  file_id    — bearing ID  (K001, KA01, …)
  experiment — Healthy | Artificial | Real
  fault_size — 0 (healthy) | 1 (artificial damage) | 2 (real damage)
  load       — 0–3  encoding of the operating condition subfolder
  label      — fault type: Normal | OR | IR   ← use as label_col in YAML
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.signal import resample_poly

from privacy_pipeline import extract_features, segment_signal


# ── Sampling-rate constants ───────────────────────────────────────────────────
# Paderborn vibration data are recorded at 64 kHz.  The privacy_pipeline feature
# extractors assume FS = 12 kHz (CWRU rate).  Resampling to 12 kHz makes the
# absolute frequency features directly comparable and gives ~85 ms segments
# (1024 / 12 kHz) instead of ~16 ms (1024 / 64 kHz).
PADERBORN_FS = 64_000
TARGET_FS    = 12_000  # must match privacy_pipeline.features.FS


# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry: (experiment, fault_type, bearing_id, zenodo_url)

BEARINGS: list[tuple[str, str, str, str]] = [
    ("Healthy",    "Normal", "K001", "https://zenodo.org/records/15845309/files/K001.rar?download=1"),
    ("Healthy",    "Normal", "K002", "https://zenodo.org/records/15845309/files/K002.rar?download=1"),
    ("Healthy",    "Normal", "K003", "https://zenodo.org/records/15845309/files/K003.rar?download=1"),
    ("Healthy",    "Normal", "K004", "https://zenodo.org/records/15845309/files/K004.rar?download=1"),
    ("Healthy",    "Normal", "K005", "https://zenodo.org/records/15845309/files/K005.rar?download=1"),
    ("Healthy",    "Normal", "K006", "https://zenodo.org/records/15845309/files/K006.rar?download=1"),
    ("Artificial", "OR",     "KA01", "https://zenodo.org/records/15845309/files/KA01.rar?download=1"),
    ("Artificial", "OR",     "KA03", "https://zenodo.org/records/15845309/files/KA03.rar?download=1"),
    ("Artificial", "OR",     "KA05", "https://zenodo.org/records/15845309/files/KA05.rar?download=1"),
    ("Artificial", "OR",     "KA06", "https://zenodo.org/records/15845309/files/KA06.rar?download=1"),
    ("Artificial", "OR",     "KA07", "https://zenodo.org/records/15845309/files/KA07.rar?download=1"),
    ("Artificial", "OR",     "KA09", "https://zenodo.org/records/15845309/files/KA09.rar?download=1"),
    ("Artificial", "IR",     "KI01", "https://zenodo.org/records/15845309/files/KI01.rar?download=1"),
    ("Artificial", "IR",     "KI03", "https://zenodo.org/records/15845309/files/KI03.rar?download=1"),
    ("Artificial", "IR",     "KI05", "https://zenodo.org/records/15845309/files/KI05.rar?download=1"),
    ("Artificial", "IR",     "KI07", "https://zenodo.org/records/15845309/files/KI07.rar?download=1"),
    ("Artificial", "IR",     "KI08", "https://zenodo.org/records/15845309/files/KI08.rar?download=1"),
    ("Real",       "OR",     "KA04", "https://zenodo.org/records/15845309/files/KA04.rar?download=1"),
    ("Real",       "OR",     "KA15", "https://zenodo.org/records/15845309/files/KA15.rar?download=1"),
    ("Real",       "OR",     "KA16", "https://zenodo.org/records/15845309/files/KA16.rar?download=1"),
    ("Real",       "OR",     "KA22", "https://zenodo.org/records/15845309/files/KA22.rar?download=1"),
    ("Real",       "OR",     "KA30", "https://zenodo.org/records/15845309/files/KA30.rar?download=1"),
    ("Real",       "IR",     "KB23", "https://zenodo.org/records/15845309/files/KB23.rar?download=1"),
    ("Real",       "IR",     "KB24", "https://zenodo.org/records/15845309/files/KB24.rar?download=1"),
    ("Real",       "IR",     "KI04", "https://zenodo.org/records/15845309/files/KI04.rar?download=1"),
    ("Real",       "IR",     "KI14", "https://zenodo.org/records/15845309/files/KI14.rar?download=1"),
    ("Real",       "IR",     "KI16", "https://zenodo.org/records/15845309/files/KI16.rar?download=1"),
    ("Real",       "IR",     "KI17", "https://zenodo.org/records/15845309/files/KI17.rar?download=1"),
    ("Real",       "IR",     "KI18", "https://zenodo.org/records/15845309/files/KI18.rar?download=1"),
    ("Real",       "IR",     "KI21", "https://zenodo.org/records/15845309/files/KI21.rar?download=1"),
    ("Real",       "OR",     "KB27", "https://zenodo.org/records/15845309/files/KB27.rar?download=1"),
]

# fault_size encoding: experiment type → integer severity level
_FAULT_SIZE: dict[str, int] = {"Healthy": 0, "Artificial": 1, "Real": 2}

# Paderborn operating condition subfolders (consistent across all bearings)
_CONDITIONS = ["N09_M07_F10", "N15_M07_F10", "N15_M01_F10", "N15_M07_F04"]
_CONDITION_LOAD: dict[str, int] = {c: i for i, c in enumerate(_CONDITIONS)}


# ── Download and extraction ────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> None:
    """Stream-download a file; retry once without SSL verification on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} …", end=" ", flush=True)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        print("(SSL fallback)", end=" ", flush=True)
        resp = requests.get(url, stream=True, verify=False, timeout=60)
        resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    received = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65_536):
            if chunk:
                f.write(chunk)
                received += len(chunk)
    mb = received / 1_048_576
    print(f"done ({mb:.1f} MB)")


def _extract(rar_path: Path, out_dir: Path) -> None:
    """
    Extract a RAR archive using the first available tool.

    Tried in order: bsdtar → unrar → unar → 7z.
    If none are found, raises RuntimeError with install instructions.

    On Fedora (no RPM Fusion required):
        sudo dnf install bsdtar
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # (executable, command)
    candidates = [
        ("bsdtar", ["bsdtar", "-xf", str(rar_path), "-C", str(out_dir)]),
        ("unrar",  ["unrar", "x", "-y", str(rar_path), str(out_dir) + "/"]),
        ("unar",   ["unar", "-o", str(out_dir), "-D", str(rar_path)]),
        ("7z",     ["7z", "x", str(rar_path), f"-o{out_dir}", "-y"]),
    ]

    for exe, cmd in candidates:
        if shutil.which(exe) is None:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        # Tool found but failed — report and try next
        print(f"    [{exe} failed] {result.stderr.strip()[:120]}")

    raise RuntimeError(
        f"No working RAR extraction tool found for {rar_path.name}.\n"
        f"Install one of the following and retry:\n"
        f"  sudo dnf install bsdtar          # Fedora standard repo\n"
        f"  sudo dnf install unrar           # requires RPM Fusion"
    )


# ── Mat file reading ───────────────────────────────────────────────────────────

def _load_vibration(mat_path: Path, target_fs: int = TARGET_FS) -> np.ndarray | None:
    """
    Return the vibration channel from one Paderborn .mat file, resampled to
    ``target_fs`` (default 12 kHz).

    The Paderborn struct stores multiple sensor channels nested inside a MATLAB
    struct array. Following the same logic as the reference loader, this function
    searches fields at struct index 1 and 2 for all arrays longer than 200 000
    samples (≈ 64 kHz × 4 s) and returns the last one, which corresponds to
    the accelerometer (vibration) channel.

    The raw signal is then resampled from 64 kHz down to ``target_fs`` using
    ``scipy.signal.resample_poly`` (polyphase filtering) so that frequency-domain
    features are directly comparable with the CWRU pipeline.
    """
    mat = loadmat(str(mat_path))
    key = [k for k in mat if not k.startswith("_")][-1]
    root = mat[key][0][0]

    found: list[np.ndarray] = []
    for field_idx in (1, 2):
        if field_idx >= len(root):
            continue
        try:
            field = root[field_idx]
            for j in field:
                for i2 in j:
                    for i3 in i2:
                        for i4 in i3:
                            arr = np.asarray(i4).ravel()
                            if arr.size > 200_000:
                                found.append(arr.astype(np.float64))
        except (TypeError, IndexError, ValueError):
            continue

    if not found:
        return None

    sig = found[-1]
    # Resample 64 kHz → target_fs (e.g. 12 kHz) via exact rational ratio
    up, down = 3, 16  # 64_000 * 3 / 16 = 12_000
    if target_fs != PADERBORN_FS:
        sig = resample_poly(sig, up=up, down=down)
    return sig


# ── Per-bearing feature extraction ────────────────────────────────────────────

_COND_RE = __import__("re").compile(r"^(N\d+_M\d+_F\d+)_")


def _process_bearing(
    bearing_id : str,
    fault_type : str,
    experiment : str,
    bearing_dir: Path,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Process all .mat files in bearing_dir (flat layout — no subdirectories).

    The operating condition is embedded in each filename:
        N15_M07_F04_KA01_8.mat  →  condition = N15_M07_F04

    Metadata dicts satisfy extract_features' required keys:
        file_id, fault_type, fault_size, load
    An extra 'experiment' key is carried through for the output CSV.
    """
    all_segments: list[np.ndarray] = []
    all_meta    : list[dict]       = []

    fault_size = _FAULT_SIZE[experiment]
    mat_files  = sorted(bearing_dir.glob("*.mat"))

    if not mat_files:
        print(f"    [warn] No .mat files found in {bearing_dir}")
        return all_segments, all_meta

    # Group files by condition for cleaner progress output
    by_condition: dict[str, list[Path]] = {}
    for mat_path in mat_files:
        m = _COND_RE.match(mat_path.name)
        cond = m.group(1) if m else "unknown"
        by_condition.setdefault(cond, []).append(mat_path)

    for cond, files in sorted(by_condition.items()):
        load_code = _CONDITION_LOAD.get(cond, -1)
        seg_count = 0

        for mat_path in files:
            signal = _load_vibration(mat_path)
            if signal is None:
                print(f"    [warn] No vibration channel: {mat_path.name}")
                continue

            segs = segment_signal(signal, seg_len=None)  # uses module default (1024)
            if not segs:
                continue

            meta = {
                "file_id"   : bearing_id,
                "fault_type": fault_type,
                "fault_size": fault_size,
                "load"      : load_code,
                "experiment": experiment,
            }
            all_segments.extend(segs)
            all_meta.extend([meta] * len(segs))
            seg_count += len(segs)

        print(f"    {cond}  →  {len(files)} files, {seg_count} segments")

    return all_segments, all_meta


# ── Stratified subset sampling ────────────────────────────────────────────────

def sample_stratified_subset(
    csv_paths   : list[Path],
    experiments : list[str] = ("Real", "Healthy"),
    n_target    : int       = 9000,
    random_state: int       = 42,
    out         : Path | None = None,
) -> pd.DataFrame:
    """
    Load one or more feature CSVs, filter to the given experiment types,
    and draw a stratified sample that preserves the original class ratio.

    Parameters
    ----------
    csv_paths    : one or more paths to feature CSVs produced by this script
    experiments  : experiment column values to include (default: Real + Healthy)
    n_target     : approximate total rows in the output (actual may differ by ±1
                   per class due to rounding)
    random_state : RNG seed — fix this for reproducibility
    out          : if given, save the subset to this path as CSV

    Returns
    -------
    Shuffled, stratified subset DataFrame

    Example
    -------
    From the command line::

        python extract_paderborn.py sample \\
            --from paderborn_features.csv paderborn_healthy.csv \\
            --n 9000 --out paderborn_subset.csv
    """
    df = pd.concat(
        [pd.read_csv(p) for p in csv_paths],
        ignore_index=True,
    )

    pool = df[df["experiment"].isin(experiments)].copy()
    if pool.empty:
        raise ValueError(
            f"No rows matched experiments {list(experiments)}. "
            f"Available: {df['experiment'].unique().tolist()}"
        )

    counts  = pool["label"].value_counts().sort_index()
    ratios  = counts / counts.sum()
    # Proportional allocation with floor, then give leftover to largest class
    alloc   = (ratios * n_target).astype(int)
    leftover = n_target - alloc.sum()
    if leftover > 0:
        alloc[counts.idxmax()] += leftover

    print(f"Pool : {len(pool):,} rows from {pool['experiment'].unique().tolist()}")
    print(f"{'Label':<12}  {'Available':>10}  {'Ratio':>7}  {'Sampled':>9}")
    print("─" * 46)
    parts = []
    for label in counts.index:
        available = int(counts[label])
        n_sample  = min(int(alloc[label]), available)
        print(f"{label:<12}  {available:>10,}  {ratios[label]:>6.1%}  {n_sample:>9,}")
        parts.append(
            pool[pool["label"] == label].sample(n=n_sample, random_state=random_state)
        )

    result = (
        pd.concat(parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    total = len(result)
    print("─" * 46)
    print(f"{'Total':<12}  {'':>10}  {'':>7}  {total:>9,}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        print(f"\nSaved → {out}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and feature-extract the Paderborn Bearing Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
subcommand: sample
  Create a stratified subset from one or more existing feature CSVs.

  python extract_paderborn.py sample \\
      --from paderborn_features.csv paderborn_healthy.csv \\
      --n 9000 --out paderborn_subset.csv

  Filters to Real + Healthy experiments (IR / OR / Normal), computes the
  original class ratio, and draws proportionally so the ratio is preserved.
  Use --experiments to change which experiment groups are included.
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── 'sample' subcommand ───────────────────────────────────────────────────
    sp = subparsers.add_parser(
        "sample",
        help="Stratified subset from existing feature CSV(s)",
    )
    sp.add_argument(
        "--from", dest="from_csvs", nargs="+", type=Path, required=True,
        metavar="CSV",
        help="One or more feature CSVs to load and merge before sampling",
    )
    sp.add_argument(
        "--n", dest="n_target", type=int, default=9000,
        help="Target total rows in the subset (default: 9000)",
    )
    sp.add_argument(
        "--experiments", nargs="+", default=["Real", "Healthy"],
        metavar="EXP",
        help="Experiment types to include (default: Real Healthy)",
    )
    sp.add_argument(
        "--out", type=Path, default=Path("paderborn_subset.csv"),
        help="Output path for the subset CSV (default: paderborn_subset.csv)",
    )
    sp.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # ── extraction arguments (default command) ────────────────────────────────
    parser.add_argument(
        "--root", type=Path,
        default=Path.home() / "Datasets" / "Paderborn",
        help="Root directory for downloaded files (default: ~/Datasets/Paderborn)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("paderborn_features.csv"),
        help="Output CSV path (default: paderborn_features.csv)",
    )
    parser.add_argument(
        "--subset", choices=["Healthy", "Artificial", "Real"],
        default=None,
        help="Process only one experiment group (default: all)",
    )
    parser.add_argument(
        "--groups", nargs="+",
        choices=["time", "frequency", "wpd", "acf"],
        default=["time", "frequency", "wpd", "acf"],
        help="Feature groups to extract (default: all four)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip download; assume RAR files are already present",
    )
    args = parser.parse_args()

    # ── Dispatch: sample subcommand ───────────────────────────────────────────
    if args.command == "sample":
        sample_stratified_subset(
            csv_paths   = args.from_csvs,
            experiments = args.experiments,
            n_target    = args.n_target,
            random_state= args.seed,
            out         = args.out,
        )
        return

    bearings = (
        [b for b in BEARINGS if b[0] == args.subset]
        if args.subset else BEARINGS
    )
    if not bearings:
        print(f"No bearings matched subset '{args.subset}'.")
        sys.exit(1)

    print(f"Processing {len(bearings)} bearing(s).")
    print(f"Feature groups : {args.groups}")
    print(f"Download root  : {args.root}")
    print(f"Output CSV     : {args.out}")
    print()

    all_segments : list[np.ndarray] = []
    all_meta     : list[dict]       = []

    for experiment, fault_type, bearing_id, url in bearings:
        rar_dir    = args.root / experiment / fault_type
        rar_path   = rar_dir / f"{bearing_id}.rar"
        bearing_dir = rar_dir / bearing_id

        print(f"[{bearing_id}]  {experiment} / {fault_type}")

        # ── Download ──────────────────────────────────────────────────────────
        if not args.no_download:
            if not rar_path.exists():
                _download(url, rar_path)
            else:
                print(f"  Already downloaded: {rar_path.name}")
        elif not rar_path.exists():
            print(f"  [skip] RAR not found: {rar_path}")
            continue

        # ── Extract ───────────────────────────────────────────────────────────
        if not bearing_dir.exists():
            print(f"  Extracting {rar_path.name} …", end=" ", flush=True)
            _extract(rar_path, rar_dir)
            print("done")
        else:
            print(f"  Already extracted: {bearing_dir.name}/")

        # ── Feature extraction ────────────────────────────────────────────────
        segs, meta = _process_bearing(
            bearing_id, fault_type, experiment, bearing_dir
        )
        all_segments.extend(segs)
        all_meta.extend(meta)
        print(f"  Subtotal: {len(segs)} segments\n")

    if not all_segments:
        print("No segments extracted. Check that the data files downloaded correctly.")
        sys.exit(1)

    # ── Build feature DataFrame ───────────────────────────────────────────────
    print(f"Extracting features for {len(all_segments)} total segments …")

    # Capture experiment before calling extract_features — the library only
    # reads its own META_COLS (file_id, fault_type, fault_size, load) from
    # each dict, so the extra 'experiment' key would be dropped otherwise.
    experiments = [m["experiment"] for m in all_meta]

    df = extract_features(all_segments, all_meta, groups=args.groups, fs=TARGET_FS)

    # Add experiment column and rename fault_type → label
    df.insert(df.columns.get_loc("fault_type"), "experiment", experiments)
    df = df.rename(columns={"fault_type": "label"})

    # Final column order: all metadata first, then features
    meta_cols = ["file_id", "experiment", "fault_size", "load", "label"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {args.out}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    print()
    print("To use with the pipeline, add to experiment.yaml:")
    print(f"  data:")
    print(f"    file_path: {args.out}")
    print(f"    meta_cols: [file_id, experiment, fault_size, load]")
    print(f"    label_col: label")
    print(f"    feature_groups: null")


if __name__ == "__main__":
    main()
