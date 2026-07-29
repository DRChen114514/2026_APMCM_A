#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for 2025 MCM A: Water Quality Prediction & Assessment.
Handles data loading, date parsing, preprocessing, and common constants.
"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

# --- Paths ---
DATA_DIR_2025 = os.path.expanduser(
    "/home/violet/Workspace/Data/2025MCM_A/附件1  2025数据集"
)
DATA_DIR_2026 = os.path.expanduser(
    "/home/violet/Workspace/Data/2025MCM_A/附件2  2026数据集"
)
OUTPUT_DIR = os.path.expanduser(
    "/home/violet/Workspace/Code/Competition_code/2025_MCM_A/output"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Chinese font support ---
_CHINESE_FONT_SETUP = False


def setup_chinese_font():
    """Configure matplotlib to render Chinese characters via Noto Sans CJK SC."""
    global _CHINESE_FONT_SETUP
    if _CHINESE_FONT_SETUP:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager

    zh_candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK TC",
        "AR PL UKai CN", "AR PL UMing CN",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in fontManager.ttflist}
    chosen = None
    for f in zh_candidates:
        if f in available:
            chosen = f
            break
    if chosen is None:
        for f in sorted(available):
            if any(k in f.lower() for k in ("cjk", "kai", "ming")):
                chosen = f
                break
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    _CHINESE_FONT_SETUP = True
    print(f"  [font] Chinese renderer: {chosen or 'fallback'}")


# --- Column definitions ---
COLS_2025 = [
    "DATE", "TIME", "RIVER_LEVEL", "RW_PUMP_DUTY", "RW_FLOW",
    "RW_NTU", "RW_CLR", "RW_PH", "FILT_NTU", "CW_WELL_LEVEL",
    "PH", "NTU", "CLR", "CL2", "FRIDE", "ALUM",
    "TW_PUMP_DUTY", "TW_FLOW", "18ML_LEVEL", "18ML_FLOW", "REMARKS"
]

COLS_2026 = [
    "TIME", "RIVER_LEVEL", "RW_PUMP_DUTY", "RW_FLOW",
    "RW_NTU", "RW_CLR", "RW_PH", "FILT_NTU", "CW_WELL_LEVEL",
    "PH", "NTU", "CLR", "CL2", "FRIDE", "ALUM",
    "TW_PUMP_DUTY", "TW_FLOW", "18ML_LEVEL", "18ML_FLOW", "REMARKS"
]

NUMERIC_COLS = [
    "RIVER_LEVEL", "RW_PUMP_DUTY", "RW_FLOW", "RW_NTU", "RW_CLR", "RW_PH",
    "FILT_NTU", "CW_WELL_LEVEL", "PH", "NTU", "CLR", "CL2",
    "FRIDE", "ALUM", "TW_PUMP_DUTY", "TW_FLOW", "18ML_LEVEL", "18ML_FLOW"
]


def parse_date_robust(date_str):
    """
    Parse the DATE field from 2025 data files.

    Handles 4 formats found across the 12 monthly files:
      1. 'YYYY-MM-DD HH:MM:SS' — days 1-12 with swapped MM/DD
         (Jan, Feb, Mar, Apr, Sep, Dec files)
      2. 'YYYY-MM-DD' (no time) — correct format
         (May, Jun, Jul, Aug files)
      3. 'DD/MM/YYYY' — correct format, days 13-end
         (all files)
      4. Integer (Excel 1900 date serial) — days 1-12
         (Oct, Nov files)

    Returns a pd.Timestamp.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    date_str = str(date_str).strip()

    # Pattern 1: 'DD/MM/YYYY' (correct, used for days 13-31)
    m1 = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if m1:
        day, month, year = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        return pd.Timestamp(year=year, month=month, day=day)

    # Pattern 2: 'YYYY-MM-DD HH:MM:SS' with swapped month-day
    m2 = re.match(r'^(\d{4})-(\d{2})-(\d{2})\s+\d{2}:\d{2}:\d{2}$', date_str)
    if m2:
        year = int(m2.group(1))
        fake_month = int(m2.group(2))  # Actually the DAY
        fake_day = int(m2.group(3))    # Actually the MONTH
        real_day = fake_month
        real_month = fake_day
        return pd.Timestamp(year=year, month=real_month, day=real_day)

    # Pattern 3: 'YYYY-MM-DD' without time (correct, no swap)
    m3 = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_str)
    if m3:
        year, month, day = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        return pd.Timestamp(year=year, month=month, day=day)

    # Pattern 4: Integer (Excel 1900 date system serial)
    try:
        serial = int(float(date_str))
        # Excel epoch: Dec 30, 1899 (serial 0)
        base = datetime(1899, 12, 30)
        dt = base + timedelta(days=serial)
        return pd.Timestamp(dt)
    except (ValueError, OverflowError):
        pass

    raise ValueError(f"Cannot parse DATE: '{date_str}'")


def parse_time_robust(time_val):
    """
    Parse TIME field (integer like 700, 900, ..., 2300, 100, 300, 500).
    Returns (hour, minute) tuple.
    """
    t = int(float(str(time_val).replace(',', '').strip()))
    hour = t // 100
    minute = t % 100
    return hour, minute


def load_2025_data():
    """
    Load and merge all 2025 monthly data files.
    Returns a DataFrame with proper datetime index and all numeric columns cleaned.
    """
    files = sorted(glob(os.path.join(DATA_DIR_2025, "JBALB_*.txt")))
    if not files:
        raise FileNotFoundError(f"No 2025 data files found in {DATA_DIR_2025}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t', encoding='utf-8', names=COLS_2025, skiprows=1,
                         na_values=['', '-', ' '])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Parse dates
    dates = []
    for _, row in df.iterrows():
        try:
            d = parse_date_robust(row["DATE"])
        except Exception:
            d = pd.NaT
        dates.append(d)
    df["parsed_date"] = dates

    # Parse times
    times = []
    for t in df["TIME"]:
        try:
            h, m = parse_time_robust(t)
            times.append((h, m))
        except Exception:
            times.append((0, 0))
    hours = [t[0] for t in times]
    minutes = [t[1] for t in times]

    # Build datetime index
    datetimes = []
    for d, h, m in zip(df["parsed_date"], hours, minutes):
        if pd.notna(d):
            datetimes.append(d + pd.Timedelta(hours=int(h), minutes=int(m)))
        else:
            datetimes.append(pd.NaT)
    df["datetime"] = pd.to_datetime(datetimes)

    # Drop rows with invalid datetime
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop helper and original date columns
    drop_cols = ["DATE", "TIME", "REMARKS", "parsed_date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Remove duplicate index entries
    df = df[~df.index.duplicated(keep='first')]

    return df


def load_2026_data():
    """
    Load 2026 data files. Each file contains ~12 time points for a partial day.
    The 2026 data has NO DATE column; date is inferred from filename.

    Returns a DataFrame with datetime index.
    """
    files = sorted(glob(os.path.join(DATA_DIR_2026, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No 2026 data files found in {DATA_DIR_2026}")

    # Map filenames to base dates
    date_map = {
        "2026年1月_15.01": "2026-01-15",
        "2026年2月_01.02": "2026-02-01",
        "2026年3月_23.03": "2026-03-23",
    }

    dfs = []
    for f in files:
        basename = os.path.basename(f).replace('.txt', '').strip()
        base_date_str = None
        for key, date_str in date_map.items():
            if key in basename:
                base_date_str = date_str
                break
        if base_date_str is None:
            base_date_str = "2026-01-01"

        df = pd.read_csv(f, sep='\t', encoding='utf-8', names=COLS_2026, skiprows=1,
                         na_values=['', '-', ' '])

        base_date = pd.Timestamp(base_date_str)

        # Parse TIME and build datetime
        datetimes = []
        for _, row in df.iterrows():
            try:
                h, m = parse_time_robust(row["TIME"])
                dt = base_date + pd.Timedelta(hours=int(h), minutes=int(m))
                # Handle midnight crossing
                if h < 6 and len(datetimes) > 0:
                    last_h = parse_time_robust(
                        df.loc[df.index[len(datetimes)-1], "TIME"] if len(datetimes) > 0 else "0"
                    )[0]
                    if last_h > 18:
                        dt = dt + pd.Timedelta(days=1)
                datetimes.append(dt)
            except Exception:
                datetimes.append(pd.NaT)
        df["datetime"] = pd.to_datetime(datetimes)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop helper columns
    drop_cols = ["TIME", "REMARKS"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Remove duplicate index entries
    df = df[~df.index.duplicated(keep='first')]

    return df


def handle_missing_and_outliers(df, columns, iqr_factor=3.0):
    """
    Handle missing values and outliers:
    1. Linear interpolation for missing
    2. Forward/backward fill for remaining
    3. IQR-based outlier replacement with NaN, then interpolation
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].ffill().bfill()

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        df.loc[outlier_mask, col] = np.nan
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].ffill().bfill()

    return df


def add_temporal_features(df):
    """Add time-based features."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df.index.month
    df['day'] = df.index.day
    return df


def add_lag_features(df, cols_to_lag, lag_steps):
    """Add lag features for specified columns."""
    df = df.copy()
    for col in cols_to_lag:
        if col not in df.columns:
            continue
        for lag in lag_steps:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.bfill().ffill()
    return df


def add_diff_features(df, cols, steps=1):
    """Add first-difference features."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_diff{steps}"] = df[col].diff(steps)
    df = df.bfill().ffill()
    return df


if __name__ == "__main__":
    print("Loading 2025 data...")
    df_2025 = load_2025_data()
    print(f"2025 shape: {df_2025.shape}")
    print(f"Date range: {df_2025.index.min()} to {df_2025.index.max()}")
    print(f"\nFirst 3 rows:\n{df_2025.head(3)}")

    print("\n" + "=" * 60)
    print("Loading 2026 data...")
    df_2026 = load_2026_data()
    print(f"2026 shape: {df_2026.shape}")
    print(f"Date range: {df_2026.index.min()} to {df_2026.index.max()}")
    print(f"\nFirst 3 rows:\n{df_2026.head(3)}")

    print("\nMissing values in 2025 key columns:")
    for col in ["RW_NTU", "FILT_NTU", "NTU", "ALUM", "RW_PH", "RW_FLOW"]:
        if col in df_2025.columns:
            print(f"  {col}: {df_2025[col].isna().sum()} missing")
