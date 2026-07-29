#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4: Water quality risk assessment system.
Uses NTU exceedance magnitude and duration to classify risk into 4 levels.
Applies entropy weighting for indicator importance.
Outputs March 2026 daily classification to Excel.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    OUTPUT_DIR, NUMERIC_COLS,
    load_2025_data, load_2026_data,
    handle_missing_and_outliers,
    setup_chinese_font,
)

OUT_DIR = os.path.join(OUTPUT_DIR, "problem4")
os.makedirs(OUT_DIR, exist_ok=True)

# National standard
NTU_LIMIT = 1.0


def compute_daily_indicators(df, ntu_col="NTU"):
    """Compute daily risk indicators from NTU time series."""
    df = df.copy()
    df['date'] = df.index.date

    daily = []
    for date, group in df.groupby('date'):
        ntu_values = group[ntu_col].dropna().values

        if len(ntu_values) == 0:
            continue

        ntu_max = np.max(ntu_values)
        ntu_mean = np.mean(ntu_values)
        ntu_std = np.std(ntu_values)

        # Exceedance indicators
        exceed_mask = ntu_values > NTU_LIMIT
        exceed_hours = np.sum(exceed_mask) * 2  # each step = 2 hours
        exceed_count = np.sum(exceed_mask)

        # Magnitude indicator: max exceedance above limit
        exceed_magnitude = max(0, ntu_max - NTU_LIMIT)

        # Duration indicator: consecutive exceedances
        max_consecutive = 0
        current_streak = 0
        for v in exceed_mask:
            if v:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0

        # Cumulative exceedance (area above limit)
        cumulative_exceed = np.sum(np.maximum(0, ntu_values - NTU_LIMIT))

        daily.append({
            "date": date,
            "NTU_max": ntu_max,
            "NTU_mean": ntu_mean,
            "NTU_std": ntu_std,
            "exceed_hours": exceed_hours,
            "exceed_count": exceed_count,
            "max_consecutive_hours": max_consecutive * 2,
            "exceed_magnitude": exceed_magnitude,
            "cumulative_exceed": cumulative_exceed,
            "n_samples": len(ntu_values)
        })

    daily_df = pd.DataFrame(daily)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    return daily_df


def normalize_indicators(daily_df):
    """Normalize indicators to [0, 1] range."""
    df = daily_df.copy()

    indicators = ["exceed_magnitude", "exceed_hours", "NTU_max", "cumulative_exceed"]
    for col in indicators:
        if col not in df.columns:
            continue
        col_max = df[col].max()
        if col_max > 0:
            df[f"{col}_norm"] = df[col] / col_max
        else:
            df[f"{col}_norm"] = 0

    # Duration index (normalized by 24 hours)
    df["duration_index"] = np.minimum(df["exceed_hours"] / 24.0, 1.0)

    # Peak intensity (capped at 3.0 NTU as high risk reference)
    df["peak_intensity"] = np.minimum(df["NTU_max"] / 3.0, 1.0)

    return df


def entropy_weight(df, indicators):
    """
    Compute entropy-based weights for indicators.
    Higher entropy = lower weight (less discriminative).
    """
    n = len(df)
    weights = {}

    for col in indicators:
        if col not in df.columns:
            continue
        values = df[col].values
        # Shift to positive
        values = values - values.min() + 1e-10
        p = values / values.sum()
        # Entropy
        entropy = -np.sum(p * np.log(p + 1e-10)) / np.log(n)
        weights[col] = 1 - entropy

    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    return weights


def classify_risk(daily_df, weights):
    """Classify each day into 4 risk levels."""
    df = daily_df.copy()

    # Indicators used for scoring
    indicator_cols = [
        "exceed_magnitude_norm",
        "duration_index",
        "peak_intensity",
        "cumulative_exceed_norm"
    ]
    indicator_cols = [c for c in indicator_cols if c in df.columns]

    if not indicator_cols:
        df["risk_score"] = 0
        df["risk_level"] = "安全"
        return df

    # Compute weighted risk score
    available_weights = {k: v for k, v in weights.items()
                         if f"{k}_norm" in df.columns or k in df.columns}

    # Map weight keys to column names
    weight_map = {
        "exceed_magnitude": "exceed_magnitude_norm",
        "exceed_hours": "duration_index",
        "NTU_max": "peak_intensity",
        "cumulative_exceed": "cumulative_exceed_norm"
    }

    df["risk_score"] = 0.0
    for key, col in weight_map.items():
        if key in available_weights and col in df.columns:
            df["risk_score"] += available_weights[key] * df[col]

    # Fallback: equal weights if entropy failed
    if df["risk_score"].max() == 0:
        for col in indicator_cols:
            if col in df.columns:
                df["risk_score"] += df[col]
        df["risk_score"] /= len(indicator_cols)

    # Classify
    scores = df["risk_score"].values
    safe_mask = (df["NTU_max"] <= NTU_LIMIT)

    # For non-safe, use percentile thresholds
    risky_scores = scores[~safe_mask]
    if len(risky_scores) > 0:
        p33 = np.percentile(risky_scores, 33)
        p67 = np.percentile(risky_scores, 67)
    else:
        p33 = 0.3
        p67 = 0.6

    def assign_level(row):
        if row["NTU_max"] <= NTU_LIMIT:
            return "安全"

        # Hard triggers
        if row["NTU_max"] > 5.0:
            return "高风险"
        if row["NTU_max"] > 3.0:
            if row["risk_score"] <= p33:
                return "中风险"
            elif row["risk_score"] <= p67:
                return "中风险"
            else:
                return "高风险"

        # Normal classification
        score = row["risk_score"]
        if score <= p33:
            return "低风险"
        elif score <= p67:
            return "中风险"
        else:
            return "高风险"

    df["risk_level"] = df.apply(assign_level, axis=1)
    return df


def main():
    print("=" * 60)
    print("Problem 4: Water Quality Risk Assessment System")
    print("=" * 60)

    # Configure Chinese font for plots
    setup_chinese_font()

    # Load 2026 data
    print("\n[1/5] Loading 2026 data...")
    df_2026 = load_2026_data()

    # Handle missing NTU values
    all_num = [c for c in NUMERIC_COLS if c in df_2026.columns]
    df_2026 = handle_missing_and_outliers(df_2026, all_num)

    print(f"  2026 data shape: {df_2026.shape}")
    print(f"  Date range: {df_2026.index.min()} to {df_2026.index.max()}")
    print(f"  NTU stats: mean={df_2026['NTU'].mean():.3f}, "
          f"max={df_2026['NTU'].max():.3f}")
    print(f"  NTU > 1.0: {(df_2026['NTU'] > NTU_LIMIT).sum()} points ("
          f"{(df_2026['NTU'] > NTU_LIMIT).mean()*100:.1f}%)")

    # Compute daily indicators
    print("\n[2/5] Computing daily risk indicators...")
    daily_df = compute_daily_indicators(df_2026)
    daily_df = normalize_indicators(daily_df)
    print(f"  Days with data: {len(daily_df)}")

    # Entropy weighting
    print("\n[3/5] Entropy-based weight determination...")
    entropy_indicators = ["exceed_magnitude", "exceed_hours", "NTU_max", "cumulative_exceed"]
    weights = entropy_weight(daily_df, entropy_indicators)

    print("\n  --- Indicator Weights (Entropy Method) ---")
    for k, v in weights.items():
        print(f"  {k:20s}: {v:.4f}")
    weight_df = pd.DataFrame(
        [{"indicator": k, "weight": v} for k, v in weights.items()]
    )
    weight_df.to_csv(os.path.join(OUT_DIR, "entropy_weights.csv"), index=False)

    # Also show equal weights for comparison
    print("\n  Equal weights (for comparison):")
    n_indicators = len(weights)
    for k in weights:
        print(f"  {k:20s}: {1/n_indicators:.4f}")

    # Risk classification
    print("\n[4/5] Classifying risk levels...")
    daily_df = classify_risk(daily_df, weights)

    # Statistics by month
    daily_df["month"] = daily_df["date"].dt.month
    level_order = ["安全", "低风险", "中风险", "高风险"]

    print("\n  --- Risk Level Distribution ---")
    total_days = len(daily_df)
    for level in level_order:
        count = (daily_df["risk_level"] == level).sum()
        pct = count / total_days * 100 if total_days > 0 else 0
        print(f"  {level:12s}: {count:3d} days ({pct:5.1f}%)")

    # Monthly breakdown
    print("\n  --- Monthly Risk Distribution ---")
    monthly_stats = []
    for month in sorted(daily_df["month"].unique()):
        month_data = daily_df[daily_df["month"] == month]
        month_total = len(month_data)
        row = {"month": int(month)}
        for level in level_order:
            count = (month_data["risk_level"] == level).sum()
            row[level] = count
        monthly_stats.append(row)
        print(f"  Month {int(month)}: ", end="")
        print(", ".join(f"{level}={row.get(level, 0)}" for level in level_order))

    monthly_df = pd.DataFrame(monthly_stats)
    monthly_df.to_csv(os.path.join(OUT_DIR, "monthly_risk_distribution.csv"), index=False)

    # March detailed classification
    print("\n[5/5] Generating March detailed classification...")
    march_data = daily_df[daily_df["month"] == 3].copy()

    # Prepare Excel output
    excel_cols = [
        "date", "NTU_max", "NTU_mean", "exceed_hours",
        "max_consecutive_hours", "risk_score", "risk_level"
    ]
    march_out = march_data[[c for c in excel_cols if c in march_data.columns]]
    march_out["date"] = march_out["date"].dt.strftime("%Y-%m-%d")

    # Add remarks for high risk or hard trigger days (English to avoid font issues in Excel)
    march_out["备注"] = ""
    for idx, row in march_out.iterrows():
        remarks_list = []
        if row["NTU_max"] > 5.0:
            remarks_list.append(f"NTU峰值={row['NTU_max']:.2f}>5.0, force High Risk")
        elif row["NTU_max"] > 3.0:
            remarks_list.append(f"NTU峰值={row['NTU_max']:.2f}>3.0")
        if row["exceed_hours"] > 6:
            remarks_list.append(f"Exceed {int(row['exceed_hours'])}h consecutively")
        march_out.at[idx, "备注"] = "; ".join(remarks_list)

    march_out.to_excel(os.path.join(OUT_DIR, "march_2026_risk_classification.xlsx"),
                       index=False)
    print(f"  March days: {len(march_out)}")
    print("  Sample:")
    print(march_out.head(5).to_string(index=False))

    # Plots
    print("\n  Generating plots...")
    setup_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chinese risk labels for plots
    zh_labels = {"安全": "安全", "低风险": "低风险",
                 "中风险": "中风险", "高风险": "高风险"}
    color_map = {"安全": "green", "低风险": "yellow",
                 "中风险": "orange", "高风险": "red"}

    # 1. NTU time series with risk background
    ax1 = axes[0, 0]
    for _, row in daily_df.iterrows():
        color = color_map.get(row["risk_level"], "gray")
        d = row["date"]
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.3, color=color)

    ax1.plot(df_2026.index, df_2026["NTU"].values, 'b-', linewidth=0.8, alpha=0.7)
    ax1.axhline(y=NTU_LIMIT, color='red', linestyle='--', label=f'NTU={NTU_LIMIT}')
    ax1.set_title("NTU时间序列与风险分级")
    ax1.set_ylabel("NTU")
    ax1.legend()

    # Add custom legend for risk levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='安全'),
        Patch(facecolor='yellow', alpha=0.3, label='低风险'),
        Patch(facecolor='orange', alpha=0.3, label='中风险'),
        Patch(facecolor='red', alpha=0.3, label='高风险'),
    ]
    ax1.legend(handles=[plt.Line2D([0], [0], color='b', linewidth=0.8, label='NTU'),
                        plt.Line2D([0], [0], color='red', linestyle='--', label=f'NTU={NTU_LIMIT}')] +
               legend_elements, loc='upper left')

    # 2. Risk level distribution pie chart
    ax2 = axes[0, 1]
    level_counts = [daily_df["risk_level"].value_counts().get(l, 0) for l in level_order]
    colors_pie = ['green', 'yellow', 'orange', 'red']
    ax2.pie(level_counts, labels=level_order, colors=colors_pie, autopct='%1.1f%%')
    ax2.set_title("风险等级分布")

    # 3. Risk score distribution
    ax3 = axes[1, 0]
    for i, level in enumerate(level_order):
        level_scores = daily_df[daily_df["risk_level"] == level]["risk_score"]
        ax3.hist(level_scores, bins=20, alpha=0.5, color=colors_pie[i], label=level)
    ax3.set_xlabel("风险分数")
    ax3.set_ylabel("频次")
    ax3.set_title("风险等级-分数分布")
    ax3.legend()

    # 4. Monthly stacked bar chart
    ax4 = axes[1, 1]
    months = sorted(daily_df["month"].unique())
    bar_data = {}
    for level in level_order:
        bar_data[level] = [daily_df[(daily_df["month"] == m) & (daily_df["risk_level"] == level)].shape[0]
                            for m in months]

    bottom = np.zeros(len(months))
    for i, level in enumerate(level_order):
        ax4.bar(months, bar_data[level], bottom=bottom, color=colors_pie[i],
                label=level, alpha=0.8)
        bottom += np.array(bar_data[level])
    ax4.set_xlabel("月份")
    ax4.set_ylabel("天数")
    ax4.set_title("月度风险等级分布")
    ax4.set_xticks(months)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "problem4_risk_assessment.png"), dpi=150)
    plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
