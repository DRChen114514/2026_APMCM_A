#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2: Dynamic time-lag model for filtered water turbidity (FILT.NTU).
Uses Cross-Correlation Function (CCF) for lag identification,
then builds a NARX neural network with identified time lags.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from scipy import signal
# Using scipy.signal.correlate instead of statsmodels CCF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    OUTPUT_DIR, NUMERIC_COLS,
    load_2025_data,
    handle_missing_and_outliers,
    setup_chinese_font,
)

OUT_DIR = os.path.join(OUTPUT_DIR, "problem2")
os.makedirs(OUT_DIR, exist_ok=True)

# Key variables for Problem 2
TARGET = "FILT_NTU"
INPUT_VARS = ["RW_NTU", "RW_PH", "ALUM", "RW_FLOW"]
MAX_LAG = 12  # max lag in time steps (24 hours)


def compute_ccf_lags(df, target, inputs, max_lag=MAX_LAG):
    """
    Compute cross-correlation function between each input and target.
    Uses pre-whitening via AR fitting to avoid spurious correlation.
    Returns a DataFrame with optimal lags and CCF values.
    """
    results = []

    # Ensure no NaN
    cols = [target] + inputs
    data = df[cols].copy()
    data = data.ffill().bfill().fillna(0)

    y = data[target].values

    for inp in inputs:
        x = data[inp].values

        # Remove trend (first difference)
        y_diff = np.diff(y, prepend=y[0])
        x_diff = np.diff(x, prepend=x[0])

        # Compute CCF at lags k=0..max_lag
        ccf_values = []
        for k in range(max_lag + 1):
            if k == 0:
                corr = np.corrcoef(x_diff, y_diff)[0, 1]
            else:
                corr = np.corrcoef(x_diff[:-k], y_diff[k:])[0, 1]
            ccf_values.append(corr)

        ccf_arr = np.array(ccf_values)
        best_lag = np.argmax(np.abs(ccf_arr))
        best_ccf = ccf_arr[best_lag]

        results.append({
            "variable": inp,
            "optimal_lag_steps": best_lag,
            "optimal_lag_hours": best_lag * 2,
            "ccf_value": best_ccf,
            "direction": "positive" if best_ccf > 0 else "negative"
        })

        # Plot CCF
        plt.figure(figsize=(8, 3))
        plt.stem(range(len(ccf_arr)), ccf_arr, basefmt=" ")
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=best_lag, color='red', linestyle='--',
                    label=f'Best lag={best_lag} ({best_lag*2}h)')
        plt.xlabel("Lag (2-hour steps)")
        plt.ylabel("CCF")
        plt.title(f"CCF: {inp} -> {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"ccf_{inp}.png"), dpi=120)
        plt.close()

    lag_df = pd.DataFrame(results)
    return lag_df


def build_narx_features(df, lag_config):
    """
    Build feature matrix for NARX model with specified lags.
    lag_config: dict {variable: [lag_steps]}
    """
    df = df.copy()
    features = []

    # Target autoregressive terms
    for lag in [1, 2, 3]:
        col_name = f"{TARGET}_lag{lag}"
        df[col_name] = df[TARGET].shift(lag)
        features.append(col_name)

    # Exogenous variables with identified lags
    for var, lags in lag_config.items():
        if var not in df.columns:
            continue
        for lag in lags:
            col_name = f"{var}_lag{lag}"
            df[col_name] = df[var].shift(lag)
            features.append(col_name)

    # Also include current values
    for var in INPUT_VARS:
        if var in df.columns:
            features.append(var)

    df = df.ffill().bfill().fillna(0)

    X = df[features].values
    y = df[TARGET].values

    return X, y, features


def train_narx(X_train, y_train, X_val, y_val):
    """Train NARX neural network."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
        verbose=False
    )
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_val = model.predict(X_val_s)

    return model, scaler, y_pred_train, y_pred_val


def linear_baseline(X_train, y_train, X_test, y_test):
    """Simple linear regression with lag features as baseline."""
    from sklearn.linear_model import LinearRegression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    return y_pred


def persistence_baseline(y_train, y_test):
    """Persistence model: f(t) = f(t-1)."""
    # Use last training value as prediction
    return np.full_like(y_test, y_train[-1])


def main():
    setup_chinese_font()
    print("=" * 60)
    print("Problem 2: Dynamic Time-Lag Model for FILT.NTU")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading and preparing data...")
    df_2025 = load_2025_data()

    # Handle missing values for key variables
    key_cols = [TARGET] + INPUT_VARS
    df_2025 = handle_missing_and_outliers(df_2025, key_cols)
    df_clean = df_2025[key_cols].copy()
    df_clean = df_clean.ffill().bfill().fillna(0)

    print(f"  Clean data shape: {df_clean.shape}")
    print(f"  FILT.NTU range: [{df_clean[TARGET].min():.2f}, "
          f"{df_clean[TARGET].max():.2f}]")

    # Step 1: CCF lag identification
    print("\n[2/5] Computing cross-correlation for lag identification...")
    lag_df = compute_ccf_lags(df_clean, TARGET, INPUT_VARS)

    print("\n  --- Identified Time Lags ---")
    print(lag_df.to_string(index=False))
    lag_df.to_csv(os.path.join(OUT_DIR, "lag_identification.csv"), index=False)

    # Build lag configuration
    lag_config = {}
    for _, row in lag_df.iterrows():
        var = row["variable"]
        opt_lag = int(row["optimal_lag_steps"])
        # Include optimal lag and adjacent lags for robustness
        lags = sorted(set([max(0, opt_lag - 1), opt_lag, min(MAX_LAG, opt_lag + 1)]))
        lag_config[var] = lags

    print("\n  Lag configuration for NARX:")
    for var, lags in lag_config.items():
        print(f"    {var}: lags {lags} (hours: {[l*2 for l in lags]})")

    # Step 2: Build NARX features
    print("\n[3/5] Building NARX features...")
    X, y, feature_names = build_narx_features(df_clean, lag_config)
    print(f"  Feature matrix: {X.shape}, {len(feature_names)} features")

    # Step 3: Split and train
    print("\n[4/5] Training NARX model...")
    n_train = int(0.7 * len(X))
    n_val = int(0.85 * len(X))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train NARX
    model, scaler, y_pred_train, y_pred_val = train_narx(
        X_train, y_train, X_val, y_val
    )

    # Predict on test set
    X_test_s = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_s)

    # Baselines
    y_pred_lin = linear_baseline(X_train, y_train, X_test, y_test)
    y_pred_pers = persistence_baseline(y_train, y_test)

    # Step 4: Evaluate
    print("\n[5/5] Model evaluation...")

    def compute_metrics(y_true, y_pred, name=""):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

    metrics = [
        compute_metrics(y_test, y_pred_test, "NARX"),
        compute_metrics(y_test, y_pred_lin, "Linear+Lags"),
        compute_metrics(y_test, y_pred_pers, "Persistence"),
    ]
    metrics_df = pd.DataFrame(metrics)
    print("\n  --- Model Comparison ---")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

    # Residual analysis
    residuals = y_test - y_pred_test

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Actual vs Predicted (test set)
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'r--')
    axes[0, 0].set_xlabel("FILT.NTU 实际值")
    axes[0, 0].set_ylabel("FILT.NTU 预测值")
    axes[0, 0].set_title(f"NARX 测试集\nRMSE={metrics[0]['RMSE']:.4f}, "
                         f"R2={metrics[0]['R2']:.3f}")

    # 2. Residuals over time (test set)
    axes[0, 1].plot(residuals, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].fill_between(range(len(residuals)), -0.1, 0.1, alpha=0.1)
    axes[0, 1].set_title("残差")
    axes[0, 1].set_xlabel("测试样本序号")
    axes[0, 1].set_ylabel("残差 (实际值 - 预测值)")

    # 3. Time series comparison (sample of test set)
    n_plot = min(200, len(y_test))
    axes[1, 0].plot(y_test[:n_plot], 'b-', label="实际值", alpha=0.8,
                    linewidth=0.8)
    axes[1, 0].plot(y_pred_test[:n_plot], 'r--', label="NARX 预测值",
                    alpha=0.8, linewidth=0.8)
    axes[1, 0].set_title("FILT.NTU 实际值 vs 预测值 (测试集)")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("时间步")
    axes[1, 0].set_ylabel("FILT.NTU")

    # 4. Residual histogram
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_title(f"残差分布\n"
                         f"均值={residuals.mean():.4f}, "
                         f"标准差={residuals.std():.4f}")
    axes[1, 1].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "problem2_results.png"), dpi=150)
    plt.close()

    # Save final lag parameters
    print("\n  --- Final Time Lag Parameters ---")
    final_lag = lag_df[["variable", "optimal_lag_steps", "optimal_lag_hours",
                         "ccf_value", "direction"]].copy()
    final_lag["physical_interpretation"] = [
        "原水浊度经混凝沉淀过滤后到达滤后检测点",
        "pH影响混凝效果，需经混合反应后才显现",
        "矾投加后经混合-絮凝-沉淀-过滤才见效果",
        "流量变化影响水力停留时间，进而影响去除效果"
    ]
    print(final_lag.to_string(index=False))
    final_lag.to_csv(os.path.join(OUT_DIR, "final_lag_parameters.csv"), index=False)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
