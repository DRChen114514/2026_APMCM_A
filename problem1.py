#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1: NTU turbidity prediction with stationarity assessment.
Approach:
  1. Stationarity test (ADF) on 2025 NTU series
  2. If stationary: mean/median as baseline prediction + bootstrap confidence intervals
  3. Rolling-window statistics for time-local prediction
  4. Time-series anomaly detection (Isolation Forest + IQR)
  5. Supplementary ML-based feature importance (Spearman, SHAP, XGBoost)
  6. Predict NTU for 2026 snapshot dates with confidence intervals
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DATA_DIR_2025, DATA_DIR_2026, OUTPUT_DIR, NUMERIC_COLS,
    load_2025_data, load_2026_data,
    handle_missing_and_outliers, add_temporal_features,
    add_lag_features, add_diff_features,
    setup_chinese_font,
)

OUT_DIR = os.path.join(OUTPUT_DIR, "problem1")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────

def bootstrap_confidence_interval(data, stat_func=np.median, n_boot=2000, alpha=0.05):
    """Bootstrap CI for a statistic of a 1-D array."""
    n = len(data)
    boot_stats = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return stat_func(data), lo, hi


def prepare_data(df, target_col="NTU"):
    """Prepare features and target (same as original for comparability)."""
    all_num = [c for c in NUMERIC_COLS if c in df.columns and c != target_col]
    df = handle_missing_and_outliers(df, all_num + [target_col])
    df = add_temporal_features(df)

    lag_config = {
        "RW_NTU": [1, 2, 3], "ALUM": [3, 4, 5, 6],
        "RW_FLOW": [1, 2], "RW_PH": [1, 2], "FILT_NTU": [1, 2],
    }
    for col, lags in lag_config.items():
        if col in df.columns:
            df = add_lag_features(df, [col], lags)
    df = add_diff_features(df, ["RW_NTU"], steps=1)

    if "RW_NTU" in df.columns and "ALUM" in df.columns:
        df["RW_NTU_x_ALUM"] = df["RW_NTU"] * df["ALUM"]
        df["ALUM_div_RW_NTU"] = df["ALUM"] / (df["RW_NTU"] + 0.01)

    df = df.dropna(subset=[target_col])

    exclude = [target_col, "CLR", "CL2", "18ML_LEVEL", "18ML_FLOW",
               "TW_PUMP_DUTY", "TW_FLOW", "CW_WELL_LEVEL",
               "RIVER_LEVEL", "RW_PUMP_DUTY"]
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].isna().sum() < len(df) * 0.5
                 and not c.startswith("_")]

    X = df[feat_cols].copy()
    y = df[target_col].copy()
    X = X.ffill().bfill().fillna(0)
    return X, y, feat_cols, df


# ── anomaly detection ────────────────────────────────────────────────

def anomaly_detection_2025(df_2025, target_col="NTU"):
    """Detect anomalous NTU points in 2025 using Isolation Forest + IQR."""
    ntu = df_2025[target_col].dropna().values.reshape(-1, 1)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = iso.fit_predict(ntu)
    iso_anomaly = iso_labels == -1

    # IQR method (supplementary)
    q1, q3 = np.percentile(ntu, 25), np.percentile(ntu, 75)
    iqr = q3 - q1
    lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    iqr_anomaly = (ntu.flatten() < lower) | (ntu.flatten() > upper)

    # Combined: anomaly if detected by either method
    combined = iso_anomaly | iqr_anomaly

    results = {
        "isolation_forest_anomalies": int(iso_anomaly.sum()),
        "iqr_anomalies": int(iqr_anomaly.sum()),
        "combined_anomalies": int(combined.sum()),
        "total_points": len(ntu),
        "anomaly_rate": combined.mean().item(),
        "iqr_lower": lower, "iqr_upper": upper,
    }
    return results, combined, df_2025.index


# ── stationarity tests ────────────────────────────────────────────────

def _adf_test_numpy(y, max_lag=None):
    """Augmented Dickey-Fuller test using pure numpy (no statsmodels).

    Regresses:  Δy_t = α + γ·y_{t-1} + Σ δ_j·Δy_{t-j} + ε_t
    Tests H0: γ=0  (unit root, non-stationary).
    """
    from numpy.linalg import lstsq as nplstsq
    y = np.asarray(y, dtype=float)
    n = len(y)
    if max_lag is None:
        max_lag = int(np.floor(12.0 * (n / 100.0) ** 0.25))
    max_lag = min(max_lag, n - 3)

    dy = np.diff(y)   # length n-1

    best_bic = np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        n_eff = n - lag - 1
        if n_eff < 30:
            continue
        # Build design matrix
        cols = [np.ones(n_eff), y[lag - 1 : n - 2]]
        for j in range(1, lag + 1):
            cols.append(dy[lag - j : n - 1 - j])
        X = np.column_stack(cols)
        try:
            coef, resid, _, _ = nplstsq(X, dy[lag:n - 1], rcond=None)
            sse = resid[0] if len(resid) > 0 else float(np.sum((dy[lag:n - 1] - X @ coef) ** 2))
            bic = n_eff * np.log(sse / n_eff) + np.log(n_eff) * (lag + 2)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except Exception:
            continue

    lag = best_lag
    n_eff = n - lag - 1

    cols = [np.ones(n_eff), y[lag - 1 : n - 2]]
    for j in range(1, lag + 1):
        cols.append(dy[lag - j : n - 1 - j])
    X = np.column_stack(cols)

    coef, resid, _, _ = nplstsq(X, dy[lag:n - 1], rcond=None)
    gamma = coef[1]
    resid_vals = dy[lag:n - 1] - X @ coef
    sigma2 = np.sum(resid_vals ** 2) / (n_eff - X.shape[1])

    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {"ADF_stat": np.nan, "ADF_pvalue": np.nan,
                "ADF_crit_5pct": np.nan, "lag_used": lag, "n_obs": n}

    se_gamma = np.sqrt(cov.diagonal()[1])
    if se_gamma <= 0 or np.isnan(se_gamma):
        return {"ADF_stat": np.nan, "ADF_pvalue": np.nan,
                "ADF_crit_5pct": np.nan, "lag_used": lag, "n_obs": n}

    adf_stat = gamma / se_gamma

    # Critical values (MacKinnon 2010 surface approximation)
    if n > 500:
        crit_5pct = -2.86
    elif n > 100:
        crit_5pct = -2.89
    else:
        crit_5pct = -2.95

    # Smooth p-value approximation
    adf_p = 1.0 / (1.0 + np.exp(-3.5 * (-adf_stat - 0.6)))

    return {"ADF_stat": adf_stat, "ADF_pvalue": adf_p,
            "ADF_crit_5pct": crit_5pct, "lag_used": lag, "n_obs": n}


def stationarity_tests(series, name="NTU"):
    """Run ADF test (pure numpy) and report stationarity."""
    clean = series.dropna().values.astype(float)
    if len(clean) < 20:
        return {"name": name, "error": "insufficient data"}

    adf = _adf_test_numpy(clean)

    # Also run Phillips-Perron style check: variance ratio
    dy = np.diff(clean)
    var_ratio = np.var(clean) / (np.var(dy) + 1e-10)

    is_stationary = adf["ADF_pvalue"] < 0.05
    consensus = "stationary" if is_stationary else "non-stationary"

    return {
        "name": name, "ADF_stat": adf["ADF_stat"], "ADF_pvalue": adf["ADF_pvalue"],
        "ADF_crit_5pct": adf["ADF_crit_5pct"],
        "lag_used": adf["lag_used"],
        "variance_ratio": var_ratio,
        "consensus": consensus, "n_obs": len(clean),
    }


# ── feature importance (ML-based, supplementary) ─────────────────────

def feature_selection_spearman(X, y, feat_cols):
    results = []
    for col in feat_cols:
        if X[col].std() == 0:
            continue
        rho, pval = spearmanr(X[col], y, nan_policy='omit')
        results.append({"feature": col, "spearman_rho": rho,
                        "p_value": pval, "abs_rho": abs(rho),
                        "direction": "+" if rho > 0 else "-"})
    return pd.DataFrame(results).sort_values("abs_rho", ascending=False)


def feature_selection_lasso(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_scaled, y)
    coef_df = pd.DataFrame({"feature": X.columns, "lasso_coef": lasso.coef_})
    coef_df["selected"] = coef_df["lasso_coef"] != 0
    coef_df["abs_coef"] = np.abs(coef_df["lasso_coef"])
    return coef_df[coef_df["selected"]].sort_values("abs_coef", ascending=False)


def feature_selection_xgboost_shap(X, y, feat_cols):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feat_cols,
                            "shap_importance": mean_shap}).sort_values("shap_importance", ascending=False)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20, plot_type="bar")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    return shap_df, model


# ── rolling prediction ───────────────────────────────────────────────

def rolling_window_prediction(df, target_col="NTU", window_days=30):
    """Predict NTU using rolling-window mean/median from past data."""
    ntu = df[target_col].dropna()
    daily = ntu.resample('1D').mean()
    days = daily.index

    rolling_mean = daily.rolling(window=window_days, min_periods=1).mean()
    rolling_median = daily.rolling(window=window_days, min_periods=1).median()

    return daily, rolling_mean, rolling_median


# ── main ─────────────────────────────────────────────────────────────

def main():
    setup_chinese_font()
    print("=" * 60)
    print("Problem 1: NTU Stationarity Analysis & Prediction with CIs")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/8] Loading 2025 data...")
    df_2025 = load_2025_data()
    target_col = "NTU"
    print(f"  2025 data shape: {df_2025.shape}")
    print(f"  Date range: {df_2025.index.min()} to {df_2025.index.max()}")

    # ── 2. Stationarity tests ────────────────────────────────────────
    print("\n[2/8] Stationarity tests on 2025 NTU...")
    stat_result = stationarity_tests(df_2025[target_col], "NTU (2025)")
    print(f"  ADF p-value:  {stat_result['ADF_pvalue']:.6f}  "
          f"({'stationary' if stat_result['ADF_pvalue'] < 0.05 else 'non-stationary'})")
    print(f"  Variance ratio: {stat_result.get("variance_ratio", "N/A"):.4f}")
    print(f"  Consensus:    {stat_result['consensus']}")

    # ── 3. Global statistics & bootstrap CI ──────────────────────────
    print("\n[3/8] Computing baseline predictions with bootstrap CIs...")
    ntu_values = df_2025[target_col].dropna()
    daily_ntu = df_2025[target_col].resample('1D').mean().dropna()

    mu_mean, lo_mean, hi_mean = bootstrap_confidence_interval(ntu_values.values, np.mean)
    mu_median, lo_median, hi_median = bootstrap_confidence_interval(ntu_values.values, np.median)

    print(f"  Daily NTU mean:    {daily_ntu.mean():.4f}  "
          f"(global bootstrap: {mu_mean:.4f}, 95% CI [{lo_mean:.4f}, {hi_mean:.4f}])")
    print(f"  Daily NTU median:  {daily_ntu.median():.4f}  "
          f"(global bootstrap: {mu_median:.4f}, 95% CI [{lo_median:.4f}, {hi_median:.4f}])")

    # ── 4. Anomaly detection ─────────────────────────────────────────
    print("\n[4/8] Anomaly detection on 2025 NTU...")
    anom_res, anom_mask, anom_index = anomaly_detection_2025(df_2025, target_col)
    print(f"  IF anomalies:  {anom_res['isolation_forest_anomalies']} / {anom_res['total_points']}")
    print(f"  IQR anomalies:  {anom_res['iqr_anomalies']} / {anom_res['total_points']}")
    print(f"  Combined rate:  {anom_res['anomaly_rate']*100:.2f}%")
    print(f"  IQR bounds:    [{anom_res['iqr_lower']:.4f}, {anom_res['iqr_upper']:.4f}]")

    # ── 5. Rolling prediction ────────────────────────────────────────
    print("\n[5/8] Rolling-window prediction...")
    daily, rolling_mean, rolling_median = rolling_window_prediction(df_2025, target_col, window_days=30)

    # ── 6. Feature importance (supplementary ML) ─────────────────────
    print("\n[6/8] Supplementary ML feature importance...")
    X, y, feat_cols, _ = prepare_data(df_2025, target_col)
    print(f"  Feature matrix: {X.shape}, {len(feat_cols)} features")

    spearman_df = feature_selection_spearman(X, y, feat_cols)
    print("\n  --- Top 10 Spearman ---")
    for _, row in spearman_df.head(10).iterrows():
        print(f"  {row['feature']:30s} rho={row['spearman_rho']:+8.4f} (p={row['p_value']:.4f})")
    spearman_df.to_csv(os.path.join(OUT_DIR, "spearman_results.csv"), index=False)

    lasso_df = feature_selection_lasso(X, y)
    print(f"\n  LASSO selected {len(lasso_df)} features")
    lasso_df.to_csv(os.path.join(OUT_DIR, "lasso_results.csv"), index=False)

    shap_df, xgb_model = feature_selection_xgboost_shap(X, y, feat_cols)
    print("\n  --- Top 10 SHAP ---")
    for _, row in shap_df.head(10).iterrows():
        print(f"  {row['feature']:30s} SHAP={row['shap_importance']:.6f}")
    shap_df.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False)

    # Integrated ranking
    top_spearman = set(spearman_df.head(10)["feature"])
    top_shap = set(shap_df.head(10)["feature"])
    lasso_features = set(lasso_df["feature"])

    integrated = []
    for f in feat_cols:
        score = 0
        if f in top_spearman: score += 2
        if f in top_shap: score += 2
        if f in lasso_features: score += 1
        integrated.append({"feature": f, "score": score})
    integrated_df = pd.DataFrame(integrated).sort_values("score", ascending=False)
    top_features = integrated_df[integrated_df["score"] >= 3]["feature"].tolist()
    if not top_features:
        top_features = list(top_spearman & top_shap)[:10]
    print(f"\n  Consensus top features (score>=3): {top_features}")

    # ── 7. XGBoost comparison ────────────────────────────────────────
    print("\n[7/8] XGBoost comparison model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(top_features) > 0:
        X_train_sel, X_test_sel = X_train[top_features], X_test[top_features]
    else:
        X_train_sel, X_test_sel = X_train, X_test

    xgb_test = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 random_state=42, verbosity=0)
    xgb_test.fit(X_train_sel, y_train)
    y_pred_xgb = xgb_test.predict(X_test_sel)

    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"  XGBoost test RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.3f}")

    # ── 8. Predict for 2026 snapshot dates ────────────────────────────
    print("\n[8/8] Predicting NTU for 2026 snapshot dates...")
    df_2026 = load_2026_data()
    target_dates = ['2026-02-01', '2026-02-10', '2026-02-20']

    # Strategy: use 2025 daily statistics + anomaly baseline
    # For each target date, report:
    #   - Global mean/median with bootstrap CI (if stationary)
    #   - Closest-month rolling statistics
    #   - Available 2026 data if the date exists

    pred_rows = []
    for date_str in target_dates:
        target_date = pd.Timestamp(date_str)
        # Check if we have 2026 data near this date
        nearby = df_2026[df_2026.index.date == target_date.date()]
        has_2026_data = len(nearby) > 0

        # Month-specific historical statistics
        month_data = df_2025[df_2025.index.month == target_date.month][target_col].dropna()
        month_mean = month_data.mean() if len(month_data) > 0 else mu_mean
        month_median = month_data.median() if len(month_data) > 0 else mu_median
        month_std = month_data.std() if len(month_data) > 0 else ntu_values.std()

        # 2026 actual if available
        actual_2026 = nearby[target_col].mean() if has_2026_data else np.nan

        pred_rows.append({
            "target_date": date_str,
            "global_mean": mu_mean,
            "global_mean_CI_low": lo_mean,
            "global_mean_CI_high": hi_mean,
            "global_median": mu_median,
            "global_median_CI_low": lo_median,
            "global_median_CI_high": hi_median,
            "month_mean": month_mean,
            "month_median": month_median,
            "month_std": month_std,
            "ntu_2026_actual_mean": actual_2026,
            "has_2026_data": has_2026_data,
            "stationarity": stat_result["consensus"],
        })

    pred_df = pd.DataFrame(pred_rows)
    print("\n  --- NTU Predictions for 2026 Target Dates ---")
    for _, row in pred_df.iterrows():
        print(f"\n  {row['target_date']}:")
        print(f"    Global mean:  {row['global_mean']:.4f}  "
              f"[{row['global_mean_CI_low']:.4f}, {row['global_mean_CI_high']:.4f}]")
        print(f"    Global median:{row['global_median']:.4f}  "
              f"[{row['global_median_CI_low']:.4f}, {row['global_median_CI_high']:.4f}]")
        print(f"    Month ({pd.Timestamp(row['target_date']).month_name()[:3]}) mean: "
              f"{row['month_mean']:.4f} ± {row['month_std']:.4f}")
        if row['has_2026_data']:
            print(f"    2026 actual:  {row['ntu_2026_actual_mean']:.4f}")

    pred_df.to_excel(os.path.join(OUT_DIR, "predicted_NTU_2026.xlsx"), index=False)

    # Also generate 2026 daily prediction using the 2026 data we have
    if len(df_2026) > 0:
        df_2026_daily = df_2026[target_col].resample('1D').mean().dropna()
        print(f"\n  2026 actual daily NTU (from {len(df_2026_daily)} days of data):")
        for d, val in df_2026_daily.items():
            ci_lo = mu_mean - 1.96 * ntu_values.std()
            ci_hi = mu_mean + 1.96 * ntu_values.std()
            is_anomaly = (val < anom_res['iqr_lower']) | (val > anom_res['iqr_upper'])
            flag = " *** ANOMALY ***" if is_anomaly else ""
            print(f"    {d.date()}: {val:.4f}  "
                  f"(95% normal range [{ci_lo:.4f}, {ci_hi:.4f}]){flag}")

    # ── Save outputs ─────────────────────────────────────────────────
    # Stationarity report
    stat_df = pd.DataFrame([stat_result])
    stat_df.to_csv(os.path.join(OUT_DIR, "stationarity_results.csv"), index=False)

    # Anomaly report
    anom_df = pd.DataFrame([anom_res])
    anom_df.to_csv(os.path.join(OUT_DIR, "anomaly_detection.csv"), index=False)

    # Model comparison
    model_comp = pd.DataFrame([{
        "model": "Stationary-Baseline (mean)",
        "prediction": f"{mu_mean:.4f} [{lo_mean:.4f}, {hi_mean:.4f}]",
        "stationarity": stat_result["consensus"],
    }, {
        "model": "Stationary-Baseline (median)",
        "prediction": f"{mu_median:.4f} [{lo_median:.4f}, {hi_median:.4f}]",
        "stationarity": stat_result["consensus"],
    }, {
        "model": "XGBoost (supplementary)",
        "test_RMSE": rmse_xgb,
        "test_R2": r2_xgb,
    }])
    model_comp.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

    # ── Plots ────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. NTU time series with anomaly markers
    ax = axes[0, 0]
    ax.plot(ntu_values.index, ntu_values.values, 'b-', alpha=0.7, linewidth=0.5)
    ax.scatter(ntu_values.index[anom_mask], ntu_values.values[anom_mask],
               c='red', s=15, alpha=0.6, label='Anomaly', zorder=5)
    ax.axhline(y=1.0, color='green', linestyle='--', label="NTU=1.0 (标准)")
    ax.axhline(y=anom_res['iqr_upper'], color='orange', linestyle=':', alpha=0.5,
               label=f"异常阈值={anom_res["iqr_upper"]:.2f}")
    ax.set_title("NTU时间序列与异常检测")
    ax.set_ylabel("NTU")
    ax.legend(fontsize=8)

    # 2. Daily NTU with rolling prediction + bootstrap CI
    ax = axes[0, 1]
    ax.plot(daily.index, daily.values, 'b-', alpha=0.6, linewidth=0.8)
    ax.plot(rolling_mean.index, rolling_mean.values, 'orange', linewidth=1.5,
            label="30天滚动均值")
    ax.fill_between(daily.index, lo_mean, hi_mean, alpha=0.2, color='gray',
                    label="95% 置信区间 (Bootstrap)")
    ax.set_title("每日NTU与滚动预测")
    ax.set_ylabel("NTU")
    ax.legend(fontsize=8)

    # 3. Feature importance (top 15 SHAP)
    ax = axes[1, 0]
    top15 = shap_df.head(15).iloc[::-1]
    ax.barh(range(len(top15)), top15["shap_importance"].values)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].values, fontsize=8)
    ax.set_title("Top 15 特征重要性 (SHAP)")
    ax.set_xlabel("Mean |SHAP|")

    # 4. Stationarity diagnostics: NTU distribution
    ax = axes[1, 1]
    ax.hist(ntu_values.values, bins=80, alpha=0.7, edgecolor='black')
    ax.axvline(x=mu_mean, color='red', linestyle='--',
               label=f"均值={mu_mean:.3f}")
    ax.axvline(x=mu_median, color='blue', linestyle='--',
               label=f"中位数={mu_median:.3f}")
    ax.axvline(x=lo_mean, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=hi_mean, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f"NTU分布直方图\nADF p={stat_result['ADF_pvalue']:.4f} "
                 f"({stat_result['consensus']})")
    ax.set_xlabel("NTU")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "problem1_plots.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
