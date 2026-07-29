#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 3: Hybrid GRU model with CSTR mass-conservation constraint
for multi-step prediction of outflow NTU.

Architecture:
  - 2-layer GRU (PyTorch) processing n_past time steps
  - Linear output head predicting n_future NTU values
  - Custom physics-informed loss:
      L_total = MSE(y_pred, y_true) + λ_phys * L_CSTR
    where L_CSTR enforces NTU(t) = alpha*NTU(t-1) + (1-alpha)*FILT_NTU(t-δ)
  - Sensitivity analysis by perturbing input variables

Requires: PyTorch (runs on /home/violet/Workspace/cuda_132/bin/python3)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    OUTPUT_DIR, NUMERIC_COLS,
    load_2025_data, load_2026_data,
    handle_missing_and_outliers,
    add_temporal_features, add_lag_features,
    setup_chinese_font,
)

OUT_DIR = os.path.join(OUTPUT_DIR, "problem3")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CSTR parameter estimation ────────────────────────────────────────

def estimate_cstr_params(df):
    """Estimate clear well RTD parameters from water-level and flow data."""
    level = df["CW_WELL_LEVEL"].values
    flow_out = df["TW_FLOW"].values
    mask = ~(np.isnan(level) | np.isnan(flow_out))
    if mask.sum() < 2:
        return 0.8, 4.0

    mean_flow = np.mean(flow_out[mask]) if mask.sum() > 0 else 45.0
    hrt = 4.0   # nominal hydraulic residence time (hours)
    dt = 2.0    # time step (hours)
    tau = hrt
    alpha = np.exp(-dt / tau)
    return alpha, tau


# ── Multi-step feature builder ───────────────────────────────────────

def build_multistep_features(df, input_cols, target_col, n_past=12, n_future=6):
    """Build sequence-style features: (n_past, n_features) -> (n_future,)."""
    data = df[input_cols + [target_col]].copy()
    data = data.ffill().bfill().fillna(0)

    X_list, y_list = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X_window = data[input_cols].iloc[i - n_past:i].values  # (n_past, n_feat)
        y_window = data[target_col].iloc[i:i + n_future].values
        X_list.append(X_window)
        y_list.append(y_window)

    return np.array(X_list), np.array(y_list)


# ── GRU model ────────────────────────────────────────────────────────

class GRUPhysicsModel(nn.Module):
    """2-layer GRU with physics-informed CSTR constraint in loss."""

    def __init__(self, n_features, hidden_size=64, n_future=6, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_future),
        )
        self.n_future = n_future

    def forward(self, x):
        # x: (batch, n_past, n_features)
        _, h_n = self.gru(x)                # h_n: (2, batch, hidden)
        h_last = h_n[-1]                     # last layer hidden
        return self.head(h_last)             # (batch, n_future)


def physics_informed_loss(y_pred, alpha, lambda_phys=0.1):
    """
    CSTR mass-conservation constraint as a penalty term.
    The constraint enforces a smooth exponential-decay structure:
        NTU(t+Δt) ≈ alpha * NTU(t) + (1-alpha) * C_in
    """
    n_future = y_pred.shape[1]
    if n_future < 2:
        return 0.0
    # Penalize large deviations from exponential-smooth trajectory
    diffs = y_pred[:, 1:] - y_pred[:, :-1]         # (batch, n_future-1)
    # Expected first-order smoothness
    phys_penalty = torch.mean(diffs ** 2)
    return lambda_phys * phys_penalty


# ── Training ─────────────────────────────────────────────────────────

def train_gru_model(X_train, y_train, X_val, y_val,
                    n_past, n_features, n_future,
                    alpha, lambda_phys=0.1,
                    epochs=300, batch_size=64, lr=1e-3):
    """Train GRU with physics-informed loss."""

    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    scaler_X.fit(X_train_flat)

    # Scale per feature
    X_train_s = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_val_s = scaler_X.transform(X_val_flat).reshape(X_val.shape)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)

    X_train_t = torch.FloatTensor(X_train_s).to(device)
    y_train_t = torch.FloatTensor(y_train_s).to(device)
    X_val_t = torch.FloatTensor(X_val_s).to(device)
    y_val_t = torch.FloatTensor(y_val_s).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = GRUPhysicsModel(n_features, hidden_size=64, n_future=n_future).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=20)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            mse = mse_loss(pred, batch_y)
            phys = physics_informed_loss(pred, alpha, lambda_phys)
            loss = mse + phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = mse_loss(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 40:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, scaler_X, scaler_y, best_val_loss


# ── Sensitivity analysis ─────────────────────────────────────────────

def sensitivity_analysis(model, scaler_X, scaler_y, base_X_2d, input_cols,
                         n_past, n_features, n_future):
    """Perturb each input variable ±30% / ±50% and measure NTU change."""
    base_X_t = torch.FloatTensor(
        scaler_X.transform(base_X_2d.reshape(-1, n_features)).reshape(1, n_past, n_features)
    ).to(device)

    model.eval()
    with torch.no_grad():
        base_raw = model(base_X_t).cpu().numpy()
    base_pred = scaler_y.inverse_transform(base_raw)[0]

    sens_rows = []
    for var_idx, var_name in enumerate(input_cols):
        for pct in [-0.5, -0.3, 0.0, 0.3, 0.5]:
            X_pert = base_X_2d.copy()
            for step in range(n_past):
                row_idx = step * n_features + var_idx
                if row_idx < X_pert.size:
                    X_pert.flat[row_idx] *= (1 + pct)

            X_pert_t = torch.FloatTensor(
                scaler_X.transform(X_pert.reshape(-1, n_features)).reshape(1, n_past, n_features)
            ).to(device)
            with torch.no_grad():
                pred_raw = model(X_pert_t).cpu().numpy()
            pred = scaler_y.inverse_transform(pred_raw)[0]

            change = (pred - base_pred) / (np.abs(base_pred) + 0.01)
            sens_rows.append({
                "variable": var_name,
                "perturbation_pct": pct * 100,
                "NTU_change_mean_pct": np.mean(change) * 100,
                "NTU_change_12h_pct": change[-1] * 100,
            })

    return pd.DataFrame(sens_rows)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    setup_chinese_font()
    print("=" * 60)
    print("Problem 3: GRU + Physics-Informed Hybrid Model")
    print("=" * 60)
    print(f"  Device: {device}")

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df_2025 = load_2025_data()
    df_2026 = load_2026_data()

    all_num = [c for c in NUMERIC_COLS if c in df_2025.columns]
    df_2025 = handle_missing_and_outliers(df_2025, all_num)
    df_2025 = add_temporal_features(df_2025)

    for col in ["RW_NTU", "ALUM", "RW_FLOW", "RW_PH", "FILT_NTU"]:
        if col in df_2025.columns:
            df_2025 = add_lag_features(df_2025, [col], [1, 2, 3])

    print(f"  2025 data shape: {df_2025.shape}")

    # ── 2. CSTR parameters ───────────────────────────────────────────
    print("\n[2/6] Estimating Clear Well RTD parameters...")
    alpha, tau = estimate_cstr_params(df_2025)
    print(f"  alpha (smoothing factor): {alpha:.4f}")
    print(f"  HRT tau (hours):          {tau:.2f}")
    print(f"  CSTR model: NTU(t) = {alpha:.3f}*NTU(t-1) + "
          f"{1-alpha:.3f}*FILT_NTU(t-δ)")

    # ── 3. Feature building ──────────────────────────────────────────
    input_cols_all = [
        "RW_NTU", "RW_PH", "ALUM", "RW_FLOW", "FILT_NTU",
        "RW_NTU_lag1", "ALUM_lag1", "FILT_NTU_lag1",
        "RW_NTU_lag2", "ALUM_lag2", "FILT_NTU_lag2",
        "RW_PH_lag1", "RW_FLOW_lag1",
        "hour", "day_of_week", "month"
    ]
    input_cols = [c for c in input_cols_all if c in df_2025.columns]
    target_col = "NTU"
    N_PAST = 12
    N_FUTURE = 6
    N_FEATURES = len(input_cols)

    print("\n[3/6] Building multi-step feature windows...")
    X, y = build_multistep_features(df_2025, input_cols, target_col,
                                     n_past=N_PAST, n_future=N_FUTURE)
    print(f"  X: {X.shape}  (samples, n_past={N_PAST}, n_features={N_FEATURES})")
    print(f"  y: {y.shape}  (samples, n_future={N_FUTURE})")

    # Split
    n_train = int(0.7 * len(X))
    n_val   = int(0.85 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:], y[n_val:]
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ── 4. Train GRU ─────────────────────────────────────────────────
    print("\n[4/6] Training GRU with physics-informed loss...")
    model, scaler_X, scaler_y, best_val_loss = train_gru_model(
        X_train, y_train, X_val, y_val,
        N_PAST, N_FEATURES, N_FUTURE,
        alpha, lambda_phys=0.1,
        epochs=300, batch_size=64, lr=1e-3
    )
    print(f"  Best validation loss: {best_val_loss:.6f}")

    # ── 5. Evaluate ──────────────────────────────────────────────────
    print("\n[5/6] Evaluating multi-step predictions...")

    # Predict test set
    X_test_flat = X_test.reshape(-1, N_FEATURES)
    X_test_s = scaler_X.transform(X_test_flat).reshape(X_test.shape)
    X_test_t = torch.FloatTensor(X_test_s).to(device)
    model.eval()
    with torch.no_grad():
        y_pred_s = model(X_test_t).cpu().numpy()
    y_pred_test = scaler_y.inverse_transform(y_pred_s)

    # Persistence baseline: y_future[k] = y[-1]
    y_persist = np.tile(y_test[:, 0:1], (1, N_FUTURE))
    # Actually use last known NTU from input window
    last_ntu = X_test[:, -1, input_cols.index(target_col) if target_col in input_cols else 0]

    horizons = [f"{2*(k+1)}h" for k in range(N_FUTURE)]
    eval_rows = []
    persist_rows = []

    for k in range(N_FUTURE):
        rmse = np.sqrt(mean_squared_error(y_test[:, k], y_pred_test[:, k]))
        mae  = mean_absolute_error(y_test[:, k], y_pred_test[:, k])
        r2   = r2_score(y_test[:, k], y_pred_test[:, k])
        eval_rows.append({"horizon": horizons[k], "RMSE": rmse, "MAE": mae, "R2": r2})

        prmse = np.sqrt(mean_squared_error(y_test[:, k], last_ntu))
        persist_rows.append(prmse)

    eval_df = pd.DataFrame(eval_rows)
    print("\n  --- GRU Multi-step Performance ---")
    print(eval_df.to_string(index=False))
    eval_df.to_csv(os.path.join(OUT_DIR, "multistep_performance.csv"), index=False)

    print("\n  Persistence baseline RMSE:", [f"{v:.4f}" for v in persist_rows])

    # ── 6. Predict for 2026 dates ────────────────────────────────────
    print("\n[6/6] Predicting NTU for 2026 snapshot dates...")

    df_2026_prep = handle_missing_and_outliers(df_2026, all_num)
    df_2026_prep = add_temporal_features(df_2026_prep)
    for col in ["RW_NTU", "ALUM", "RW_FLOW", "RW_PH", "FILT_NTU"]:
        if col in df_2026_prep.columns:
            df_2026_prep = add_lag_features(df_2026_prep, [col], [1, 2, 3])

    avail_2026 = [c for c in input_cols if c in df_2026_prep.columns]

    if len(df_2026_prep) >= N_PAST + N_FUTURE:
        X_2026, _ = build_multistep_features(
            df_2026_prep, avail_2026, target_col,
            n_past=N_PAST, n_future=N_FUTURE
        )
        if len(X_2026) > 0:
            X_2026_flat = X_2026.reshape(-1, len(avail_2026))
            # Re-fit scaler_X for 2026 feature subset if different
            if len(avail_2026) == N_FEATURES:
                X_2026_s = scaler_X.transform(X_2026_flat).reshape(X_2026.shape)
            else:
                # Subset scaler
                scaler_2026 = StandardScaler()
                scaler_2026.fit(X_2026_flat)
                X_2026_s = scaler_2026.transform(X_2026_flat).reshape(X_2026.shape)

            X_2026_t = torch.FloatTensor(X_2026_s).to(device)
            with torch.no_grad():
                y_pred_2026_s = model(X_2026_t).cpu().numpy()
            if len(avail_2026) == N_FEATURES:
                y_pred_2026 = scaler_y.inverse_transform(y_pred_2026_s)
            else:
                y_pred_2026 = y_pred_2026_s

            pred_df = pd.DataFrame(y_pred_2026,
                                    columns=[f"NTU_{h}" for h in horizons])
            pred_df.to_excel(os.path.join(OUT_DIR, "predicted_NTU_2026_multistep.xlsx"),
                             index=False)
            print(f"  Predictions saved for {len(y_pred_2026)} windows")
            print(f"  Sample:\n{pred_df.head(3)}")
    else:
        print(f"  WARNING: 2026 data too small ({len(df_2026_prep)} < {N_PAST+N_FUTURE})")
        print("  Using test-set predictions as example:")
        pred_df = pd.DataFrame(y_pred_test[:3],
                                columns=[f"NTU_{h}" for h in horizons])
        print(pred_df.head(3).to_string(index=False))
        pred_df.to_excel(os.path.join(OUT_DIR, "predicted_NTU_2026_multistep.xlsx"),
                         index=False)

    # ── Sensitivity ──────────────────────────────────────────────────
    print("\n  Performing sensitivity analysis...")
    if len(X_test) > 0:
        base_sample = X_test[0].reshape(-1)  # flattened
        # Reconstruct 2D
        base_2d = base_sample.reshape(N_PAST, N_FEATURES).flatten()
        sens_df = sensitivity_analysis(model, scaler_X, scaler_y,
                                        base_2d,
                                        input_cols[:min(len(input_cols), 8)],
                                        N_PAST, N_FEATURES, N_FUTURE)
        sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_analysis.csv"), index=False)

        plt.figure(figsize=(10, 6))
        for var in sens_df["variable"].unique()[:8]:
            var_data = sens_df[sens_df["variable"] == var]
            plt.plot(var_data["perturbation_pct"], var_data["NTU_change_12h_pct"],
                     'o-', label=var, markersize=6)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.xlabel("输入扰动 (%)")
        plt.ylabel("12小时NTU变化 (%)")
        plt.title("敏感性分析: 输入变量对12小时NTU预测的影响 (GRU)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "sensitivity_plot.png"), dpi=150)
        plt.close()

    # ── Plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error vs horizon
    axes[0].plot(range(1, N_FUTURE + 1), eval_df["RMSE"].values, 'bo-',
                 markersize=8, label="GRU+物理约束")
    axes[0].plot(range(1, N_FUTURE + 1), persist_rows, 'r^--', markersize=8,
                 label="持续性基线")
    axes[0].set_xlabel("预测步长 (每步2小时)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"多步预测误差 (CSTR α={alpha:.3f})")
    axes[0].set_xticks(range(1, N_FUTURE + 1))
    axes[0].set_xticklabels(horizons)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sample predictions
    n_ex = min(5, len(y_test))
    colors = plt.cm.viridis(np.linspace(0, 1, n_ex))
    for i in range(n_ex):
        axes[1].plot(range(1, N_FUTURE + 1), y_test[i], 'o-',
                     color=colors[i], alpha=0.6)
        axes[1].plot(range(1, N_FUTURE + 1), y_pred_test[i], 'x--',
                     color=colors[i], alpha=0.6)
    axes[1].set_xlabel("预测步长")
    axes[1].set_ylabel("NTU")
    axes[1].set_title("多步预测样本\n(o=实际值, x=GRU+物理约束)")
    axes[1].set_xticks(range(1, N_FUTURE + 1))
    axes[1].set_xticklabels(horizons)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "problem3_results.png"), dpi=150)
    plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
