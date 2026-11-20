
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

print("==========================================================")
print("   INDUSTRIAL POWER LOAD FORECASTING & SHAP PIPELINE      ")
print("==========================================================\n")

# 1. Generate Realistic Hourly Industrial Feeder Telemetry Dataset
rng = np.random.default_rng(42)
n_hours = 24 * 120  # 120 days (2,880 hourly intervals)
timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")

hour = timestamps.hour
day_of_week = timestamps.dayofweek
is_weekend = (day_of_week >= 5).astype(int)

# Ambient temperature (°C) with diurnal cycle + seasonal drift
ambient_temp_c = (
    26.0
    + 8.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    + 5.0 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365))
    + rng.normal(0, 1.8, n_hours)
)

# Active industrial shift factor & voltage stability
shift_load_factor = np.where(is_weekend == 1, 0.58, np.where((hour >= 8) & (hour <= 20), 1.18, 0.88))
bus_voltage_kv = 132.0 + rng.normal(0, 1.4, n_hours) - 0.04 * np.maximum(0, ambient_temp_c - 32)
reactive_power_mvar = 14.0 * shift_load_factor + rng.normal(0, 1.5, n_hours)

# Target: Active Power Load (MW)
base_load_mw = 85.0
load_mw = (
    base_load_mw * shift_load_factor
    + 1.35 * np.maximum(0, ambient_temp_c - 24)
    + 0.65 * reactive_power_mvar
    + 10.0 * np.sin(2 * np.pi * (hour - 14) / 24)
    + rng.normal(0, 2.5, n_hours)
)

df = pd.DataFrame({
    "timestamp": timestamps,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "ambient_temp_c": np.round(ambient_temp_c, 2),
    "bus_voltage_kv": np.round(bus_voltage_kv, 2),
    "reactive_power_mvar": np.round(reactive_power_mvar, 2),
    "load_mw": np.round(load_mw, 2),
})

# 2. Time-Series Lag, Rolling & Thermal Stress Feature Engineering
df["load_lag_1h"] = df["load_mw"].shift(1)
df["load_lag_24h"] = df["load_mw"].shift(24)
df["load_lag_168h"] = df["load_mw"].shift(168)
df["load_roll_mean_24h"] = df["load_mw"].shift(1).rolling(24).mean()
df["load_roll_std_24h"] = df["load_mw"].shift(1).rolling(24).std()
df["temp_roll_mean_12h"] = df["ambient_temp_c"].rolling(12).mean()
df["apparent_power_mva"] = np.sqrt(df["load_lag_1h"] ** 2 + df["reactive_power_mvar"] ** 2)
df["thermal_load_index"] = df["ambient_temp_c"] * (df["load_lag_1h"] / 100.0)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df_model = df.dropna().reset_index(drop=True)
df_model.to_csv("data/industrial_load_telemetry.csv", index=False)
print(f"[INFO] Saved telemetry dataset with {df_model.shape[0]} rows and {df_model.shape[1]} columns.")

# 3. Chronological Train / Test Split (Last 20% for out-of-time evaluation)
feature_cols = [
    "hour", "day_of_week", "is_weekend", "ambient_temp_c",
    "bus_voltage_kv", "reactive_power_mvar", "load_lag_1h",
    "load_lag_24h", "load_lag_168h", "load_roll_mean_24h",
    "load_roll_std_24h", "temp_roll_mean_12h", "apparent_power_mva",
    "thermal_load_index", "hour_sin", "hour_cos"
]
target_col = "load_mw"

split_idx = int(len(df_model) * 0.8)
X_train, y_train = df_model[feature_cols].iloc[:split_idx], df_model[target_col].iloc[:split_idx]
X_test, y_test = df_model[feature_cols].iloc[split_idx:], df_model[target_col].iloc[split_idx:]
ts_test = df_model["timestamp"].iloc[split_idx:]

# 4. Train LightGBM & XGBoost Ensemble
model_lgb = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
)
pred_lgb = model_lgb.predict(X_test)

if HAS_XGB:
    model_xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred_xgb = model_xgb.predict(X_test)
    y_pred = 0.5 * pred_lgb + 0.5 * pred_xgb
else:
    y_pred = pred_lgb

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"[METRICS] Holdout Test -> RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | R2: {r2:.4f}")

# Save forecast audit
forecast_out = pd.DataFrame({
    "timestamp": ts_test.values,
    "actual_load_mw": y_test.values,
    "predicted_load_mw": np.round(y_pred, 2),
    "residual_mw": np.round(y_test.values - y_pred, 2)
})
forecast_out.to_csv("data/load_forecast_results.csv", index=False)

# 5. Visual 1: Actual vs. Predicted Load Forecast
plt.figure(figsize=(11, 5))
plt.plot(ts_test.iloc[:240], y_test.iloc[:240], label="Actual Load (MW)", linewidth=1.8)
plt.plot(ts_test.iloc[:240], y_pred[:240], label="Ensemble Forecast (MW)", linestyle="--", linewidth=1.8)
plt.title("1. Industrial Feeder Load Forecast (First 10 Days of Holdout Window)", fontsize=11, fontweight="bold")
plt.xlabel("Timestamp")
plt.ylabel("Active Power Load (MW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/load_forecast_comparison.png", dpi=300)
plt.close()

# 6. Visual 2: Residual Distribution
residuals = y_test.values - y_pred
plt.figure(figsize=(9, 5))
plt.hist(residuals, bins=35, edgecolor="black", alpha=0.8)
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
plt.title("2. Out-of-Time Forecast Residual Distribution (MW)", fontsize=11, fontweight="bold")
plt.xlabel("Prediction Error (Actual - Predicted MW)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/load_residual_distribution.png", dpi=300)
plt.close()

# 7. Visual 3: SHAP Global Feature Importance
X_shap = X_test.sample(n=min(400, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer(X_shap)

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title("3. SHAP Global Feature Importance for Industrial Load Forecast", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/shap_load_drivers.png", dpi=300)
plt.close()

print("[SUCCESS] Generated telemetry data, forecast CSV, and 3 diagnostic charts in 'visuals/'!")
EOF