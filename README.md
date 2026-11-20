# Industrial Power Grid & Feeder Load Forecasting (ML & SHAP)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Forecasting-green)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red)](https://shap.readthedocs.io/)

## Project Overview
An end-to-end time-series machine learning pipeline for **hourly industrial feeder load forecasting**. It engineers lag features (`1h`, `24h`, `168h`), rolling statistics, apparent power (`MVA`), and thermal load stress indicators, trains a `LightGBM` + `XGBoost` ensemble on a strict chronological split, and explains feature drivers using **SHAP**.

## Visual Diagnostics

### 1. Actual vs. Forecasted Load (MW)
![Actual vs Forecasted Load](visuals/load_forecast_comparison.png)

### 2. Forecast Residual Distribution
![Residual Distribution](visuals/load_residual_distribution.png)

### 3. SHAP Global Feature Importance
![SHAP Feature Importance](visuals/shap_load_drivers.png)

## Quickstart
```bash
pip install -r requirements.txt
python src/power_load_forecasting.py
